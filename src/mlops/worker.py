import asyncio
import csv
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from aio_pika.exceptions import AMQPError
from faststream import AckPolicy
from faststream.asgi import AsgiFastStream, make_ping_asgi
from faststream.rabbit import Channel, RabbitBroker, RabbitMessage, RabbitQueue
from faststream.rabbit.opentelemetry import RabbitTelemetryMiddleware
from faststream.rabbit.schemas import QueueType
from opentelemetry import propagate, trace
from pamqp.commands import Basic
from sqlalchemy.dialects.postgresql import insert

from mlops.config import Settings
from mlops.core import telemetry
from mlops.core.logging import configure_logging, task_id
from mlops.db import task_results
from mlops.db.session import create_db_engine
from mlops.messaging import TaskMessage
from mlops.model import load_registered_model, single_binary_prediction
from mlops.producer.metrics import peak_rss_mb
from mlops.producer.publisher import PublishUncertain, QueueConfig, retry_delay

settings = Settings()
configure_logging()
telemetry.configure_telemetry("mlops-worker")
queue = QueueConfig(settings.queue_name, settings.queue_max_length or None)
broker = RabbitBroker(
    settings.rabbitmq_url,
    logger=logging.getLogger("mlops.broker"),
    log_level=logging.getLogger().level,
    middlewares=[RabbitTelemetryMiddleware()],
    graceful_timeout=settings.inference_delay_seconds + 30,
    default_channel=Channel(
        prefetch_count=settings.prefetch_count, publisher_confirms=True, on_return_raises=True
    ),
)
engine = loaded = writer = metrics_file = None
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan():
    global engine, loaded, writer, metrics_file
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
    engine = create_db_engine()
    handle = None
    try:
        loaded = await asyncio.to_thread(load_registered_model, settings)
        logger.info(
            "worker_started",
            extra={
                "model_alias": settings.mlflow_model_alias,
                "model_version": loaded.version,
                "prefetch_count": settings.prefetch_count,
            },
        )
        await broker.connect()
        await broker.declare_queue(RabbitQueue(queue.name + ".dlq", durable=True))
        await broker.declare_queue(
            RabbitQueue(
                queue.name + ".retry",
                QueueType.QUORUM,
                arguments={
                    "x-message-ttl": 100,
                    "x-overflow": "reject-publish",
                    "x-dead-letter-exchange": "",
                    "x-dead-letter-routing-key": queue.name,
                    "x-dead-letter-strategy": "at-least-once",
                },
            )
        )
        if settings.worker_metrics_file:
            path = Path(settings.worker_metrics_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = await asyncio.to_thread(path.open, "w", newline="", encoding="utf-8")
            writer = csv.writer(handle, lineterminator="\n")
            metrics_file = handle
            writer.writerow(
                [
                    "timestamp_utc",
                    "worker_id",
                    "task_id",
                    "task_latency_seconds",
                    "processing_seconds",
                    "worker_peak_rss_mb",
                ]
            )
            handle.flush()
        yield
    finally:
        if handle:
            handle.close()
        writer = None
        await engine.dispose()
        logger.info("shutdown_completed")


app = AsgiFastStream(
    broker, lifespan=lifespan, asgi_routes=[("/health", make_ping_asgi(broker, timeout=1))]
)


@app.on_shutdown
async def stopping():
    logger.info("shutdown_started")


@app.after_startup
async def ready():
    if settings.worker_ready_file:
        await asyncio.to_thread(
            Path(settings.worker_ready_file).write_text, "ready", encoding="utf-8"
        )


@broker.subscriber(
    RabbitQueue(queue.name, durable=True, arguments=queue.arguments),
    ack_policy=AckPolicy.MANUAL,
    decoder=lambda message: message.body,
)
async def handle_task(body: bytes, message: RabbitMessage) -> None:
    token = task_id.set(message.message_id)
    attempt = message.headers.get("attempt", 0)
    delivery_attempt = attempt + 1 if isinstance(attempt, int) and attempt >= 0 else 3
    span = trace.get_current_span()
    span.set_attribute("task.id", message.message_id or "unknown")
    span.set_attribute("messaging.delivery.attempt", delivery_attempt)
    telemetry.in_flight.add(1)
    if attempt or getattr(message.raw_message, "redelivered", False):
        telemetry.redeliveries.add(1)
    try:
        age = telemetry.message_age(message.headers, "published_at_epoch")
        if age is not None:
            now = time.time_ns()
            start_ns = max(0, now - int(age * 1e9))
            waiting = telemetry.tracer.start_span(
                "queue.wait",
                context=propagate.extract(message.headers),
                start_time=start_ns,
                attributes={"task.id": message.message_id or "unknown"},
            )
            waiting.end(end_time=now)
        logger.info("task_received", extra={"attempt": delivery_attempt})
        await _handle_task(body, message)
    finally:
        telemetry.in_flight.add(-1)
        task_id.reset(token)


async def _handle_task(body: bytes, message: RabbitMessage) -> None:
    attempt = message.headers.get("attempt", 0)
    attempt = attempt if isinstance(attempt, int) and attempt >= 0 else 2
    try:
        task = TaskMessage(**json.loads(body))
        start = time.monotonic()
        try:
            with telemetry.tracer.start_as_current_span(
                "inference",
                attributes={
                    "model.alias": settings.mlflow_model_alias,
                    "model.version": loaded.version,
                    "messaging.delivery.attempt": attempt + 1,
                },
            ):
                await asyncio.sleep(settings.inference_delay_seconds)
                # to_thread copies the current contextvars, including the active OTel span.
                prediction = single_binary_prediction(
                    await asyncio.to_thread(
                        loaded.model.predict, pd.DataFrame({"text": [task.text]})
                    )
                )
        finally:
            duration = time.monotonic() - start
            telemetry.inference_duration.record(duration)
        with telemetry.tracer.start_as_current_span("db.save_result"):
            async with engine.begin() as connection:
                saved = await connection.scalar(
                    insert(task_results)
                    .values(
                        task_id=task.task_id,
                        text=task.text,
                        prediction=prediction,
                        model_name=loaded.name,
                        model_version=loaded.version,
                        processing_seconds=duration,
                    )
                    .on_conflict_do_nothing(index_elements=["task_id"])
                    .returning(task_results.c.task_id)
                )
    except Exception as error:
        telemetry.record_error(error)
        telemetry.processed.add(1, {"result": "error"})
        logger.exception("task_failed", extra={"attempt": attempt + 1})
        target = queue.name + (".retry" if attempt < 2 else ".dlq")
        await transfer_failed_task(body, message, target, attempt + 1)
        if attempt >= 2:
            telemetry.processed.add(1, {"result": "dlq"})
        logger.warning(
            "task_dlq" if attempt >= 2 else "task_retry",
            extra={"attempt": attempt + 1, "target": target},
        )
    else:
        telemetry.processed.add(1, {"result": "ok" if saved else "duplicate"})
        age = telemetry.message_age(message.headers, "created_at_epoch")
        if age is not None:
            telemetry.task_duration.record(age)
        logger.info(
            "task_completed",
            extra={
                "model_version": loaded.version,
                "duration_seconds": duration,
                "duplicate": not bool(saved),
            },
        )
        if saved and writer:
            writer.writerow(
                [
                    datetime.now(UTC).isoformat(),
                    os.getpid(),
                    task.task_id,
                    age if age is not None else "",
                    duration,
                    peak_rss_mb(),
                ]
            )
            metrics_file.flush()
    await message.ack()


async def transfer_failed_task(
    body: bytes, message: RabbitMessage, target: str, attempt: int
) -> None:
    """Keep one delivery in flight until its persistent handoff is confirmed.

    Transport retries do not consume inference attempts. Cancellation propagates
    so shutdown can close the channel and return the unacked delivery to RabbitMQ.
    """
    failures = 0
    headers = {**message.headers, "attempt": attempt, "published_at_epoch": repr(time.time())}
    while True:
        try:
            async with asyncio.timeout(3):
                confirmation = await broker.publish(
                    body,
                    queue=target,
                    persist=True,
                    timeout=3,
                    content_type="application/json",
                    message_id=message.message_id,
                    headers=headers,
                )
            if not isinstance(confirmation, Basic.Ack):
                raise PublishUncertain(f"Retry/DLQ publication not confirmed: {target}")
            return
        except (AMQPError, PublishUncertain, TimeoutError, ConnectionError, OSError) as error:
            failures += 1
            delay = retry_delay(failures)
            logger.warning(
                "Handoff failed target=%s error=%s; retry in %.3fs",
                target,
                type(error).__name__,
                delay,
            )
            await asyncio.sleep(delay)
