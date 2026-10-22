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
from faststream import AckPolicy
from faststream.asgi import AsgiFastStream, make_ping_asgi
from faststream.rabbit import Channel, RabbitBroker, RabbitMessage, RabbitQueue
from faststream.rabbit.schemas import QueueType
from pamqp.commands import Basic
from sqlalchemy.dialects.postgresql import insert

from mlops.config import Settings
from mlops.db import task_results
from mlops.db.session import create_db_engine
from mlops.messaging import TaskMessage
from mlops.model import load_registered_model, single_binary_prediction
from mlops.producer.metrics import peak_rss_mb
from mlops.producer.publisher import PublishUncertain, QueueConfig

settings = Settings()
queue = QueueConfig(settings.queue_name, settings.queue_max_length or None)
broker = RabbitBroker(
    settings.rabbitmq_url,
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


app = AsgiFastStream(
    broker, lifespan=lifespan, asgi_routes=[("/health", make_ping_asgi(broker, timeout=1))]
)


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
    attempt = message.headers.get("attempt", 0)
    attempt = attempt if isinstance(attempt, int) and attempt >= 0 else 2
    try:
        task = TaskMessage(**json.loads(body))
        start = time.monotonic()
        await asyncio.sleep(settings.inference_delay_seconds)
        prediction = single_binary_prediction(
            await asyncio.to_thread(loaded.model.predict, pd.DataFrame({"text": [task.text]}))
        )
        duration = time.monotonic() - start
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
        target = queue.name + (".retry" if attempt < 2 else ".dlq")
        confirmation = await broker.publish(
            body,
            queue=target,
            persist=True,
            timeout=3,
            content_type="application/json",
            message_id=message.message_id,
            headers={**message.headers, "attempt": attempt + 1},
        )
        if not isinstance(confirmation, Basic.Ack):
            raise PublishUncertain(f"Retry/DLQ publication not confirmed: {target}") from error
        logger.warning(
            "Task failed attempt=%s target=%s error=%s", attempt + 1, target, type(error).__name__
        )
    else:
        if saved and writer:
            created = message.headers.get("created_at_epoch")
            writer.writerow(
                [
                    datetime.now(UTC).isoformat(),
                    os.getpid(),
                    task.task_id,
                    max(0, time.time() - float(created)) if created else "",
                    duration,
                    peak_rss_mb(),
                ]
            )
            metrics_file.flush()
    await message.ack()
