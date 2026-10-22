import argparse
import asyncio
import csv
import json
import os
import signal
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from mlops.messaging import TaskMessage
from mlops.producer.metrics import peak_rss_mb
from mlops.producer.publisher import QueueConfig


async def consume(
    *,
    rabbitmq_url: str,
    queue_name: str,
    max_length: int | None,
    delay_seconds: float,
    worker_id: str,
    results_file: str,
    ready_file: str | None = None,
) -> None:
    import aio_pika

    if delay_seconds <= 0:
        raise ValueError("delay_seconds must be positive")
    queue_config = QueueConfig(queue_name, max_length)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    path = Path(results_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    in_flight: set[asyncio.Task] = set()
    connection = await aio_pika.connect_robust(rabbitmq_url)
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        queue = await channel.declare_queue(
            queue_config.name, durable=True, arguments=queue_config.arguments
        )
        handle = await asyncio.to_thread(path.open, "w", newline="", encoding="utf-8")
        with handle:
            writer = csv.writer(handle)
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

            async def on_message(message: aio_pika.IncomingMessage) -> None:
                current = asyncio.current_task()
                if current is not None:
                    in_flight.add(current)
                try:
                    async with message.process(requeue=True):
                        task = TaskMessage(**json.loads(message.body))
                        processing_start = time.monotonic()
                        await asyncio.sleep(delay_seconds)
                        processing_seconds = time.monotonic() - processing_start
                        created = (message.headers or {}).get("created_at_epoch")
                        latency = max(0.0, time.time() - float(created)) if created else ""
                        writer.writerow(
                            [
                                datetime.now(UTC).isoformat(),
                                worker_id,
                                task.task_id,
                                latency,
                                round(processing_seconds, 4),
                                round(peak_rss_mb(), 3),
                            ]
                        )
                        handle.flush()
                finally:
                    if current is not None:
                        in_flight.discard(current)

            consumer_tag = await queue.consume(on_message, no_ack=False)
            if ready_file:
                await asyncio.to_thread(Path(ready_file).write_text, "ready", encoding="utf-8")
            print(
                f"Demo worker {worker_id}: queue={queue_name}, delay={delay_seconds}s", flush=True
            )
            await stop.wait()
            await queue.cancel(consumer_tag)
            if in_flight:
                await asyncio.gather(*in_flight)
            print(f"Demo worker {worker_id}: stopped gracefully", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic slow consumer for backpressure demos")
    parser.add_argument("--queue", default="inference_tasks")
    parser.add_argument("--mode", choices=("bounded", "baseline"), default="bounded")
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--worker-id", default="worker-1")
    parser.add_argument("--results-file", default="metrics/demo_worker.csv")
    parser.add_argument("--ready-file", default=None)
    parser.add_argument(
        "--rabbitmq-url", default=os.getenv("RABBITMQ_URL", "amqp://guest:guest@127.0.0.1:5672/")
    )
    args = parser.parse_args()
    asyncio.run(
        consume(
            rabbitmq_url=args.rabbitmq_url,
            queue_name=args.queue,
            max_length=args.max_length if args.mode == "bounded" else None,
            delay_seconds=args.delay,
            worker_id=args.worker_id,
            results_file=args.results_file,
            ready_file=args.ready_file,
        )
    )


if __name__ == "__main__":
    main()
