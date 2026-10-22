import argparse
import asyncio
import csv
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from mlops.producer.metrics import MetricsSampler
from mlops.producer.publisher import PublishStats, QueueConfig, RabbitPublisher, publish_test_tasks

SCENARIOS = {"baseline": ("baseline", 1), "bounded1": ("bounded", 1), "bounded2": ("bounded", 2)}


def read_results(directory: Path, name: str) -> list[dict[str, str]]:
    rows = []
    for path in sorted(directory.glob(f"{name}_worker*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def check_workers(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is not None:
            raise RuntimeError(f"Worker exited early: {process.returncode}; inspect worker logs")


async def run_scenario(args: argparse.Namespace) -> None:
    name = args.scenario
    mode, workers = SCENARIOS[name]
    directory = args.output_dir
    queue = f"bp_exp_{name}_{uuid4().hex[:12]}"
    max_length = args.max_length if mode == "bounded" else None
    config = QueueConfig(queue, max_length)
    processes = []
    logs = []
    ready_files = []
    stats = PublishStats()
    started = time.monotonic()
    drained = False
    consumed = []
    print(f"{name}: FastStream/MLflow worker, workers={workers}, queue={queue}", flush=True)

    async with RabbitPublisher(args.rabbitmq_url, config) as observer:
        try:
            for index in range(1, workers + 1):
                log = (directory / f"{name}_worker{index}.log").open("w")
                logs.append(log)
                ready = directory / f"{name}_worker{index}.ready"
                ready_files.append(ready)
                command = [sys.executable, "-m", "uvicorn", "mlops.worker:app", "--port", "0"]
                worker_env = dict(
                    os.environ,
                    RABBITMQ_URL=args.rabbitmq_url,
                    QUEUE_NAME=queue,
                    QUEUE_MAX_LENGTH=str(max_length or 0),
                    INFERENCE_DELAY_SECONDS=str(args.delay),
                    WORKER_METRICS_FILE=str(directory / f"{name}_worker{index}.csv"),
                    WORKER_READY_FILE=str(ready),
                    MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD="false",
                )
                processes.append(
                    await asyncio.to_thread(
                        subprocess.Popen, command, stdout=log, stderr=log, env=worker_env
                    )
                )
            async with asyncio.timeout(min(args.timeout, 30)):
                while not all(path.exists() for path in ready_files):
                    check_workers(processes)
                    await asyncio.sleep(0.05)

            async with MetricsSampler(
                path=directory / f"{name}_producer.csv",
                stats=stats,
                queue_name=queue,
                interval_seconds=args.sample_interval,
                management_url=args.management_url,
                management_user=args.management_user,
                management_password=args.management_password,
                vhost=unquote(urlsplit(args.rabbitmq_url).path[1:]) or "/",
                ready_probe=observer.ready_depth,
            ):
                started = time.monotonic()
                async with asyncio.timeout(args.timeout):
                    await publish_test_tasks(
                        rabbitmq_url=args.rabbitmq_url,
                        queue_name=queue,
                        max_length=max_length,
                        count=args.count,
                        batch_id=name,
                        outbox_path=directory / f"{name}.sqlite3",
                        progress_every=args.count,
                        max_wait_seconds=min(args.timeout, 60),
                        metrics=stats,
                    )
                    publish_seconds = time.monotonic() - started
                    while True:
                        check_workers(processes)
                        consumed = await asyncio.to_thread(read_results, directory, name)
                        if (
                            len({row["task_id"] for row in consumed}) >= args.count
                            and await observer.ready_depth() == 0
                        ):
                            break
                        await asyncio.sleep(0.1)
            drained = True
            completion_seconds = time.monotonic() - started
        finally:
            for process in processes:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
            for process in processes:
                try:
                    await asyncio.to_thread(process.wait, timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait)
            for log in logs:
                log.close()
            for ready in ready_files:
                ready.unlink(missing_ok=True)

            if drained:
                await observer.delete_empty_queue()
                for suffix in (".retry", ".dlq"):
                    channel = await observer._channel.get_underlay_channel()
                    state = await channel.queue_declare(queue=queue + suffix, passive=True)
                    if state.message_count or state.consumer_count:
                        raise RuntimeError(f"Experiment queue is not empty: {queue + suffix}")
                    await observer._channel.queue_delete(
                        queue + suffix, if_empty=False, if_unused=False
                    )

    consumed = read_results(directory, name)
    with closing(sqlite3.connect(directory / f"{name}.sqlite3")) as db:
        expected = {
            row[0]
            for row in db.execute(
                "SELECT task_id FROM producer_tasks WHERE confirmed_at IS NOT NULL"
            )
        }
    actual = {row["task_id"] for row in consumed}
    if len(expected) != args.count or actual != expected:
        raise RuntimeError("Consumer UUIDs do not match the confirmed SQLite outbox")
    from sqlalchemy import select

    from mlops.db import task_results
    from mlops.db.session import create_db_engine

    engine = create_db_engine()
    try:
        async with engine.connect() as connection:
            saved = set(
                (
                    await connection.execute(
                        select(task_results.c.task_id).where(task_results.c.task_id.in_(expected))
                    )
                ).scalars()
            )
        if saved != expected:
            raise RuntimeError("PostgreSQL results do not match the producer outbox")
    finally:
        await engine.dispose()
    summary = {
        "scenario": name,
        "consumer": "faststream_mlflow",
        "queue": queue,
        "workers": workers,
        "count": args.count,
        "max_length": max_length,
        "delay_seconds": args.delay,
        "published": stats.published,
        "rejected": stats.rejected,
        "retries": stats.retries,
        "duplicates": len(consumed) - len(actual),
        "database_results": len(saved),
        "publish_seconds": publish_seconds,
        "completion_seconds": completion_seconds,
        "sample_interval_seconds": args.sample_interval,
        "note": "Management statistics are sampled asynchronously; RSS is process peak, broker memory is whole node.",
    }
    (directory / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare three real RabbitMQ/FastStream/MLflow scenarios"
    )
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scenario", choices=SCENARIOS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--rabbitmq-url", default=os.getenv("RABBITMQ_URL", "amqp://guest:guest@127.0.0.1:5673/")
    )
    parser.add_argument("--management-url", default="http://127.0.0.1:15673")
    parser.add_argument("--management-user", default="guest")
    parser.add_argument("--management-password", default="guest")
    args = parser.parse_args()
    if (
        args.count <= 0
        or args.max_length <= 0
        or any(
            not math.isfinite(v) or v <= 0 for v in (args.delay, args.timeout, args.sample_interval)
        )
    ):
        parser.error(
            "count, delay, max-length, timeout and sample-interval must be finite and positive"
        )
    if args.scenario:
        asyncio.run(run_scenario(args))
        return
    directory = args.output_dir or Path("metrics") / datetime.now(UTC).strftime(
        "mlflow-%Y%m%d-%H%M%S"
    )
    directory.mkdir(parents=True, exist_ok=False)
    for name in SCENARIOS:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--scenario",
            name,
            "--output-dir",
            str(directory),
            "--count",
            str(args.count),
            "--delay",
            str(args.delay),
            "--max-length",
            str(args.max_length),
            "--timeout",
            str(args.timeout),
            "--sample-interval",
            str(args.sample_interval),
            "--rabbitmq-url",
            args.rabbitmq_url,
            "--management-url",
            args.management_url,
            "--management-user",
            args.management_user,
            "--management-password",
            args.management_password,
        ]
        subprocess.run(command, check=True, timeout=args.timeout + 60)
    from plot_backpressure import plot

    plot(directory)
    print(f"MLflow experiment completed: {directory}")


if __name__ == "__main__":
    main()
