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
            raise RuntimeError(
                f"Synthetic consumer exited early: {process.returncode}; inspect worker logs"
            )


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
    print(f"{name}: SYNTHETIC consumer, workers={workers}, queue={queue}", flush=True)

    async with RabbitPublisher(args.rabbitmq_url, config) as observer:
        try:
            for index in range(1, workers + 1):
                log = (directory / f"{name}_worker{index}.log").open("w")
                logs.append(log)
                ready = directory / f"{name}_worker{index}.ready"
                ready_files.append(ready)
                command = [
                    sys.executable,
                    "-m",
                    "mlops.producer.slow_consumer",
                    "--queue",
                    queue,
                    "--mode",
                    mode,
                    "--max-length",
                    str(args.max_length),
                    "--delay",
                    str(args.delay),
                    "--worker-id",
                    f"worker-{index}",
                    "--results-file",
                    str(directory / f"{name}_worker{index}.csv"),
                    "--ready-file",
                    str(ready),
                    "--rabbitmq-url",
                    args.rabbitmq_url,
                ]
                processes.append(
                    await asyncio.to_thread(subprocess.Popen, command, stdout=log, stderr=log)
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

            if drained:
                await observer.delete_empty_queue()

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
    summary = {
        "scenario": name,
        "consumer": "synthetic_sleep",
        "queue": queue,
        "workers": workers,
        "count": args.count,
        "max_length": max_length,
        "delay_seconds": args.delay,
        "published": stats.published,
        "rejected": stats.rejected,
        "retries": stats.retries,
        "duplicates": len(consumed) - len(actual),
        "publish_seconds": publish_seconds,
        "completion_seconds": time.monotonic() - started,
        "sample_interval_seconds": args.sample_interval,
        "note": "Management statistics are sampled asynchronously; RSS is process peak, broker memory is whole node.",
    }
    (directory / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure three synthetic-consumer scenarios against real RabbitMQ.\n\nEach scenario runs in its own process, with a fresh queue and outbox. Sampling\ncontinues through consumption; results are checked against the durable outbox.\nThese measurements do NOT validate FastStream, MLflow or PostgreSQL.\n"
    )
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scenario", choices=SCENARIOS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--rabbitmq-url", default=os.getenv("RABBITMQ_URL", "amqp://guest:guest@127.0.0.1:5672/")
    )
    parser.add_argument("--management-url", default="http://127.0.0.1:15672")
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
        "synthetic-%Y%m%d-%H%M%S"
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
    print(f"Synthetic experiment completed: {directory}")


if __name__ == "__main__":
    main()
