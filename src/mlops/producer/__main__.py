import argparse
import asyncio
import math
import os
import sys
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from mlops.producer.metrics import MetricsSampler
from mlops.producer.publisher import PublishStats, publish_test_tasks


def positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish a resumable inference batch to RabbitMQ")
    parser.add_argument("--count", type=positive_int, default=1000)
    parser.add_argument("--mode", choices=("bounded", "baseline"), default="bounded")
    parser.add_argument(
        "--queue", default=None, help="defaults to inference_tasks / inference_tasks_baseline"
    )
    parser.add_argument("--max-length", type=positive_int, default=20)
    parser.add_argument("--text", default="The Earth is flat.")
    parser.add_argument("--progress-every", type=positive_int, default=100)
    parser.add_argument("--outbox", default=".producer-outbox.sqlite3")
    parser.add_argument(
        "--batch-id", default=None, help="reuse this ID to resume an interrupted batch"
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=float,
        default=60.0,
        help="finite time limit per task (default 60); pending task remains in SQLite",
    )
    parser.add_argument("--metrics-file", default=None)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument(
        "--rabbitmq-url", default=os.getenv("RABBITMQ_URL", "amqp://guest:guest@127.0.0.1:5672/")
    )
    parser.add_argument("--management-url", default="http://127.0.0.1:15672")
    parser.add_argument("--management-user", default="guest")
    parser.add_argument("--management-password", default="guest")
    return parser


async def _run(args: argparse.Namespace, batch_id: str, metrics_file: str) -> PublishStats:
    queue = args.queue or (
        "inference_tasks" if args.mode == "bounded" else "inference_tasks_baseline"
    )
    max_length = args.max_length if args.mode == "bounded" else None
    stats = PublishStats()
    async with MetricsSampler(
        path=metrics_file,
        stats=stats,
        queue_name=queue,
        management_url=args.management_url,
        management_user=args.management_user,
        management_password=args.management_password,
        interval_seconds=args.sample_interval,
        vhost=unquote(urlsplit(args.rabbitmq_url).path[1:]) or "/",
    ):
        return await publish_test_tasks(
            rabbitmq_url=args.rabbitmq_url,
            queue_name=queue,
            max_length=max_length,
            count=args.count,
            text=args.text,
            progress_every=args.progress_every,
            outbox_path=args.outbox,
            batch_id=batch_id,
            max_wait_seconds=args.max_wait_seconds,
            metrics=stats,
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(not math.isfinite(v) or v <= 0 for v in (args.sample_interval, args.max_wait_seconds)):
        print("sample interval and max wait must be positive", file=sys.stderr)
        return 2
    batch_id = args.batch_id or f"batch-{uuid4().hex[:12]}"
    metrics_file = args.metrics_file or f"metrics/{batch_id}.csv"
    print(f"Batch ID: {batch_id} (reuse --batch-id to resume after interruption)", flush=True)
    print(f"Metrics: {metrics_file}", flush=True)
    try:
        stats = asyncio.run(_run(args, batch_id, metrics_file))
    except KeyboardInterrupt:
        print("Interrupted; unconfirmed task is retained in the SQLite outbox", file=sys.stderr)
        return 130
    except Exception as error:
        print(
            f"Publishing stopped: {type(error).__name__}: {error}. "
            f"Resume with --batch-id {batch_id}",
            file=sys.stderr,
        )
        return 1
    print(
        f"Done: {stats.published} confirmed this run in {stats.elapsed_seconds:.2f}s "
        f"({stats.tasks_per_second:.2f} tasks/s), "
        f"NACKs={stats.rejected}, retries={stats.retries}, "
        f"mean time from creation to confirm={stats.mean_publish_latency_seconds:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
