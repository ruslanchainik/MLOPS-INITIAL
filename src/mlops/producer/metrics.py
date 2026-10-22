import asyncio
import base64
import csv
import json
import math
import resource
import sys
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from aio_pika.exceptions import AMQPError

from mlops.producer.publisher import PublishStats

FIELDS = (
    "timestamp_utc",
    "elapsed_seconds",
    "generated_total",
    "generation_per_second",
    "published_total",
    "publish_per_second",
    "rejected_total",
    "retry_total",
    "queue_ready",
    "queue_ready_amqp",
    "queue_unacked",
    "queue_total",
    "broker_memory_mb",
    "producer_peak_rss_mb",
    "mean_publish_latency_seconds",
)


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def fetch_queue_depth(
    *, url: str, user: str, password: str, queue_name: str, vhost: str = "/"
) -> tuple[int | None, int | None, int | None]:
    endpoint = f"{url.rstrip('/')}/api/queues/{quote(vhost, safe='')}/{quote(queue_name, safe='')}"
    credentials = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
    request = Request(endpoint, headers={"Authorization": f"Basic {credentials}"})
    try:
        with urlopen(request, timeout=2) as response:
            data = json.load(response)
    except HTTPError, URLError, TimeoutError, ValueError, OSError:
        return None, None, None
    return data.get("messages_ready"), data.get("messages_unacknowledged"), data.get("messages")


def fetch_broker_memory_mb(*, url: str, user: str, password: str) -> float | None:
    credentials = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
    request = Request(
        f"{url.rstrip('/')}/api/nodes",
        headers={"Authorization": f"Basic {credentials}"},
    )
    try:
        with urlopen(request, timeout=2) as response:
            nodes = json.load(response)
        return round(sum(node.get("mem_used", 0) for node in nodes) / (1024 * 1024), 3)
    except HTTPError, URLError, TimeoutError, ValueError, OSError, TypeError:
        return None


class MetricsSampler:
    def __init__(
        self,
        *,
        path: str | Path,
        stats: PublishStats,
        queue_name: str,
        management_url: str = "http://127.0.0.1:15672",
        management_user: str = "guest",
        management_password: str = "guest",
        interval_seconds: float = 1.0,
        vhost: str = "/",
        ready_probe: Callable[[], Awaitable[int]] | None = None,
    ):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and positive")
        self.path = Path(path)
        self.stats = stats
        self.queue_name = queue_name
        self.management_url = management_url
        self.management_user = management_user
        self.management_password = management_password
        self.interval_seconds = interval_seconds
        self.vhost = vhost
        self.ready_probe = ready_probe
        self._start = monotonic()
        self._last_sample = self._start
        self._last_generated = 0
        self._last_published = 0
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._file = None
        self._writer = None

    async def __aenter__(self) -> MetricsSampler:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS, lineterminator="\n")
        self._writer.writeheader()
        await self._sample()
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *_: object) -> None:
        self._stop.set()
        if self._task:
            await self._task
        await self._sample()
        if self._file:
            self._file.close()

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
                return
            except TimeoutError:
                await self._sample()

    async def _sample(self) -> None:
        ready_amqp = None
        if self.ready_probe is not None:
            with suppress(TimeoutError, AMQPError, ConnectionError, OSError):
                ready_amqp = await self.ready_probe()
        ready, unacked, total = await asyncio.to_thread(
            fetch_queue_depth,
            url=self.management_url,
            user=self.management_user,
            password=self.management_password,
            queue_name=self.queue_name,
            vhost=self.vhost,
        )
        broker_memory = await asyncio.to_thread(
            fetch_broker_memory_mb,
            url=self.management_url,
            user=self.management_user,
            password=self.management_password,
        )
        now = monotonic()
        dt = max(now - self._last_sample, 1e-9)
        assert self._writer is not None
        self._writer.writerow(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": round(now - self._start, 3),
                "generated_total": self.stats.generated,
                "generation_per_second": round(
                    (self.stats.generated - self._last_generated) / dt, 3
                ),
                "published_total": self.stats.published,
                "publish_per_second": round((self.stats.published - self._last_published) / dt, 3),
                "rejected_total": self.stats.rejected,
                "retry_total": self.stats.retries,
                "queue_ready": ready,
                "queue_ready_amqp": ready_amqp,
                "queue_unacked": unacked,
                "queue_total": total,
                "broker_memory_mb": broker_memory,
                "producer_peak_rss_mb": round(peak_rss_mb(), 3),
                "mean_publish_latency_seconds": round(self.stats.mean_publish_latency_seconds, 4),
            }
        )
        self._file.flush()
        self._last_sample = now
        self._last_generated = self.stats.generated
        self._last_published = self.stats.published
