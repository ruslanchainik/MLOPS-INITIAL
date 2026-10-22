import asyncio
import math
import random
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter, time
from typing import Any

import aio_pika
from aio_pika.exceptions import (
    AMQPConnectionError,
    ChannelInvalidStateError,
    ChannelPreconditionFailed,
    DeliveryError,
    PublishError,
)
from pamqp.commands import Basic

from mlops.messaging import TaskMessage
from mlops.producer.outbox import PendingTask, SQLiteOutbox


class QueueOverloaded(Exception):
    pass


class PublishUncertain(Exception):
    pass


class QueueConfigurationError(Exception):
    pass


class UnroutableTask(Exception):
    pass


@dataclass(frozen=True, slots=True)
class QueueConfig:
    name: str = "inference_tasks"
    max_length: int | None = 20

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name.strip() != self.name:
            raise ValueError("queue name must be nonempty and have no surrounding spaces")
        if self.max_length is not None and (
            type(self.max_length) is not int or self.max_length <= 0
        ):
            raise ValueError("max_length must be positive or None for baseline mode")

    @property
    def arguments(self) -> dict[str, Any] | None:
        if self.max_length is None:
            return None
        return {"x-max-length": self.max_length, "x-overflow": "reject-publish"}


@dataclass(slots=True)
class PublishStats:
    published: int = 0
    generated: int = 0
    rejected: int = 0
    retries: int = 0
    elapsed_seconds: float = 0.0
    publish_latency_sum_seconds: float = 0.0

    @property
    def tasks_per_second(self) -> float:
        return self.published / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def mean_publish_latency_seconds(self) -> float:
        return self.publish_latency_sum_seconds / self.published if self.published else 0.0


class RabbitPublisher:
    def __init__(self, rabbitmq_url: str, queue: QueueConfig, *, confirm_timeout: float = 10):
        if not math.isfinite(confirm_timeout) or confirm_timeout <= 0:
            raise ValueError("confirm_timeout must be finite and positive")
        self.rabbitmq_url = rabbitmq_url
        self.queue = queue
        self.confirm_timeout = confirm_timeout
        self._connection: Any = None
        self._channel: Any = None

    async def __aenter__(self) -> RabbitPublisher:
        self._connection = await aio_pika.connect_robust(self.rabbitmq_url, timeout=10)
        try:
            self._channel = await self._connection.channel(
                publisher_confirms=True, on_return_raises=True
            )
            await self._channel.declare_queue(
                self.queue.name, durable=True, arguments=self.queue.arguments
            )
        except ChannelPreconditionFailed as exc:
            await self._close()
            raise QueueConfigurationError(
                f"Queue {self.queue.name!r} already has incompatible arguments. "
                "Use a new versioned queue name and migrate/drain the old queue; "
                "never delete a nonempty queue automatically. "
                f"Requested arguments: {self.queue.arguments!r}"
            ) from exc
        except BaseException:
            await self._close()
            raise
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._close()

    async def _close(self) -> None:
        try:
            if self._channel is not None:
                await self._channel.close()
        finally:
            if self._connection is not None:
                await self._connection.close()

    async def ready_depth(self) -> int:
        async with asyncio.timeout(3):
            channel = await self._channel.get_underlay_channel()
            result = await channel.queue_declare(queue=self.queue.name, passive=True)
        return result.message_count

    async def delete_empty_queue(self) -> None:
        await self._channel.queue_delete(self.queue.name, if_empty=True, if_unused=True)

    async def publish_once(
        self, task: TaskMessage, *, created_at_epoch: float | None = None
    ) -> None:
        if self._channel is None:
            raise RuntimeError("Publisher is not connected")
        message = aio_pika.Message(
            body=task.to_bytes(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=task.task_id,
            headers={"created_at_epoch": repr(created_at_epoch)}
            if created_at_epoch is not None
            else {},
        )
        try:
            async with asyncio.timeout(self.confirm_timeout):
                confirm = await self._channel.default_exchange.publish(
                    message,
                    routing_key=self.queue.name,
                    mandatory=True,
                    timeout=self.confirm_timeout,
                )
        except PublishError as exc:
            raise UnroutableTask(f"Mandatory publish returned for {self.queue.name!r}") from exc
        except DeliveryError as exc:
            if isinstance(exc.frame, (Basic.Nack, Basic.Reject)):
                raise QueueOverloaded(f"RabbitMQ refused task {task.task_id}") from exc
            raise PublishUncertain(f"Unrecognized confirm for task {task.task_id}") from exc
        except (
            TimeoutError,
            AMQPConnectionError,
            ChannelInvalidStateError,
            ConnectionError,
            OSError,
        ) as exc:
            raise PublishUncertain(f"Confirm unavailable for task {task.task_id}") from exc
        if isinstance(confirm, (Basic.Nack, Basic.Reject)):
            raise QueueOverloaded(f"RabbitMQ NACKed task {task.task_id}")
        if not isinstance(confirm, Basic.Ack):
            raise PublishUncertain(f"No positive broker confirm for task {task.task_id}")


def retry_delay(retries: int, *, base: float = 0.02, cap: float = 0.5) -> float:
    return min(cap, base * (2 ** min(max(retries - 1, 0), 30)) * random.uniform(0.8, 1.2))


async def publish_test_tasks(
    *,
    rabbitmq_url: str,
    queue_name: str = "inference_tasks",
    max_length: int | None = 20,
    count: int = 1000,
    text: str = "The Earth is flat.",
    progress_every: int = 100,
    outbox_path: str | Path = ".producer-outbox.sqlite3",
    batch_id: str = "demo",
    max_wait_seconds: float = 60.0,
    metrics: PublishStats | None = None,
) -> PublishStats:
    if count <= 0 or progress_every <= 0:
        raise ValueError("count and progress_every must be positive")
    if str(outbox_path) == ":memory:":
        raise ValueError("Batch publishing requires a disk-backed outbox")
    if not math.isfinite(max_wait_seconds) or max_wait_seconds <= 0:
        raise ValueError("max_wait_seconds must be finite and positive")
    queue = QueueConfig(name=queue_name, max_length=max_length)
    stats = metrics if metrics is not None else PublishStats()
    started = perf_counter()
    try:
        with SQLiteOutbox(outbox_path) as outbox:
            outbox.open_batch(
                batch_id=batch_id,
                count=count,
                text=text,
                queue_name=queue.name,
                max_length=queue.max_length,
            )
            async with RabbitPublisher(rabbitmq_url, queue) as publisher:
                while True:
                    pending, is_new = outbox.next_task(batch_id)
                    if pending is None:
                        break
                    if is_new:
                        stats.generated += 1
                    await _publish_with_backpressure(
                        publisher, pending, stats, max_wait_seconds=max_wait_seconds
                    )
                    outbox.mark_confirmed(batch_id, pending.message.task_id)
                    stats.published += 1
                    stats.publish_latency_sum_seconds += max(0.0, time() - pending.created_at_epoch)
                    if pending.sequence % progress_every == 0 or pending.sequence == count:
                        print(
                            f"Confirmed {pending.sequence}/{count}; "
                            f"broker rejections={stats.rejected}",
                            flush=True,
                        )
    finally:
        stats.elapsed_seconds = perf_counter() - started
    return stats


async def _publish_with_backpressure(
    publisher: RabbitPublisher,
    pending: PendingTask,
    stats: PublishStats,
    *,
    max_wait_seconds: float,
) -> None:
    deadline = monotonic() + max_wait_seconds
    attempts = 0
    while True:
        try:
            async with asyncio.timeout_at(deadline):
                await publisher.publish_once(
                    pending.message, created_at_epoch=pending.created_at_epoch
                )
            return
        except (QueueOverloaded, PublishUncertain, TimeoutError) as exc:
            if isinstance(exc, QueueOverloaded):
                stats.rejected += 1
            stats.retries += 1
            attempts += 1
            remaining = deadline - monotonic()
            if remaining > 0:
                await asyncio.sleep(min(retry_delay(attempts), remaining))
            if monotonic() >= deadline:
                raise TimeoutError(
                    f"Broker did not accept task {pending.message.task_id} "
                    f"within {max_wait_seconds}s; resume with the same --batch-id"
                ) from exc
