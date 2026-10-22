import asyncio
import json
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import aio_pika
import pytest
from aio_pika.exceptions import DeliveryError
from pamqp.commands import Basic

from mlops.messaging import TaskMessage
from mlops.producer import publisher as producer
from mlops.producer.metrics import peak_rss_mb
from mlops.producer.outbox import PendingTask, SQLiteOutbox


def test_process_peak_memory_is_available():
    assert peak_rss_mb() > 0


@pytest.fixture
def publisher(monkeypatch):
    channel = Mock(declare_queue=AsyncMock(), close=AsyncMock())
    channel.default_exchange.publish = AsyncMock(return_value=Basic.Ack())
    connection = Mock(channel=AsyncMock(return_value=channel), close=AsyncMock())
    monkeypatch.setattr(aio_pika, "connect_robust", AsyncMock(return_value=connection))
    return producer.RabbitPublisher("amqp://test/", producer.QueueConfig())


async def test_message_and_publisher_contract(publisher):
    tasks = [TaskMessage.new("test") for _ in range(1000)]
    assert len({task.task_id for task in tasks}) == 1000
    async with publisher:
        await publisher.publish_once(tasks[0], created_at_epoch=123.456)
        publisher._connection.channel.assert_awaited_once_with(
            publisher_confirms=True, on_return_raises=True
        )
        publisher._channel.declare_queue.assert_awaited_once_with(
            "inference_tasks",
            durable=True,
            arguments={"x-max-length": 20, "x-overflow": "reject-publish"},
        )
        call = publisher._channel.default_exchange.publish.await_args
        message = call.args[0]
        assert json.loads(message.body) == {"task_id": tasks[0].task_id, "text": "test"}
        assert message.message_id == tasks[0].task_id
        assert message.delivery_mode == aio_pika.DeliveryMode.PERSISTENT
        assert message.headers["created_at_epoch"] == "123.456"
        assert call.kwargs["mandatory"] and call.kwargs["routing_key"] == "inference_tasks"
    publisher._connection.close.assert_awaited_once()


@pytest.mark.parametrize(
    "response",
    [
        Basic.Nack(),
        Basic.Reject(),
        DeliveryError(None, Basic.Nack()),
        DeliveryError(None, Basic.Reject()),
        None,
        True,
        False,
        object(),
        DeliveryError(None, Basic.Return()),
        TimeoutError(),
        ConnectionError(),
    ],
)
async def test_only_positive_ack_succeeds(publisher, response):
    async with publisher:
        publish = publisher._channel.default_exchange.publish
        if isinstance(response, Exception):
            publish.side_effect = response
        else:
            publish.return_value = response
        frame = response.frame if isinstance(response, DeliveryError) else response
        expected = (
            producer.QueueOverloaded
            if isinstance(frame, (Basic.Nack, Basic.Reject))
            else producer.PublishUncertain
        )
        with pytest.raises(expected):
            await publisher.publish_once(TaskMessage.new("test"))


@pytest.mark.parametrize("task_id,text", [("invalid", "test"), (str(uuid4()), " ")])
def test_invalid_message(task_id, text):
    with pytest.raises(ValueError):
        TaskMessage(task_id, text)


def test_outbox_resume_and_lock(tmp_path):
    path = tmp_path / "outbox.sqlite3"
    with SQLiteOutbox(path) as outbox:
        outbox.open_batch(batch_id="b", count=1, text="test", queue_name="q", max_length=2)
        pending, _ = outbox.next_task("b")
        with pytest.raises(RuntimeError), SQLiteOutbox(path):
            pass
    with SQLiteOutbox(path) as resumed:
        assert resumed.next_task("b") == (pending, False)
        resumed.mark_confirmed("b", pending.message.task_id)
        assert resumed.count_confirmed("b") == 1 and resumed.next_task("b") == (None, False)


async def test_backoff_and_expired_retry_budget(monkeypatch):
    monkeypatch.setattr(producer.random, "uniform", lambda *_: 1.2)
    delays = [producer.retry_delay(attempt) for attempt in range(1, 100)]
    assert 0 < delays[0] < delays[1] < delays[2] <= max(delays) <= 0.5
    clock = [asyncio.get_running_loop().time()]
    monkeypatch.setattr(producer, "monotonic", lambda: clock[0])

    async def sleep(delay):
        clock[0] += delay

    monkeypatch.setattr(producer.asyncio, "sleep", sleep)
    publisher = Mock(publish_once=AsyncMock(side_effect=producer.QueueOverloaded()))
    pending = PendingTask(TaskMessage.new("test"), 0, 1)
    with pytest.raises(TimeoutError):
        await producer._publish_with_backpressure(
            publisher, pending, producer.PublishStats(), max_wait_seconds=0.01
        )
    publisher.publish_once.assert_awaited_once()
