import asyncio
import gc
from contextlib import closing
from uuid import uuid4

import aio_pika
import docker
import pytest
from testcontainers.community.rabbitmq import RabbitMqContainer

from mlops.messaging import TaskMessage
from mlops.producer import publisher as producer
from mlops.producer.outbox import SQLiteOutbox

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def broker():
    try:
        with closing(docker.from_env()) as client:
            client.ping()
    except docker.errors.DockerException as error:
        pytest.skip(f"Docker unavailable: {error}")
    with RabbitMqContainer("rabbitmq:4.1-management") as container:
        yield container


@pytest.fixture(autouse=True)
async def no_unhandled_futures():
    loop = asyncio.get_running_loop()
    previous, errors = loop.get_exception_handler(), []

    def record_error(loop, context):
        errors.append(context)
        loop.default_exception_handler(context)

    loop.set_exception_handler(record_error)
    try:
        yield
        gc.collect()
        await asyncio.sleep(0.05)
        assert not errors, errors
    finally:
        loop.set_exception_handler(previous)


def broker_url(broker):
    return f"amqp://guest:guest@{broker.get_container_host_ip()}:{broker.get_exposed_port(5672)}/"


async def test_overflow_outbox_and_resume_1000_tasks(broker, tmp_path):
    url, name = broker_url(broker), f"resume_{uuid4().hex}"
    path = tmp_path / "outbox.sqlite3"
    options = dict(rabbitmq_url=url, queue_name=name, max_length=2, count=1000, outbox_path=path)
    failed = producer.PublishStats()
    with pytest.raises(TimeoutError):
        await producer.publish_test_tasks(**options, max_wait_seconds=0.05, metrics=failed)
    assert failed.published == 2 and failed.rejected > 0
    with SQLiteOutbox(path) as outbox:
        pending, is_new = outbox.next_task("demo")
        assert not is_new and pending.sequence == 3
        assert outbox.db.execute("SELECT COUNT(*) FROM producer_tasks").fetchone()[0] == 3
    async with producer.RabbitPublisher(url, producer.QueueConfig(name, 2)) as publisher:
        assert await publisher.ready_depth() == 2
        with pytest.raises(producer.QueueOverloaded):
            await publisher.publish_once(pending.message)
        with pytest.raises(producer.QueueConfigurationError):
            async with producer.RabbitPublisher(url, producer.QueueConfig(name, 3)):
                pass
        assert await publisher.ready_depth() == 2
    connection = await aio_pika.connect(url)
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        queue = await channel.get_queue(name)
        received = []

        async def consume():
            async with queue.iterator() as messages:
                async for message in messages:
                    async with message.process():
                        assert message.delivery_mode == aio_pika.DeliveryMode.PERSISTENT
                        received.append(message.message_id)
                        await asyncio.sleep(0.001)
                    if len(received) == 1000:
                        return

        consumer = asyncio.create_task(consume())
        try:
            stats = await producer.publish_test_tasks(**options, max_wait_seconds=10)
            await asyncio.wait_for(consumer, timeout=15)
        finally:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        assert stats.published == 998 and stats.generated == 997
        with SQLiteOutbox(path) as outbox:
            expected = {row[0] for row in outbox.db.execute("SELECT task_id FROM producer_tasks")}
            assert outbox.count_confirmed("demo") == 1000
        assert len(received) == len(set(received)) == 1000 and set(received) == expected
        assert received[2] == pending.message.task_id
        await queue.delete(if_empty=True)


async def test_connection_recovery_and_persistence(broker):
    url, name = broker_url(broker), f"recovery_{uuid4().hex}"
    task = TaskMessage.new("persistent")
    async with producer.RabbitPublisher(
        url, producer.QueueConfig(name, 2), confirm_timeout=0.5
    ) as publisher:
        await publisher.publish_once(task)
        raw = broker.get_wrapped_container()
        stopped = await asyncio.to_thread(raw.exec_run, ["rabbitmqctl", "stop_app"])
        assert stopped.exit_code == 0, stopped.output
        try:
            with pytest.raises(producer.PublishUncertain):
                await publisher.publish_once(TaskMessage.new("offline"))
        finally:
            started = await asyncio.to_thread(raw.exec_run, ["rabbitmqctl", "start_app"])
            assert started.exit_code == 0, started.output
            await asyncio.to_thread(broker.readiness_probe)
        await asyncio.wait_for(publisher._channel.ready(), timeout=20)
        connection = await aio_pika.connect(url)
        async with connection:
            queue = await (await connection.channel()).get_queue(name)
            message = await queue.get(timeout=5)
            assert message.message_id == task.task_id
            await message.ack()
            await queue.delete(if_empty=True)
        with pytest.raises(producer.UnroutableTask):
            await publisher.publish_once(task)
