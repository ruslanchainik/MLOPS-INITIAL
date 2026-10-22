import asyncio
import threading
from unittest.mock import AsyncMock, Mock

import pytest
from aio_pika.exceptions import DeliveryError
from faststream.rabbit import TestRabbitBroker
from pamqp.commands import Basic

from mlops import worker
from mlops.messaging import TaskMessage
from mlops.model import LoadedModel
from mlops.producer.publisher import PublishUncertain


@pytest.fixture
def runtime(monkeypatch):
    connection = AsyncMock()
    transaction = AsyncMock()
    transaction.__aenter__.return_value = connection
    engine = Mock(begin=Mock(return_value=transaction))
    model = Mock()
    model.predict.return_value = ["true"]
    monkeypatch.setattr(worker, "engine", engine)
    monkeypatch.setattr(worker, "loaded", LoadedModel(model, "claim-veracity", "7"))
    monkeypatch.setattr(worker.settings, "inference_delay_seconds", 0)
    return model, connection, transaction


async def test_handler_predicts_in_thread_and_commits(runtime):
    model, connection, transaction = runtime
    threads = []
    model.predict.side_effect = lambda frame: threads.append(threading.get_ident()) or ["true"]
    task = TaskMessage.new("test")
    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish({"task_id": task.task_id, "text": task.text}, worker.queue.name)
    assert len(threads) == 1 and threads[0] != threading.get_ident()
    connection.scalar.assert_awaited_once()
    sql = connection.scalar.await_args.args[0]
    assert "ON CONFLICT (task_id) DO NOTHING" in str(sql)
    transaction.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.parametrize("attempt,target", [(0, ".retry"), (1, ".retry"), (2, ".dlq")])
async def test_poison_message_has_bounded_redeliveries(runtime, monkeypatch, attempt, target):
    async with TestRabbitBroker(worker.broker) as broker:
        send = broker.publish
        ack = AsyncMock()
        monkeypatch.setattr(worker.RabbitMessage, "ack", ack)
        publish = AsyncMock(return_value=Basic.Ack())
        monkeypatch.setattr(broker, "publish", publish)
        await send(b"invalid json", worker.queue.name, headers={"attempt": attempt})
        call = publish.await_args
        assert call.kwargs["queue"] == worker.queue.name + target
        assert call.kwargs["headers"]["attempt"] == attempt + 1
        assert call.kwargs["persist"] is True
        ack.assert_awaited_once()
    runtime[1].scalar.assert_not_awaited()


@pytest.mark.parametrize("attempt", [0, 2])
@pytest.mark.parametrize(
    "outcome",
    [
        None,
        False,
        True,
        Basic.Nack(),
        Basic.Reject(),
        DeliveryError(None, Basic.Nack()),
        TimeoutError(),
        ConnectionError(),
    ],
)
async def test_failed_handoff_is_not_acknowledged(runtime, monkeypatch, attempt, outcome):
    async with TestRabbitBroker(worker.broker) as broker:
        send = broker.publish
        ack = AsyncMock()
        monkeypatch.setattr(worker.RabbitMessage, "ack", ack)
        failed = isinstance(outcome, Exception)
        publish = AsyncMock(return_value=outcome, side_effect=outcome if failed else None)
        monkeypatch.setattr(broker, "publish", publish)
        with pytest.raises(type(outcome) if failed else PublishUncertain):
            await send(b"invalid json", worker.queue.name, headers={"attempt": attempt})
        ack.assert_not_awaited()
        runtime[1].scalar.assert_not_awaited()


@pytest.mark.parametrize("attempt", [0, 2])
async def test_ack_waits_for_publication_confirm(runtime, monkeypatch, attempt):
    async with TestRabbitBroker(worker.broker) as broker:
        send = broker.publish
        ack = AsyncMock()
        monkeypatch.setattr(worker.RabbitMessage, "ack", ack)
        confirmation = asyncio.get_running_loop().create_future()
        publishing = asyncio.Event()

        async def publish(*args, **kwargs):
            publishing.set()
            return await confirmation

        monkeypatch.setattr(broker, "publish", publish)
        async with asyncio.TaskGroup() as group:
            delivery = group.create_task(
                send(b"invalid json", worker.queue.name, headers={"attempt": attempt})
            )
            await asyncio.wait_for(publishing.wait(), timeout=1)
            ack.assert_not_awaited()
            assert not delivery.done()
            confirmation.set_result(Basic.Ack())
        ack.assert_awaited_once()


async def test_worker_loads_alias_once_at_startup(runtime, monkeypatch):
    from faststream import TestApp

    engine = worker.engine
    engine.dispose = AsyncMock()
    loader = Mock(return_value=worker.loaded)
    monkeypatch.setattr(worker, "create_db_engine", Mock(return_value=engine))
    monkeypatch.setattr(worker, "load_registered_model", loader)
    monkeypatch.setattr(worker.broker, "declare_queue", AsyncMock())
    async with TestRabbitBroker(worker.broker), TestApp(worker.app):
        for _ in range(2):
            task = TaskMessage.new("test")
            await worker.broker.publish(
                {"task_id": task.task_id, "text": task.text}, worker.queue.name
            )
    loader.assert_called_once_with(worker.settings)
    assert runtime[0].predict.call_count == 2
    engine.dispose.assert_awaited_once()
