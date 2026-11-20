import json
import logging
import os
import subprocess
import sys
import textwrap

from opentelemetry.sdk.trace import TracerProvider

from mlops.core.logging import ContextFormatter, HealthFilter, task_id
from mlops.core.telemetry import histogram_bounds


def test_json_logs_bind_context_and_redact_secrets(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "test-password")
    record = logging.LogRecord(
        "test",
        logging.ERROR,
        __file__,
        1,
        "Failed postgresql://user:test-password@db/name",
        (),
        None,
    )
    token = task_id.set("task-123")
    provider = TracerProvider()
    try:
        with provider.get_tracer("test").start_as_current_span("inference") as span:
            log = json.loads(ContextFormatter().format(record))
            assert log["trace_id"] == f"{span.get_span_context().trace_id:032x}"
            assert log["span_id"] == f"{span.get_span_context().span_id:016x}"
        assert log["task_id"] == "task-123" and log["level"] == "ERROR"
        assert "test-password" not in json.dumps(log)
        assert "user:" not in log["message"]
    finally:
        task_id.reset(token)
        provider.shutdown()
    assert task_id.get() is None


def test_health_noise_is_filtered_and_buckets_follow_inference():
    health = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, "%s", ("ip", "GET", "/healthz", "1.1", 200), None
    )
    assert not HealthFilter().filter(health)
    assert 3 in histogram_bounds(3) and 6 in histogram_bounds(3)
    assert histogram_bounds(0.05) != histogram_bounds(3)


def test_api_broker_thread_database_share_trace_and_retries_keep_context():
    # Separate process isolates the global OTel providers from the rest of the suite.
    script = textwrap.dedent("""
        import asyncio
        import os
        os.environ.pop('OTEL_SDK_DISABLED', None)
        from unittest.mock import AsyncMock, Mock
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        reader = InMemoryMetricReader()
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
        os.environ['OTEL_SDK_DISABLED'] = 'true'

        import aio_pika
        from faststream.rabbit import TestRabbitBroker
        from httpx import ASGITransport, AsyncClient
        from pamqp.commands import Basic
        from mlops import main, worker
        from mlops.core import telemetry
        from mlops.core.logging import task_id
        from mlops.model import LoadedModel
        from mlops.producer.publisher import RabbitPublisher, QueueConfig, QueueOverloaded

        import logging
        logging.getLogger('sqlalchemy.engine.Engine').info('private_input_payload')
        logging.getLogger('example_library').warning('library_event')

        async def run():
            main.initialize_database = AsyncMock()
            transaction = AsyncMock()
            connection = AsyncMock()
            connection.scalar.return_value = 'saved'
            transaction.__aenter__.return_value = connection
            worker.engine = Mock(begin=Mock(return_value=transaction))
            model = Mock()
            seen = []
            def predict(frame):
                seen.append((trace.get_current_span().get_span_context(), task_id.get()))
                return ['true']
            model.predict.side_effect = predict
            worker.loaded = LoadedModel(model, 'claim-veracity', '7')
            worker.settings.inference_delay_seconds = 0

            channel = Mock(declare_queue=AsyncMock(), close=AsyncMock())
            raw_connection = Mock(channel=AsyncMock(return_value=channel), close=AsyncMock())
            aio_pika.connect_robust = AsyncMock(return_value=raw_connection)
            async with TestRabbitBroker(worker.broker) as broker:
                async def deliver(message, **kwargs):
                    await broker.publish(message.body, queue=worker.queue.name,
                                         headers=message.headers, message_id=message.message_id)
                    return Basic.Ack()
                channel.default_exchange.publish = AsyncMock(side_effect=deliver)
                async with main.app.router.lifespan_context(main.app):
                    async with AsyncClient(transport=ASGITransport(app=main.app), base_url='http://test') as client:
                        response = await client.post('/api/v1/tasks', json={'text': 'test'})
                        assert response.status_code == 202, response.text
                        accepted = response.json()['task_id']
                        channel.default_exchange.publish.side_effect = QueueOverloaded()
                        response = await client.post('/api/v1/tasks', json={'text': 'test'})
                        assert response.status_code == 503
                        assert (await client.get('/healthz')).status_code == 200
                        assert (await client.patch('/api/v1/tasks', json={'text': 'test'})).status_code == 405

                # Poison delivery carries the API trace through the built-in retry publisher.
                spans = exporter.get_finished_spans()
                api = next(s for s in spans if s.name == 'POST /api/v1/tasks')
                from opentelemetry.propagate import inject
                headers = {}
                with trace.use_span(trace.NonRecordingSpan(api.context)):
                    inject(headers)
                send = broker.publish
                handoff = AsyncMock(return_value=Basic.Ack())
                broker.publish = handoff
                await send(b'invalid json', queue=worker.queue.name, message_id='poison', headers=headers)
                assert handoff.await_args.kwargs['headers']['traceparent'] == headers['traceparent']
                broker.publish = send

            spans = exporter.get_finished_spans()
            api = next(s for s in spans if s.name == 'POST /api/v1/tasks' and s.status.status_code.name != 'ERROR')
            for name in ('publish', 'inference', 'db.save_result', 'queue.wait'):
                assert any(s.name == name and s.context.trace_id == api.context.trace_id for s in spans), name
            inference = next(s for s in spans if s.name == 'inference')
            assert seen[0][0].span_id == inference.context.span_id
            assert seen[0][1] == accepted
            assert any(s.name.endswith(' process') and s.context.trace_id == api.context.trace_id for s in spans)
            assert any(s.name == 'POST /api/v1/tasks' and s.status.status_code.name == 'ERROR' for s in spans)
            assert task_id.get() is None
            data = reader.get_metrics_data()
            instruments = [m for r in data.resource_metrics for scope in r.scope_metrics for m in scope.metrics]
            requests = next(m for m in instruments if m.name == 'mlops_api_requests')
            assert {p.attributes['status_code'] for p in requests.data.data_points} == {'200', '202', '405', '503'}
            assert {p.attributes['method'] for p in requests.data.data_points} == {'GET', 'POST', 'OTHER'}
            assert all('task_id' not in p.attributes and 'trace_id' not in p.attributes
                       for m in instruments for p in m.data.data_points)

        asyncio.run(run())
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "OTEL_SDK_DISABLED": "true"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "private_input_payload" not in result.stdout
    logs = [json.loads(line) for line in result.stdout.splitlines() if line]
    assert any(log.get("logger") == "example_library" for log in logs)
