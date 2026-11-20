"""OTLP export runs in bounded SDK background queues, never on the inference path."""

import math
import os
import socket
import time

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from mlops import __version__

_providers = []
tracer = trace.get_tracer("mlops")
meter = metrics.get_meter("mlops")
api_requests = meter.create_counter("mlops_api_requests")
api_duration = meter.create_histogram("mlops_api_duration", unit="s")
generated = meter.create_counter("mlops_source_generated")
published = meter.create_counter("mlops_source_published")
rejected = meter.create_counter("mlops_source_rejected")
wait_time = meter.create_counter("mlops_source_wait", unit="s")
processed = meter.create_counter("mlops_worker_processed")
redeliveries = meter.create_counter("mlops_worker_redeliveries")
in_flight = meter.create_up_down_counter("mlops_worker_in_flight")
inference_duration = meter.create_histogram("mlops_inference_duration", unit="s")
task_duration = meter.create_histogram("mlops_task_duration", unit="s")


def histogram_bounds(delay: float) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1,
                2.5,
                5,
                10,
                30,
                60,
                *[max(delay, 0.001) * factor for factor in (0.5, 1, 1.5, 2, 5, 10)],
            }
        )
    )


def message_age(headers: dict, field: str) -> float | None:
    """Untrusted timing metadata must not break message acknowledgement."""
    try:
        timestamp = float(headers[field])
        if math.isfinite(timestamp) and timestamp > 0:
            return max(0, time.time() - timestamp)
    except KeyError, TypeError, ValueError:
        pass
    return None


def configure_telemetry(service: str) -> None:
    os.environ.setdefault("OTEL_SERVICE_NAME", service)
    if _providers or os.getenv("OTEL_SDK_DISABLED", "true").lower() == "true":
        return
    attributes = {
        "service.name": os.environ["OTEL_SERVICE_NAME"],
        "service.version": __version__,
        "service.instance.id": f"{socket.gethostname()}:{os.getpid()}",
        "deployment.environment.name": os.getenv("DEPLOYMENT_ENVIRONMENT", "local"),
    }
    attributes.update(Resource.create().attributes)
    resource = Resource(attributes)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(timeout=1),
            max_queue_size=512,
            max_export_batch_size=128,
            schedule_delay_millis=1000,
        )
    )
    trace.set_tracer_provider(provider)
    delay = float(os.getenv("INFERENCE_DELAY_SECONDS", "0.05"))
    metric_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                OTLPMetricExporter(timeout=1),
                export_interval_millis=1000,
                export_timeout_millis=1500,
            )
        ],
        views=[
            View(
                instrument_name=name,
                aggregation=ExplicitBucketHistogramAggregation(boundaries=histogram_bounds(delay)),
            )
            for name in ("mlops_inference_duration", "mlops_api_duration", "mlops_task_duration")
        ],
    )
    metrics.set_meter_provider(metric_provider)
    _providers.extend((provider, metric_provider))
    SystemMetricsInstrumentor(config={"process.memory.usage": None}).instrument()
    meter.create_observable_gauge(
        "mlops_queue_capacity",
        callbacks=[
            lambda _: [
                Observation(
                    int(os.getenv("QUEUE_MAX_LENGTH", "20")),
                    {"queue": os.getenv("QUEUE_NAME", "inference_tasks")},
                )
            ]
        ],
    )


def flush_telemetry() -> None:
    """Publish the final counters of the short-lived batch source."""
    for provider in _providers:
        provider.force_flush(timeout_millis=2000)


def record_error(error: Exception) -> None:
    span = trace.get_current_span()
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, type(error).__name__))
