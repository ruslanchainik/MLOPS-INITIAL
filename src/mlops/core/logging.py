import logging
import os
import re
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

from opentelemetry import trace
from pythonjsonlogger.json import JsonFormatter

task_id: ContextVar[str | None] = ContextVar("task_id", default=None)


class ContextFormatter(JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record.pop("color_message", None)
        log_record.update(
            timestamp=datetime.fromtimestamp(record.created, UTC).isoformat(),
            level=record.levelname,
            service=os.getenv("OTEL_SERVICE_NAME", "mlops"),
            env=os.getenv("DEPLOYMENT_ENVIRONMENT", "local"),
            logger=record.name,
            task_id=task_id.get() or getattr(record, "message_id", None) or None,
        )
        span = trace.get_current_span().get_span_context()
        if span.is_valid:
            log_record.update(trace_id=f"{span.trace_id:032x}", span_id=f"{span.span_id:016x}")

    def serialize_log_record(self, log_record):
        output = super().serialize_log_record(log_record)
        output = re.sub(r"(\w+(?:\+\w+)?://)[^\s/@]+:[^\s/@]+@", r"\1***@", output)
        for name, value in os.environ.items():
            if (
                value
                and len(value) >= 4
                and any(s in name for s in ("PASSWORD", "SECRET", "TOKEN"))
            ):
                output = output.replace(value, "***")
        return output


class HealthFilter(logging.Filter):
    def filter(self, record):
        return not (
            record.name == "uvicorn.access"
            and isinstance(record.args, tuple)
            and len(record.args) >= 3
            and str(record.args[2]).split("?", 1)[0] in {"/healthz", "/health", "/api/v1/health"}
        )


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    handler.setFormatter(ContextFormatter())
    handler.addFilter(HealthFilter())
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(), handlers=[handler], force=True
    )
    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.handlers.clear()
            logger.propagate = True
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
