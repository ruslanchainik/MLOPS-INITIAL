import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from sqlalchemy.ext.asyncio import async_sessionmaker

from mlops.api.healthz import router as healthz_router
from mlops.api.process import router as process_router
from mlops.api.v1.router import router as v1_router
from mlops.config import Settings
from mlops.core import telemetry
from mlops.core.logging import configure_logging, task_id
from mlops.db.session import create_db_engine, initialize_database
from mlops.producer.publisher import QueueConfig, RabbitPublisher

configure_logging()
telemetry.configure_telemetry("mlops-api")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_db_engine()

    app.state.session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )

    logger.info("startup")

    try:
        settings = Settings()
        await initialize_database(engine)
        async with RabbitPublisher(
            settings.rabbitmq_url,
            QueueConfig(settings.queue_name, settings.queue_max_length or None),
            confirm_timeout=1,
        ) as publisher:
            app.state.publisher = publisher
            yield
    finally:
        logger.info("shutdown_started")
        app.state.publisher = None
        await engine.dispose()
        logger.info("shutdown_completed")


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next,
):
    start = perf_counter()

    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        duration = perf_counter() - start
        route = getattr(request.scope.get("route"), "path", "unmatched")
        method = request.method if request.method in {"GET", "POST"} else "OTHER"
        labels = {"route": route, "method": method, "status_code": status}
        telemetry.api_requests.add(1, labels)
        telemetry.api_duration.record(duration, {"route": route, "method": method})
        if route not in {"/healthz", "/api/v1/health"}:
            token = task_id.set(getattr(request.state, "task_id", None))
            try:
                logger.info(
                    "request_completed", extra={**labels, "duration_ms": round(duration * 1000, 2)}
                )
            finally:
                task_id.reset(token)


app.include_router(healthz_router)
app.include_router(process_router)
app.include_router(v1_router)
FastAPIInstrumentor.instrument_app(
    app, excluded_urls="healthz,api/v1/health", exclude_spans=["receive", "send"]
)
