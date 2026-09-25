import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from sqlalchemy.ext.asyncio import async_sessionmaker

from mlops.api.healthz import router as healthz_router
from mlops.api.v1.router import router as v1_router
from mlops.core.logging import configure_logging
from mlops.db.session import create_db_engine


configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_db_engine()

    app.state.session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )

    logger.info("Application started")

    try:
        yield
    finally:
        await engine.dispose()
        logger.info("Application stopped")


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next,
):
    start = perf_counter()

    response = await call_next(request)

    duration_ms = round(
        (perf_counter() - start) * 1000,
        2,
    )

    logger.info(
        "request_completed method=%s path=%s status_code=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


app.include_router(healthz_router)
app.include_router(v1_router)
