import logging
from time import perf_counter

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from mlops.db.session import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health(
    db: AsyncSession = Depends(get_db),
):
    start = perf_counter()

    try:
        result = await db.execute(text("SELECT version()"))

        postgres_version = result.scalar_one()

        response_time_ms = round(
            (perf_counter() - start) * 1000,
            2,
        )

        logger.info(
            "PostgreSQL health check passed response_time_ms=%s",
            response_time_ms,
        )

        return {
            "status": "ok",
            "components": [
                {
                    "name": "postgresql",
                    "status": "up",
                    "version": postgres_version,
                    "response_time_ms": response_time_ms,
                }
            ],
        }

    except Exception:
        response_time_ms = round(
            (perf_counter() - start) * 1000,
            2,
        )

        logger.exception(
            "PostgreSQL health check failed response_time_ms=%s",
            response_time_ms,
        )

        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "components": [
                    {
                        "name": "postgresql",
                        "status": "down",
                        "version": None,
                        "response_time_ms": response_time_ms,
                    }
                ],
            },
        )
