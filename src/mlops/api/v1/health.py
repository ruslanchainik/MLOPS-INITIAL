import asyncio
import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from mlops import __version__
from mlops.db.session import get_db


router = APIRouter()
logger = logging.getLogger(__name__)

DB_CHECK_TIMEOUT = 3


async def check_postgres(db: AsyncSession) -> dict:
    start = time.perf_counter()

    try:
        async with asyncio.timeout(DB_CHECK_TIMEOUT):
            result = await db.execute(text("SELECT version()"))
            pg_version = result.scalar_one()

        elapsed = time.perf_counter() - start
        response_time_ms = round(elapsed * 1000, 2)

        logger.info(
            "PostgreSQL health check passed response_time_ms=%s",
            response_time_ms,
        )

        return {
            "name": "postgresql",
            "status": "up",
            "version": pg_version,
            "response_time_ms": response_time_ms,
        }

    except Exception as error:
        elapsed = time.perf_counter() - start
        response_time_ms = round(elapsed * 1000, 2)

        error_message = (
            "Database check timed out"
            if isinstance(error, TimeoutError)
            else "Database unavailable"
        )

        logger.warning(
            "PostgreSQL health check failed error=%s response_time_ms=%s",
            error_message,
            response_time_ms,
        )

        return {
            "name": "postgresql",
            "status": "down",
            "error": error_message,
            "response_time_ms": response_time_ms,
        }


@router.get("/health")
async def health_check(
    db: Annotated[AsyncSession, Depends(get_db)],
):
    check = [await check_postgres(db)]

    status = (
        "ok"
        if all(component["status"] == "up" for component in check)
        else "degraded"
    )

    return {
        "status": status,
        "components": check,
        "app_version": __version__,
    }
