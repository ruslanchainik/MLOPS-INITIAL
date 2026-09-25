import asyncio
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from mlops import __version__
from mlops.db.session import get_db

router = APIRouter()
DB_CHECK_TIMEOUT = 3


async def check_postgres(db: AsyncSession) -> dict:
    start = time.perf_counter()
    try:
        async with asyncio.timeout(DB_CHECK_TIMEOUT):
            result = await db.execute(text("SELECT version()"))
            pg_version = result.scalar_one()
        elapsed = time.perf_counter() - start
        return {
            "name": "postgresql",
            "status": "up",
            "version": pg_version,
            "response_time_ms": round(elapsed * 1000, 2),
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "name": "postgresql",
            "status": "down",
            "error": "Database check timed out"
            if isinstance(e, TimeoutError)
            else "Database unavailable",
            "response_time_ms": round(elapsed * 1000, 2),
        }


@router.get("/health")
async def health_check(db: Annotated[AsyncSession, Depends(get_db)]):
    check = [await check_postgres(db)]

    status = "ok" if all(c["status"] == "up" for c in check) else "degraded"

    return {"status": status, "components": check, "app_version": __version__}
