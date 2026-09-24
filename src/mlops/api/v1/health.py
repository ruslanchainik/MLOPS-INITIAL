import time
from importlib.metadata import version as pkg_version
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from mlops.db.session import get_db

router = APIRouter()


async def check_postgres(db: AsyncSession) -> dict:
    start = time.perf_counter()
    try:
        result = await db.execute(text("Select version()"))
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
            "error": str(e),
            "response_time_ms": round(elapsed * 1000, 2),
        }


@router.get("/health")
async def health_check(db: Annotated[AsyncSession, Depends(get_db)]):
    check = [await check_postgres(db)]

    status = "ok" if all(c["status"] == "up" for c in check) else "degraded"

    return {"status": status, "components": check, "app_version": pkg_version("mlops")}
