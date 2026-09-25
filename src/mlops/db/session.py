import os
from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine


def create_db_engine() -> AsyncEngine:
    url = URL.create(
        "postgresql+asyncpg",
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", "5432")),
        database=os.environ["POSTGRES_DB"],
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    return create_async_engine(url, pool_pre_ping=True, connect_args={"timeout": 3})


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_factory() as session:
        yield session
