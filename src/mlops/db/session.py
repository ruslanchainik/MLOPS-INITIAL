from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from mlops.config import settings

engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
)

session_factory = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession]:
    async with session_factory() as session:
        yield session


async def close_db() -> None:
    await engine.dispose()
