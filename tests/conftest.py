import os
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from mlops.db.session import get_db
from mlops.main import app

for name, value in {
    "DB_HOST": "127.0.0.1",
    "DB_PORT": "5432",
    "POSTGRES_DB": "test",
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
}.items():
    os.environ.setdefault(name, value)


@pytest.fixture
async def client(monkeypatch):
    # Unit tests use mocked sessions; no connection is opened during startup.
    for name, value in {
        "DB_HOST": "127.0.0.1",
        "DB_PORT": "5432",
        "POSTGRES_DB": "test",
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
    }.items():
        monkeypatch.setenv(name, value)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as test_client,
    ):
        yield test_client


@pytest.fixture
def db(monkeypatch):
    session = AsyncMock(spec=AsyncSession)

    async def override_get_db():
        yield session

    # Restore the original dependency after each test.
    monkeypatch.setitem(app.dependency_overrides, get_db, override_get_db)
    return session
