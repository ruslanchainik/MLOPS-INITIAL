from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from mlops import main


async def test_session_closes_after_database_error(client, monkeypatch):
    session = AsyncMock(spec=AsyncSession)
    session.execute.side_effect = TimeoutError()
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(main.app.state, "session_factory", Mock(return_value=context))

    response = await client.get("/api/v1/health")

    assert response.json()["status"] == "degraded"
    session.execute.assert_awaited_once()
    context.__aexit__.assert_awaited_once()


async def test_engine_disposed_even_after_lifespan_error(monkeypatch):
    engine = Mock()
    engine.dispose = AsyncMock()
    monkeypatch.setattr(main, "create_db_engine", Mock(return_value=engine))

    with pytest.raises(RuntimeError, match="test failure"):
        async with main.lifespan(main.app):
            raise RuntimeError("test failure")

    engine.dispose.assert_awaited_once()
