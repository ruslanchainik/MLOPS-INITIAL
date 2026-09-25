import asyncio
import math
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import OperationalError

from mlops.api.v1 import health
from mlops.db.session import get_db
from mlops.main import app


async def test_healthz_does_not_need_database(client, monkeypatch):
    async def unavailable_db():
        raise AssertionError("The liveness endpoint must not access the database")

    monkeypatch.setitem(app.dependency_overrides, get_db, unavailable_db)

    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_health_with_available_database(client, db, project_version):
    db.execute.return_value = Mock()
    db.execute.return_value.scalar_one.return_value = "PostgreSQL 17.6"

    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_version"] == project_version
    assert len(body["components"]) == 1
    component = body["components"][0]
    assert component["name"] == "postgresql"
    assert component["status"] == "up"
    assert component["version"] == "PostgreSQL 17.6"
    assert "error" not in component
    assert math.isfinite(component["response_time_ms"])
    assert component["response_time_ms"] >= 0
    db.execute.assert_awaited_once()


@pytest.mark.parametrize(
    "error",
    [
        OperationalError("SELECT version()", {}, Exception("database unavailable")),
        TimeoutError("database request timed out"),
    ],
    ids=["connection-error", "timeout"],
)
async def test_health_with_unavailable_database(client, db, project_version, error):
    db.execute.side_effect = error

    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["app_version"] == project_version
    assert len(body["components"]) == 1
    component = body["components"][0]
    assert component["name"] == "postgresql"
    assert component["status"] == "down"
    expected_error = (
        "Database check timed out" if isinstance(error, TimeoutError) else "Database unavailable"
    )
    assert component["error"] == expected_error
    assert "version" not in component
    assert math.isfinite(component["response_time_ms"])
    assert component["response_time_ms"] >= 0
    db.execute.assert_awaited_once()


@pytest.fixture
def project_version():
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(project_file.read_text(encoding="utf-8"))
    return project["project"]["version"]


async def test_version_matches_project_without_mocks(client, project_version):
    response = await client.get("/api/v1/version")

    assert response.status_code == 200
    assert response.json() == {"version": project_version}


async def test_health_limits_database_wait(client, db, monkeypatch):
    async def hanging_query(*args):
        await asyncio.Event().wait()

    db.execute.side_effect = hanging_query
    monkeypatch.setattr(health, "DB_CHECK_TIMEOUT", 0.01)

    response = await asyncio.wait_for(client.get("/api/v1/health"), timeout=1)

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["components"][0]["error"] == "Database check timed out"
