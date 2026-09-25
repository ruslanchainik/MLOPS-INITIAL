from fastapi.testclient import TestClient

from mlops.db.session import get_db
from mlops.main import app


class FakeResult:
    def scalar_one(self):
        return "PostgreSQL test"


class WorkingDatabase:
    async def execute(self, query):
        return FakeResult()


class BrokenDatabase:
    async def execute(self, query):
        raise RuntimeError("Database unavailable")


async def working_db():
    yield WorkingDatabase()


async def broken_db():
    yield BrokenDatabase()


def test_health_database_up():
    app.dependency_overrides[get_db] = working_db

    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"

    component = data["components"][0]

    assert component["name"] == "postgresql"
    assert component["status"] == "up"
    assert component["version"] == "PostgreSQL test"
    assert component["response_time_ms"] >= 0


def test_health_database_down():
    app.dependency_overrides[get_db] = broken_db

    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503

    data = response.json()

    assert data["status"] == "degraded"

    component = data["components"][0]

    assert component["name"] == "postgresql"
    assert component["status"] == "down"
    assert component["version"] is None
