import os

import pytest
from httpx import AsyncClient


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_DB_TESTS") != "1", reason="Requires running Docker Compose")
async def test_running_app_checks_real_postgres():
    base_url = os.environ.get("TEST_BASE_URL", "http://127.0.0.1:8001")
    # Local Compose requests should not use proxy settings from the environment.
    async with AsyncClient(base_url=base_url, timeout=10, trust_env=False) as client:
        response = await client.get("/api/v1/health")
        version_response = await client.get("/api/v1/version")

    assert response.status_code == 200, f"{response.url}: {response.status_code} {response.text}"
    assert version_response.status_code == 200, version_response.text
    body = response.json()
    assert body["status"] == "ok", body
    assert body["app_version"] == version_response.json()["version"]
    assert len(body["components"]) == 1
    component = body["components"][0]
    assert component["name"] == "postgresql"
    assert component["status"] == "up"
    assert component["version"].startswith("PostgreSQL ")
    assert component["response_time_ms"] >= 0
