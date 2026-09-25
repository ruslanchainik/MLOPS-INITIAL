from fastapi.testclient import TestClient

from mlops.main import app

client = TestClient(app)


def test_version_success():
    response = client.get("/api/v1/version")

    assert response.status_code == 200
    assert "version" in response.json()
