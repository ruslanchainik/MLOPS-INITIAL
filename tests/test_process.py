from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from mlops.config import Settings
from mlops.model import ModelLoadError, load_registered_model, single_binary_prediction
from mlops.producer.publisher import PublishUncertain, QueueOverloaded


async def test_process_publishes_without_inference(client, mock_services):
    response = await client.post("/api/v1/tasks", json={"text": "The Earth is flat."})
    assert response.status_code == 202
    task = mock_services.publish_once.await_args.args[0]
    assert response.json() == {"task_id": task.task_id}
    assert task.text == "The Earth is flat."
    mock_services.publish_once.assert_awaited_once()


async def test_old_process_route_is_removed(client, mock_services):
    assert (await client.post("/process", json={"text": "test"})).status_code == 404
    mock_services.publish_once.assert_not_awaited()


@pytest.mark.parametrize("text", ["", "  "])
async def test_process_rejects_empty_text(client, text):
    assert (await client.post("/api/v1/tasks", json={"text": text})).status_code == 422


@pytest.mark.parametrize("error", [QueueOverloaded(), PublishUncertain()])
async def test_process_sheds_load_without_retry(client, mock_services, error):
    mock_services.publish_once.side_effect = error
    response = await client.post("/api/v1/tasks", json={"text": "test"})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    mock_services.publish_once.assert_awaited_once()
    assert isinstance(response.json()["detail"], dict) == isinstance(error, PublishUncertain)


@pytest.mark.parametrize("result", [[True], [1], ["TRUE"], ["true", "false"], "true", []])
def test_ambiguous_predictions_are_rejected(result):
    with pytest.raises(ValueError):
        single_binary_prediction(result)


def test_pandas_series_prediction_is_accepted():
    assert single_binary_prediction(pd.Series(["true"], index=[7])) == "true"


def test_loader_pins_alias_to_specific_version(monkeypatch):
    from mlops import model as model_module

    version = SimpleNamespace(version="7")
    client = Mock()
    client.get_model_version_by_alias.return_value = version
    mock_client = Mock(return_value=client)
    mock_load = Mock(return_value=Mock())
    monkeypatch.setattr(model_module, "MlflowClient", mock_client)
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model", mock_load)
    monkeypatch.setattr(model_module.mlflow, "set_tracking_uri", Mock())

    loaded = load_registered_model(Settings())

    mock_client.assert_called_once_with(tracking_uri="http://mlflow.test:5000")
    client.get_model_version_by_alias.assert_called_once_with("claim-veracity", "champion")
    mock_load.assert_called_once_with("models:/claim-veracity/7")
    assert loaded.version == "7"


@pytest.mark.parametrize(
    "error", [ConnectionError("MLflow unavailable"), LookupError("alias missing")]
)
def test_loader_reports_registry_failures(monkeypatch, error):
    from mlops import model as model_module

    client = Mock()
    client.get_model_version_by_alias.side_effect = error
    monkeypatch.setattr(model_module, "MlflowClient", Mock(return_value=client))
    monkeypatch.setattr(model_module.mlflow, "set_tracking_uri", Mock())
    with pytest.raises(ModelLoadError, match="Could not resolve MLflow model"):
        load_registered_model(Settings())


def test_loader_reports_artifact_or_dependency_failure(monkeypatch):
    from mlops import model as model_module

    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="3")
    monkeypatch.setattr(model_module, "MlflowClient", Mock(return_value=client))
    monkeypatch.setattr(model_module.mlflow, "set_tracking_uri", Mock())
    monkeypatch.setattr(
        model_module.mlflow.pyfunc, "load_model", Mock(side_effect=ImportError("missing package"))
    )
    with pytest.raises(ModelLoadError, match="models:/claim-veracity/3"):
        load_registered_model(Settings())
