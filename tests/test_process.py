import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from mlops import main
from mlops.api.process import single_binary_prediction
from mlops.config import Settings
from mlops.model import LoadedModel, ModelLoadError, load_registered_model


async def test_process_uses_loaded_model_and_dataframe(client, mock_registered_model):
    response = await client.post("/process", json={"text": "The Earth is flat."})

    assert response.status_code == 200
    assert response.json() == {
        "prediction": "false",
        "model_name": "claim-veracity",
        "model_version": "2",
    }
    frame = mock_registered_model.return_value.model.predict.call_args.args[0]
    assert isinstance(frame, pd.DataFrame)
    assert frame.to_dict(orient="list") == {"text": ["The Earth is flat."]}


@pytest.mark.parametrize("text", ["", "  \t  "])
async def test_process_rejects_empty_text(client, text):
    response = await client.post("/process", json={"text": text})
    assert response.status_code == 422


async def test_process_rejects_invalid_json(client):
    response = await client.post(
        "/process", content='{"text":', headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


async def test_model_is_loaded_once_for_many_requests(client, mock_registered_model):
    for _ in range(3):
        assert (await client.post("/process", json={"text": "A claim"})).status_code == 200
    mock_registered_model.assert_called_once()
    assert mock_registered_model.return_value.model.predict.call_count == 3


async def test_predict_does_not_block_event_loop(client, mock_registered_model):
    caller_thread = threading.get_ident()
    predict_thread = []

    def predict(frame):
        predict_thread.append(threading.get_ident())
        return ["true"]

    mock_registered_model.return_value.model.predict.side_effect = predict
    response = await client.post("/process", json={"text": "A claim"})
    assert response.json()["prediction"] == "true"
    assert predict_thread[0] != caller_thread


@pytest.mark.parametrize("result", [[True], [1], ["TRUE"], ["true", "false"], "true", []])
def test_ambiguous_predictions_are_rejected(result):
    with pytest.raises(ValueError):
        single_binary_prediction(result)


def test_pandas_series_prediction_is_accepted():
    assert single_binary_prediction(pd.Series(["true"], index=[7])) == "true"


async def test_predict_error_is_private(client, mock_registered_model):
    mock_registered_model.return_value.model.predict.side_effect = RuntimeError("secret details")
    response = await client.post("/process", json={"text": "A claim"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}
    assert "secret details" not in response.text


async def test_inference_log_does_not_contain_claim(client, mock_registered_model, caplog):
    claim = "A private claim that must stay out of logs"
    mock_registered_model.return_value.model.predict.side_effect = RuntimeError(claim)

    response = await client.post("/process", json={"text": claim})

    assert response.status_code == 500
    assert claim not in caplog.text


async def test_invalid_model_output_is_private(client, mock_registered_model):
    mock_registered_model.return_value.model.predict.return_value = [0]
    response = await client.post("/process", json={"text": "A claim"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}


async def test_process_reports_unavailable_model(client):
    main.app.state.loaded_model = None
    response = await client.post("/process", json={"text": "A claim"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Model unavailable"}


async def test_alias_change_takes_effect_after_restart(monkeypatch, mock_registered_model):
    from httpx import ASGITransport, AsyncClient

    version = "2"
    model = Mock()
    model.predict.return_value = ["true"]

    def loader(settings):
        return LoadedModel(model=model, name=settings.mlflow_model_name, version=version)

    monkeypatch.setattr(main, "load_registered_model", Mock(side_effect=loader))
    async with (
        main.lifespan(main.app),
        AsyncClient(transport=ASGITransport(app=main.app), base_url="http://test") as client,
    ):
        first = await client.post("/process", json={"text": "A claim"})
        version = "1"
        unchanged = await client.post("/process", json={"text": "A claim"})
    async with (
        main.lifespan(main.app),
        AsyncClient(transport=ASGITransport(app=main.app), base_url="http://test") as client,
    ):
        second = await client.post("/process", json={"text": "A claim"})

    assert first.json()["model_version"] == "2"
    assert unchanged.json()["model_version"] == "2"
    assert second.json()["model_version"] == "1"
    assert main.load_registered_model.call_count == 2


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


async def test_startup_fails_without_model_and_disposes_database(
    monkeypatch, mock_registered_model
):
    engine = Mock()
    engine.dispose = AsyncMock()
    monkeypatch.setattr(main, "create_db_engine", Mock(return_value=engine))
    mock_registered_model.side_effect = ModelLoadError("No champion alias")

    with pytest.raises(ModelLoadError, match="No champion alias"):
        async with main.lifespan(main.app):
            pass
    engine.dispose.assert_awaited_once()
    assert main.app.state.loaded_model is None
