from unittest.mock import Mock

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from experiments import train
from mlflow.models import infer_signature


@pytest.fixture
def training_frame():
    return pd.DataFrame(
        {
            "text": [f"verified evidence confirms the claim {index}" for index in range(6)]
            + [f"fabricated rumor contradicts the records {index}" for index in range(12)],
            "label": ["true"] * 6 + ["false"] * 12,
        }
    )


@pytest.mark.parametrize("name", [train.BASELINE, "tfidf_logistic_regression", "tfidf_linear_svc"])
def test_saved_model_accepts_dataframe_and_returns_binary_strings(name, training_frame, tmp_path):
    model = train.build_models()[name]
    model.fit(training_frame[["text"]], training_frame["label"])
    example = pd.DataFrame({"text": ["Example claim"]})
    batch = pd.DataFrame(
        {"text": ["verified evidence confirms", "fabricated rumor contradicts", "Unseen words"]}
    )
    expected = train.predict_labels(model, batch)
    path = tmp_path / "model"

    mlflow.sklearn.save_model(
        model,
        path=str(path),
        serialization_format="cloudpickle",
        signature=infer_signature(example, train.predict_labels(model, example)),
        input_example=example,
        pip_requirements=train.MODEL_REQUIREMENTS,
    )
    loaded = mlflow.pyfunc.load_model(str(path))

    assert train.predict_labels(loaded, batch) == expected
    assert train.predict_labels(loaded, example)[0] in {"true", "false"}
    if name == train.BASELINE:
        assert expected == ["false"] * 3
    else:
        assert expected[:2] == ["true", "false"]


def test_metrics_report_minority_class_failure():
    metrics = train.calculate_metrics(["true", "false", "false", "false"], ["false"] * 4)

    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx(3 / 7)
    assert metrics["f1_true"] == 0
    assert metrics["recall_true"] == 0
    assert metrics["f1_false"] == pytest.approx(6 / 7)


@pytest.mark.parametrize(
    "predictions",
    [[True], [0], ["TRUE"], "true", [], ["true", "false"], {"false": 1}, None],
)
def test_prediction_contract_rejects_invalid_model_output(predictions):
    model = Mock()
    model.predict.return_value = predictions

    with pytest.raises(ValueError, match="Model must return"):
        train.predict_labels(model, pd.DataFrame({"text": ["Example claim"]}))


def test_champion_selection_uses_validation_and_allows_baseline_to_win():
    baseline = {"candidate": train.BASELINE, "validation_macro_f1": 0.6, "test_macro_f1": 0.1}
    learned = {"candidate": "learned", "validation_macro_f1": 0.5, "test_macro_f1": 0.9}

    assert train.select_best([baseline, learned]) is baseline
    learned["validation_macro_f1"] = 0.7
    assert train.select_best([baseline, learned]) is learned


def test_loading_prepared_data_preserves_string_labels_and_rejects_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "PREPARED_DIR", tmp_path)
    for split in ("train", "validation", "test"):
        pd.DataFrame(
            {"text": [f"{split} first claim", f"{split} second claim"], "label": ["true", "false"]}
        ).to_csv(tmp_path / f"{split}.csv", index=False)

    frames = train.load_splits()

    assert frames["train"]["label"].tolist() == ["true", "false"]
    frames["test"].loc[0, "text"] = " TRAIN   FIRST CLAIM "
    frames["test"].to_csv(tmp_path / "test.csv", index=False)
    with pytest.raises(ValueError, match="overlaps another split"):
        train.load_splits()
