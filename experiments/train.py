"""Compare binary MultiFC models and publish a Registry champion.

poetry run python experiments/train.py
"""

import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "multi_fc"
PREPARED_DIR = DATA_DIR / "prepared"
OUTPUT_DIR = DATA_DIR / "training"
EXPERIMENT = "multi_fc_binary"
MODEL_NAME = "claim-veracity"
MODEL_ALIAS = "champion"
LABELS = ("true", "false")
RANDOM_STATE = 42
BASELINE = "baseline_most_frequent"
MODEL_REQUIREMENTS = [
    f"{package}=={version(package)}"
    for package in ("mlflow", "scikit-learn", "pandas", "numpy", "scipy", "cloudpickle")
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_splits() -> dict[str, pd.DataFrame]:
    frames = {}
    keys = {}
    for name in ("train", "validation", "test"):
        path = PREPARED_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                "Prepared data not found; run experiments/prepare_data.py first"
            )
        frame = pd.read_csv(path, dtype={"text": "str", "label": "str"})
        if list(frame.columns) != ["text", "label"] or frame.isna().any().any():
            raise ValueError(f"{name} must contain non-null text and label columns")
        if set(frame["label"]) != set(LABELS) or frame["text"].str.strip().eq("").any():
            raise ValueError(f"{name} must contain both string labels and non-empty texts")
        normalized = (
            frame["text"]
            .str.normalize("NFKC")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.casefold()
        )
        if normalized.duplicated().any():
            raise ValueError(f"Duplicate texts in prepared {name}")
        keys[name] = set(normalized)
        if any(keys[name] & keys[previous] for previous in frames):
            raise ValueError(f"Prepared {name} overlaps another split")
        frames[name] = frame
    return frames


def build_models() -> dict[str, DummyClassifier | Pipeline]:
    models = {BASELINE: DummyClassifier(strategy="most_frequent")}
    classifiers = {
        "tfidf_logistic_regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "tfidf_linear_svc": LinearSVC(
            C=1.0, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE
        ),
    }
    for name, classifier in classifiers.items():
        # A string column selector passes the text Series, not the DataFrame's column names.
        features = ColumnTransformer(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True
                    ),
                    "text",
                )
            ]
        )
        models[name] = Pipeline([("features", features), ("classifier", classifier)])
    return models


def predict_labels(model, frame: pd.DataFrame) -> list[str]:
    predictions = model.predict(frame[["text"]])
    if isinstance(predictions, (str, bytes, dict, pd.DataFrame)):
        raise ValueError("Model must return a sequence of string labels")
    try:
        values = list(predictions)
    except TypeError as error:
        raise ValueError("Model must return a sequence of string labels") from error
    if len(values) != len(frame) or any(
        not isinstance(value, str) or value not in LABELS for value in values
    ):
        raise ValueError("Model must return one 'true' or 'false' string per input row")
    return values


def calculate_metrics(labels, predictions) -> dict[str, float]:
    report = classification_report(
        labels, predictions, labels=list(LABELS), output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(LABELS), average="macro")),
        **{
            f"{metric}_{label}": float(report[label][field])
            for label in LABELS
            for metric, field in (
                ("precision", "precision"),
                ("recall", "recall"),
                ("f1", "f1-score"),
            )
        },
    }


def evaluate(model, frame: pd.DataFrame, split: str) -> tuple[dict[str, float], list[str]]:
    start = perf_counter()
    predictions = predict_labels(model, frame)
    metrics = {
        f"{split}_{key}": value
        for key, value in calculate_metrics(frame["label"], predictions).items()
    }
    metrics[f"{split}_predict_seconds"] = perf_counter() - start
    mlflow.log_metrics(metrics)
    mlflow.log_dict(
        classification_report(
            frame["label"], predictions, labels=list(LABELS), output_dict=True, zero_division=0
        ),
        f"{split}/classification_report.json",
    )
    outcomes = frame.assign(prediction=predictions)
    mlflow.log_table(
        outcomes.loc[outcomes["label"] != outcomes["prediction"]].head(50),
        f"{split}/errors.json",
    )

    figure, ax = plt.subplots(figsize=(5, 4))
    try:
        ConfusionMatrixDisplay.from_predictions(
            frame["label"], predictions, labels=list(LABELS), cmap="Blues", colorbar=False, ax=ax
        )
        ax.set_title(f"{split.capitalize()} confusion matrix")
        figure.tight_layout()
        mlflow.log_figure(figure, f"{split}/confusion_matrix.png")
    finally:
        plt.close(figure)
    return metrics, predictions


def log_dataset(frame: pd.DataFrame, split: str) -> None:
    path = PREPARED_DIR / f"{split}.csv"
    dataset = mlflow.data.from_pandas(
        frame, source=path.resolve().as_uri(), targets="label", name=f"multi_fc_binary_{split}"
    )
    mlflow.log_input(dataset, context="training" if split == "train" else split)
    mlflow.log_params({f"sha256_{split}": file_sha256(path), f"rows_{split}": len(frame)})


def train_candidate(name: str, model, frames: dict[str, pd.DataFrame]) -> dict:
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"task": "binary_claim_veracity", "candidate": name})
        mlflow.log_params({"random_state": RANDOM_STATE, "label_rule": "exact true/false only"})
        classifier = model.named_steps["classifier"] if isinstance(model, Pipeline) else model
        mlflow.log_params(
            {f"classifier_{key}": value for key, value in classifier.get_params().items()}
        )
        if isinstance(model, Pipeline):
            vectorizer = model.named_steps["features"].transformers[0][1]
            mlflow.log_params(
                {f"tfidf_{key}": value for key, value in vectorizer.get_params().items()}
            )
        for split in ("train", "validation"):
            log_dataset(frames[split], split)
        mlflow.log_artifact(str(PREPARED_DIR / "summary.json"), "data")
        mlflow.log_artifact(__file__, "code")
        mlflow.log_artifact(str(Path(__file__).with_name("prepare_data.py")), "code")

        start = perf_counter()
        model.fit(frames["train"][["text"]], frames["train"]["label"])
        mlflow.log_metric("fit_seconds", perf_counter() - start)
        metrics, predictions = evaluate(model, frames["validation"], "validation")

        example = pd.DataFrame({"text": ["Example claim"]})
        expected = predict_labels(model, example)
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
            signature=infer_signature(example, expected),
            input_example=example,
            pip_requirements=MODEL_REQUIREMENTS,
        )
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        if (
            predict_labels(loaded, example) != expected
            or predict_labels(loaded, frames["validation"]) != predictions
        ):
            raise ValueError("Reloaded model predictions differ from the trained model")
        mlflow.log_dict(
            {
                "input": example.to_dict(orient="list"),
                "prediction": expected,
                "validation_matches": True,
            },
            "model_contract.json",
        )
        print(f"{name}: validation macro F1 = {metrics['validation_macro_f1']:.4f}")
        return {
            "candidate": name,
            "run_id": run.info.run_id,
            "model_uri": info.model_uri,
            **metrics,
        }


def select_best(results: list[dict]) -> dict:
    # A tie keeps the first candidate in the fixed experiment order.
    return max(results, key=lambda result: result["validation_macro_f1"])


def main() -> None:
    frames = load_splits()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    # Keep artifact downloads behind MLflow; RustFS hostnames are internal to Compose.
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    results = [train_candidate(name, model, frames) for name, model in build_models().items()]
    best = select_best(results)
    for result in results:
        # Always register both learned models; include the baseline if it wins.
        if result["candidate"] == BASELINE and result is not best:
            continue
        registered = mlflow.register_model(
            result["model_uri"], MODEL_NAME, tags={"candidate": result["candidate"]}
        )
        result["registry_version"] = str(registered.version)
        with mlflow.start_run(run_id=result["run_id"]):
            mlflow.set_tags({"registry_model": MODEL_NAME, "registry_version": registered.version})

    model_uri = f"models:/{MODEL_NAME}/{best['registry_version']}"
    selected = mlflow.pyfunc.load_model(model_uri)
    with mlflow.start_run(run_id=best["run_id"]):
        log_dataset(frames["test"], "test")
        test_metrics, _ = evaluate(selected, frames["test"], "test")
        summary = {
            "selection_metric": "validation_macro_f1",
            "candidate": best["candidate"],
            "run_id": best["run_id"],
            "model_name": MODEL_NAME,
            "model_version": best["registry_version"],
            "model_alias": MODEL_ALIAS,
            "validation_macro_f1": best["validation_macro_f1"],
            **test_metrics,
        }
        comparison = pd.DataFrame(results)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(OUTPUT_DIR / "comparison.csv", index=False)
        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        mlflow.log_table(comparison, "comparison.json")
        mlflow.log_dict(summary, "training_summary.json")
        # Promote only after saving, reloading and evaluating the exact Registry version.
        client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, best["registry_version"])

    print(f"Champion: {best['candidate']} ({model_uri})")
    print(f"Final test macro F1: {test_metrics['test_macro_f1']:.4f}")


if __name__ == "__main__":
    # MLflow prints Unicode run links, including on legacy Windows consoles.
    sys.stdout.reconfigure(errors="replace")
    main()
