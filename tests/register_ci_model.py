"""Register a small model in the isolated CI Registry before starting the API.

poetry run python tests/register_ci_model.py
"""

import os
import sys
from importlib.metadata import version

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier


def main() -> None:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    name = os.environ["MLFLOW_MODEL_NAME"]
    alias = os.environ["MLFLOW_MODEL_ALIAS"]
    mlflow.set_experiment("ci_inference")
    frame = pd.DataFrame({"text": ["A true claim", "A false claim"]})
    model = DummyClassifier(strategy="constant", constant="false")
    model.fit(frame, ["true", "false"])

    with mlflow.start_run(run_name="ci_constant_model"):
        mlflow.set_tag("purpose", "API integration fixture, not a quality experiment")
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
            signature=infer_signature(frame, model.predict(frame)),
            input_example=frame,
            pip_requirements=[
                f"{package}=={version(package)}"
                for package in ("mlflow", "scikit-learn", "pandas", "numpy", "scipy", "cloudpickle")
            ],
        )
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        if list(loaded.predict(pd.DataFrame({"text": ["Example claim"]}))) != ["false"]:
            raise ValueError("CI model does not satisfy the DataFrame/string prediction contract")
        registered = mlflow.register_model(info.model_uri, name)
        MlflowClient().set_registered_model_alias(name, alias, registered.version)
        print(f"CI model: {name} version {registered.version}, alias {alias}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    main()
