"""Load one immutable Registry version for the lifetime of the application."""

import logging
from dataclasses import dataclass

import mlflow
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

from mlops.config import Settings

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """The configured Registry model cannot be used at startup."""


@dataclass(frozen=True)
class LoadedModel:
    model: PyFuncModel
    name: str
    version: str


def load_registered_model(settings: Settings) -> LoadedModel:
    """Resolve the alias first, then load that exact version via pyfunc."""
    name = settings.mlflow_model_name
    alias = settings.mlflow_model_alias
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    logger.info("Loading MLflow model name=%s alias=%s", name, alias)

    try:
        version = MlflowClient(
            tracking_uri=settings.mlflow_tracking_uri
        ).get_model_version_by_alias(name, alias)
    except Exception as error:
        logger.exception("Could not resolve MLflow model name=%s alias=%s", name, alias)
        raise ModelLoadError(f"Could not resolve MLflow model {name!r} alias {alias!r}") from error

    model_uri = f"models:/{name}/{version.version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as error:
        logger.exception("Could not load MLflow model uri=%s", model_uri)
        raise ModelLoadError(f"Could not load MLflow model {model_uri!r}") from error

    logger.info("Loaded MLflow model name=%s version=%s", name, version.version)
    return LoadedModel(model=model, name=name, version=str(version.version))
