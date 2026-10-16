"""Inference for the model loaded during FastAPI startup."""

import logging
from time import perf_counter
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from mlops.model import LoadedModel

router = APIRouter()
logger = logging.getLogger(__name__)


class ProcessRequest(BaseModel):
    text: Annotated[str, Field(min_length=1)]

    @field_validator("text")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class ProcessResponse(BaseModel):
    prediction: str
    model_name: str
    model_version: str


def single_binary_prediction(predictions: object) -> str:
    """Accept a one-row sequence without coercing booleans or numeric labels."""
    if isinstance(predictions, (str, bytes, dict, pd.DataFrame)):
        raise ValueError("Model must return a sequence of one class label")
    try:
        if len(predictions) != 1:  # type: ignore[arg-type]
            raise ValueError("Model must return exactly one prediction")
        value = next(iter(predictions))  # type: ignore[arg-type]
    except (TypeError, StopIteration) as error:
        raise ValueError("Model must return a sequence of one class label") from error
    if not isinstance(value, str) or value not in {"true", "false"}:
        raise ValueError("Model prediction must be the string 'true' or 'false'")
    return value


@router.post("/process", response_model=ProcessResponse)
async def process(payload: ProcessRequest, request: Request) -> ProcessResponse:
    loaded: LoadedModel | None = getattr(request.app.state, "loaded_model", None)
    if loaded is None:
        raise HTTPException(status_code=503, detail="Model unavailable")

    frame = pd.DataFrame({"text": [payload.text]})
    start = perf_counter()
    try:
        raw_prediction = await run_in_threadpool(loaded.model.predict, frame)
        prediction = single_binary_prediction(raw_prediction)
    except Exception as error:
        logger.error(
            "Inference failed model_name=%s model_version=%s error_type=%s duration_ms=%.2f",
            loaded.name,
            loaded.version,
            type(error).__name__,
            (perf_counter() - start) * 1000,
        )
        raise HTTPException(status_code=500, detail="Prediction failed") from error

    logger.info(
        "Inference completed model_name=%s model_version=%s duration_ms=%.2f",
        loaded.name,
        loaded.version,
        (perf_counter() - start) * 1000,
    )
    return ProcessResponse(
        prediction=prediction,
        model_name=loaded.name,
        model_version=loaded.version,
    )
