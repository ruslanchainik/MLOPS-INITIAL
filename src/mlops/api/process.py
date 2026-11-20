import logging
import math
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from mlops.config import Settings
from mlops.core import telemetry
from mlops.core.logging import task_id
from mlops.messaging import TaskMessage
from mlops.producer.publisher import PublishUncertain, QueueOverloaded, UnroutableTask

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


@router.post("/api/v1/tasks", status_code=202)
async def process(payload: ProcessRequest, request: Request) -> dict[str, str]:
    task = TaskMessage.new(payload.text)
    request.state.task_id = task.task_id
    token = task_id.set(task.task_id)
    retry_after = str(max(1, math.ceil(Settings().inference_delay_seconds)))
    try:
        await request.app.state.publisher.publish_once(task)
    except (QueueOverloaded, UnroutableTask) as error:
        telemetry.record_error(error)
        logger.warning("task_rejected", extra={"status_code": 503, "retry_after": retry_after})
        raise HTTPException(
            503, "Task not accepted", headers={"Retry-After": retry_after}
        ) from error
    except PublishUncertain as error:
        telemetry.record_error(error)
        logger.warning("publication_uncertain", extra={"status_code": 503})
        raise HTTPException(
            503,
            {"message": "Publication outcome unknown", "task_id": task.task_id},
            headers={"Retry-After": retry_after},
        ) from error
    finally:
        task_id.reset(token)
    return {"task_id": task.task_id}
