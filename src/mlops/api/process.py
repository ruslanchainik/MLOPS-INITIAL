import math
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from mlops.config import Settings
from mlops.messaging import TaskMessage
from mlops.producer.publisher import PublishUncertain, QueueOverloaded, UnroutableTask

router = APIRouter()


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
    retry_after = str(max(1, math.ceil(Settings().inference_delay_seconds)))
    try:
        await request.app.state.publisher.publish_once(task)
    except (QueueOverloaded, UnroutableTask) as error:
        raise HTTPException(
            503, "Task not accepted", headers={"Retry-After": retry_after}
        ) from error
    except PublishUncertain as error:
        raise HTTPException(
            503,
            {"message": "Publication outcome unknown", "task_id": task.task_id},
            headers={"Retry-After": retry_after},
        ) from error
    return {"task_id": task.task_id}
