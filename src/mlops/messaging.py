import json
from dataclasses import dataclass
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class TaskMessage:
    task_id: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or str(UUID(self.task_id)) != self.task_id:
            raise ValueError("task_id must be a canonical UUID string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Task text must not be empty")

    @classmethod
    def new(cls, text: str) -> TaskMessage:
        return cls(task_id=str(uuid4()), text=text)

    def to_bytes(self) -> bytes:
        return json.dumps(
            {"task_id": self.task_id, "text": self.text},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
