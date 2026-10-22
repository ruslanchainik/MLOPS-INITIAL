from sqlalchemy import Column, DateTime, Float, MetaData, String, Table, Text, func
from sqlalchemy.dialects.postgresql import UUID

metadata = MetaData()
task_results = Table(
    "task_results",
    metadata,
    Column("task_id", UUID(as_uuid=False), primary_key=True),
    Column("text", Text, nullable=False),
    Column("prediction", String, nullable=False),
    Column("model_name", String, nullable=False),
    Column("model_version", String, nullable=False),
    Column("processing_seconds", Float, nullable=False),
    Column("completed_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)
