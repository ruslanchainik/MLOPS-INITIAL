from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://mlops:mlops@localhost:5432/mlops"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_model_name: str = "claim-veracity"
    mlflow_model_alias: str = "champion"
    rabbitmq_url: str = "amqp://guest:guest@127.0.0.1:5673/"
    queue_name: str = "inference_tasks"
    queue_max_length: int = Field(default=20, ge=0)
    prefetch_count: int = Field(default=1, ge=1)
    inference_delay_seconds: float = Field(default=0.05, ge=0, allow_inf_nan=False)
    worker_metrics_file: str | None = None
    worker_ready_file: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
