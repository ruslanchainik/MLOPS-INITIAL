from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://mlops:mlops@localhost:5432/mlops"
    mlflow_tracking_uri: str
    mlflow_model_name: str = "claim-veracity"
    mlflow_model_alias: str = "champion"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
