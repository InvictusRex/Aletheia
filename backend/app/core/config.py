"""Application configuration (Phase 0).

Environment-based settings with safe defaults. Secrets are never committed;
copy ``.env.example`` to ``.env`` locally and configure values there.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Backend settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = Field(default="aletheia-backend")
    environment: str = Field(default="local")

    # PostgreSQL + pgvector. In Docker Compose the host is ``db``;
    # override with ``DATABASE_URL`` for local runs.
    database_url: str = Field(
        default="postgresql+psycopg://aletheia:aletheia@db:5432/aletheia"
    )

    # Future extension points (Phase 3/5). Safe placeholders only.
    llm_provider: str = Field(default="gemini")
    gemini_api_key: str = Field(default="")
    aws_region: str = Field(default="")
    textract_role_arn: str = Field(default="")

    # Configurable dataset location; starter datasets live outside the repo.
    dataset_dir: str = Field(default="../starter-datasets")

    # Phase 1 provenance-first PDF ingestion (PLAN §39): extraction quality routing.
    extraction_min_chars: int = Field(default=50)
    extraction_quality_threshold: float = Field(default=0.35)


settings = Settings()
