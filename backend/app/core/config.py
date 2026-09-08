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

    # Future LLM extension point. Safe placeholder only.
    llm_provider: str = Field(default="gemini")
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-3.5-flash")

    # Fact extraction bounds (evidence-scoped LLM calls only).
    fact_chunk_max_chars: int = Field(default=6000)
    fact_llm_timeout_s: int = Field(default=60)
    fact_llm_max_retries: int = Field(default=2)

    # Local OCR fallback (PaddleOCR): no credentials, no cloud calls.
    # Enabled by default; the pipeline auto-disables OCR at runtime when
    # the engine or its model weights are unavailable.
    ocr_enabled: bool = Field(default=True)
    ocr_provider: str = Field(default="paddleocr")
    ocr_language: str = Field(default="en")
    ocr_allow_model_download: bool = Field(default=False)
    ocr_dpi: int = Field(default=200)

    # Configurable dataset location; starter datasets live outside the repo.
    dataset_dir: str = Field(default="../starter-datasets")

    # Structured table evidence (PLAN §40). Kill-switch for table processing.
    tables_enabled: bool = Field(default=True)

    # Phase 1 provenance-first PDF ingestion (PLAN §39): extraction quality routing.
    extraction_min_chars: int = Field(default=50)
    extraction_quality_threshold: float = Field(default=0.35)


settings = Settings()
