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

    # LLM provider (Groq, OpenAI-compatible). Safe placeholders only.
    llm_provider: str = Field(default="groq")
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="openai/gpt-oss-120b")

    # Local Ollama inference (no API key, no cloud calls). The host owns
    # the model; the backend only talks HTTP to OLLAMA_BASE_URL. From
    # Docker Desktop, host.docker.internal reaches the Windows host.
    ollama_base_url: str = Field(default="http://host.docker.internal:11434")
    ollama_model: str = Field(default="qwen3.5:9b")
    ollama_think: bool | None = Field(default=None)
    ollama_timeout_s: int = Field(default=120)
    ollama_max_retries: int = Field(default=2)
    ollama_backoff_base_s: float = Field(default=1.0)
    ollama_backoff_max_s: float = Field(default=30.0)
    ollama_keep_alive: int = Field(default=-1)

    # Fact extraction bounds (evidence-scoped LLM calls only). The chunk
    # bound keeps normal requests near ~1.5-2K total tokens against the
    # 8K TPM tier: one bounded chunk per serial request, no packing.
    fact_chunk_max_chars: int = Field(default=8000)
    # Operability bound: limits extraction to representative chunks when
    # set; None processes all eligible chunks. Bounded by default because
    # exhaustive extraction over a 100-page report is thousands of serial
    # requests; select_representative_chunks spreads the sample across
    # the document and prefers table/numeric-dense content.
    fact_max_chunks_per_document: int | None = Field(default=60)
    fact_llm_timeout_s: int = Field(default=60)
    # Model context window. This drives BOTH chunk sizing and the num_ctx
    # sent to Ollama -- a server that is not told num_ctx silently
    # truncates to its own small default and the model then emits
    # unparseable JSON. Raise both this and fact_chunk_max_chars together:
    # whichever is smaller binds.
    llm_context_tokens: int = Field(default=16384)
    extraction_reserved_output_tokens: int = Field(default=4096)
    extraction_safety_margin_tokens: int = Field(default=200)

    # Groq request management (free-tier aware). Kept explicit and
    # separate: RPM spacing and retry/backoff bounds are independent
    # knobs. Requests are paced only by start-interval; per-request
    # token totals are usage-logged for observability.
    groq_min_request_interval_s: float = Field(default=2.5)
    groq_max_retries: int = Field(default=5)
    groq_backoff_base_s: float = Field(default=1.0)
    groq_backoff_max_s: float = Field(default=60.0)

    # Internal ML service (PaddleOCR + MiniLM over HTTP on the Compose
    # network). Local laptop runs without Compose can point this at a
    # locally started ML service or leave it unreachable (OCR/embedding
    # degrade to native extraction / lexical ranking).
    ml_service_url: str = Field(default="http://ml:8001")

    # Candidate discovery + relationship reasoning (deterministic-first).
    matching_top_k: int = Field(default=10)
    matching_similarity_floor: float = Field(default=0.25)
    matching_numeric_tolerance: float = Field(default=0.01)
    matching_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

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
