"""Document model (PLAN Section 6)."""

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class IngestionStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    filename: str
    title: str | None = None
    source: str | None = None

    page_count: int = Field(ge=0)

    file_sha256: str
    ingestion_status: IngestionStatus = IngestionStatus.UPLOADED
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
