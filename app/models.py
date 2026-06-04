from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EventType(StrEnum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class EventMetadata(BaseModel):
    queue_depth: int | None = None
    sku_zone: str | None = None
    session_seq: int | None = None
    source: str | None = None
    track_id: str | None = None


class VisitorEvent(BaseModel):
    event_id: str = Field(min_length=8)
    store_id: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    visitor_id: str = Field(min_length=1)
    event_type: EventType
    timestamp: datetime
    zone_id: str | None = None
    dwell_ms: int = Field(ge=0)
    is_staff: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("zone_id")
    @classmethod
    def zone_is_required_for_zone_events(cls, value: str | None, info: Any) -> str | None:
        event_type = info.data.get("event_type")
        if event_type in {EventType.ZONE_ENTER, EventType.ZONE_EXIT, EventType.ZONE_DWELL} and not value:
            raise ValueError("zone_id is required for zone events")
        return value


class IngestResponse(BaseModel):
    accepted: int
    duplicates: int
    rejected: int
    errors: list[dict[str, Any]]


class StoreMetrics(BaseModel):
    store_id: str
    as_of: datetime | None
    unique_visitors: int
    conversion_rate: float
    avg_dwell_ms_by_zone: dict[str, float]
    current_queue_depth: int
    abandonment_rate: float
    purchases: int


class HealthResponse(BaseModel):
    status: str
    database: str
    stores: dict[str, dict[str, Any]]
