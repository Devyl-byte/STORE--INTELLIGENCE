from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


def iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class EventEmitter:
    store_id: str
    camera_id: str
    source: str
    session_seq: dict[str, int] = field(default_factory=dict)

    def event(
        self,
        visitor_id: str,
        event_type: str,
        timestamp: datetime,
        zone_id: str | None = None,
        dwell_ms: int = 0,
        is_staff: bool = False,
        confidence: float = 0.74,
        queue_depth: int | None = None,
        sku_zone: str | None = None,
        track_id: str | None = None,
    ) -> dict:
        self.session_seq[visitor_id] = self.session_seq.get(visitor_id, 0) + 1
        return {
            "event_id": str(uuid.uuid4()),
            "store_id": self.store_id,
            "camera_id": self.camera_id,
            "visitor_id": visitor_id,
            "event_type": event_type,
            "timestamp": iso(timestamp),
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": is_staff,
            "confidence": round(max(0.0, min(confidence, 1.0)), 3),
            "metadata": {
                "queue_depth": queue_depth,
                "sku_zone": sku_zone,
                "session_seq": self.session_seq[visitor_id],
                "source": self.source,
                "track_id": track_id,
            },
        }


def write_jsonl(events: Iterable[dict], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, separators=(",", ":")) + "\n")
            count += 1
    return count
