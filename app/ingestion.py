from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.database import connect, insert_event, insert_rejection
from app.models import IngestResponse, VisitorEvent


MAX_BATCH_SIZE = 500


def ingest_batch(payload: list[dict[str, Any]]) -> IngestResponse:
    if len(payload) > MAX_BATCH_SIZE:
        payload = payload[:MAX_BATCH_SIZE]
    accepted = duplicates = rejected = 0
    errors: list[dict[str, Any]] = []
    with connect() as conn:
        for idx, item in enumerate(payload):
            try:
                event = VisitorEvent.model_validate(item)
            except ValidationError as exc:
                rejected += 1
                error = {"index": idx, "error": exc.errors()}
                errors.append(error)
                insert_rejection(conn, item, str(exc))
                continue
            if insert_event(conn, event):
                accepted += 1
            else:
                duplicates += 1
    return IngestResponse(accepted=accepted, duplicates=duplicates, rejected=rejected, errors=errors)
