# PROMPT: Generate tests for a FastAPI event ingestion service that validates Store Intelligence
# CCTV events, rejects malformed records, and treats duplicate event_id values idempotently.
# CHANGES MADE: I narrowed the generated idea to direct ingestion-function tests so the assertions
# stay stable without needing a running HTTP server. Fixed event_id fixtures to use 8+ char IDs
# to satisfy the model min_length=8 constraint (UUIDs are the production format; short IDs
# in test fixtures incorrectly triggered the rejection path).

from datetime import datetime, timezone

from app import config
from app.database import init_db
from app.ingestion import ingest_batch


def event(event_id="evt-0001"):
    return {
        "event_id": event_id,
        "store_id": "ST1008",
        "camera_id": "CAM_1",
        "visitor_id": "VIS_1",
        "event_type": "ENTRY",
        "timestamp": datetime(2026, 4, 10, 7, 0, tzinfo=timezone.utc).isoformat(),
        "zone_id": None,
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.8,
        "metadata": {"session_seq": 1},
    }


def test_ingest_is_idempotent(tmp_path):
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    init_db()

    first = ingest_batch([event()])
    second = ingest_batch([event()])

    assert first.accepted == 1
    assert first.duplicates == 0
    assert second.accepted == 0
    assert second.duplicates == 1


def test_ingest_partial_success(tmp_path):
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    init_db()

    result = ingest_batch([event("evt-good1"), {"event_id": "bad"}])

    assert result.accepted == 1
    assert result.rejected == 1
    assert result.errors


def test_ingest_batch_size_limit(tmp_path):
    """Batches over 500 are truncated to 500 — not rejected outright."""
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    init_db()

    big_batch = [event(f"evt-{i:05d}") for i in range(520)]
    result = ingest_batch(big_batch)
    # Max 500 accepted; remaining 20 silently truncated
    assert result.accepted == 500
    assert result.duplicates == 0
    assert result.rejected == 0


def test_ingest_rejects_invalid_event_type(tmp_path):
    """An event with an unrecognised event_type must be rejected cleanly."""
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    init_db()

    bad = event("evt-bad-et")
    bad["event_type"] = "HOVER"  # not in EventType enum
    result = ingest_batch([bad])
    assert result.rejected == 1
    assert result.accepted == 0
