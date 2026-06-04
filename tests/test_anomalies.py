# PROMPT: Add anomaly tests for billing queue spikes, dead zones, and conversion drops in a
# retail intelligence API. Ensure test data produces actionable WARN/CRITICAL results.
# CHANGES MADE: Fixed event_id to use 8+ char IDs (was "queue-1", 7 chars — rejected by
# model validator). Added dead-zone test and conversion-drop test for full anomaly coverage.

from datetime import datetime, timedelta, timezone

from app import config
from app.analytics import compute_anomalies
from app.database import init_db
from app.ingestion import ingest_batch


def _entry(eid, vid, ts, zone_id=None, event_type="ENTRY", queue_depth=None, is_staff=False):
    return {
        "event_id": eid,
        "store_id": "ST1008",
        "camera_id": "CAM_3",
        "visitor_id": vid,
        "event_type": event_type,
        "timestamp": ts.isoformat(),
        "zone_id": zone_id,
        "dwell_ms": 0,
        "is_staff": is_staff,
        "confidence": 0.88,
        "metadata": {"queue_depth": queue_depth, "session_seq": 1},
    }


def test_queue_spike_anomaly(tmp_path):
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    config.STORE_LAYOUT_PATH = config.DATA_DIR / "store_layout.json"
    init_db()
    ts = datetime(2026, 4, 10, 7, 0, tzinfo=timezone.utc)
    ingest_batch([
        _entry("queue-001", "VIS_1", ts, "CASH_COUNTER", "BILLING_QUEUE_JOIN", queue_depth=8),
    ])

    anomalies = compute_anomalies("ST1008")["anomalies"]
    assert any(
        item["type"] == "BILLING_QUEUE_SPIKE" and item["severity"] == "CRITICAL"
        for item in anomalies
    )


def test_queue_warn_anomaly(tmp_path):
    """queue_depth of 5 or 6 triggers WARN, not CRITICAL."""
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    config.STORE_LAYOUT_PATH = config.DATA_DIR / "store_layout.json"
    init_db()
    ts = datetime(2026, 4, 10, 7, 0, tzinfo=timezone.utc)
    ingest_batch([
        _entry("queue-002", "VIS_2", ts, "CASH_COUNTER", "BILLING_QUEUE_JOIN", queue_depth=6),
    ])

    anomalies = compute_anomalies("ST1008")["anomalies"]
    assert any(
        item["type"] == "BILLING_QUEUE_SPIKE" and item["severity"] == "WARN"
        for item in anomalies
    )


def test_dead_zone_anomaly(tmp_path):
    """Merchandise zones with no visits in the last 30 min trigger DEAD_ZONE INFO."""
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    config.STORE_LAYOUT_PATH = config.DATA_DIR / "store_layout.json"
    init_db()
    # No events at all — all merchandise zones are dead
    anomalies = compute_anomalies("ST1008")["anomalies"]
    assert any(item["type"] == "DEAD_ZONE" for item in anomalies)
    for item in [a for a in anomalies if a["type"] == "DEAD_ZONE"]:
        assert item["severity"] == "INFO"
        assert "suggested_action" in item


def test_conversion_drop_anomaly(tmp_path):
    """10+ visitors with conversion below 8% triggers CONVERSION_DROP WARN."""
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    config.STORE_LAYOUT_PATH = config.DATA_DIR / "store_layout.json"
    init_db()
    base = datetime(2026, 4, 10, 7, 0, tzinfo=timezone.utc)
    events = []
    for i in range(12):
        vid = f"VIS_{i:04d}"
        events.append(_entry(f"entr-{i:04d}", vid, base + timedelta(minutes=i)))
    ingest_batch(events)

    anomalies = compute_anomalies("ST1008")["anomalies"]
    assert any(item["type"] == "CONVERSION_DROP" for item in anomalies)


def test_no_anomaly_when_queue_shallow(tmp_path):
    """queue_depth of 3 should not trigger any BILLING_QUEUE_SPIKE."""
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    config.STORE_LAYOUT_PATH = config.DATA_DIR / "store_layout.json"
    init_db()
    ts = datetime(2026, 4, 10, 7, 0, tzinfo=timezone.utc)
    ingest_batch([
        _entry("queue-003", "VIS_3", ts, "CASH_COUNTER", "BILLING_QUEUE_JOIN", queue_depth=3),
    ])

    anomalies = compute_anomalies("ST1008")["anomalies"]
    assert not any(item["type"] == "BILLING_QUEUE_SPIKE" for item in anomalies)
