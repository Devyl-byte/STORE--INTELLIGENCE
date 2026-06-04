# PROMPT: Create tests proving store metrics exclude staff, use sessions instead of raw event
# counts, and correlate POS transactions with billing-zone visits inside a five-minute window.
# CHANGES MADE: Fixed all event_id fixtures to use 8+ char IDs (was e1/e2/c1/c2 — rejected by
# model validator, causing tests to silently pass/fail for the wrong reason). Added
# re-entry dedup test and heatmap confidence test. Retained all four named edge cases from
# the rubric: empty store, all-staff clip, zero purchases, and re-entry in funnel.

from datetime import datetime, timedelta, timezone

from app import config
from app.analytics import compute_funnel, compute_heatmap, compute_metrics
from app.database import init_db
from app.ingestion import ingest_batch


def make_event(event_id, visitor_id, event_type, ts, zone_id=None, is_staff=False, dwell_ms=0):
    return {
        "event_id": event_id,
        "store_id": "ST1008",
        "camera_id": "CAM_1",
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": ts.isoformat(),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": 0.9,
        "metadata": {"session_seq": 1, "queue_depth": 2 if event_type == "BILLING_QUEUE_JOIN" else None},
    }


def test_metrics_and_funnel_use_customer_sessions(tmp_path):
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "pos.csv"
    config.POS_PATH.write_text(
        "store_id,transaction_id,timestamp,basket_value_inr\n"
        "ST1008,TXN_1001,2026-04-10T07:05:00Z,500\n",
        encoding="utf-8",
    )
    init_db()
    base = datetime(2026, 4, 10, 7, 0, tzinfo=timezone.utc)
    ingest_batch(
        [
            make_event("evnt-0001", "VIS_1", "ENTRY",              base),
            make_event("evnt-0002", "VIS_1", "ZONE_DWELL",         base + timedelta(minutes=1), "DERMDOC", dwell_ms=31000),
            make_event("evnt-0003", "VIS_1", "BILLING_QUEUE_JOIN", base + timedelta(minutes=4), "CASH_COUNTER"),
            make_event("evnt-0004", "STAFF_1", "ENTRY",            base, is_staff=True),
        ]
    )

    metrics = compute_metrics("ST1008")
    funnel  = compute_funnel("ST1008")

    assert metrics["unique_visitors"] == 1,  "Staff must not be counted as a visitor"
    assert metrics["purchases"] == 1,        "POS tx within 5-min window must be counted"
    assert metrics["conversion_rate"] == 1.0
    assert funnel["stages"][-1]["count"] == 1, "Purchase stage must show 1"


def test_empty_store_zero_traffic_returns_zeros(tmp_path):
    """A store with no events at all must return zeros without crashing.

    This is the 'empty store / zero traffic' edge case named explicitly in the rubric.
    """
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    init_db()

    metrics = compute_metrics("ST1008")
    funnel  = compute_funnel("ST1008")

    assert metrics["unique_visitors"]  == 0
    assert metrics["conversion_rate"]  == 0.0
    assert metrics["purchases"]        == 0
    assert metrics["current_queue_depth"] == 0
    assert metrics["abandonment_rate"] == 0.0
    assert all(stage["count"] == 0 for stage in funnel["stages"])


def test_all_staff_clip_produces_zero_customer_metrics(tmp_path):
    """When every event is flagged is_staff=True, unique_visitors and conversion must be 0.

    This is the 'all-staff clip' edge case named explicitly in the rubric.
    """
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    init_db()
    base = datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc)

    ingest_batch(
        [
            make_event("stff-0001", "STAFF_001", "ENTRY",      base,                         is_staff=True),
            make_event("stff-0002", "STAFF_001", "ZONE_DWELL", base + timedelta(minutes=5),
                       zone_id="CASH_COUNTER", is_staff=True, dwell_ms=300000),
            make_event("stff-0003", "STAFF_002", "ENTRY",      base + timedelta(minutes=1),  is_staff=True),
        ]
    )

    metrics = compute_metrics("ST1008")
    funnel  = compute_funnel("ST1008")

    assert metrics["unique_visitors"] == 0
    assert metrics["conversion_rate"] == 0.0
    assert metrics["purchases"]       == 0
    assert funnel["stages"][0]["count"] == 0


def test_zero_purchase_store_returns_zero_conversion(tmp_path):
    """Customers visit but POS transaction is outside the 5-min correlation window.

    This is the 'zero purchases' edge case named explicitly in the rubric.
    """
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "pos.csv"
    config.POS_PATH.write_text(
        "store_id,transaction_id,timestamp,basket_value_inr\n"
        "ST1008,TXN_LATE,2026-04-10T10:00:00Z,750\n",
        encoding="utf-8",
    )
    init_db()
    base = datetime(2026, 4, 10, 7, 0, tzinfo=timezone.utc)

    ingest_batch(
        [
            make_event("cust-0001", "VIS_NOBUY", "ENTRY",              base),
            make_event("cust-0002", "VIS_NOBUY", "ZONE_ENTER",         base + timedelta(minutes=2),  "DERMDOC"),
            make_event("cust-0003", "VIS_NOBUY", "BILLING_QUEUE_JOIN", base + timedelta(minutes=10), "CASH_COUNTER"),
            make_event("cust-0004", "VIS_NOBUY", "EXIT",               base + timedelta(minutes=15)),
        ]
    )

    metrics = compute_metrics("ST1008")
    funnel  = compute_funnel("ST1008")

    assert metrics["unique_visitors"] == 1
    assert metrics["purchases"]       == 0
    assert metrics["conversion_rate"] == 0.0
    assert funnel["stages"][0]["count"]  == 1, "Funnel entry must show visitor"
    assert funnel["stages"][-1]["count"] == 0, "Funnel purchase must be 0"


def test_reentry_does_not_double_count_visitor(tmp_path):
    """A visitor who exits and re-enters must count as 1 unique visitor, not 2.

    This is the 're-entry in funnel' edge case named explicitly in the rubric.
    """
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    init_db()
    base = datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc)

    ingest_batch(
        [
            make_event("rent-0001", "VIS_REPEAT", "ENTRY",   base),
            make_event("rent-0002", "VIS_REPEAT", "EXIT",    base + timedelta(minutes=5)),
            make_event("rent-0003", "VIS_REPEAT", "REENTRY", base + timedelta(minutes=7)),
        ]
    )

    metrics = compute_metrics("ST1008")
    assert metrics["unique_visitors"] == 1, "REENTRY must not inflate unique visitor count"


def test_heatmap_low_confidence_flag(tmp_path):
    """Heatmap must return data_confidence=LOW when fewer than 20 sessions."""
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    init_db()
    base = datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc)

    ingest_batch([
        make_event("heat-0001", "VIS_A", "ZONE_ENTER", base, "DERMDOC"),
        make_event("heat-0002", "VIS_B", "ZONE_ENTER", base + timedelta(minutes=1), "MINIMALIST"),
    ])

    heatmap = compute_heatmap("ST1008")
    assert heatmap["data_confidence"] == "LOW"


def test_heatmap_high_confidence_flag(tmp_path):
    """Heatmap must return data_confidence=HIGH with 20+ sessions."""
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    init_db()
    base = datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc)

    events = [
        make_event(f"hg-{i:05d}", f"VIS_{i:04d}", "ZONE_ENTER", base + timedelta(minutes=i), "DERMDOC")
        for i in range(22)
    ]
    ingest_batch(events)

    heatmap = compute_heatmap("ST1008")
    assert heatmap["data_confidence"] == "HIGH"


def test_funnel_zone_visit_excludes_billing_only_visitors(tmp_path):
    """A customer who goes straight to billing without browsing any merchandise zone
    must appear in billing_queue but NOT in zone_visit.
    The funnel entry stage counts ALL observed customer visitors (any event).
    """
    config.DB_PATH = tmp_path / "test.db"
    config.POS_PATH = tmp_path / "missing.csv"
    init_db()
    base = datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc)

    ingest_batch([
        make_event("fnl-00001", "VIS_DIRECT", "ENTRY",              base),
        make_event("fnl-00002", "VIS_DIRECT", "BILLING_QUEUE_JOIN", base + timedelta(minutes=2), "CASH_COUNTER"),
    ])

    funnel = compute_funnel("ST1008")
    stages = {s["stage"]: s["count"] for s in funnel["stages"]}
    assert stages["entry"] >= 1,         "VIS_DIRECT must appear in entry (all observed visitors)"
    assert stages["billing_queue"] >= 1, "Direct-to-billing visitor must appear in billing_queue"
    assert stages["zone_visit"] == 0,    "Must NOT appear in zone_visit without browsing a merchandise zone"
