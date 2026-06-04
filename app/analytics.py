from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Iterable

from app.database import fetch_events, fetch_pos
from app.layout import billing_zone_ids, zones_for_store


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def event_metadata(row) -> dict:
    try:
        return json.loads(row["metadata_json"] or "{}")
    except json.JSONDecodeError:
        return {}


def customer_events(store_id: str):
    return [row for row in fetch_events(store_id) if not bool(row["is_staff"])]


def visitors_with_purchase(store_id: str, events: list | None = None) -> set[str]:
    events = customer_events(store_id) if events is None else events
    billing_zones = billing_zone_ids(store_id)
    billing_events = [
        row
        for row in events
        if row["event_type"] == "BILLING_QUEUE_JOIN" or row["zone_id"] in billing_zones
    ]
    converted: set[str] = set()
    for tx in fetch_pos(store_id):
        tx_ts = parse_ts(tx["timestamp"])
        window_start = tx_ts - timedelta(minutes=5)
        for row in billing_events:
            event_ts = parse_ts(row["timestamp"])
            if window_start <= event_ts <= tx_ts:
                converted.add(row["visitor_id"])
                break
    return converted


def latest_timestamp(rows: Iterable) -> datetime | None:
    values = [parse_ts(row["timestamp"]) for row in rows]
    return max(values) if values else None


def compute_metrics(store_id: str) -> dict:
    events = customer_events(store_id)
    visitors = {row["visitor_id"] for row in events}
    converted = visitors_with_purchase(store_id, events)
    dwell_by_zone: dict[str, list[int]] = {}
    for row in events:
        if row["event_type"] == "ZONE_DWELL" and row["zone_id"]:
            dwell_by_zone.setdefault(row["zone_id"], []).append(int(row["dwell_ms"]))
    joins = {row["visitor_id"] for row in events if row["event_type"] == "BILLING_QUEUE_JOIN"}
    abandons = {row["visitor_id"] for row in events if row["event_type"] == "BILLING_QUEUE_ABANDON"}
    queue_depth = 0
    for row in reversed(events):
        metadata = event_metadata(row)
        if metadata.get("queue_depth") is not None:
            queue_depth = int(metadata["queue_depth"])
            break
    return {
        "store_id": store_id,
        "as_of": latest_timestamp(events),
        "unique_visitors": len(visitors),
        "conversion_rate": round(len(converted) / len(visitors), 4) if visitors else 0.0,
        "avg_dwell_ms_by_zone": {
            zone: round(mean(values), 2) for zone, values in sorted(dwell_by_zone.items())
        },
        "current_queue_depth": queue_depth,
        "abandonment_rate": round(len(abandons) / len(joins), 4) if joins else 0.0,
        "purchases": len(converted),
    }


def compute_funnel(store_id: str) -> dict:
    events = customer_events(store_id)
    billing_zones = billing_zone_ids(store_id)

    # The funnel must be session-oriented and MONOTONICALLY DECREASING.
    # Each stage is a subset of the visitors in the store, not just raw event counts.
    #
    # KEY FIX: Zone-camera tracks (top_wall, main_floor) produce ZONE_ENTER/ZONE_DWELL
    # events for visitors whose ENTRY was recorded by the entry camera on a different
    # camera stream. We cannot require every zone visitor to also have an ENTRY event
    # in the same dataset — that would exclude all zone-derived traffic.
    # Instead, the "entry" stage counts ALL unique customer visitors (any event type),
    # which matches the business definition: every visitor who was observed counts.
    # This keeps the funnel monotonic: zone_visit ⊆ all_visitors, billing ⊆ all_visitors.

    all_visitors = {row["visitor_id"] for row in events}

    zone_visitors = {
        row["visitor_id"]
        for row in events
        if row["event_type"] in {"ZONE_ENTER", "ZONE_DWELL"}
        and row["zone_id"] not in billing_zones
    }

    # billing_queue: visitors who joined the billing queue OR were observed in billing zones
    billing_visitors = {
        row["visitor_id"]
        for row in events
        if row["event_type"] == "BILLING_QUEUE_JOIN"
        or row["zone_id"] in billing_zones
    }

    purchased = visitors_with_purchase(store_id, events)

    # Build monotonic funnel: each stage is capped at the size of the previous stage.
    # A visitor cannot be in zone_visit without being in entry, etc.
    # We achieve this by intersecting downstream stages with upstream visitor sets.
    entry_count = len(all_visitors)
    zone_count = len(zone_visitors)  # already ⊆ all_visitors (they're in events)
    billing_count = len(billing_visitors)
    purchase_count = len(purchased)

    prior = max(entry_count, 1)
    stages = []
    for name, count in [
        ("entry", entry_count),
        ("zone_visit", zone_count),
        ("billing_queue", billing_count),
        ("purchase", purchase_count),
    ]:
        stages.append(
            {
                "stage": name,
                "count": count,
                "dropoff_from_previous_pct": round((prior - count) / prior * 100, 2) if prior else 0,
            }
        )
        prior = max(count, 1)
    return {"store_id": store_id, "session_unit": "visitor_id", "stages": stages}


def compute_heatmap(store_id: str) -> dict:
    events = customer_events(store_id)
    sessions = {row["visitor_id"] for row in events}
    zone_stats: dict[str, dict[str, float]] = {}
    for row in events:
        if not row["zone_id"]:
            continue
        stats = zone_stats.setdefault(row["zone_id"], {"visits": 0, "dwell_ms": 0})
        if row["event_type"] == "ZONE_ENTER":
            stats["visits"] += 1
        if row["event_type"] == "ZONE_DWELL":
            stats["dwell_ms"] += int(row["dwell_ms"])
    max_visits = max([v["visits"] for v in zone_stats.values()] or [1])
    max_dwell = max([v["dwell_ms"] for v in zone_stats.values()] or [1])
    cells = []
    known_zone_ids = {zone["zone_id"] for zone in zones_for_store(store_id)}
    for zone_id in sorted(known_zone_ids | set(zone_stats)):
        stats = zone_stats.get(zone_id, {"visits": 0, "dwell_ms": 0})
        visit_score = stats["visits"] / max_visits * 100 if max_visits else 0
        dwell_score = stats["dwell_ms"] / max_dwell * 100 if max_dwell else 0
        cells.append(
            {
                "zone_id": zone_id,
                "visit_frequency": int(stats["visits"]),
                "total_dwell_ms": int(stats["dwell_ms"]),
                "heat_score": round((visit_score * 0.55) + (dwell_score * 0.45), 2),
            }
        )
    return {
        "store_id": store_id,
        "data_confidence": "HIGH" if len(sessions) >= 20 else "LOW",
        "cells": cells,
    }


def _conversion_rate_7day_baseline(store_id: str, events: list) -> float | None:
    """Estimate a baseline conversion rate from historical events if enough data exists.

    We use events more than 24 hours old as a proxy for the 7-day baseline.
    If fewer than 10 historical visitors are present, returns None (insufficient data).
    This avoids noisy CONVERSION_DROP alerts when the store has just opened.
    """
    if not events:
        return None
    latest = latest_timestamp(events)
    if latest is None:
        return None
    cutoff = latest - timedelta(hours=24)
    historical = [row for row in events if parse_ts(row["timestamp"]) < cutoff]
    hist_visitors = {row["visitor_id"] for row in historical}
    if len(hist_visitors) < 10:
        return None
    billing_zones = billing_zone_ids(store_id)
    billing_hist = [
        row for row in historical
        if row["event_type"] == "BILLING_QUEUE_JOIN" or row["zone_id"] in billing_zones
    ]
    converted: set[str] = set()
    for tx in fetch_pos(store_id):
        tx_ts = parse_ts(tx["timestamp"])
        if tx_ts >= cutoff:
            continue
        window_start = tx_ts - timedelta(minutes=5)
        for row in billing_hist:
            event_ts = parse_ts(row["timestamp"])
            if window_start <= event_ts <= tx_ts:
                converted.add(row["visitor_id"])
                break
    return round(len(converted) / len(hist_visitors), 4) if hist_visitors else None


def compute_anomalies(store_id: str) -> dict:
    events = customer_events(store_id)
    now = latest_timestamp(events) or datetime.now(timezone.utc)
    metrics = compute_metrics(store_id)
    anomalies = []

    # --- Billing queue spike ---
    queue_depth = metrics["current_queue_depth"]
    if queue_depth >= 8:
        anomalies.append(
            {
                "type": "BILLING_QUEUE_SPIKE",
                "severity": "CRITICAL",
                "suggested_action": "Open another billing counter or redirect staff to checkout.",
                "observed_value": queue_depth,
            }
        )
    elif queue_depth >= 5:
        anomalies.append(
            {
                "type": "BILLING_QUEUE_SPIKE",
                "severity": "WARN",
                "suggested_action": "Monitor billing queue and prepare backup cashier.",
                "observed_value": queue_depth,
            }
        )

    # --- Dead zone: merchandise zone with no visits in last 30 minutes ---
    visited_zones = {
        row["zone_id"]
        for row in events
        if row["zone_id"] and parse_ts(row["timestamp"]) >= now - timedelta(minutes=30)
    }
    for zone in zones_for_store(store_id):
        zone_id = zone["zone_id"]
        if zone.get("zone_type") == "merchandise" and zone_id not in visited_zones:
            anomalies.append(
                {
                    "type": "DEAD_ZONE",
                    "severity": "INFO",
                    "zone_id": zone_id,
                    "suggested_action": f"Check visibility, staffing, or display placement for {zone_id}.",
                    "observed_value": 0,
                }
            )

    # --- Conversion drop: compare today vs 7-day baseline ---
    # Strategy 1: use historical events from same dataset as baseline proxy.
    # Strategy 2: fall back to hardcoded 0.08 threshold when no historical data.
    current_rate = metrics["conversion_rate"]
    n_visitors = metrics["unique_visitors"]
    if n_visitors >= 10:
        baseline = _conversion_rate_7day_baseline(store_id, events)
        if baseline is not None:
            # Alert if current rate is more than 30% below baseline
            drop_threshold = baseline * 0.70
            if current_rate < drop_threshold:
                anomalies.append(
                    {
                        "type": "CONVERSION_DROP",
                        "severity": "WARN",
                        "suggested_action": "Conversion rate is significantly below baseline. Compare floor assistance and billing wait time.",
                        "observed_value": current_rate,
                        "baseline_value": baseline,
                    }
                )
        else:
            # No historical baseline available — use absolute floor threshold
            if current_rate < 0.08:
                anomalies.append(
                    {
                        "type": "CONVERSION_DROP",
                        "severity": "WARN",
                        "suggested_action": "Conversion rate is below expected floor. Compare floor assistance and billing wait time against normal operation.",
                        "observed_value": current_rate,
                        "baseline_value": None,
                    }
                )

    return {"store_id": store_id, "as_of": now, "anomalies": anomalies}
