from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app import config
from app.models import VisitorEvent


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    visitor_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    zone_id TEXT,
    dwell_ms INTEGER NOT NULL,
    is_staff INTEGER NOT NULL,
    confidence REAL NOT NULL,
    metadata_json TEXT NOT NULL,
    raw_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_store_ts ON events(store_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_store_visitor ON events(store_id, visitor_id);
CREATE TABLE IF NOT EXISTS rejected_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at TEXT NOT NULL,
    error TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pos_transactions (
    transaction_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    basket_value_inr REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pos_store_ts ON pos_transactions(store_id, timestamp);
"""


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_parent(config.DB_PATH)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
    seed_pos_transactions(config.POS_PATH)


def seed_pos_transactions(path: Path) -> int:
    """Seed POS transactions from CSV.

    Supports two schemas:
    - Challenge canonical: store_id, transaction_id, timestamp, basket_value_inr
    - Uploaded sample:     order_id, order_date, order_time, store_id, product_id,
                           brand_name, total_amount  (DD-MM-YYYY date format)
    """
    if not path.exists():
        return 0
    # Use pipeline.pos reader so both schemas are normalized identically
    try:
        from pipeline.pos import read_pos as _read_pos
        rows = _read_pos(path)
    except ImportError:
        # Fallback: raw CSV (canonical schema only)
        rows = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "store_id": row.get("store_id", ""),
                    "transaction_id": row.get("transaction_id", ""),
                    "timestamp": row.get("timestamp", ""),
                    "basket_value_inr": float(row.get("basket_value_inr") or 0),
                })
    if not rows:
        return 0
    with connect() as conn:
        existing = conn.execute("SELECT COUNT(*) AS n FROM pos_transactions").fetchone()["n"]
        if existing:
            return 0
        count = 0
        for row in rows:
            if not row.get("transaction_id") or not row.get("timestamp"):
                continue
            conn.execute(
                """
                INSERT OR IGNORE INTO pos_transactions
                (transaction_id, store_id, timestamp, basket_value_inr)
                VALUES (?, ?, ?, ?)
                """,
                (
                    row["transaction_id"],
                    row["store_id"],
                    row["timestamp"],
                    float(row.get("basket_value_inr") or 0),
                ),
            )
            count += 1
        return count


def insert_event(conn: sqlite3.Connection, event: VisitorEvent) -> bool:
    payload = event.model_dump(mode="json")
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO events
        (event_id, store_id, camera_id, visitor_id, event_type, timestamp, zone_id,
         dwell_ms, is_staff, confidence, metadata_json, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.event_id,
            event.store_id,
            event.camera_id,
            event.visitor_id,
            event.event_type.value,
            event.timestamp.isoformat().replace("+00:00", "Z"),
            event.zone_id,
            event.dwell_ms,
            int(event.is_staff),
            event.confidence,
            json.dumps(payload["metadata"], separators=(",", ":")),
            json.dumps(payload, separators=(",", ":")),
        ),
    )
    return cur.rowcount == 1


def insert_rejection(conn: sqlite3.Connection, payload: object, error: str) -> None:
    conn.execute(
        "INSERT INTO rejected_events(received_at, error, payload_json) VALUES (?, ?, ?)",
        (datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), error, json.dumps(payload, default=str)),
    )


def fetch_events(store_id: str) -> list[sqlite3.Row]:
    with connect() as conn:
        return list(
            conn.execute(
                "SELECT * FROM events WHERE store_id = ? ORDER BY timestamp, event_id",
                (store_id,),
            )
        )


def fetch_store_ids() -> list[str]:
    with connect() as conn:
        ids = [row["store_id"] for row in conn.execute("SELECT DISTINCT store_id FROM events")]
        ids += [row["store_id"] for row in conn.execute("SELECT DISTINCT store_id FROM pos_transactions")]
        return sorted(set(ids))


def fetch_pos(store_id: str) -> list[sqlite3.Row]:
    with connect() as conn:
        return list(
            conn.execute(
                "SELECT * FROM pos_transactions WHERE store_id = ? ORDER BY timestamp",
                (store_id,),
            )
        )
