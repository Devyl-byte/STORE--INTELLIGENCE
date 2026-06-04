from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _normalize_row(row: dict) -> dict | None:
    """Normalize a POS row to canonical {store_id, transaction_id, timestamp, basket_value_inr}.

    Supports two schemas:
    - Challenge PS schema:  store_id, transaction_id, timestamp, basket_value_inr
    - Uploaded sample POS:  order_id, order_date, order_time, store_id, product_id,
                            brand_name, total_amount
    Returns None for rows that cannot be parsed.
    """
    # Schema 1 — canonical challenge format
    if "transaction_id" in row and "timestamp" in row:
        return {
            "store_id": row.get("store_id", ""),
            "transaction_id": row["transaction_id"],
            "timestamp": row["timestamp"],
            "basket_value_inr": float(row.get("basket_value_inr") or 0),
        }

    # Schema 2 — uploaded sample format (order_id, order_date, order_time, total_amount)
    if "order_id" in row and "order_date" in row and "order_time" in row:
        try:
            # order_date is "10-04-2026", order_time is "12:15:05"
            date_str = row["order_date"].strip()
            time_str = row["order_time"].strip()
            # Convert DD-MM-YYYY to YYYY-MM-DD
            parts = date_str.split("-")
            if len(parts) == 3:
                if len(parts[2]) == 4:  # DD-MM-YYYY
                    iso_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
                else:  # YYYY-MM-DD already
                    iso_date = date_str
            else:
                return None
            timestamp = f"{iso_date}T{time_str}Z"
            return {
                "store_id": row.get("store_id", ""),
                "transaction_id": str(row["order_id"]),
                "timestamp": timestamp,
                "basket_value_inr": float(row.get("total_amount") or 0),
            }
        except (ValueError, KeyError):
            return None

    return None


def read_pos(path: Path, store_id: str | None = None) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    rows = []
    for raw in raw_rows:
        normalized = _normalize_row(raw)
        if normalized:
            rows.append(normalized)

    if store_id:
        rows = [row for row in rows if row.get("store_id") == store_id]

    rows.sort(key=lambda row: row["timestamp"])
    return rows
