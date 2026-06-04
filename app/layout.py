from __future__ import annotations

import json
from functools import lru_cache

from app import config


@lru_cache
def load_layout() -> dict:
    if not config.STORE_LAYOUT_PATH.exists():
        return {"stores": []}
    return json.loads(config.STORE_LAYOUT_PATH.read_text(encoding="utf-8"))


def zones_for_store(store_id: str) -> list[dict]:
    for store in load_layout().get("stores", []):
        if store.get("store_id") == store_id:
            return store.get("zones", [])
    return []


def billing_zone_ids(store_id: str) -> set[str]:
    return {
        zone["zone_id"]
        for zone in zones_for_store(store_id)
        if zone.get("zone_type") in {"billing", "cash_counter"}
        or "BILL" in zone.get("zone_id", "")
        or "CASH" in zone.get("zone_id", "")
    }
