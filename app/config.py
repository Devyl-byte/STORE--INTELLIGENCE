from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "store_intelligence.db"))
STORE_LAYOUT_PATH = Path(os.getenv("STORE_LAYOUT_PATH", DATA_DIR / "store_layout.json"))
POS_PATH = Path(os.getenv("POS_PATH", DATA_DIR / "pos_transactions_canonical.csv"))
STALE_FEED_SECONDS = int(os.getenv("STALE_FEED_SECONDS", "600"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
