# Store Intelligence

End-to-end submission for the Purplle Store Intelligence challenge. Transforms
CCTV clips and POS data into structured visitor events, a production REST API,
and a live web dashboard.

## Quick Start (5 commands)

    git clone <repo-url> store-intelligence
    cd store-intelligence
    docker compose up --build
    python -m pipeline.detect --input "path/to/CCTV-clips" --output outputs/events.jsonl
    python -m pipeline.replay --events outputs/events.jsonl

The API is live at http://localhost:8000 after step 3. Steps 4-5 feed real
video-derived events into it. The `outputs/events.jsonl` file included in the
repo already contains 258 pre-generated events so the API endpoints are
queryable immediately after `docker compose up`.

## Optional: YOLO upgrade (adds ~800MB, improves detection accuracy)

    INSTALL_YOLO=true docker compose build
    docker compose up

## API Endpoints

    curl http://localhost:8000/health
    curl http://localhost:8000/stores/ST1008/metrics
    curl http://localhost:8000/stores/ST1008/funnel
    curl http://localhost:8000/stores/ST1008/heatmap
    curl http://localhost:8000/stores/ST1008/anomalies

## Live Dashboard

Web dashboard (auto-refreshes every 3 seconds):
    http://localhost:8000/dashboard

Terminal dashboard:
    python -m pipeline.dashboard --iterations 5

## Ingest pre-generated events

    python -m pipeline.replay --events outputs/events.jsonl

## Run tests

    python -m pytest

27 tests, all required edge cases: empty store, all-staff clip, zero purchases,
re-entry dedup, queue spike (WARN + CRITICAL), dead zone, conversion drop.

## Architecture

- `pipeline/detect.py` — frame sampling (every 3s, 640px), OpenCV HOG +
  motion contours, centroid tracker, staff heuristic, re-entry detection,
  event emission. Upgrades to YOLOv8n+ByteTrack when ultralytics is installed.
- `pipeline/replay.py` — POSTs events.jsonl to /events/ingest in batches.
- `app/main.py` — FastAPI application, structured JSON logging, 503 on DB fail.
- `app/analytics.py` — metrics, funnel, heatmap, anomaly computation.
- `app/database.py` — SQLite-backed event store, idempotent ingest.
- `data/store_layout.json` — zone/camera definitions for ST1008.
- `data/pos_transactions_canonical.csv` — 24 real POS transactions.
- `outputs/events.jsonl` / `event_log.jsonl` — mandatory challenge deliverable
  (258 events, all 8 event types, correct schema).
- `docs/DESIGN.md`, `docs/CHOICES.md` — architecture and AI decision logs.