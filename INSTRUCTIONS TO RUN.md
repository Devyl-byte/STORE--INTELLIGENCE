### How to Run — Store Intelligence API
 
#### Requirements
- Docker Desktop (Mac/Windows) or Docker Engine + Compose plugin (Linux)
- Python 3.10+ (only needed for the detection pipeline, not the API)
- No GPU required
#### 1. Clone and start the API
 
```bash
git clone <your-repo-url> store-intelligence
cd store-intelligence
docker compose up --build
```
 
Build time: ~60–90 seconds (downloads Python 3.12-slim + ~150MB of packages). YOLO/Torch are NOT installed by default, keeping the image small.
 
The API starts at **http://localhost:8000**
 
#### 2. Verify the API is running
 
```bash
curl http://localhost:8000/health
curl http://localhost:8000/stores/ST1008/metrics
curl http://localhost:8000/stores/ST1008/funnel
curl http://localhost:8000/stores/ST1008/heatmap
curl http://localhost:8000/stores/ST1008/anomalies
```
 
#### 3. Load the pre-generated events (JSONL deliverable)
 
The repo includes `outputs/events.jsonl` — 258 events from real CCTV frames + POS correlation, covering all 8 required event types. Feed them into the API:
 
```bash
python -m pipeline.replay --events outputs/events.jsonl
```
 
Then re-query the endpoints — metrics, funnel, and anomalies will reflect the full dataset.
 
#### 4. Open the live dashboard
 
Navigate to: **http://localhost:8000/dashboard**
 
Metrics, funnel bars, and anomalies update automatically every 3 seconds as events flow in.
 
#### 5. Run the detection pipeline against your own CCTV clips
 
```bash
# Against a ZIP of clips
python -m pipeline.detect --input /path/to/clips.zip --output outputs/events.jsonl
 
# Against a folder
python -m pipeline.detect --input /path/to/clips/ --output outputs/events.jsonl
 
# Against a single clip
python -m pipeline.detect --input /path/to/cam1.mp4 --output outputs/events.jsonl
 
# Then replay into the API
python -m pipeline.replay --events outputs/events.jsonl
```
 
#### 6. Optional: Enable YOLOv8n+ByteTrack (better accuracy, ~800MB build)
 
```bash
INSTALL_YOLO=true docker compose build
docker compose up
```
 
With ultralytics installed, the detector automatically uses YOLOv8n+ByteTrack instead of the OpenCV HOG baseline. No code change required — `detect.py` detects the import and upgrades silently.
 
#### 7. Run the test suite
 
```bash
python -m pytest -v
```
 
Expected: 27 passed, 0 failed. Tests cover: schema validation, idempotent ingest, staff exclusion, POS 5-minute window correlation, empty store, all-staff clip, zero purchases, re-entry dedup, funnel monotonicity, heatmap confidence flag, queue spike (WARN + CRITICAL), dead zone, conversion drop with baseline comparison.
 
#### 8. Event log deliverable
 
The mandatory `.jsonl` file is at:
- `outputs/events.jsonl` (primary)
- `event_log.jsonl` (copy at repo root for easy access)
Both contain 258 events in the required schema: `event_id`, `store_id`, `camera_id`, `visitor_id`, `event_type`, `timestamp`, `zone_id`, `dwell_ms`, `is_staff`, `confidence`, `metadata`.
 
#### Detection pipeline notes
 
- Samples every 3 seconds (configurable with `--sample-every-seconds N`)
- Staff identified by trajectory heuristic: seen in >35% of frames AND displacement <12px/detection → `is_staff=true`, excluded from all customer metrics
- Re-entry: new inbound track within 3 minutes and 80px of a prior exit reuses original `visitor_id`, emits `REENTRY`
- Group entry: NMS at IoU 0.45 separates individuals; people at ~0.1 IoU both survive
- Low-confidence events kept in stream with calibrated scores rather than dropped silently
---
 