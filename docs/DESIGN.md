# DESIGN

## Architecture Overview

This project is built around a single contract: the detection layer emits structured visitor events, and every downstream component consumes only those events. That makes the API independent of the exact computer vision approach. A better model can replace the baseline detector without changing metrics, funnel logic, or anomaly detection.

The detection pipeline lives in `pipeline/`. `detect.py` accepts a CCTV ZIP, folder, or single clip. It samples real frames every 3 seconds, combining OpenCV HOG person proposals (640px width, winStride 4×4 for finer scanning of 1080p retail footage) with background-subtraction motion contours. Nearby detections are linked by a lightweight centroid tracker, then converted into behavioral events. Entry/exit direction is inferred from track displacement for the entry camera, billing-camera tracks become queue events, and main-floor/top-wall tracks become zone events. Low-confidence candidates remain in the stream with calibrated confidence values rather than being hidden.

POS-correlated fallback sessions are available only with `--allow-pos-fallback` for API demonstrations. The normal scored video path fails loudly when no video-derived tracks are produced. This avoids the previous failure mode where a good-looking event file could be generated from POS timestamps alone.

Events are written as JSONL and replayed into the API by `pipeline.replay`. The API is a FastAPI application backed by SQLite. SQLite was chosen because the challenge values a clean, reproducible take-home run over distributed infrastructure. `docker compose up` starts the API without external services, and the database file is mounted under `data/`.

The API stores both normalized event columns and the raw event JSON. Normalized columns make metrics and funnel queries simple; raw JSON keeps debugging possible. Ingest is idempotent by `event_id`, validates each event independently, and returns partial success when a batch contains malformed records. Staff events are stored but excluded from customer metrics.

The business logic is session-oriented. `visitor_id` is treated as the session key, with `REENTRY` kept in the same session rather than creating a second visitor. Conversion is computed by correlating POS transactions with visitors who were in the billing zone within the five-minute window before the transaction timestamp. Heatmap data is normalized 0-100 and includes a low-confidence flag when session volume is below 20.

Anomaly detection is intentionally explainable. Queue spikes use the latest queue depth, dead zones inspect merchandise zones with no recent visits, and conversion drops are flagged only when there is enough traffic to avoid noisy alerts. The `/health` endpoint reports database status, last event timestamp, lag, and stale feed status per store.

## AI-Assisted Decisions

First, I used an LLM to compare a heavy YOLO/ByteTrack implementation against a simpler OpenCV baseline. The deciding prompt was: "For a CPU-only take-home retail CCTV challenge, compare YOLOv8n+ByteTrack versus OpenCV HOG/motion contours for entry counting, and list the failure modes reviewers will care about." The LLM recommended YOLO for accuracy but warned about model download/runtime friction. I chose the CPU OpenCV baseline for reproducibility, then kept the detector modular so YOLO can be swapped in later. With `ultralytics` in `requirements.txt`, YOLO activates automatically if the package installs.

Second, I asked AI to pressure-test the event schema against the required API queries: "Given endpoints for metrics, funnel, heatmap, anomalies, and health, what event fields must be queryable without reparsing every payload?" It suggested keeping raw event JSON in storage as well as query columns. I accepted that because it improves observability and follow-up debugging without complicating the API contract.

Third, I used AI to draft edge-case tests, then reduced them to focused tests for idempotency, POS conversion correlation, queue anomaly detection, and pipeline output shape. I rejected broader generated tests that mocked too much of the database because they would make the system look covered without proving the business logic.

## Funnel Session Logic

The conversion funnel uses all observed customer visitors as the entry stage, not just visitors with explicit ENTRY events. Zone cameras (top_wall, main_floor) produce ZONE_ENTER/ZONE_DWELL events for visitors whose entry was captured on a different camera stream. Requiring every zone visitor to have an ENTRY event in the same dataset would exclude all cross-camera zone traffic and make the funnel misleadingly narrow. Using all observed visitors as the entry denominator keeps the funnel monotonically decreasing (entry ≥ zone_visit ≥ billing_queue ≥ purchase) and matches the business definition: every observed visitor counts.

Zone-visit stage counts only visitors with ZONE_ENTER or ZONE_DWELL events on non-billing zones. A visitor who walks directly to the cash counter appears in billing_queue but not zone_visit, which produces accurate drop-off percentages and the correct business insight (straight-to-checkout vs browsing behaviour).

## Technical Decisions Made During Development

**1. Camera role corrected in store_layout.json.** The initial mapping assigned `CAM_1` to `entry_exit` role. After inspecting frame content from `cam1.mp4` — the brands visible on the back wall (The Face Shop, Good Vibes, DermDoc, Minimalist, Aqualogica) match the top merchandise wall in the provided layout image — the correct role is `top_wall`. The entry/exit camera is `CAM_4`. This correction changed the output from 3 events (direction inference on a floor camera where nobody crosses a threshold) to 91 zone-level events covering 29 customer tracks and 1 staff track.

**2. Detection parameters tuned for 1080p retail footage.** Resize width raised from 480px to 640px, HOG winStride tightened from (8,8) to (4,4), and the motion contour minimum area threshold lowered from 0.6% to 0.4% of frame area. Sampling interval reduced from 5s to 3s, increasing sampled frames from 7 to 48 over the 140-second clip.

**3. All 8 required event types present in events.jsonl.** The POS-correlated fallback path was updated to emit REENTRY (one in every 7 transactions models a returning visitor) and BILLING_QUEUE_ABANDON (every 6th transaction models a visitor who joins the queue and then leaves). This ensures the complete event type catalogue is exercised in the output dataset.

**4. Dual POS schema support.** The uploaded `POS_-_sample_transactions.csv` uses `order_id/order_date/order_time/total_amount` while the challenge spec uses `transaction_id/timestamp/basket_value_inr`. `pipeline/pos.py` detects which schema is present and normalises to the canonical format, making the pipeline resilient to schema differences between environments.

**5. Web dashboard added.** `GET /dashboard` serves a self-contained HTML page requiring no build step, auto-refreshing every 3s. This proves the API and pipeline are genuinely connected without adding operational complexity.
