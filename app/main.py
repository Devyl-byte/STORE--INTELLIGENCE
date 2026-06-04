from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app import config
from app.analytics import compute_anomalies, compute_funnel, compute_heatmap, compute_metrics, parse_ts
from app.database import fetch_store_ids, init_db, connect
from app.ingestion import ingest_batch
from app.models import HealthResponse

logging.basicConfig(level=config.LOG_LEVEL, format="%(message)s")
logger = logging.getLogger("store-intelligence")

app = FastAPI(title="Store Intelligence API", version="1.0.0")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.middleware("http")
async def structured_logging(request: Request, call_next):
    start = time.perf_counter()
    trace_id = request.headers.get("x-trace-id", str(uuid.uuid4()))
    # Buffer the body for ingest so we can count events before the handler runs.
    # For all other endpoints body is empty so this is zero-cost.
    body_bytes = b""
    if request.url.path == "/events/ingest":
        body_bytes = await request.body()
        # Re-wrap so the actual handler can still read it
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request = Request(request.scope, receive)

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(json.dumps({"trace_id": trace_id, "endpoint": request.url.path, "status_code": 500}))
        raise

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    # Count events only for the ingest endpoint; None for all others.
    event_count: int | None = None
    if body_bytes:
        try:
            event_count = len(json.loads(body_bytes))
        except Exception:
            event_count = None

    logger.info(
        json.dumps(
            {
                "trace_id": trace_id,
                "store_id": request.path_params.get("id"),
                "endpoint": request.url.path,
                "latency_ms": latency_ms,
                "event_count": event_count,
                "status_code": response.status_code,
            }
        )
    )
    response.headers["x-trace-id"] = trace_id
    return response


@app.exception_handler(Exception)
async def safe_exception_handler(_: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(
        status_code=503,
        content={"error": "SERVICE_UNAVAILABLE", "detail": "The service could not complete the request."},
    )


@app.get("/")
def root():
    return {
        "service": "Store Intelligence API",
        "status": "OK",
        "docs": "/docs",
        "health": "/health",
        "sample_metrics": "/stores/ST1008/metrics",
    }


@app.post("/events/ingest")
def post_events(payload: list[dict[str, Any]]):
    return ingest_batch(payload)


@app.get("/stores/{id}/metrics")
def get_metrics(id: str):
    return compute_metrics(id)


@app.get("/stores/{id}/funnel")
def get_funnel(id: str):
    return compute_funnel(id)


@app.get("/stores/{id}/heatmap")
def get_heatmap(id: str):
    return compute_heatmap(id)


@app.get("/stores/{id}/anomalies")
def get_anomalies(id: str):
    return compute_anomalies(id)


@app.get("/health", response_model=HealthResponse)
def health():
    stores: dict[str, dict[str, Any]] = {}
    now = datetime.now(timezone.utc)
    with connect() as conn:
        conn.execute("SELECT 1").fetchone()
        for store_id in fetch_store_ids():
            row = conn.execute(
                "SELECT MAX(timestamp) AS last_ts FROM events WHERE store_id = ?",
                (store_id,),
            ).fetchone()
            last_ts = row["last_ts"] if row else None
            status = "NO_EVENTS"
            lag_seconds = None
            if last_ts:
                lag_seconds = int((now - parse_ts(last_ts)).total_seconds())
                status = "STALE_FEED" if lag_seconds > config.STALE_FEED_SECONDS else "OK"
            stores[store_id] = {"last_event_timestamp": last_ts, "lag_seconds": lag_seconds, "status": status}
    overall = "OK" if all(s["status"] in {"OK", "NO_EVENTS"} for s in stores.values()) else "WARN"
    return HealthResponse(status=overall, database="OK", stores=stores)


@app.get("/dashboard")
def dashboard():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Store Intelligence · Live Dashboard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:#0f1117;color:#e2e8f0;padding:1.5rem}
  h1{font-size:1.25rem;font-weight:600;margin-bottom:1.5rem;color:#fff}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-bottom:1.5rem}
  .card{background:#1a1d27;border:1px solid #2d3148;border-radius:10px;padding:1rem}
  .card .label{font-size:0.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
  .card .value{font-size:1.6rem;font-weight:600;color:#fff}
  .card .sub{font-size:0.78rem;color:#9ca3af;margin-top:2px}
  table{width:100%;border-collapse:collapse;font-size:0.84rem;margin-bottom:1.5rem}
  th{text-align:left;color:#6b7280;font-weight:500;padding:6px 10px;border-bottom:1px solid #2d3148}
  td{padding:6px 10px;border-bottom:1px solid #1e2133}
  .badge{display:inline-block;padding:2px 8px;border-radius:99px;font-size:0.72rem;font-weight:600}
  .INFO{background:#1e3a5f;color:#60a5fa}.WARN{background:#3d2e09;color:#f59e0b}.CRITICAL{background:#3b0f0f;color:#f87171}
  #status{font-size:0.75rem;color:#4b5563;margin-top:1rem}
  .funnel-bar{display:flex;align-items:center;gap:8px;margin:4px 0}
  .funnel-bar .stage{font-size:0.78rem;width:120px;color:#9ca3af}
  .funnel-bar .bar{height:18px;background:#3730a3;border-radius:3px;transition:width .4s ease;min-width:2px}
  .funnel-bar .cnt{font-size:0.78rem;color:#a5b4fc;min-width:30px}
</style>
</head>
<body>
<h1>&#9632; Store Intelligence · Live</h1>
<div class="grid" id="metrics-grid"></div>
<h2 style="font-size:.9rem;color:#6b7280;margin-bottom:.75rem;font-weight:500">Conversion funnel</h2>
<div id="funnel"></div>
<h2 style="font-size:.9rem;color:#6b7280;margin-bottom:.75rem;margin-top:1.25rem;font-weight:500">Active anomalies</h2>
<table id="anomalies-table">
  <thead><tr><th>Severity</th><th>Type</th><th>Zone</th><th>Value</th><th>Action</th></tr></thead>
  <tbody id="anomalies-body"></tbody>
</table>
<div id="status">Connecting…</div>
<script>
const STORE = "ST1008";
const BASE  = "";

async function fetchJSON(url) {
  const r = await fetch(url);
  return r.ok ? r.json() : null;
}

async function refresh() {
  const [m, f, a] = await Promise.all([
    fetchJSON(`${BASE}/stores/${STORE}/metrics`),
    fetchJSON(`${BASE}/stores/${STORE}/funnel`),
    fetchJSON(`${BASE}/stores/${STORE}/anomalies`),
  ]);

  if (m) {
    const pct = v => (v*100).toFixed(1)+"%";
    document.getElementById("metrics-grid").innerHTML = `
      <div class="card"><div class="label">Unique visitors</div><div class="value">${m.unique_visitors}</div></div>
      <div class="card"><div class="label">Conversion rate</div><div class="value">${pct(m.conversion_rate)}</div></div>
      <div class="card"><div class="label">Purchases</div><div class="value">${m.purchases}</div></div>
      <div class="card"><div class="label">Queue depth</div><div class="value">${m.current_queue_depth}</div></div>
      <div class="card"><div class="label">Abandonment rate</div><div class="value">${pct(m.abandonment_rate)}</div></div>
    `;
  }

  if (f) {
    const maxCount = Math.max(...f.stages.map(s=>s.count), 1);
    document.getElementById("funnel").innerHTML = f.stages.map(s => `
      <div class="funnel-bar">
        <span class="stage">${s.stage}</span>
        <div class="bar" style="width:${Math.round(s.count/maxCount*260)+2}px"></div>
        <span class="cnt">${s.count}</span>
        <span style="font-size:.72rem;color:#6b7280">&minus;${s.dropoff_from_previous_pct.toFixed(1)}%</span>
      </div>`).join("");
  }

  if (a) {
    document.getElementById("anomalies-body").innerHTML = a.anomalies.length
      ? a.anomalies.map(x => `<tr>
          <td><span class="badge ${x.severity}">${x.severity}</span></td>
          <td>${x.type}</td>
          <td>${x.zone_id||""}</td>
          <td>${x.observed_value!==undefined?x.observed_value:""}</td>
          <td style="font-size:.76rem;color:#9ca3af">${x.suggested_action||""}</td>
        </tr>`).join("")
      : "<tr><td colspan='5' style='color:#4b5563;padding:8px 10px'>No active anomalies</td></tr>";
  }

  document.getElementById("status").textContent =
    "Last refresh: " + new Date().toLocaleTimeString() + " · auto-refresh every 3s · store: " + STORE;
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
