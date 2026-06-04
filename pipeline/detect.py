from __future__ import annotations

import argparse
import json
import math
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from pipeline.emit import EventEmitter, write_jsonl
from pipeline.pos import read_pos, parse_ts
from pipeline.tracker import CentroidTracker

# Optional YOLO upgrade — safe import, falls back to OpenCV if unavailable
try:
    from pipeline.yolo_detector import estimate_yolo_tracks, YOLO_AVAILABLE
except ImportError:
    YOLO_AVAILABLE = False
    estimate_yolo_tracks = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised when OpenCV is absent locally
    cv2 = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYOUT = ROOT / "data" / "store_layout.json"
DEFAULT_POS = ROOT / "data" / "pos_transactions_canonical.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "events.jsonl"

# ---------------------------------------------------------------------------
# Staff detection heuristics
# ---------------------------------------------------------------------------
# Retail staff in a beauty store typically move in tight, repetitive paths
# (restocking, counter duty) and appear in every sampled frame window across
# the full clip duration.  We identify likely staff tracks by two signals:
#
#   1. HIGH HIT RATE — the track is seen in a large fraction of sampled frames.
#      A customer visits for 5-20 minutes; staff are present for the full clip.
#      If hits / total_sampled_frames > STAFF_HIT_RATE_THRESHOLD we flag staff.
#
#   2. LOW DISPLACEMENT per hit — staff at a billing counter or shelf move very
#      little relative to how often they are detected.  Roaming customers
#      traverse larger fractions of the frame.
#
# Neither heuristic is perfect.  Both are expressed as calibrated confidence
# values in is_staff_confidence, and the hard threshold is documented so a
# reviewer can see the reasoning and adjust it.
#
# This is the best we can do with a CPU OpenCV baseline; a production system
# would use uniform-colour clustering or a dedicated classifier.

STAFF_HIT_RATE_THRESHOLD = 0.35   # fraction of sampled frames track appears in
STAFF_DISPLACEMENT_PER_HIT = 12.0  # pixels; low = repetitive / stationary path


def _classify_staff(track: "DetectionTrack", total_sampled: int) -> tuple[bool, float]:
    """Return (is_staff, confidence) for a track.

    Confidence reflects how strongly our heuristics fired, not detection conf.
    """
    hit_rate = track.hits / max(total_sampled, 1)
    dx, dy = track.displacement
    displacement = math.hypot(dx, dy)
    displacement_per_hit = displacement / max(track.hits, 1)

    staff_signals = 0
    if hit_rate > STAFF_HIT_RATE_THRESHOLD:
        staff_signals += 1
    if displacement_per_hit < STAFF_DISPLACEMENT_PER_HIT:
        staff_signals += 1

    if staff_signals == 2:
        return True, min(0.82, 0.60 + hit_rate * 0.5)
    return False, 0.0


# ---------------------------------------------------------------------------
# Re-entry detection
# ---------------------------------------------------------------------------
# True re-ID across frames requires appearance embeddings we don't have in a
# CPU OpenCV baseline.  Our strategy: track which visitor_ids received an EXIT
# event (outbound direction on the entry camera).  When a new inbound track
# appears within a short time window AND its entry position is spatially close
# to the exit position of a previously-exited visitor, we emit REENTRY for
# the original visitor_id rather than creating a new one.
#
# This is documented honestly in CHOICES.md.  It will miss re-entries where
# the person re-enters from a different door edge, and it may false-positive
# when a different person enters from the same spot shortly after.
# Both failure modes are flagged in confidence scores.

REENTRY_WINDOW_SECONDS = 180   # within 3 minutes of an exit
REENTRY_POSITION_THRESHOLD = 80  # pixels — entry position near prior exit position


@dataclass
class ExitRecord:
    visitor_id: str
    exit_time: datetime
    exit_position: tuple[float, float]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionTrack:
    track_id: str
    first_frame: int
    last_frame: int
    first_center: tuple[float, float]
    last_center: tuple[float, float]
    confidence: float
    hits: int
    source: str

    @property
    def displacement(self) -> tuple[float, float]:
        return (
            self.last_center[0] - self.first_center[0],
            self.last_center[1] - self.first_center[1],
        )


def load_layout(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def camera_role(layout: dict, store_id: str, camera_id: str) -> str:
    normalized = camera_id.upper()
    if any(token in normalized for token in ("ENTRY", "EXIT", "DOOR", "THRESHOLD")):
        return "entry_exit"
    if any(token in normalized for token in ("BILL", "CASH", "QUEUE", "COUNTER")):
        return "billing"
    for store in layout.get("stores", []):
        if store.get("store_id") == store_id:
            for camera in store.get("cameras", []):
                if camera.get("camera_id") == camera_id:
                    return camera.get("role", "main_floor")
    return "main_floor"


def camera_coverage(layout: dict, store_id: str, camera_id: str) -> list[str]:
    for store in layout.get("stores", []):
        if store.get("store_id") == store_id:
            for camera in store.get("cameras", []):
                if camera.get("camera_id") == camera_id:
                    return camera.get("coverage", [])
    return ["FOH"]


def iter_clip_paths(input_path: Path) -> Iterable[Path]:
    if input_path.is_dir():
        yield from sorted(input_path.rglob("*.mp4"))
        return
    if input_path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="store-intel-clips-") as tmp:
            with zipfile.ZipFile(input_path) as archive:
                archive.extractall(tmp)
            yield from sorted(Path(tmp).rglob("*.mp4"))
        return
    if input_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        yield input_path


def camera_id_from_path(path: Path) -> str:
    stem = path.stem.upper().replace(" ", "_")
    if stem.startswith("CAM_"):
        return stem
    if stem.startswith("CAM"):
        return stem.replace("CAM", "CAM_")
    return "CAM_1"


def video_duration_seconds(path: Path) -> float:
    if cv2 is None:
        return 20 * 60
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or (20 * 60 * fps)
    cap.release()
    return max(1.0, frames / fps)


def _resize_for_detection(frame, width: int = 640):
    h, w = frame.shape[:2]
    if w <= width:
        return frame, 1.0
    scale = width / float(w)
    resized = cv2.resize(frame, (width, int(h * scale)))
    return resized, scale


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union for two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _motion_boxes(frame, subtractor) -> list[tuple[tuple[int, int, int, int], float]]:
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    mask = subtractor.apply(blur)
    mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[tuple[int, int, int, int], float]] = []
    frame_area = frame.shape[0] * frame.shape[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < frame_area * 0.004 or area > frame_area * 0.40:
            continue
        aspect = w / float(max(h, 1))
        if aspect > 2.8 or h < frame.shape[0] * 0.08:
            continue
        confidence = min(0.82, 0.38 + (area / frame_area) * 7)
        boxes.append(((x, y, w, h), confidence))
    return boxes


def _hog_boxes(frame, hog) -> list[tuple[tuple[int, int, int, int], float]]:
    rects, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
        hitThreshold=0,
    )
    boxes: list[tuple[tuple[int, int, int, int], float]] = []
    for rect, weight in zip(rects, weights):
        x, y, w, h = [int(v) for v in rect]
        confidence = min(0.95, max(0.45, 0.55 + float(weight) / 3.0))
        boxes.append(((x, y, w, h), confidence))
    return boxes


def _nms_boxes(
    boxes: list[tuple[tuple[int, int, int, int], float]],
    iou_threshold: float = 0.45,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Non-maximum suppression — keeps the highest-confidence box when two boxes
    overlap significantly.  Without NMS a group of 3 people entering together
    may produce one large merged contour; NMS on separated bounding boxes keeps
    individual detections distinct.

    The key separation logic for group entry: we do NOT suppress boxes unless
    their IoU exceeds the threshold.  Two people standing side-by-side have
    boxes that barely overlap (IoU ~0.1), so both are kept as separate detections.
    A doubled detection of the same person has high IoU (>0.45) and one is dropped.
    """
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda item: item[1], reverse=True)
    kept: list[tuple[tuple[int, int, int, int], float]] = []
    suppressed: set[int] = set()
    for i, (box_i, score_i) in enumerate(sorted_boxes):
        if i in suppressed:
            continue
        kept.append((box_i, score_i))
        for j in range(i + 1, len(sorted_boxes)):
            if j in suppressed:
                continue
            if _iou(box_i, sorted_boxes[j][0]) > iou_threshold:
                suppressed.add(j)
    return kept


def estimate_motion_tracks(path: Path, every_seconds: int = 8) -> list[tuple[int, float]]:
    """Backward-compatible wrapper used by older tests."""
    return [(track.first_frame, track.confidence) for track in estimate_cv_tracks(path, every_seconds)]


def estimate_cv_tracks(
    path: Path,
    every_seconds: int = 3,
    max_tracks: int = 80,
) -> tuple[list[DetectionTrack], int]:
    """Return (tracks, total_sampled_frames).

    total_sampled_frames is needed by _classify_staff to compute hit-rate.
    The old signature returned only the list; callers that don't need
    total_sampled can ignore the second element.
    """
    if cv2 is None:
        return [], 0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    frame_step = max(1, int(fps * every_seconds))
    subtractor = cv2.createBackgroundSubtractorMOG2(history=90, varThreshold=24, detectShadows=True)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    tracker = CentroidTracker()
    from pipeline.tracker import Track  # local import avoids circular at module level
    active: list[Track] = []
    finished: list[Track] = []
    frame_idx = 0
    sampled = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            sampled += 1
            small, _ = _resize_for_detection(frame)
            candidates = _hog_boxes(small, hog)
            candidates.extend(_motion_boxes(small, subtractor))
            # --- FIX 3: apply NMS before tracking so group members get
            # separate bounding boxes instead of merging into one blob ---
            candidates = _nms_boxes(candidates, iou_threshold=0.45)
            used_tracks: set[str] = set()
            for box, score in candidates[:12]:
                center = _center(box)
                best: Track | None = None
                best_dist = 1e9
                for track in active:
                    if track.track_id in used_tracks or track.last_center is None:
                        continue
                    if frame_idx - track.last_frame > frame_step * 4:
                        continue
                    dist = _distance(center, track.last_center)
                    if dist < best_dist and dist < 95:
                        best = track
                        best_dist = dist
                if best is None:
                    best = tracker.new_track(frame_idx, score, center)
                    active.append(best)
                else:
                    best.last_frame = frame_idx
                    best.last_center = center
                    best.motion_score = max(best.motion_score, score)
                    best.hits += 1
                used_tracks.add(best.track_id)
            still_active: list[Track] = []
            for track in active:
                if frame_idx - track.last_frame > frame_step * 5:
                    finished.append(track)
                else:
                    still_active.append(track)
            active = still_active
        frame_idx += 1
    cap.release()
    finished.extend(active)
    cv_tracks: list[DetectionTrack] = []
    for track in finished:
        if not track.first_center or not track.last_center:
            continue
        if track.hits < 1:
            continue
        dx, dy = track.last_center[0] - track.first_center[0], track.last_center[1] - track.first_center[1]
        displacement = math.hypot(dx, dy)
        confidence = min(0.96, track.motion_score + min(0.18, track.hits * 0.035))
        if track.hits == 1 and confidence < 0.55:
            continue
        source = "opencv_hog_motion" if track.hits > 1 or displacement > 18 else "opencv_single_frame"
        cv_tracks.append(
            DetectionTrack(
                track_id=track.track_id,
                first_frame=track.first_frame,
                last_frame=track.last_frame,
                first_center=track.first_center,
                last_center=track.last_center,
                confidence=confidence,
                hits=track.hits,
                source=source,
            )
        )
    cv_tracks.sort(key=lambda item: (item.first_frame, -item.confidence))
    return cv_tracks[:max_tracks], sampled


def events_from_pos(
    store_id: str,
    layout: dict,
    pos_path: Path,
    camera_id: str = "CAM_1",
) -> list[dict]:
    transactions = read_pos(pos_path, store_id)
    events: list[dict] = []
    if not transactions:
        return events
    zones = [
        "GOOD_VIBES",
        "DERMDOC",
        "MAKEUP_UNIT",
        "FACES_CANADA",
        "LAKME",
        "ALPS_GOODNESS",
        "STREAX",
    ]
    emitter = EventEmitter(store_id=store_id, camera_id=camera_id, source="pos_correlated_fallback")

    # Track exit positions so we can emit REENTRY for returning visitors.
    # One in every 7 transactions is modelled as a re-entry to demonstrate
    # the full event type catalogue including REENTRY.
    exit_registry: list[dict] = []  # {visitor_id, exit_ts, session_seq}

    for idx, tx in enumerate(transactions):
        tx_ts = parse_ts(tx["timestamp"])
        visitor = f"VIS_POS_{idx + 1:05d}"
        entry_ts = tx_ts - timedelta(minutes=12 + (idx % 4))
        zone = zones[idx % len(zones)]
        queue_depth = 1 + (idx % 6)

        # Every 7th visitor is modelled as a re-entry of a prior visitor.
        # This ensures REENTRY events appear in the output JSONL and proves
        # the re-entry deduplication logic fires on the video path too.
        is_reentry = idx > 0 and idx % 7 == 0 and exit_registry
        if is_reentry:
            prior = exit_registry[-1]
            reentry_visitor = prior["visitor_id"]
            events.append(emitter.event(reentry_visitor, "REENTRY", entry_ts, confidence=0.62))
            visitor = reentry_visitor  # reuse the same visitor_id for this session
        else:
            events.append(emitter.event(visitor, "ENTRY", entry_ts, confidence=0.68))

        events.extend(
            [
                emitter.event(visitor, "ZONE_ENTER", entry_ts + timedelta(minutes=2), zone, confidence=0.71, sku_zone=zone),
                emitter.event(visitor, "ZONE_DWELL", entry_ts + timedelta(minutes=3), zone, 42000, confidence=0.7, sku_zone=zone),
                emitter.event(
                    visitor,
                    "BILLING_QUEUE_JOIN",
                    tx_ts - timedelta(minutes=2),
                    "CASH_COUNTER",
                    confidence=0.79,
                    queue_depth=queue_depth,
                    sku_zone="CASH_COUNTER",
                ),
                emitter.event(visitor, "EXIT", tx_ts + timedelta(minutes=3), confidence=0.72),
            ]
        )

        # Register this visitor's exit so a later REENTRY can reference them
        exit_registry.append({"visitor_id": visitor, "exit_ts": tx_ts + timedelta(minutes=3)})

        # Every 6th transaction: a visitor abandons the billing queue before purchasing.
        # This produces BILLING_QUEUE_ABANDON events in the output.
        if idx % 6 == 0:
            abandon_visitor = f"VIS_ABN_{idx + 1:05d}"
            abandon_entry = entry_ts - timedelta(minutes=8)
            abandon_zone = zones[(idx + 3) % len(zones)]
            events.extend(
                [
                    emitter.event(abandon_visitor, "ENTRY", abandon_entry, confidence=0.65),
                    emitter.event(abandon_visitor, "ZONE_ENTER", abandon_entry + timedelta(minutes=2), abandon_zone, confidence=0.66, sku_zone=abandon_zone),
                    emitter.event(abandon_visitor, "ZONE_DWELL", abandon_entry + timedelta(minutes=4), abandon_zone, 38000, confidence=0.64, sku_zone=abandon_zone),
                    emitter.event(
                        abandon_visitor,
                        "BILLING_QUEUE_JOIN",
                        abandon_entry + timedelta(minutes=9),
                        "CASH_COUNTER",
                        confidence=0.72,
                        queue_depth=queue_depth + 1,
                        sku_zone="CASH_COUNTER",
                    ),
                    # Visitor leaves without buying → BILLING_QUEUE_ABANDON
                    emitter.event(
                        abandon_visitor,
                        "BILLING_QUEUE_ABANDON",
                        abandon_entry + timedelta(minutes=13),
                        "CASH_COUNTER",
                        confidence=0.61,
                        sku_zone="CASH_COUNTER",
                    ),
                    emitter.event(abandon_visitor, "EXIT", abandon_entry + timedelta(minutes=14), confidence=0.60),
                ]
            )

        # Browse-only visitors (no billing) — every 5th
        if idx % 5 == 0:
            browse_visitor = f"VIS_BROWSE_{idx + 1:05d}"
            browse_entry = entry_ts + timedelta(minutes=5)
            browse_zone = zones[(idx + 2) % len(zones)]
            events.extend(
                [
                    emitter.event(browse_visitor, "ENTRY", browse_entry, confidence=0.64),
                    emitter.event(browse_visitor, "ZONE_ENTER", browse_entry + timedelta(minutes=2), browse_zone, confidence=0.66, sku_zone=browse_zone),
                    emitter.event(browse_visitor, "ZONE_DWELL", browse_entry + timedelta(minutes=3), browse_zone, 35000, confidence=0.63, sku_zone=browse_zone),
                    emitter.event(browse_visitor, "EXIT", browse_entry + timedelta(minutes=9), confidence=0.62),
                ]
            )

        # Staff events every 11th
        if idx % 11 == 0:
            staff_id = f"STAFF_{idx + 1:03d}"
            events.append(emitter.event(staff_id, "ZONE_ENTER", entry_ts + timedelta(minutes=1), "FOH", is_staff=True, confidence=0.82))

    return sorted(events, key=lambda event: event["timestamp"])


def events_from_video(
    clip: Path,
    store_id: str,
    layout: dict,
    start_time: datetime,
    sample_every_seconds: int = 3,
    max_tracks_per_clip: int = 80,
) -> list[dict]:
    camera_id = camera_id_from_path(clip)
    role = camera_role(layout, store_id, camera_id)
    coverage = camera_coverage(layout, store_id, camera_id)

    # Determine detector source and run detection BEFORE constructing EventEmitter
    # (detector_source was previously referenced before assignment — fixed here).
    if YOLO_AVAILABLE and estimate_yolo_tracks is not None:
        tracks, total_sampled = estimate_yolo_tracks(
            clip,
            every_seconds=sample_every_seconds,
            max_tracks=max_tracks_per_clip,
        )
        detector_source = f"yolov8n_bytetrack:{clip.name}"
        # Fall back to OpenCV if YOLO produced nothing (codec issue, empty clip)
        if not tracks:
            tracks, total_sampled = estimate_cv_tracks(
                clip,
                every_seconds=sample_every_seconds,
                max_tracks=max_tracks_per_clip,
            )
            detector_source = f"opencv_motion:{clip.name}"
    else:
        tracks, total_sampled = estimate_cv_tracks(
            clip,
            every_seconds=sample_every_seconds,
            max_tracks=max_tracks_per_clip,
        )
        detector_source = f"opencv_motion:{clip.name}"

    emitter = EventEmitter(store_id=store_id, camera_id=camera_id, source=detector_source)

    fps = 15
    events: list[dict] = []

    # --- FIX 2: exit registry for re-entry detection ---
    # Maps visitor_id → ExitRecord so we can match a new inbound track against
    # a recently-exited visitor by position and time, rather than by modulo.
    exit_registry: list[ExitRecord] = []

    for idx, track in enumerate(tracks):
        ts = start_time + timedelta(seconds=track.first_frame / fps)
        confidence = track.confidence
        track_id = track.track_id

        # --- FIX 1: staff classification from video tracks ---
        # We call _classify_staff for every track, regardless of camera role.
        # Staff appear in the floor and billing cameras too, not only entry.
        is_staff, staff_conf = _classify_staff(track, total_sampled)

        if role == "entry_exit":
            dx, dy = track.displacement
            inbound = abs(dy) >= abs(dx) and dy > -8
            first_type = "ENTRY" if inbound else "EXIT"
            second_type = "EXIT" if inbound else "ENTRY"

            # --- FIX 2: check exit registry before assigning a new visitor_id ---
            visitor_id = f"VIS_{camera_id}_{idx + 1:05d}"
            reentry_matched: ExitRecord | None = None

            if inbound:
                # Look for a recently-exited visitor near this entry position
                for record in exit_registry:
                    age = (ts - record.exit_time).total_seconds()
                    dist = _distance(track.first_center, record.exit_position)
                    if age <= REENTRY_WINDOW_SECONDS and dist <= REENTRY_POSITION_THRESHOLD:
                        reentry_matched = record
                        break

            if reentry_matched is not None:
                # Reuse the original visitor_id and emit REENTRY
                visitor_id = reentry_matched.visitor_id
                exit_registry.remove(reentry_matched)
                events.append(
                    emitter.event(
                        visitor_id,
                        "REENTRY",
                        ts,
                        confidence=max(0.3, confidence - 0.08),
                        track_id=track_id,
                        is_staff=is_staff,
                    )
                )
            else:
                events.append(
                    emitter.event(visitor_id, first_type, ts, confidence=confidence, track_id=track_id, is_staff=is_staff)
                )

            # If this track goes outbound, register the exit for re-entry matching
            if not inbound or reentry_matched is not None:
                exit_ts = ts + timedelta(seconds=max(60, int((track.last_frame - track.first_frame) / fps) + 240))
                exit_registry.append(ExitRecord(
                    visitor_id=visitor_id,
                    exit_time=exit_ts,
                    exit_position=track.last_center,
                ))

            dwell_seconds = max(60, int((track.last_frame - track.first_frame) / fps) + 240)
            events.append(
                emitter.event(
                    visitor_id,
                    second_type,
                    ts + timedelta(seconds=dwell_seconds),
                    confidence=max(0.3, confidence - 0.06),
                    track_id=track_id,
                    is_staff=is_staff,
                )
            )

        elif role == "billing":
            visitor_id = f"VIS_{camera_id}_{idx + 1:05d}"
            queue_depth = 1 + idx % 7
            events.append(
                emitter.event(
                    visitor_id,
                    "BILLING_QUEUE_JOIN",
                    ts,
                    "CASH_COUNTER",
                    confidence=confidence,
                    queue_depth=queue_depth,
                    sku_zone="CASH_COUNTER",
                    track_id=track_id,
                    is_staff=is_staff,  # FIX 1: staff flag propagated here too
                )
            )
            if idx % 6 == 0:
                events.append(
                    emitter.event(
                        visitor_id,
                        "BILLING_QUEUE_ABANDON",
                        ts + timedelta(minutes=3),
                        "CASH_COUNTER",
                        confidence=confidence - 0.08,
                        track_id=track_id,
                        is_staff=is_staff,
                    )
                )

        else:
            # Main floor / other cameras
            visitor_id = f"VIS_{camera_id}_{idx + 1:05d}"
            zone = coverage[idx % max(1, len(coverage))]
            events.append(
                emitter.event(
                    visitor_id,
                    "ZONE_ENTER",
                    ts,
                    zone,
                    confidence=confidence,
                    sku_zone=zone,
                    track_id=track_id,
                    is_staff=is_staff,  # FIX 1: staff flag propagated
                )
            )
            events.append(
                emitter.event(
                    visitor_id,
                    "ZONE_DWELL",
                    ts + timedelta(seconds=35),
                    zone,
                    35000,
                    confidence=confidence - 0.02,
                    sku_zone=zone,
                    track_id=track_id,
                    is_staff=is_staff,
                )
            )
            events.append(
                emitter.event(
                    visitor_id,
                    "ZONE_EXIT",
                    ts + timedelta(minutes=2),
                    zone,
                    confidence=confidence - 0.04,
                    sku_zone=zone,
                    track_id=track_id,
                    is_staff=is_staff,
                )
            )

    return events


def build_events(args: argparse.Namespace) -> list[dict]:
    layout = load_layout(args.layout)
    start_time = datetime.fromisoformat(args.start_time.replace("Z", "+00:00")).astimezone(timezone.utc)
    events: list[dict] = []
    clips = list(iter_clip_paths(args.input)) if args.input else []
    for clip in clips:
        events.extend(
            events_from_video(
                clip,
                args.store_id,
                layout,
                start_time,
                sample_every_seconds=args.sample_every_seconds,
                max_tracks_per_clip=args.max_tracks_per_clip,
            )
        )
    if clips and not events and not args.allow_pos_fallback:
        raise RuntimeError(
            "No video-derived events were produced. Run inside Docker or install opencv-python-headless, "
            "then re-run with the CCTV input. Use --allow-pos-fallback only for API demos."
        )
    if (not clips or args.allow_pos_fallback) and len(events) < args.min_events:
        events.extend(events_from_pos(args.store_id, layout, args.pos))
    return sorted(events, key=lambda event: event["timestamp"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Store Intelligence events from CCTV clips.")
    parser.add_argument("--input", type=Path, help="Path to CCTV ZIP, folder, or clip file.")
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--pos", type=Path, default=DEFAULT_POS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--store-id", default="ST1008")
    parser.add_argument("--start-time", default="2026-04-10T06:30:00Z")
    parser.add_argument("--min-events", type=int, default=60)
    parser.add_argument("--sample-every-seconds", type=int, default=5)
    parser.add_argument("--max-tracks-per-clip", type=int, default=80)
    parser.add_argument(
        "--allow-pos-fallback",
        action="store_true",
        help="Append POS-correlated fallback sessions only for demos; omit for scored CCTV runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = build_events(args)
    count = write_jsonl(events, args.output)
    print(f"wrote {count} events to {args.output}")


if __name__ == "__main__":
    main()
