"""
Stream receiver server — run on your laptop.

Accepts JPEG frames POSTed from Raspberry Pis (raw_stream_demo.py,
object_detection_demo.py, or segmentation_demo_overlay.py) and serves a single
dashboard page that lets the user switch between individual cameras or monitor
every camera together in a responsive grid.

Usage:
  python stream_receiver_server.py [--host 0.0.0.0] [--port 9000]
  # or: uvicorn stream_receiver_server:app --host 0.0.0.0 --port 9000

Then open http://localhost:9000 in a browser. On each Pi, set STREAM_SERVER_URL
to your laptop's IP (e.g. http://192.168.1.100:9000) and STREAM_CAMERA_ID so
frames are tagged with the camera id.
"""

import argparse
import asyncio
import base64
import json
import math
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

app = FastAPI(title="Stream receiver")

# Latest JPEG frame per camera_id: {camera_id: (jpeg_bytes, receive_time)}
_latest = {}
# Per-camera (receive_time, capture_time) for last N frames for metrics
_metrics_history = {}  # camera_id -> list of (receive_time, capture_time), max 60
# Per-camera stream state for event detection
_stream_was_active = {}  # camera_id -> bool
# Per-camera event log: list of { "time": unix_ts, "message": str }
_event_log = {}  # camera_id -> list, max MAX_LOG_ENTRIES
# Global system log and alert-frame cache for supervisor operational activity.
_system_log = []
_alert_frames = {}
_next_system_event_id = 1
OPERATIONAL_MODES = {"free", "safety", "search", "investigation"}
PLACEHOLDER_OPERATIONAL_MODES = {"investigation"}
SAFETY_HAZARD_NAMES = {
    "fire_smoke": "Fire Hazard",
    "work_zone_encroachment": "Work-Zone Intrusion",
    "machine_obstacle_proximity": "Obstacle Hazard",
    "after_hours_intrusion": "Unauthorized Entry",
}
SAFETY_WARNING_KEYS = {"machine_obstacle_proximity"}
_system_state = {
    "mode": "free",
    "objective": None,
    "scanner_running": False,
    "placeholder": False,
    "zones": [],
    "updated_at": None,
    "safety_status": "clear",
    "active_safety_hazards": [],
    "safety_updated_at": None,
}
_lock = threading.Lock()
MJPEG_BOUNDARY = "frame"
MAX_METRICS_SAMPLES = 60
MAX_LOG_ENTRIES = 80
MAX_SYSTEM_LOG_ENTRIES = 200
STREAM_STALE_SEC = 2.0

CAMERAS_FILE = Path(__file__).with_name("cameras.json")


def _load_registry() -> dict:
    """Load camera registry from cameras.json. Returns {id: {name, location, ...}}."""
    try:
        with open(CAMERAS_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    registry = {}
    for cam in data.get("cameras", []):
        try:
            cid = int(cam["id"])
        except (KeyError, TypeError, ValueError):
            continue
        registry[cid] = {
            "name": cam.get("name", f"Camera {cid}"),
            "location": cam.get("location", ""),
            "zone": str(cam.get("zone") or "Unassigned").strip() or "Unassigned",
            "pi_host": cam.get("pi_host"),
        }
    return registry


CAMERA_REGISTRY = _load_registry()
ZONE_CAMERAS: dict[str, list[int]] = {}
for _camera_id, _camera in sorted(CAMERA_REGISTRY.items()):
    ZONE_CAMERAS.setdefault(_camera["zone"], []).append(_camera_id)
_system_state["zones"] = [
    {
        "zone": zone,
        "camera_ids": list(camera_ids),
        "mode": "free",
        "objective": None,
        "scanner_running": False,
        "placeholder": False,
    }
    for zone, camera_ids in sorted(ZONE_CAMERAS.items())
]


def _get_args():
    p = argparse.ArgumentParser(description="Receive camera stream from Pi and serve viewer")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=9000, help="Bind port")
    return p.parse_args()


def _update_metrics(camera_id: int, receive_time: float, capture_time):
    if capture_time is None:
        return
    if camera_id not in _metrics_history:
        _metrics_history[camera_id] = []
    _metrics_history[camera_id].append((receive_time, capture_time))
    if len(_metrics_history[camera_id]) > MAX_METRICS_SAMPLES:
        _metrics_history[camera_id] = _metrics_history[camera_id][-MAX_METRICS_SAMPLES:]


def _log_event(camera_id: int, message: str) -> None:
    if camera_id not in _event_log:
        _event_log[camera_id] = []
    _event_log[camera_id].append({"time": time.time(), "message": message})
    if len(_event_log[camera_id]) > MAX_LOG_ENTRIES:
        _event_log[camera_id] = _event_log[camera_id][-MAX_LOG_ENTRIES:]


def _log_system_event(payload: dict) -> dict:
    global _next_system_event_id

    entry = {
        "id": _next_system_event_id,
        "time": time.time(),
        "kind": str(payload.get("kind") or "system"),
        "level": str(payload.get("level") or "info"),
        "message": str(payload.get("message") or ""),
    }
    _next_system_event_id += 1

    camera_id = payload.get("camera_id")
    if camera_id is not None:
        try:
            entry["camera_id"] = int(camera_id)
            camera = CAMERA_REGISTRY.get(entry["camera_id"])
            if camera is not None:
                entry["zone"] = camera["zone"]
        except (TypeError, ValueError):
            pass

    confidence = payload.get("confidence")
    if confidence is not None:
        try:
            entry["confidence"] = round(float(confidence), 3)
        except (TypeError, ValueError):
            pass

    for field in ("hazard_key", "hazard_name", "cause", "reason"):
        value = payload.get(field)
        if value is not None:
            entry[field] = str(value)

    audible = payload.get("audible")
    if isinstance(audible, bool):
        entry["audible"] = audible

    frame_b64 = payload.get("frame_jpeg_b64")
    if frame_b64:
        try:
            _alert_frames[entry["id"]] = base64.b64decode(frame_b64, validate=True)
            entry["frame_url"] = f"/system/alerts/{entry['id']}.jpg"
        except Exception:
            pass

    _system_log.append(entry)
    if len(_system_log) > MAX_SYSTEM_LOG_ENTRIES:
        removed = _system_log[:-MAX_SYSTEM_LOG_ENTRIES]
        del _system_log[:-MAX_SYSTEM_LOG_ENTRIES]
        for old_entry in removed:
            _alert_frames.pop(old_entry["id"], None)
    return dict(entry)


def _compute_metrics(camera_id: int) -> dict:
    with _lock:
        samples = list(_metrics_history.get(camera_id) or [])
        latest = _latest.get(camera_id)
    last_receive = latest[1] if latest else None
    stream_active = (
        last_receive is not None and (time.time() - last_receive) < STREAM_STALE_SEC
    )
    with _lock:
        was_active = _stream_was_active.get(camera_id, False)
        if was_active and not stream_active:
            _log_event(camera_id, "Stream stopped")
            _stream_was_active[camera_id] = False
        elif not was_active and stream_active:
            _stream_was_active[camera_id] = True
    out = {
        "camera_id": camera_id,
        "stream_active": stream_active,
        "delay_ms": None,
        "jitter_ms": None,
        "fps": None,
    }
    if len(samples) < 2:
        return out
    delays = [(r - c) * 1000 for r, c in samples]
    receive_times = [r for r, _ in samples]
    delay_ms = delays[-1]
    avg_delay = sum(delays) / len(delays)
    variance = sum((d - avg_delay) ** 2 for d in delays) / len(delays)
    jitter_ms = math.sqrt(variance) if variance >= 0 else 0
    intervals = [receive_times[i + 1] - receive_times[i] for i in range(len(receive_times) - 1)]
    avg_interval = sum(intervals) / len(intervals) if intervals else 0
    fps = 1.0 / avg_interval if avg_interval > 0 else None
    out["delay_ms"] = round(delay_ms, 1)
    out["jitter_ms"] = round(jitter_ms, 1)
    out["fps"] = round(fps, 1) if fps is not None else None
    return out


@app.post("/frame")
async def receive_frame(request: Request):
    """Accept a JPEG frame from a Pi. Optional X-Camera-Id, X-Capture-Time (unix ts)."""
    camera_id = request.headers.get("X-Camera-Id") or request.query_params.get("camera_id", "0")
    try:
        camera_id = int(camera_id)
    except ValueError:
        camera_id = 0
    capture_time = request.headers.get("X-Capture-Time")
    if capture_time is not None:
        try:
            capture_time = float(capture_time)
        except ValueError:
            capture_time = None
    data = await request.body()
    if not data:
        return JSONResponse({"status": "error", "error": "empty body"}, status_code=400)
    receive_time = time.time()
    with _lock:
        was_active = _stream_was_active.get(camera_id, False)
        if not was_active:
            _log_event(camera_id, "Stream started")
            _stream_was_active[camera_id] = True
        _latest[camera_id] = (data, receive_time)
        _update_metrics(camera_id, receive_time, capture_time)
    return JSONResponse({"status": "ok", "camera_id": camera_id})


@app.get("/cameras")
async def list_cameras():
    """Return union of registered cameras and any that have ever streamed to us."""
    camera_registry = _load_registry()
    with _lock:
        seen_ids = set(_latest.keys()) | set(_event_log.keys()) | set(_metrics_history.keys())
        last_seen_map = {cid: t[1] for cid, t in _latest.items()}
    all_ids = sorted(set(camera_registry.keys()) | seen_ids)
    now = time.time()
    cameras = []
    for cid in all_ids:
        info = camera_registry.get(cid, {})
        last_seen = last_seen_map.get(cid)
        cameras.append({
            "camera_id": cid,
            "name": info.get("name", f"Camera {cid}"),
            "location": info.get("location", ""),
            "zone": info.get("zone", "Unassigned"),
            "pi_host": info.get("pi_host"),
            "registered": cid in camera_registry,
            "last_seen": last_seen,
            "stream_active": last_seen is not None and (now - last_seen) < STREAM_STALE_SEC,
        })
    return JSONResponse({"cameras": cameras})


@app.get("/metrics")
async def metrics(camera_id: int = 0):
    return JSONResponse(_compute_metrics(camera_id))


@app.get("/log")
async def get_log(camera_id: int = 0):
    with _lock:
        entries = list(_event_log.get(camera_id) or [])
    return JSONResponse(
        [{"time": e["time"], "message": e["message"]} for e in entries]
    )


@app.post("/log/clear")
async def clear_log(camera_id: int = 0):
    with _lock:
        _event_log[camera_id] = []
    return JSONResponse({"status": "ok", "camera_id": camera_id})


@app.get("/latest_frame/{camera_id}")
async def latest_frame(camera_id: int):
    with _lock:
        frame = _latest.get(camera_id)
    if frame is None:
        return Response(status_code=404)
    jpeg_bytes, recv_ts = frame
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"X-Receive-Time": str(recv_ts)},
    )


def _system_state_copy_locked() -> dict:
    state = dict(_system_state)
    state["zones"] = [dict(zone_state) for zone_state in _system_state.get("zones") or []]
    state["active_safety_hazards"] = [
        dict(hazard) for hazard in _system_state.get("active_safety_hazards") or []
    ]
    return state


@app.get("/system/state")
async def get_system_state():
    with _lock:
        state = _system_state_copy_locked()
    return JSONResponse(state)


@app.post("/system/state")
async def update_system_state(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "error": "Request body must be JSON."}, status_code=400)

    raw_zone_states = payload.get("zones")
    if raw_zone_states is not None and not isinstance(raw_zone_states, list):
        return JSONResponse(
            {"status": "error", "error": "zones must be an array."},
            status_code=400,
        )

    zone_lookup = {zone.casefold(): zone for zone in ZONE_CAMERAS}
    normalized_updates: dict[str, dict] = {}
    if raw_zone_states is not None:
        for raw_state in raw_zone_states:
            if not isinstance(raw_state, dict):
                return JSONResponse(
                    {"status": "error", "error": "Every zone state must be an object."},
                    status_code=400,
                )
            raw_zone = str(raw_state.get("zone") or "").strip()
            zone = zone_lookup.get(raw_zone.casefold())
            if zone is None:
                return JSONResponse(
                    {"status": "error", "error": f"Unknown zone: {raw_zone}"},
                    status_code=400,
                )
            if zone in normalized_updates:
                return JSONResponse(
                    {"status": "error", "error": f"Duplicate zone state: {zone}"},
                    status_code=400,
                )
            mode = str(raw_state.get("mode") or "free").strip().lower()
            if mode not in OPERATIONAL_MODES:
                return JSONResponse(
                    {"status": "error", "error": f"Unknown operational mode: {mode}"},
                    status_code=400,
                )
            objective = raw_state.get("objective")
            if objective is not None:
                objective = str(objective).strip() or None
            if mode in {"free", "safety"}:
                objective = None
            if mode == "search" and objective is None:
                return JSONResponse(
                    {"status": "error", "error": f"Search Mode requires an objective for {zone}."},
                    status_code=400,
                )
            normalized_updates[zone] = {
                "zone": zone,
                "camera_ids": list(ZONE_CAMERAS[zone]),
                "mode": mode,
                "objective": objective,
                "scanner_running": (
                    bool(raw_state.get("scanner_running"))
                    if mode in {"search", "safety"}
                    else False
                ),
                "placeholder": mode in PLACEHOLDER_OPERATIONAL_MODES,
            }
    else:
        # Backward-compatible whole-site update: apply the requested mode to every zone.
        mode = str(payload.get("mode") or "free").strip().lower()
        if mode not in OPERATIONAL_MODES:
            return JSONResponse(
                {"status": "error", "error": f"Unknown operational mode: {mode}"},
                status_code=400,
            )
        objective = payload.get("objective")
        if objective is not None:
            objective = str(objective).strip() or None
        if mode in {"free", "safety"}:
            objective = None
        if mode == "search" and objective is None:
            return JSONResponse(
                {"status": "error", "error": "Search Mode requires an objective."},
                status_code=400,
            )
        normalized_updates = {
            zone: {
                "zone": zone,
                "camera_ids": list(camera_ids),
                "mode": mode,
                "objective": objective,
                "scanner_running": (
                    bool(payload.get("scanner_running"))
                    if mode in {"search", "safety"}
                    else False
                ),
                "placeholder": mode in PLACEHOLDER_OPERATIONAL_MODES,
            }
            for zone, camera_ids in ZONE_CAMERAS.items()
        }

    with _lock:
        existing = {
            state["zone"]: dict(state)
            for state in _system_state.get("zones") or []
        }
        existing.update(normalized_updates)
        zone_states = [existing[zone] for zone in sorted(existing)]
        modes = {state["mode"] for state in zone_states}
        objectives = {state["objective"] for state in zone_states}
        _system_state["mode"] = next(iter(modes)) if len(modes) == 1 else "mixed"
        _system_state["objective"] = (
            next(iter(objectives)) if len(modes) == 1 and len(objectives) == 1 else None
        )
        _system_state["scanner_running"] = any(
            state["scanner_running"] for state in zone_states
        )
        _system_state["placeholder"] = bool(zone_states) and all(
            state["placeholder"] for state in zone_states
        )
        _system_state["zones"] = zone_states
        _system_state["updated_at"] = time.time()
        state = _system_state_copy_locked()
    return JSONResponse({"status": "ok", "state": state})


@app.post("/system/safety/hazard")
async def latch_safety_hazard(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "error": "Request body must be JSON."}, status_code=400)

    hazard_key = str(payload.get("hazard_key") or "").strip()
    hazard_name = SAFETY_HAZARD_NAMES.get(hazard_key)
    if hazard_name is None:
        return JSONResponse(
            {"status": "error", "error": f"Unknown safety hazard: {hazard_key}"},
            status_code=400,
        )
    try:
        camera_id = int(payload["camera_id"])
    except (KeyError, TypeError, ValueError):
        return JSONResponse(
            {"status": "error", "error": "camera_id must be an integer."},
            status_code=400,
        )

    cause = str(payload.get("cause") or "").strip() or "Visible evidence triggered this safety check."
    zone = CAMERA_REGISTRY.get(camera_id, {}).get("zone", "Unassigned")
    is_warning = hazard_key in SAFETY_WARNING_KEYS
    message = (
        f"WARNING — {hazard_name} in {zone} on camera {camera_id}: {cause}"
        if is_warning
        else f"STOP WORK — {hazard_name} in {zone} on camera {camera_id}: {cause}"
    )
    now = time.time()
    event_payload = {
        "kind": "safety_warning" if is_warning else "safety_alert",
        "level": "warning" if is_warning else "critical",
        "audible": not is_warning,
        "message": message,
        "hazard_key": hazard_key,
        "hazard_name": hazard_name,
        "cause": cause,
        "camera_id": camera_id,
        "zone": zone,
        "confidence": payload.get("confidence"),
        "frame_jpeg_b64": payload.get("frame_jpeg_b64"),
    }

    with _lock:
        event = _log_system_event(event_payload)
        if is_warning:
            state = _system_state_copy_locked()
            return JSONResponse({"status": "ok", "event": event, "state": state})

        hazard = {
            "hazard_key": hazard_key,
            "hazard_name": hazard_name,
            "cause": cause,
            "camera_id": camera_id,
            "zone": zone,
            "confidence": event.get("confidence"),
            "detected_at": now,
            "event_id": event["id"],
        }
        active = [
            item
            for item in _system_state.get("active_safety_hazards") or []
            if not (
                item.get("hazard_key") == hazard_key
                and item.get("camera_id") == camera_id
            )
        ]
        active.append(hazard)
        _system_state["active_safety_hazards"] = active[-50:]
        _system_state["safety_status"] = "hazard"
        _system_state["safety_updated_at"] = now
        state = _system_state_copy_locked()
    return JSONResponse({"status": "ok", "event": event, "state": state})


@app.post("/system/safety/clear")
async def clear_safety_hazard_state(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    reason = str(payload.get("reason") or "").strip() or "Cleared explicitly by the operator."
    now = time.time()
    with _lock:
        _system_state["safety_status"] = "clear"
        _system_state["active_safety_hazards"] = []
        _system_state["safety_updated_at"] = now
        event = _log_system_event(
            {
                "kind": "safety_clear",
                "level": "info",
                "message": f"Construction safety state cleared by operator. {reason}",
                "reason": reason,
            }
        )
        state = _system_state_copy_locked()
    return JSONResponse({"status": "ok", "event": event, "state": state})


@app.get("/system/log")
async def get_system_log():
    with _lock:
        entries = [dict(entry) for entry in _system_log]
    return JSONResponse(entries)


@app.post("/system/log")
async def append_system_log(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "error": "Request body must be JSON."}, status_code=400)
    with _lock:
        entry = _log_system_event(payload)
    return JSONResponse({"status": "ok", "entry": entry})


@app.post("/system/log/clear")
async def clear_system_log():
    with _lock:
        _system_log.clear()
        _alert_frames.clear()
    return JSONResponse({"status": "ok"})


@app.get("/system/alerts/{event_id}.jpg")
async def get_alert_frame(event_id: int):
    with _lock:
        frame = _alert_frames.get(event_id)
    if frame is None:
        return Response(status_code=404)
    return Response(content=frame, media_type="image/jpeg")


async def _mjpeg_stream(camera_id: int):
    """Async generator that yields MJPEG multipart chunks for the given camera."""
    last_sent_ts = None
    while True:
        with _lock:
            frame = _latest.get(camera_id)
        if frame is None:
            await asyncio.sleep(0.1)
            continue
        jpeg_bytes, recv_ts = frame
        if recv_ts == last_sent_ts:
            await asyncio.sleep(1 / 30)
            continue
        last_sent_ts = recv_ts
        yield (
            b"--" + MJPEG_BOUNDARY.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n"
            + jpeg_bytes + b"\r\n"
        )
        await asyncio.sleep(1 / 30)


@app.get("/stream", response_class=Response)
@app.get("/stream/{camera_id}", response_class=Response)
async def stream(camera_id: str = "0"):
    try:
        cid = int(camera_id)
    except ValueError:
        cid = 0
    return StreamingResponse(
        _mjpeg_stream(cid),
        media_type="multipart/x-mixed-replace; boundary=" + MJPEG_BOUNDARY,
    )


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <title>Camera stream</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; }
    :root {
      --bg-start: #0f0f14;
      --bg-end: #1a1a2e;
      --card-bg: rgba(255, 255, 255, 0.04);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #00d4aa;
      --accent-dim: #00a884;
      --text: #e8e8ed;
      --text-muted: #8b8b9a;
      --glow: rgba(0, 212, 170, 0.15);
    }
    body {
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(145deg, var(--bg-start) 0%, #16162a 40%, var(--bg-end) 100%);
      color: var(--text);
      font-family: 'DM Sans', system-ui, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px;
      overflow-x: hidden;
    }
    .page-header {
      width: 100%;
      max-width: 1200px;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
      flex-wrap: wrap;
    }
    .page-header .title-block h1 {
      margin: 0;
      font-size: 1.5rem;
      font-weight: 600;
      letter-spacing: 0;
      background: linear-gradient(135deg, #fff 0%, var(--accent) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .page-header .sub {
      margin-top: 4px;
      font-size: 0.875rem;
      color: var(--text-muted);
    }
    .view-tabs {
      width: 100%;
      max-width: 1200px;
      display: flex;
      gap: 8px;
      margin-bottom: 18px;
    }
    .view-tab {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 38px;
      padding: 0 14px;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      background: rgba(255,255,255,0.03);
      color: var(--text-muted);
      font: inherit;
      font-size: 0.875rem;
      font-weight: 600;
      cursor: pointer;
      transition: color 0.2s, background 0.2s, border-color 0.2s;
    }
    .view-tab.active {
      color: var(--text);
      background: rgba(0, 212, 170, 0.12);
      border-color: rgba(0, 212, 170, 0.35);
    }
    .view-tab-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
    }
    .alarm-audio-button {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 38px;
      margin-left: auto;
      padding: 0 14px;
      border: 1px solid rgba(255, 209, 102, 0.35);
      border-radius: 8px;
      background: rgba(255, 209, 102, 0.1);
      color: #ffd166;
      font: inherit;
      font-size: 0.8125rem;
      font-weight: 600;
      cursor: pointer;
    }
    .alarm-audio-button .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
    }
    .alarm-audio-button.armed {
      border-color: rgba(52, 211, 153, 0.35);
      background: rgba(16, 185, 129, 0.12);
      color: #67e8a5;
    }
    .alarm-audio-button.pending {
      border-color: rgba(248, 113, 113, 0.72);
      background: rgba(127, 29, 29, 0.38);
      color: #fecaca;
    }
    .global-safety-banner {
      width: 100%;
      max-width: 1200px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
      padding: 14px 16px;
      border: 1px solid rgba(248, 113, 113, 0.8);
      border-radius: 10px;
      background: linear-gradient(135deg, rgba(153, 27, 27, 0.92), rgba(69, 10, 10, 0.9));
      box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.2), 0 12px 34px rgba(127, 29, 29, 0.34);
    }
    .global-safety-banner[hidden] {
      display: none;
    }
    .global-safety-title {
      color: #fff;
      font-size: 1rem;
      font-weight: 800;
      letter-spacing: 0.04em;
    }
    .global-safety-detail {
      margin-top: 3px;
      color: #fecaca;
      font-size: 0.8125rem;
      line-height: 1.35;
    }
    .global-safety-action {
      flex-shrink: 0;
      min-height: 34px;
      padding: 0 12px;
      border: 1px solid rgba(255,255,255,0.42);
      border-radius: 7px;
      background: rgba(255,255,255,0.1);
      color: #fff;
      font: inherit;
      font-size: 0.75rem;
      font-weight: 700;
      cursor: pointer;
    }
    .global-safety-warning-banner {
      width: 100%;
      max-width: 1200px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
      padding: 14px 16px;
      border: 1px solid rgba(251, 191, 36, 0.72);
      border-radius: 10px;
      background: linear-gradient(135deg, rgba(120, 53, 15, 0.9), rgba(69, 26, 3, 0.82));
      box-shadow: 0 0 0 1px rgba(245, 158, 11, 0.14), 0 12px 30px rgba(120, 53, 15, 0.24);
    }
    .global-safety-warning-banner[hidden] {
      display: none;
    }
    .global-safety-warning-title {
      color: #fef3c7;
      font-size: 1rem;
      font-weight: 800;
      letter-spacing: 0.03em;
    }
    .global-safety-warning-detail {
      margin-top: 3px;
      color: #fde68a;
      font-size: 0.8125rem;
      line-height: 1.35;
    }
    .global-safety-warning-action {
      flex-shrink: 0;
      min-height: 34px;
      padding: 0 12px;
      border: 1px solid rgba(254, 243, 199, 0.48);
      border-radius: 7px;
      background: rgba(255,255,255,0.08);
      color: #fef3c7;
      font: inherit;
      font-size: 0.75rem;
      font-weight: 700;
      cursor: pointer;
    }
    .tab-view {
      display: none;
      width: 100%;
      max-width: 1200px;
    }
    .tab-view.active {
      display: block;
    }

    /* ---------- Camera picker (custom dropdown) ---------- */
    .cam-picker {
      position: relative;
      min-width: 280px;
    }
    .cam-picker-label {
      display: block;
      font-size: 0.6875rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--text-muted);
      margin-bottom: 8px;
    }
    .cam-picker-button {
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 16px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      color: var(--text);
      font-family: inherit;
      font-size: 0.9375rem;
      font-weight: 500;
      cursor: pointer;
      text-align: left;
      transition: background 0.2s, border-color 0.2s, box-shadow 0.2s;
      box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }
    .cam-picker-button:hover {
      background: rgba(255,255,255,0.06);
      border-color: rgba(0, 212, 170, 0.35);
    }
    .cam-picker.open .cam-picker-button {
      border-color: rgba(0, 212, 170, 0.55);
      box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.15), 0 4px 18px rgba(0,0,0,0.25);
    }
    .cam-picker-button .cam-label {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
      flex: 1;
    }
    .cam-picker-button .cam-label .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      flex-shrink: 0;
      background: var(--text-muted);
    }
    .cam-picker-button .cam-label .dot.live {
      background: var(--accent);
      box-shadow: 0 0 10px rgba(0, 212, 170, 0.75);
      animation: pulse-green 2s ease-in-out infinite;
    }
    .cam-picker-button .cam-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .cam-picker-button .cam-meta {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-left: 6px;
    }
    .cam-picker-button .chevron {
      flex-shrink: 0;
      transition: transform 0.2s ease;
      color: var(--text-muted);
    }
    .cam-picker.open .cam-picker-button .chevron {
      transform: rotate(180deg);
      color: var(--accent);
    }
    .cam-picker-menu {
      position: absolute;
      top: calc(100% + 8px);
      left: 0;
      right: 0;
      background: rgba(22, 22, 42, 0.98);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 6px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.45);
      z-index: 50;
      max-height: 320px;
      overflow-y: auto;
      opacity: 0;
      visibility: hidden;
      transform: translateY(-4px);
      pointer-events: none;
      transition: opacity 0.15s ease, transform 0.15s ease;
    }
    .cam-picker.open .cam-picker-menu {
      opacity: 1;
      visibility: visible;
      transform: translateY(0);
      pointer-events: auto;
    }
    .cam-option {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 12px;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.15s;
    }
    .cam-option:hover {
      background: rgba(255,255,255,0.06);
    }
    .cam-option:focus-visible {
      outline: 2px solid rgba(0, 212, 170, 0.75);
      outline-offset: -2px;
      background: rgba(255,255,255,0.07);
    }
    .cam-option.selected {
      background: rgba(0, 212, 170, 0.12);
      border: 1px solid rgba(0, 212, 170, 0.35);
      padding: 9px 11px;
    }
    .cam-option.grid-option {
      margin-bottom: 6px;
      border-bottom: 1px solid var(--card-border);
      border-bottom-left-radius: 4px;
      border-bottom-right-radius: 4px;
    }
    .cam-option.grid-option.selected {
      border-bottom-color: rgba(0, 212, 170, 0.35);
    }
    .grid-option-icon {
      width: 16px;
      height: 16px;
      flex-shrink: 0;
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 2px;
      padding: 1px;
    }
    .grid-option-icon span {
      border-radius: 1px;
      background: var(--text-muted);
    }
    .cam-option.grid-option.selected .grid-option-icon span {
      background: var(--accent);
    }
    .cam-option .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--text-muted);
      flex-shrink: 0;
    }
    .cam-option .dot.live {
      background: var(--accent);
      box-shadow: 0 0 10px rgba(0, 212, 170, 0.7);
    }
    .cam-option .cam-text {
      flex: 1;
      min-width: 0;
    }
    .cam-option .cam-text .line1 {
      font-size: 0.9375rem;
      font-weight: 500;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .cam-option .cam-text .line2 {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 2px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .cam-option .badge {
      font-size: 0.6875rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 2px 8px;
      border-radius: 999px;
      color: var(--text-muted);
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--card-border);
    }
    .cam-option .badge.live {
      color: var(--accent);
      background: rgba(0, 212, 170, 0.14);
      border-color: rgba(0, 212, 170, 0.35);
    }
    .cam-picker-menu .empty {
      padding: 14px 12px;
      color: var(--text-muted);
      font-size: 0.875rem;
      font-style: italic;
    }

    /* ---------- Layout ---------- */
    .layout {
      display: flex;
      gap: 24px;
      align-items: flex-start;
      width: 100%;
      max-width: 1200px;
    }
    .right-col {
      display: flex;
      flex-direction: column;
      gap: 24px;
      width: 280px;
      flex-shrink: 0;
    }
    .stream-box {
      flex: 1;
      min-width: 0;
      min-height: 420px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 12px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255,255,255,0.03) inset;
      position: relative;
      overflow: hidden;
    }
    .stream-box::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--glow), transparent);
      opacity: 0.6;
    }
    .stream-box-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 4px 12px 0;
      margin-bottom: 8px;
      border-bottom: 1px solid var(--card-border);
      flex-shrink: 0;
    }
    .stream-box-top .label {
      font-size: 0.75rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
    }
    .stream-status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      z-index: 1;
      box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
    }
    .stream-status.live {
      background: rgba(0, 212, 170, 0.18);
      border: 1px solid rgba(0, 212, 170, 0.35);
      color: #00d4aa;
    }
    .stream-status.live .dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #00d4aa;
      box-shadow: 0 0 12px rgba(0, 212, 170, 0.7);
      animation: pulse-green 2s ease-in-out infinite;
    }
    .stream-status.no-stream {
      background: rgba(239, 68, 68, 0.18);
      border: 1px solid rgba(239, 68, 68, 0.35);
      color: #ef4444;
    }
    .stream-status.no-stream .dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #ef4444;
    }
    @keyframes pulse-green {
      0%, 100% { opacity: 1; box-shadow: 0 0 12px rgba(0, 212, 170, 0.7); }
      50% { opacity: 0.85; box-shadow: 0 0 6px rgba(0, 212, 170, 0.5); }
    }
    .stream-box img {
      width: 100%;
      min-height: 360px;
      max-height: 75vh;
      object-fit: contain;
      display: block;
      border-radius: 8px;
      background: #000;
    }
    .stream-box .no-cam {
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text-muted);
      min-height: 360px;
      background: #000;
      border-radius: 8px;
      font-size: 0.9375rem;
    }
    .layout[hidden],
    .camera-grid[hidden] {
      display: none;
    }
    .camera-grid {
      --grid-columns: 2;
      width: 100%;
      display: grid;
      grid-template-columns: repeat(var(--grid-columns), minmax(0, 1fr));
      gap: 18px;
      align-items: start;
    }
    .camera-grid-empty {
      grid-column: 1 / -1;
      min-height: 320px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 32px;
      border: 1px solid var(--card-border);
      border-radius: 16px;
      background: var(--card-bg);
      color: var(--text-muted);
      text-align: center;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .camera-tile {
      min-width: 0;
      padding: 10px;
      border: 1px solid var(--card-border);
      border-radius: 16px;
      background: var(--card-bg);
      overflow: hidden;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255,255,255,0.03) inset;
    }
    .camera-tile-header {
      min-width: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 4px 4px 10px;
    }
    .camera-tile-heading {
      min-width: 0;
    }
    .camera-tile-name {
      overflow: hidden;
      color: var(--text);
      font-size: 0.9375rem;
      font-weight: 600;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .camera-tile-meta {
      margin-top: 2px;
      overflow: hidden;
      color: var(--text-muted);
      font-size: 0.75rem;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .camera-tile-status {
      flex-shrink: 0;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 9px;
      border: 1px solid rgba(239, 68, 68, 0.35);
      border-radius: 999px;
      background: rgba(239, 68, 68, 0.18);
      color: #ef4444;
      font-size: 0.6875rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .camera-tile-status .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: currentColor;
    }
    .camera-tile.live .camera-tile-status {
      border-color: rgba(0, 212, 170, 0.35);
      background: rgba(0, 212, 170, 0.18);
      color: var(--accent);
    }
    .camera-tile.live .camera-tile-status .dot {
      box-shadow: 0 0 9px rgba(0, 212, 170, 0.7);
      animation: pulse-green 2s ease-in-out infinite;
    }
    .camera-tile-media {
      position: relative;
      width: 100%;
      height: clamp(210px, 28vh, 290px);
      overflow: hidden;
      border-radius: 9px;
      background: #000;
    }
    .camera-tile-media img {
      position: absolute;
      inset: 0;
      z-index: 1;
      width: 100%;
      height: 100%;
      display: none;
      object-fit: contain;
      background: #000;
    }
    .camera-tile.live .camera-tile-media img {
      display: block;
    }
    .camera-tile-placeholder {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #6f6f7c;
      font-size: 0.8125rem;
      letter-spacing: 0.02em;
    }
    .camera-tile.live .camera-tile-placeholder {
      display: none;
    }
    .camera-tile.stream-error .camera-tile-status {
      border-color: rgba(255, 209, 102, 0.35);
      background: rgba(255, 209, 102, 0.12);
      color: #ffd166;
    }
    .camera-tile.stream-error .camera-tile-media img {
      display: none;
    }
    .camera-tile.stream-error .camera-tile-placeholder {
      display: flex;
      color: #9a895d;
    }
    .panel {
      width: 100%;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px 20px;
      background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, transparent 100%);
      border-bottom: 1px solid var(--card-border);
      font-size: 0.8125rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .btn-clear {
      padding: 4px 10px;
      font-size: 0.6875rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-muted);
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--card-border);
      border-radius: 6px;
      cursor: pointer;
      font-family: inherit;
      transition: color 0.2s, background 0.2s, border-color 0.2s;
    }
    .btn-clear:hover {
      color: var(--text);
      background: rgba(255,255,255,0.1);
      border-color: rgba(255,255,255,0.15);
    }
    .perf-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9375rem;
    }
    .perf-table th {
      text-align: left;
      padding: 14px 20px;
      font-weight: 500;
      color: var(--text-muted);
      border-bottom: 1px solid var(--card-border);
    }
    .perf-table td {
      padding: 14px 20px;
      border-bottom: 1px solid var(--card-border);
    }
    .perf-table tr:last-child td { border-bottom: none; }
    .perf-table .value {
      font-weight: 600;
      color: var(--accent);
      font-variant-numeric: tabular-nums;
    }
    .perf-table .value.none {
      color: var(--text-muted);
      font-weight: 500;
    }
    .perf-table .unit {
      margin-left: 4px;
      font-size: 0.8125rem;
      color: var(--text-muted);
      font-weight: 400;
    }
    .log-box {
      max-height: 200px;
      overflow-y: auto;
      padding: 12px;
      background: rgba(0, 0, 0, 0.25);
      font-size: 0.8125rem;
      font-family: 'DM Sans', system-ui, sans-serif;
    }
    .log-box::-webkit-scrollbar { width: 6px; }
    .log-box::-webkit-scrollbar-track { background: rgba(255,255,255,0.04); border-radius: 3px; }
    .log-box::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 3px; }
    .log-entry {
      display: flex;
      gap: 10px;
      padding: 6px 0;
      border-bottom: 1px solid rgba(255,255,255,0.04);
      line-height: 1.4;
    }
    .log-entry:last-child { border-bottom: none; }
    .log-entry .ts {
      flex-shrink: 0;
      color: var(--text-muted);
      font-variant-numeric: tabular-nums;
    }
    .log-entry .msg {
      color: var(--text);
      word-break: break-word;
    }
    .log-entry.stream-started .msg { color: var(--accent); }
    .log-entry.stream-stopped .msg { color: #ef4444; }
    .log-box .empty {
      color: var(--text-muted);
      font-style: italic;
      padding: 8px 0;
    }
    .system-view-shell {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .system-overview {
      display: grid;
      grid-template-columns: 160px minmax(180px, 1fr) 160px minmax(260px, 1.2fr);
      gap: 12px;
    }
    .system-stat {
      min-height: 82px;
      padding: 14px 16px;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      background: rgba(255,255,255,0.04);
    }
    .system-stat .label {
      font-size: 0.6875rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
    }
    .system-stat .value {
      margin-top: 8px;
      font-size: 1rem;
      font-weight: 700;
      color: var(--text);
    }
    .system-stat .value.search {
      color: #7dd3fc;
    }
    .system-stat .value.safety-clear {
      color: #67e8a5;
    }
    .system-stat .value.safety-hazard {
      color: #ff6b6b;
    }
    .system-stat.safety-state {
      transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
    }
    .system-overview.hazard .system-stat.safety-state {
      border-color: rgba(239, 68, 68, 0.72);
      background: linear-gradient(135deg, rgba(127, 29, 29, 0.44), rgba(69, 10, 10, 0.24));
      box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.15), 0 10px 30px rgba(127, 29, 29, 0.2);
    }
    .safety-detail {
      margin-top: 7px;
      color: var(--text-muted);
      font-size: 0.75rem;
      line-height: 1.35;
    }
    .system-overview.hazard .safety-detail {
      color: #fecaca;
    }
    .system-stat.objective .value {
      font-size: 0.9375rem;
      font-weight: 500;
      line-height: 1.4;
    }
    .zone-state-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .zone-state-card {
      padding: 14px 16px;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      background: rgba(255,255,255,0.04);
    }
    .zone-state-card.search { border-color: rgba(56, 189, 248, 0.48); }
    .zone-state-card.safety { border-color: rgba(52, 211, 153, 0.42); }
    .zone-state-card.hazard {
      border-color: rgba(239, 68, 68, 0.76);
      background: linear-gradient(135deg, rgba(127, 29, 29, 0.4), rgba(69, 10, 10, 0.2));
    }
    .zone-state-heading {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
    }
    .zone-state-name { font-size: 0.9375rem; font-weight: 750; }
    .zone-state-cameras { color: var(--text-muted); font-size: 0.6875rem; }
    .zone-state-mode {
      margin-top: 8px;
      color: #7dd3fc;
      font-size: 0.8125rem;
      font-weight: 700;
    }
    .zone-state-card.free .zone-state-mode { color: #67e8a5; }
    .zone-state-card.investigation .zone-state-mode { color: #c4b5fd; }
    .zone-state-card.hazard .zone-state-mode { color: #fecaca; }
    .zone-state-detail {
      margin-top: 5px;
      min-height: 2.7em;
      color: var(--text-muted);
      font-size: 0.75rem;
      line-height: 1.35;
    }
    .system-log {
      width: 100%;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      overflow: hidden;
      background: rgba(255,255,255,0.04);
    }
    .system-log-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--card-border);
      background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, transparent 100%);
    }
    .system-log-title {
      font-size: 0.8125rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .system-log-list {
      max-height: min(68vh, 720px);
      overflow-y: auto;
      padding: 10px 18px 18px;
      background: rgba(0,0,0,0.18);
    }
    .system-log-empty {
      padding: 18px 0 10px;
      color: var(--text-muted);
      font-style: italic;
    }
    .system-entry {
      display: grid;
      grid-template-columns: 72px 76px 1fr;
      gap: 12px;
      align-items: start;
      padding: 12px 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .system-entry:last-child {
      border-bottom: none;
    }
    .system-entry .time {
      color: var(--text-muted);
      font-size: 0.8125rem;
      font-variant-numeric: tabular-nums;
    }
    .system-entry .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      border: 1px solid var(--card-border);
      background: rgba(255,255,255,0.05);
      color: var(--text-muted);
      font-size: 0.6875rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .system-entry .content {
      min-width: 0;
      color: var(--text);
      line-height: 1.45;
    }
    .system-entry.alert .badge {
      color: #7dd3fc;
      border-color: rgba(56, 189, 248, 0.42);
      background: rgba(56, 189, 248, 0.14);
    }
    .system-entry.alert .content strong {
      color: #fff;
      font-weight: 700;
    }
    .system-entry.safety-alert {
      margin: 8px 0;
      padding: 14px 12px;
      border: 1px solid rgba(239, 68, 68, 0.5);
      border-radius: 8px;
      background: linear-gradient(135deg, rgba(127, 29, 29, 0.4), rgba(69, 10, 10, 0.16));
    }
    .system-entry.safety-alert .badge {
      color: #fff;
      border-color: rgba(248, 113, 113, 0.75);
      background: #b91c1c;
    }
    .system-entry.safety-alert .content strong {
      color: #fecaca;
      font-weight: 800;
    }
    .system-entry.safety-warning {
      margin: 8px 0;
      padding: 14px 12px;
      border: 1px solid rgba(245, 158, 11, 0.5);
      border-radius: 8px;
      background: linear-gradient(135deg, rgba(120, 53, 15, 0.34), rgba(69, 26, 3, 0.12));
    }
    .system-entry.safety-warning .badge {
      color: #fef3c7;
      border-color: rgba(251, 191, 36, 0.72);
      background: rgba(180, 83, 9, 0.76);
    }
    .system-entry.safety-warning .content strong {
      color: #fde68a;
      font-weight: 800;
    }
    .system-entry.safety-clear .badge {
      color: #67e8a5;
      border-color: rgba(52, 211, 153, 0.4);
      background: rgba(16, 185, 129, 0.12);
    }
    .system-entry.warning .badge {
      color: #ffd166;
      border-color: rgba(255, 209, 102, 0.35);
      background: rgba(255, 209, 102, 0.12);
    }
    .system-entry img {
      display: block;
      width: min(360px, 100%);
      margin-top: 12px;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.12);
      background: #000;
    }
    @media (max-width: 860px) {
      .layout { flex-direction: column; }
      .right-col { width: 100%; }
      .cam-picker { min-width: 0; width: 100%; }
      .camera-grid { --grid-columns: 2 !important; }
      .camera-grid[data-count="1"] { --grid-columns: 1 !important; }
      .system-overview { grid-template-columns: 1fr; }
      .system-entry { grid-template-columns: 1fr; gap: 8px; }
    }
    @media (max-width: 720px) {
      .camera-grid { --grid-columns: 1 !important; }
      .camera-tile-media {
        height: auto;
        aspect-ratio: 16 / 9;
      }
    }
    @media (max-width: 620px) {
      body { padding: 16px; }
      .view-tabs { flex-wrap: wrap; }
      .alarm-audio-button { width: 100%; margin-left: 0; justify-content: center; }
      .global-safety-banner { align-items: stretch; flex-direction: column; }
      .global-safety-action { width: 100%; }
      .global-safety-warning-banner { align-items: stretch; flex-direction: column; }
      .global-safety-warning-action { width: 100%; }
    }
  </style>
</head>
<body>
  <header class="page-header">
    <div class="title-block">
      <h1>Operations center</h1>
      <p class="sub" id="cam-subtitle">Select a camera to view its feed.</p>
    </div>
    <div class="cam-picker" id="cam-picker">
      <span class="cam-picker-label" id="cam-picker-label">Camera view</span>
      <button type="button" class="cam-picker-button" id="cam-picker-button" aria-haspopup="listbox" aria-expanded="false" aria-labelledby="cam-picker-label cam-button-name">
        <span class="cam-label">
          <span class="dot" id="cam-button-dot"></span>
          <span class="cam-name" id="cam-button-name">Loading…</span>
          <span class="cam-meta" id="cam-button-meta"></span>
        </span>
        <svg class="chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
      </button>
      <div class="cam-picker-menu" id="cam-picker-menu" role="listbox" aria-labelledby="cam-picker-label" aria-hidden="true"></div>
    </div>
  </header>
  <nav class="view-tabs" aria-label="Dashboard views">
    <button type="button" class="view-tab active" id="view-tab-live" data-view="live">
      <span class="view-tab-dot"></span>
      <span>Live Feed</span>
    </button>
    <button type="button" class="view-tab" id="view-tab-system" data-view="system">
      <span class="view-tab-dot"></span>
      <span>System Logs</span>
    </button>
    <button type="button" class="alarm-audio-button" id="alarm-audio-button" aria-pressed="false" title="Enable dashboard alert sounds">
      <span class="dot"></span>
      <span id="alarm-audio-label">Enable alert sound</span>
    </button>
  </nav>
  <div class="global-safety-banner" id="global-safety-banner" role="alert" aria-live="assertive" hidden>
    <div>
      <div class="global-safety-title">STOP WORK — Safety hazard active</div>
      <div class="global-safety-detail" id="global-safety-detail">Operator clearance required.</div>
    </div>
    <button type="button" class="global-safety-action" id="global-safety-action">View safety details</button>
  </div>
  <div class="global-safety-warning-banner" id="global-safety-warning-banner" role="status" aria-live="polite" hidden>
    <div>
      <div class="global-safety-warning-title">Obstacle warning</div>
      <div class="global-safety-warning-detail" id="global-safety-warning-detail">A machine-obstacle proximity risk was detected.</div>
    </div>
    <button type="button" class="global-safety-warning-action" id="global-safety-warning-action">View details</button>
  </div>
  <section class="tab-view active" id="live-view">
  <div class="layout" id="single-camera-layout">
    <div class="stream-box">
      <div class="stream-box-top">
        <span class="label" id="stream-title-label">Live feed</span>
        <div class="stream-status no-stream" id="stream-status">
          <span class="dot"></span>
          <span id="stream-status-text">No stream</span>
        </div>
      </div>
      <img id="stream-img" alt="Camera video stream" style="display:none;" />
      <div class="no-cam" id="no-cam-placeholder">No camera selected.</div>
    </div>
    <div class="right-col">
      <div class="panel">
        <div class="panel-header">Performance</div>
        <table class="perf-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th style="text-align: right;">Value</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Delay</td>
              <td style="text-align: right;"><span class="value none" id="delay">--</span><span class="unit">ms</span></td>
            </tr>
            <tr>
              <td>Jitter</td>
              <td style="text-align: right;"><span class="value none" id="jitter">--</span><span class="unit">ms</span></td>
            </tr>
            <tr>
              <td>FPS</td>
              <td style="text-align: right;"><span class="value none" id="fps">--</span></td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="panel">
        <div class="panel-header">
          <span>Event log</span>
          <button type="button" class="btn-clear" id="log-clear-btn">Clear</button>
        </div>
        <div class="log-box" id="log-box">
          <div class="empty">No events yet.</div>
        </div>
      </div>
    </div>
  </div>
  <div class="camera-grid" id="camera-grid" role="region" aria-label="All camera feeds" hidden></div>
  </section>
  <section class="tab-view" id="system-view">
    <div class="system-view-shell">
      <div class="system-overview" id="system-overview">
        <div class="system-stat">
          <div class="label">Operational mode</div>
          <div class="value" id="system-mode-value">Free</div>
        </div>
        <div class="system-stat objective">
          <div class="label">Mode objective</div>
          <div class="value" id="system-objective-value">None</div>
        </div>
        <div class="system-stat">
          <div class="label">Mode status</div>
          <div class="value" id="system-status-value">No workflow</div>
        </div>
        <div class="system-stat safety-state">
          <div class="label">Construction safety</div>
          <div class="value safety-clear" id="system-safety-value">Clear for construction</div>
          <div class="safety-detail" id="system-safety-detail">No active safety hazards.</div>
        </div>
      </div>
      <div class="zone-state-grid" id="zone-state-grid" aria-label="Zone operational states"></div>
      <div class="system-log">
        <div class="system-log-header">
          <div class="system-log-title">System log</div>
          <button type="button" class="btn-clear" id="system-log-clear-btn">Clear</button>
        </div>
        <div class="system-log-list" id="system-log-list">
          <div class="system-log-empty">No system events yet.</div>
        </div>
      </div>
    </div>
  </section>
  <script>
    (function () {
      let cameras = [];
      let selectedId = null;
      let feedMode = 'single';
      let activeDashboardView = 'live';
      let camerasRequestInFlight = false;

      const picker = document.getElementById('cam-picker');
      const button = document.getElementById('cam-picker-button');
      const menu = document.getElementById('cam-picker-menu');
      const btnDot = document.getElementById('cam-button-dot');
      const btnName = document.getElementById('cam-button-name');
      const btnMeta = document.getElementById('cam-button-meta');
      const subtitle = document.getElementById('cam-subtitle');
      const streamTitleLabel = document.getElementById('stream-title-label');
      const streamImg = document.getElementById('stream-img');
      const noCam = document.getElementById('no-cam-placeholder');
      const statusEl = document.getElementById('stream-status');
      const statusText = document.getElementById('stream-status-text');
      const liveView = document.getElementById('live-view');
      const systemView = document.getElementById('system-view');
      const singleCameraLayout = document.getElementById('single-camera-layout');
      const cameraGrid = document.getElementById('camera-grid');
      const viewTabs = Array.from(document.querySelectorAll('.view-tab'));
      const systemModeValue = document.getElementById('system-mode-value');
      const systemObjectiveValue = document.getElementById('system-objective-value');
      const systemStatusValue = document.getElementById('system-status-value');
      const systemOverview = document.getElementById('system-overview');
      const systemSafetyValue = document.getElementById('system-safety-value');
      const systemSafetyDetail = document.getElementById('system-safety-detail');
      const zoneStateGrid = document.getElementById('zone-state-grid');
      const systemLogList = document.getElementById('system-log-list');
      const globalSafetyBanner = document.getElementById('global-safety-banner');
      const globalSafetyDetail = document.getElementById('global-safety-detail');
      const globalSafetyWarningBanner = document.getElementById('global-safety-warning-banner');
      const globalSafetyWarningDetail = document.getElementById('global-safety-warning-detail');
      const alarmAudioButton = document.getElementById('alarm-audio-button');
      const alarmAudioLabel = document.getElementById('alarm-audio-label');
      const dashboardStartedAt = Date.now() / 1000;
      let systemLogInitialized = false;
      let lastAlertEventId = 0;
      let lastWarningEventId = 0;
      let lastSoundedAlertEventId = 0;
      let pendingAlarmEventId = 0;
      let alarmContext = null;
      let safetyHazardActive = false;
      let warningBannerTimer = null;

      function setView(view) {
        const showSystem = view === 'system';
        activeDashboardView = showSystem ? 'system' : 'live';
        liveView.classList.toggle('active', !showSystem);
        systemView.classList.toggle('active', showSystem);
        viewTabs.forEach(tab => tab.classList.toggle('active', tab.dataset.view === view));
        syncStreamConnections();
      }

      function hideSafetyWarning() {
        globalSafetyWarningBanner.hidden = true;
        if (warningBannerTimer !== null) {
          clearTimeout(warningBannerTimer);
          warningBannerTimer = null;
        }
      }

      function showSafetyWarning(entry) {
        if (!entry || safetyHazardActive) return;
        const location = [entry.zone, entry.camera_id != null ? 'Camera ' + entry.camera_id : '']
          .filter(Boolean)
          .join(' · ');
        globalSafetyWarningDetail.textContent = (entry.cause || entry.message || 'Machine-obstacle proximity risk detected.')
          + (location ? ' — ' + location : '');
        globalSafetyWarningBanner.hidden = false;
        if (warningBannerTimer !== null) clearTimeout(warningBannerTimer);
        warningBannerTimer = setTimeout(hideSafetyWarning, 12000);
      }

      viewTabs.forEach(tab => {
        tab.addEventListener('click', () => setView(tab.dataset.view));
      });

      function updateAlarmAudioStatus() {
        const armed = Boolean(alarmContext && alarmContext.state === 'running');
        const pending = pendingAlarmEventId > lastSoundedAlertEventId;
        alarmAudioButton.classList.toggle('armed', armed);
        alarmAudioButton.classList.toggle('pending', pending && !armed);
        alarmAudioButton.setAttribute('aria-pressed', armed ? 'true' : 'false');
        alarmAudioLabel.textContent = armed
          ? 'Alert sound on'
          : (pending ? 'Enable sound — alert pending' : 'Enable alert sound');
      }

      function playPendingAlarm() {
        if (pendingAlarmEventId <= lastSoundedAlertEventId) return true;
        if (!alarmContext || alarmContext.state !== 'running') {
          updateAlarmAudioStatus();
          return false;
        }
        const eventId = pendingAlarmEventId;
        const now = alarmContext.currentTime;
        [0, 0.22, 0.44].forEach((offset) => {
          const osc = alarmContext.createOscillator();
          const gain = alarmContext.createGain();
          osc.type = 'square';
          osc.frequency.setValueAtTime(880, now + offset);
          gain.gain.setValueAtTime(0.0001, now + offset);
          gain.gain.exponentialRampToValueAtTime(0.13, now + offset + 0.02);
          gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.16);
          osc.connect(gain);
          gain.connect(alarmContext.destination);
          osc.start(now + offset);
          osc.stop(now + offset + 0.18);
        });
        lastSoundedAlertEventId = eventId;
        if (pendingAlarmEventId <= lastSoundedAlertEventId) pendingAlarmEventId = 0;
        updateAlarmAudioStatus();
        return true;
      }

      async function prepareAlarmAudio() {
        const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCtor) {
          alarmAudioLabel.textContent = 'Alert sound unavailable';
          alarmAudioButton.disabled = true;
          return false;
        }
        if (!alarmContext || alarmContext.state === 'closed') {
          alarmContext = new AudioContextCtor();
          alarmContext.addEventListener('statechange', updateAlarmAudioStatus);
        }
        try {
          if (alarmContext.state !== 'running') await alarmContext.resume();
        } catch (_) {
          updateAlarmAudioStatus();
          return false;
        }
        updateAlarmAudioStatus();
        if (alarmContext.state === 'running') playPendingAlarm();
        return alarmContext.state === 'running';
      }

      function unlockAlarmAudio() {
        void prepareAlarmAudio();
      }

      document.addEventListener('pointerdown', unlockAlarmAudio, { passive: true });
      document.addEventListener('keydown', (event) => {
        if (!event.repeat) unlockAlarmAudio();
      });
      alarmAudioButton.addEventListener('click', unlockAlarmAudio);
      document.getElementById('global-safety-action').addEventListener('click', () => setView('system'));
      document.getElementById('global-safety-warning-action').addEventListener('click', () => setView('system'));

      function parseUrlSelection() {
        if (window.location.hash === '#view=grid' || window.location.hash === '#grid') {
          return { mode: 'grid', cameraId: null };
        }
        const m = window.location.hash.match(/#cam=(-?\d+)/);
        if (m) return { mode: 'single', cameraId: parseInt(m[1], 10) };
        const params = new URLSearchParams(window.location.search);
        if (params.has('camera_id')) {
          const v = parseInt(params.get('camera_id'), 10);
          if (!Number.isNaN(v)) return { mode: 'single', cameraId: v };
        }
        return null;
      }

      function setUrlSelection(mode, id) {
        const newHash = mode === 'grid' ? '#view=grid' : '#cam=' + id;
        if (window.location.hash !== newHash) {
          history.replaceState(null, '', window.location.pathname + window.location.search + newHash);
        }
      }

      function findCamera(id) {
        return cameras.find(c => c.camera_id === id) || null;
      }

      function renderButton() {
        if (feedMode === 'grid') {
          const liveCount = cameras.filter(cam => cam.stream_active).length;
          btnDot.className = 'dot' + (liveCount ? ' live' : '');
          btnName.textContent = 'Grid View';
          btnMeta.textContent = cameras.length ? liveCount + ' / ' + cameras.length + ' live' : '';
          subtitle.textContent = cameras.length
            ? 'All camera feeds · ' + liveCount + ' of ' + cameras.length + ' live'
            : 'No cameras registered yet.';
          return;
        }
        const cam = findCamera(selectedId);
        if (!cam) {
          btnDot.className = 'dot';
          btnName.textContent = cameras.length ? 'Choose a camera…' : 'No cameras';
          btnMeta.textContent = '';
          subtitle.textContent = 'Select a camera to view its feed.';
          return;
        }
        btnDot.className = 'dot' + (cam.stream_active ? ' live' : '');
        btnName.textContent = cam.name || ('Camera ' + cam.camera_id);
        btnMeta.textContent = '#' + cam.camera_id;
        const parts = [];
        if (cam.zone) parts.push(cam.zone);
        if (cam.location) parts.push(cam.location);
        if (cam.pi_host) parts.push('Pi ' + cam.pi_host);
        subtitle.textContent = parts.length ? parts.join(' · ') : ('Camera ' + cam.camera_id);
        streamTitleLabel.textContent = (cam.name || ('Camera ' + cam.camera_id)) + ' · Live feed';
      }

      function renderMenu() {
        const liveCount = cameras.filter(cam => cam.stream_active).length;
        const gridSelected = feedMode === 'grid';
        const gridOption = (
          '<div class="cam-option grid-option' + (gridSelected ? ' selected' : '') + '" role="option" tabindex="-1" aria-selected="' + gridSelected + '" data-view="grid">' +
            '<span class="grid-option-icon" aria-hidden="true"><span></span><span></span><span></span><span></span></span>' +
            '<div class="cam-text">' +
              '<div class="line1">Grid View</div>' +
              '<div class="line2">Watch all camera feeds</div>' +
            '</div>' +
            '<span class="badge' + (liveCount ? ' live' : '') + '">' + (cameras.length ? liveCount + ' / ' + cameras.length + ' live' : 'Empty') + '</span>' +
          '</div>'
        );
        const cameraOptions = cameras.map(cam => {
          const live = cam.stream_active;
          const selected = feedMode === 'single' && cam.camera_id === selectedId;
          const line2Parts = [];
          if (cam.zone) line2Parts.push(cam.zone);
          if (cam.location) line2Parts.push(cam.location);
          if (cam.pi_host) line2Parts.push('Pi ' + cam.pi_host);
          if (!cam.registered) line2Parts.push('unregistered');
          return (
            '<div class="cam-option' + (selected ? ' selected' : '') + '" role="option" tabindex="-1" aria-selected="' + selected + '" data-cid="' + cam.camera_id + '">' +
              '<span class="dot' + (live ? ' live' : '') + '" aria-hidden="true"></span>' +
              '<div class="cam-text">' +
                '<div class="line1">' + escapeHtml(cam.name || ('Camera ' + cam.camera_id)) + ' <span style="color:var(--text-muted);font-weight:400;">#' + cam.camera_id + '</span></div>' +
                (line2Parts.length ? '<div class="line2">' + escapeHtml(line2Parts.join(' · ')) + '</div>' : '') +
              '</div>' +
              '<span class="badge' + (live ? ' live' : '') + '">' + (live ? 'Live' : 'Idle') + '</span>' +
            '</div>'
          );
        }).join('');
        menu.innerHTML = gridOption + (cameraOptions || '<div class="empty">No cameras registered yet.</div>');
        const gridEl = menu.querySelector('[data-view="grid"]');
        if (gridEl) {
          gridEl.addEventListener('click', () => {
            closeMenu(true);
            selectGridView();
          });
        }
        menu.querySelectorAll('.cam-option[data-cid]').forEach(el => {
          el.addEventListener('click', () => {
            const cid = parseInt(el.getAttribute('data-cid'), 10);
            closeMenu(true);
            selectCamera(cid);
          });
        });
      }

      function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, c => ({
          '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
        })[c]);
      }

      function openMenu(focusOption) {
        picker.classList.add('open');
        button.setAttribute('aria-expanded', 'true');
        menu.setAttribute('aria-hidden', 'false');
        if (focusOption) {
          const target = menu.querySelector('[aria-selected="true"]') || menu.querySelector('.cam-option');
          if (target) target.focus();
        }
      }
      function closeMenu(returnFocus) {
        picker.classList.remove('open');
        button.setAttribute('aria-expanded', 'false');
        menu.setAttribute('aria-hidden', 'true');
        if (returnFocus) button.focus();
      }
      function toggleMenu() { picker.classList.contains('open') ? closeMenu() : openMenu(); }

      button.addEventListener('click', (e) => { e.stopPropagation(); toggleMenu(); });
      button.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          e.preventDefault();
          openMenu(true);
        } else if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          picker.classList.contains('open') ? closeMenu(true) : openMenu(true);
        }
      });
      menu.addEventListener('keydown', (e) => {
        const options = Array.from(menu.querySelectorAll('.cam-option'));
        const currentIndex = options.indexOf(document.activeElement);
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Home' || e.key === 'End') {
          e.preventDefault();
          let nextIndex = currentIndex;
          if (e.key === 'Home') nextIndex = 0;
          else if (e.key === 'End') nextIndex = options.length - 1;
          else if (e.key === 'ArrowDown') nextIndex = Math.min(currentIndex + 1, options.length - 1);
          else nextIndex = Math.max(currentIndex - 1, 0);
          if (options[nextIndex]) options[nextIndex].focus();
        } else if ((e.key === 'Enter' || e.key === ' ') && document.activeElement.classList.contains('cam-option')) {
          e.preventDefault();
          document.activeElement.click();
        } else if (e.key === 'Escape') {
          e.preventDefault();
          closeMenu(true);
        }
      });
      document.addEventListener('click', (e) => {
        if (!picker.contains(e.target)) closeMenu();
      });
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeMenu();
      });

      function disconnectImage(img) {
        img.removeAttribute('src');
        img.dataset.connected = 'false';
      }

      function disconnectSingleStream() {
        disconnectImage(streamImg);
        delete streamImg.dataset.cameraId;
      }

      function disconnectGridStreams() {
        cameraGrid.querySelectorAll('.camera-tile-media img').forEach(disconnectImage);
      }

      function streamsCanPlay() {
        return activeDashboardView === 'live' && !document.hidden;
      }

      function syncSingleStream(forceReconnect) {
        const cam = findCamera(selectedId);
        if (feedMode !== 'single' || selectedId == null) {
          disconnectSingleStream();
          return;
        }
        if (!cam || !cam.stream_active) {
          disconnectSingleStream();
          streamImg.style.display = 'none';
          noCam.textContent = 'No stream';
          noCam.style.display = 'flex';
          return;
        }
        if (!streamsCanPlay()) {
          disconnectSingleStream();
          return;
        }
        const cameraKey = String(selectedId);
        streamImg.style.display = '';
        noCam.style.display = 'none';
        if (forceReconnect || streamImg.dataset.cameraId !== cameraKey || !streamImg.getAttribute('src')) {
          disconnectSingleStream();
          streamImg.dataset.cameraId = cameraKey;
          streamImg.dataset.connected = 'true';
          streamImg.src = '/stream/' + selectedId + '?t=' + Date.now();
        }
      }

      function createGridTile(cameraId) {
        const tile = document.createElement('article');
        tile.className = 'camera-tile';
        tile.dataset.cameraId = String(cameraId);
        tile.innerHTML = (
          '<div class="camera-tile-header">' +
            '<div class="camera-tile-heading">' +
              '<div class="camera-tile-name"></div>' +
              '<div class="camera-tile-meta"></div>' +
            '</div>' +
            '<div class="camera-tile-status"><span class="dot" aria-hidden="true"></span><span class="text">No stream</span></div>' +
          '</div>' +
          '<div class="camera-tile-media">' +
            '<img alt="" />' +
            '<div class="camera-tile-placeholder">No stream</div>' +
          '</div>'
        );
        const img = tile.querySelector('img');
        img.dataset.connected = 'false';
        img.addEventListener('error', () => {
          disconnectImage(img);
          img.dataset.retryAfter = String(Date.now() + 3000);
          tile.classList.add('stream-error');
          tile.querySelector('.camera-tile-status .text').textContent = 'Reconnecting';
          tile.querySelector('.camera-tile-placeholder').textContent = 'Reconnecting…';
        });
        return tile;
      }

      function updateGridTile(tile, cam) {
        const name = cam.name || ('Camera ' + cam.camera_id);
        const meta = [];
        if (cam.zone) meta.push(cam.zone);
        if (cam.location) meta.push(cam.location);
        if (cam.pi_host) meta.push('Pi ' + cam.pi_host);
        if (!cam.registered) meta.push('unregistered');
        tile.querySelector('.camera-tile-name').textContent = name + '  #' + cam.camera_id;
        tile.querySelector('.camera-tile-meta').textContent = meta.join(' · ') || 'Camera ' + cam.camera_id;
        const active = Boolean(cam.stream_active);
        tile.classList.toggle('live', active);
        const img = tile.querySelector('img');
        img.alt = 'Live stream from ' + name + (cam.location ? ' at ' + cam.location : '');
        const shouldConnect = active && feedMode === 'grid' && streamsCanPlay();
        const retryAfter = Number(img.dataset.retryAfter || 0);
        const waitingToRetry = shouldConnect && retryAfter > Date.now();
        tile.classList.toggle('stream-error', waitingToRetry);
        tile.querySelector('.camera-tile-status .text').textContent = waitingToRetry ? 'Reconnecting' : (active ? 'Live' : 'No stream');
        tile.querySelector('.camera-tile-placeholder').textContent = waitingToRetry ? 'Reconnecting…' : 'No stream';
        if (shouldConnect && !waitingToRetry && (img.dataset.connected !== 'true' || !img.getAttribute('src'))) {
          delete img.dataset.retryAfter;
          img.dataset.connected = 'true';
          img.src = '/stream/' + cam.camera_id + '?t=' + Date.now();
        } else if (!shouldConnect) {
          delete img.dataset.retryAfter;
          tile.classList.remove('stream-error');
          if (img.dataset.connected === 'true' || img.getAttribute('src')) disconnectImage(img);
        }
      }

      function renderGrid() {
        const columns = cameras.length <= 1 ? 1 : (cameras.length <= 4 ? 2 : 3);
        cameraGrid.dataset.count = String(cameras.length);
        cameraGrid.style.setProperty('--grid-columns', String(columns));
        const existing = new Map();
        cameraGrid.querySelectorAll('.camera-tile').forEach(tile => {
          existing.set(Number(tile.dataset.cameraId), tile);
        });
        const empty = cameraGrid.querySelector('.camera-grid-empty');
        if (empty) empty.remove();

        if (!cameras.length) {
          disconnectGridStreams();
          existing.forEach(tile => tile.remove());
          const message = document.createElement('div');
          message.className = 'camera-grid-empty';
          message.textContent = 'No cameras registered yet.';
          cameraGrid.appendChild(message);
          return;
        }

        cameras.forEach((cam, index) => {
          let tile = existing.get(cam.camera_id);
          if (!tile) tile = createGridTile(cam.camera_id);
          updateGridTile(tile, cam);
          const currentTileAtIndex = cameraGrid.children[index];
          if (currentTileAtIndex !== tile) cameraGrid.insertBefore(tile, currentTileAtIndex || null);
          existing.delete(cam.camera_id);
        });
        existing.forEach(tile => {
          const img = tile.querySelector('img');
          if (img) disconnectImage(img);
          tile.remove();
        });
      }

      function syncStreamConnections(forceSingleReconnect) {
        if (feedMode === 'grid') {
          disconnectSingleStream();
          renderGrid();
        } else {
          disconnectGridStreams();
          syncSingleStream(Boolean(forceSingleReconnect));
        }
      }

      function selectGridView() {
        feedMode = 'grid';
        setUrlSelection('grid', null);
        singleCameraLayout.hidden = true;
        cameraGrid.hidden = false;
        renderButton();
        renderMenu();
        if (activeDashboardView !== 'live') setView('live');
        else syncStreamConnections();
      }

      function selectCamera(id) {
        if (id == null) return;
        if (feedMode === 'single' && id === selectedId) {
          renderButton();
          renderMenu();
          if (activeDashboardView !== 'live') setView('live');
          else syncStreamConnections();
          return;
        }
        feedMode = 'single';
        selectedId = id;
        setUrlSelection('single', id);
        singleCameraLayout.hidden = false;
        cameraGrid.hidden = true;
        // Reset UI values
        document.getElementById('delay').textContent = '--';
        document.getElementById('delay').className = 'value none';
        document.getElementById('jitter').textContent = '--';
        document.getElementById('jitter').className = 'value none';
        document.getElementById('fps').textContent = '--';
        document.getElementById('fps').className = 'value none';
        document.getElementById('log-box').innerHTML = '<div class="empty">Loading…</div>';
        statusEl.className = 'stream-status no-stream';
        statusText.textContent = 'No stream';
        renderButton();
        renderMenu();
        if (activeDashboardView !== 'live') setView('live');
        else syncStreamConnections(true);
        refreshMetrics();
        refreshLog();
      }

      async function loadCameras() {
        if (camerasRequestInFlight) return;
        camerasRequestInFlight = true;
        try {
          const res = await fetch('/cameras');
          if (!res.ok) throw new Error('Unable to load cameras');
          const data = await res.json();
          cameras = Array.isArray(data.cameras) ? data.cameras : [];
        } catch (e) {
          const urlSelection = parseUrlSelection();
          if (selectedId == null && feedMode === 'single' && urlSelection && urlSelection.mode === 'grid') {
            selectGridView();
          } else {
            renderButton();
            renderMenu();
            if (feedMode === 'grid') renderGrid();
          }
          return;
        } finally {
          camerasRequestInFlight = false;
        }
        if (selectedId == null && feedMode === 'single') {
          const urlSelection = parseUrlSelection();
          if (urlSelection && urlSelection.mode === 'grid') {
            selectGridView();
          } else if (urlSelection && findCamera(urlSelection.cameraId)) {
            selectCamera(urlSelection.cameraId);
          } else if (cameras.length) {
            const live = cameras.find(c => c.stream_active);
            selectCamera((live || cameras[0]).camera_id);
          } else {
            streamImg.style.display = 'none';
            noCam.textContent = 'No camera selected.';
            noCam.style.display = 'flex';
            renderButton();
            renderMenu();
          }
        } else if (feedMode === 'single' && selectedId != null && !findCamera(selectedId)) {
          if (cameras.length) {
            const live = cameras.find(c => c.stream_active);
            selectCamera((live || cameras[0]).camera_id);
          } else {
            selectedId = null;
            disconnectSingleStream();
            streamImg.style.display = 'none';
            noCam.textContent = 'No camera selected.';
            noCam.style.display = 'flex';
            renderButton();
            renderMenu();
          }
        } else {
          renderButton();
          if (!picker.classList.contains('open')) renderMenu();
          syncStreamConnections();
        }
      }

      function refreshMetrics() {
        if (feedMode !== 'single' || selectedId == null) return;
        const cid = selectedId;
        fetch('/metrics?camera_id=' + cid)
          .then(r => r.json())
          .then(d => {
            if (feedMode !== 'single' || cid !== selectedId) return;
            const delay = document.getElementById('delay');
            delay.textContent = '--';
            delay.className = 'value none';
            const jitter = document.getElementById('jitter');
            jitter.textContent = d.jitter_ms != null ? d.jitter_ms : '--';
            jitter.className = 'value' + (d.jitter_ms == null ? ' none' : '');
            const fps = document.getElementById('fps');
            fps.textContent = d.fps != null ? d.fps : '--';
            fps.className = 'value' + (d.fps == null ? ' none' : '');
            if (d.stream_active) {
              statusEl.className = 'stream-status live';
              statusText.textContent = 'Live';
            } else {
              statusEl.className = 'stream-status no-stream';
              statusText.textContent = 'No stream';
            }
            const cam = findCamera(cid);
            if (cam && cam.stream_active !== d.stream_active) {
              cam.stream_active = d.stream_active;
              renderButton();
              if (!picker.classList.contains('open')) renderMenu();
              syncSingleStream();
            }
          })
          .catch(() => {});
      }

      function formatLogTime(ts) {
        const d = new Date(ts * 1000);
        return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
      }

      function refreshLog() {
        if (feedMode !== 'single' || selectedId == null) return;
        const cid = selectedId;
        fetch('/log?camera_id=' + cid)
          .then(r => r.json())
          .then(entries => {
            if (feedMode !== 'single' || cid !== selectedId) return;
            const box = document.getElementById('log-box');
            if (!entries.length) {
              box.innerHTML = '<div class="empty">No events yet.</div>';
              return;
            }
            box.innerHTML = entries.map(e => {
              let cls = '';
              if (e.message.indexOf('started') !== -1) cls = 'stream-started';
              else if (e.message.indexOf('stopped') !== -1) cls = 'stream-stopped';
              return '<div class="log-entry ' + cls + '"><span class="ts">' + formatLogTime(e.time) + '</span><span class="msg">' + escapeHtml(e.message) + '</span></div>';
            }).join('');
            box.scrollTop = box.scrollHeight;
          })
          .catch(() => {});
      }

      function renderSystemEntry(entry) {
        const level = entry.level || 'info';
        const isSearchAlert = entry.kind === 'alert';
        const isSafetyAlert = entry.kind === 'safety_alert';
        const isSafetyWarning = entry.kind === 'safety_warning';
        const isSafetyClear = entry.kind === 'safety_clear';
        const rowClass = 'system-entry' + (
          isSafetyAlert ? ' safety-alert' : (
            isSafetyWarning ? ' safety-warning' : (
              isSearchAlert ? ' alert' : (
                isSafetyClear ? ' safety-clear' : (level === 'warning' ? ' warning' : '')
              )
            )
          )
        );
        const label = isSafetyAlert
          ? 'Stop work'
          : (isSafetyWarning ? 'Obstacle warning' : (isSearchAlert ? 'Match' : (isSafetyClear ? 'Clear' : (level === 'warning' ? 'Warning' : (entry.kind || 'System')))));
        const body = (isSearchAlert || isSafetyAlert || isSafetyWarning)
          ? '<strong>' + escapeHtml(entry.message || '') + '</strong>'
          : escapeHtml(entry.message || '');
        const frameAlt = isSafetyAlert
          ? 'Safety hazard frame'
          : (isSafetyWarning ? 'Safety warning frame' : 'Search match frame');
        const frame = entry.frame_url
          ? '<img src="' + escapeHtml(entry.frame_url) + '" alt="' + frameAlt + '" loading="lazy">'
          : '';
        return (
          '<div class="' + rowClass + '">' +
            '<div class="time">' + formatLogTime(entry.time) + '</div>' +
            '<div><span class="badge">' + escapeHtml(label) + '</span></div>' +
            '<div class="content">' + body + frame + '</div>' +
          '</div>'
        );
      }

      function refreshSystemState() {
        fetch('/system/state')
          .then(r => r.json())
          .then(state => {
            const knownModes = ['free', 'safety', 'search', 'investigation', 'mixed'];
            const mode = knownModes.includes(state.mode) ? state.mode : 'free';
            systemModeValue.textContent = mode.charAt(0).toUpperCase() + mode.slice(1);
            systemModeValue.className = 'value' + (mode === 'search' || mode === 'mixed' ? ' search' : '');
            systemObjectiveValue.textContent = mode === 'mixed' ? 'See zones below' : (state.objective || 'None');
            systemStatusValue.textContent = mode === 'mixed'
              ? 'Independent zone operation'
              : (mode === 'search' || mode === 'safety')
              ? (state.scanner_running ? 'Scanning' : 'Scanner stopped')
              : (mode === 'free' ? 'No workflow' : 'Placeholder');

            const safetyHazard = state.safety_status === 'hazard';
            safetyHazardActive = safetyHazard;
            if (safetyHazard) hideSafetyWarning();
            const activeHazards = Array.isArray(state.active_safety_hazards)
              ? state.active_safety_hazards
              : [];
            systemOverview.classList.toggle('hazard', safetyHazard);
            systemSafetyValue.textContent = safetyHazard ? 'STOP WORK — Hazard active' : 'Clear for construction';
            systemSafetyValue.className = 'value ' + (safetyHazard ? 'safety-hazard' : 'safety-clear');
            globalSafetyBanner.hidden = !safetyHazard;
            if (safetyHazard && activeHazards.length) {
              const names = [...new Set(activeHazards.map(item => item.hazard_name).filter(Boolean))];
              const detail = names.join(' · ') || 'Operator clearance required.';
              systemSafetyDetail.textContent = detail;
              globalSafetyDetail.textContent = detail + ' — Operator clearance required.';
            } else {
              systemSafetyDetail.textContent = safetyHazard
                ? 'Operator clearance required.'
                : 'No active safety hazards.';
              globalSafetyDetail.textContent = 'Operator clearance required.';
            }
            document.title = safetyHazard ? 'STOP WORK — Operations center' : 'Camera stream';

            const zoneStates = Array.isArray(state.zones) ? state.zones : [];
            const hazardZones = new Set(activeHazards.map(item => item.zone).filter(Boolean));
            zoneStateGrid.innerHTML = zoneStates.map(zoneState => {
              const zoneMode = ['free', 'safety', 'search', 'investigation'].includes(zoneState.mode)
                ? zoneState.mode
                : 'free';
              const isHazard = hazardZones.has(zoneState.zone);
              const cameraIds = Array.isArray(zoneState.camera_ids) ? zoneState.camera_ids : [];
              const detail = isHazard
                ? 'STOP WORK — Operator clearance required.'
                : (zoneMode === 'search'
                    ? (zoneState.objective || 'Search objective not set')
                    : (zoneMode === 'safety'
                        ? (zoneState.scanner_running ? 'Monitoring active safety hazards' : 'Safety scanner stopped')
                        : (zoneMode === 'free' ? 'Live view with no automated workflow' : 'Placeholder mode')));
              return (
                '<div class="zone-state-card ' + escapeHtml(zoneMode) + (isHazard ? ' hazard' : '') + '">' +
                  '<div class="zone-state-heading">' +
                    '<span class="zone-state-name">' + escapeHtml(zoneState.zone || 'Zone') + '</span>' +
                    '<span class="zone-state-cameras">Cameras ' + escapeHtml(cameraIds.join(', ') || 'none') + '</span>' +
                  '</div>' +
                  '<div class="zone-state-mode">' + escapeHtml(zoneMode.charAt(0).toUpperCase() + zoneMode.slice(1)) + ' Mode</div>' +
                  '<div class="zone-state-detail">' + escapeHtml(detail) + '</div>' +
                '</div>'
              );
            }).join('');
          })
          .catch(() => {});
      }

      function refreshSystemLog() {
        fetch('/system/log')
          .then(r => r.json())
          .then(entries => {
            const alerts = entries.filter(entry => (
              entry.kind === 'alert' || entry.kind === 'safety_alert'
            ) && entry.audible !== false);
            const safetyAlerts = entries.filter(entry => entry.kind === 'safety_alert');
            const warnings = entries.filter(entry => entry.kind === 'safety_warning');
            const newestAlertId = alerts.reduce((maxId, entry) => Math.max(maxId, entry.id || 0), 0);
            const newestSafetyAlertId = safetyAlerts.reduce((maxId, entry) => Math.max(maxId, entry.id || 0), 0);
            const newestWarningId = warnings.reduce((maxId, entry) => Math.max(maxId, entry.id || 0), 0);
            const alertArrivedAfterDashboardLoad = !systemLogInitialized && alerts.some(
              entry => Number(entry.time || 0) >= dashboardStartedAt
            );
            const safetyAlertArrivedAfterDashboardLoad = !systemLogInitialized && safetyAlerts.some(
              entry => Number(entry.time || 0) >= dashboardStartedAt
            );
            const warningArrivedAfterDashboardLoad = !systemLogInitialized && warnings.some(
              entry => Number(entry.time || 0) >= dashboardStartedAt
            );
            const safetyAlertIsNew = (
              systemLogInitialized && newestSafetyAlertId > lastAlertEventId
            ) || safetyAlertArrivedAfterDashboardLoad;
            const warningIsNew = (
              systemLogInitialized && newestWarningId > lastWarningEventId
            ) || warningArrivedAfterDashboardLoad;
            if ((systemLogInitialized && newestAlertId > lastAlertEventId) || alertArrivedAfterDashboardLoad) {
              pendingAlarmEventId = Math.max(pendingAlarmEventId, newestAlertId);
              playPendingAlarm();
            }
            if (safetyAlertIsNew) {
              hideSafetyWarning();
            } else if (warningIsNew && !safetyHazardActive) {
              showSafetyWarning(warnings[warnings.length - 1]);
            }
            lastAlertEventId = Math.max(lastAlertEventId, newestAlertId);
            lastWarningEventId = Math.max(lastWarningEventId, newestWarningId);
            systemLogInitialized = true;

            if (!entries.length) {
              systemLogList.innerHTML = '<div class="system-log-empty">No system events yet.</div>';
              return;
            }
            systemLogList.innerHTML = entries.map(renderSystemEntry).join('');
            systemLogList.scrollTop = systemLogList.scrollHeight;
          })
          .catch(() => {});
      }

      document.getElementById('log-clear-btn').addEventListener('click', () => {
        if (feedMode !== 'single' || selectedId == null) return;
        fetch('/log/clear?camera_id=' + selectedId, { method: 'POST' })
          .then(() => refreshLog())
          .catch(() => {});
      });

      document.getElementById('system-log-clear-btn').addEventListener('click', () => {
        fetch('/system/log/clear', { method: 'POST' })
          .then(() => {
            lastAlertEventId = 0;
            lastWarningEventId = 0;
            lastSoundedAlertEventId = 0;
            pendingAlarmEventId = 0;
            hideSafetyWarning();
            updateAlarmAudioStatus();
            systemLogInitialized = false;
            refreshSystemLog();
          })
          .catch(() => {});
      });

      window.addEventListener('hashchange', () => {
        const selection = parseUrlSelection();
        if (!selection) return;
        if (selection.mode === 'grid') {
          if (feedMode !== 'grid') selectGridView();
        } else if (findCamera(selection.cameraId) && (feedMode !== 'single' || selection.cameraId !== selectedId)) {
          selectCamera(selection.cameraId);
        }
      });
      document.addEventListener('visibilitychange', syncStreamConnections);
      window.addEventListener('pageshow', syncStreamConnections);
      window.addEventListener('pagehide', () => {
        disconnectSingleStream();
        disconnectGridStreams();
      });

      loadCameras();
      setInterval(loadCameras, 1000);
      setInterval(refreshMetrics, 500);
      setInterval(refreshLog, 1000);
      refreshSystemState();
      refreshSystemLog();
      setInterval(refreshSystemState, 1000);
      setInterval(refreshSystemLog, 1000);
    })();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)


if __name__ == "__main__":
    import uvicorn
    args = _get_args()
    print(f"Stream receiver: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
