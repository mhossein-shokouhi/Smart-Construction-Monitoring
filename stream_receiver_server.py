"""
Stream receiver server — run on your laptop.

Accepts JPEG frames POSTed from Raspberry Pis (raw_stream_demo.py,
object_detection_demo.py, or segmentation_demo_overlay.py) and serves a single
dashboard page that lets the user switch between cameras via a dropdown menu.

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
# Global system log and alert-frame cache for supervisor/emergency activity.
_system_log = []
_alert_frames = {}
_next_system_event_id = 1
_system_state = {
    "mode": "free",
    "emergency_intent": None,
    "scanner_running": False,
    "updated_at": None,
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
            "pi_host": cam.get("pi_host"),
        }
    return registry


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
        except (TypeError, ValueError):
            pass

    confidence = payload.get("confidence")
    if confidence is not None:
        try:
            entry["confidence"] = round(float(confidence), 3)
        except (TypeError, ValueError):
            pass

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


@app.get("/system/state")
async def get_system_state():
    with _lock:
        state = dict(_system_state)
    return JSONResponse(state)


@app.post("/system/state")
async def update_system_state(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "error": "Request body must be JSON."}, status_code=400)

    with _lock:
        _system_state["mode"] = str(payload.get("mode") or "free")
        _system_state["emergency_intent"] = payload.get("emergency_intent")
        _system_state["scanner_running"] = bool(payload.get("scanner_running"))
        _system_state["updated_at"] = time.time()
        state = dict(_system_state)
    return JSONResponse({"status": "ok", "state": state})


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
      transform: translateY(-4px);
      pointer-events: none;
      transition: opacity 0.15s ease, transform 0.15s ease;
    }
    .cam-picker.open .cam-picker-menu {
      opacity: 1;
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
    .cam-option.selected {
      background: rgba(0, 212, 170, 0.12);
      border: 1px solid rgba(0, 212, 170, 0.35);
      padding: 9px 11px;
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
      grid-template-columns: 180px 1fr 180px;
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
    .system-stat .value.emergency {
      color: #ff8d8d;
    }
    .system-stat.intent .value {
      font-size: 0.9375rem;
      font-weight: 500;
      line-height: 1.4;
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
      color: #ff8d8d;
      border-color: rgba(239, 68, 68, 0.45);
      background: rgba(239, 68, 68, 0.14);
    }
    .system-entry.alert .content strong {
      color: #fff;
      font-weight: 700;
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
      .system-overview { grid-template-columns: 1fr; }
      .system-entry { grid-template-columns: 1fr; gap: 8px; }
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
      <span class="cam-picker-label">Camera</span>
      <button type="button" class="cam-picker-button" id="cam-picker-button" aria-haspopup="listbox" aria-expanded="false">
        <span class="cam-label">
          <span class="dot" id="cam-button-dot"></span>
          <span class="cam-name" id="cam-button-name">Loading…</span>
          <span class="cam-meta" id="cam-button-meta"></span>
        </span>
        <svg class="chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
      </button>
      <div class="cam-picker-menu" id="cam-picker-menu" role="listbox"></div>
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
  </nav>
  <section class="tab-view active" id="live-view">
  <div class="layout">
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
  </section>
  <section class="tab-view" id="system-view">
    <div class="system-view-shell">
      <div class="system-overview">
        <div class="system-stat">
          <div class="label">Agentic mode</div>
          <div class="value" id="system-mode-value">Free</div>
        </div>
        <div class="system-stat intent">
          <div class="label">Emergency intent</div>
          <div class="value" id="system-intent-value">None</div>
        </div>
        <div class="system-stat">
          <div class="label">VLM scanner</div>
          <div class="value" id="system-scanner-value">Idle</div>
        </div>
      </div>
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
      const viewTabs = Array.from(document.querySelectorAll('.view-tab'));
      const systemModeValue = document.getElementById('system-mode-value');
      const systemIntentValue = document.getElementById('system-intent-value');
      const systemScannerValue = document.getElementById('system-scanner-value');
      const systemLogList = document.getElementById('system-log-list');
      let systemLogInitialized = false;
      let lastAlertEventId = 0;
      let alarmContext = null;

      function setView(view) {
        const showSystem = view === 'system';
        liveView.classList.toggle('active', !showSystem);
        systemView.classList.toggle('active', showSystem);
        viewTabs.forEach(tab => tab.classList.toggle('active', tab.dataset.view === view));
      }

      viewTabs.forEach(tab => {
        tab.addEventListener('click', () => setView(tab.dataset.view));
      });

      function prepareAlarmAudio() {
        const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCtor) return null;
        if (!alarmContext) alarmContext = new AudioContextCtor();
        if (alarmContext.state === 'suspended') alarmContext.resume().catch(() => {});
        return alarmContext;
      }

      document.addEventListener('pointerdown', prepareAlarmAudio, { once: true });

      function playAlarm() {
        const ctx = prepareAlarmAudio();
        if (!ctx) return;
        const now = ctx.currentTime;
        [0, 0.22, 0.44].forEach((offset) => {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.type = 'square';
          osc.frequency.setValueAtTime(880, now + offset);
          gain.gain.setValueAtTime(0.0001, now + offset);
          gain.gain.exponentialRampToValueAtTime(0.13, now + offset + 0.02);
          gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.16);
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start(now + offset);
          osc.stop(now + offset + 0.18);
        });
      }

      function parseUrlCamera() {
        const m = window.location.hash.match(/#cam=(-?\d+)/);
        if (m) return parseInt(m[1], 10);
        const params = new URLSearchParams(window.location.search);
        if (params.has('camera_id')) {
          const v = parseInt(params.get('camera_id'), 10);
          if (!Number.isNaN(v)) return v;
        }
        return null;
      }

      function setUrlCamera(id) {
        if (id == null) return;
        const newHash = '#cam=' + id;
        if (window.location.hash !== newHash) {
          history.replaceState(null, '', window.location.pathname + window.location.search + newHash);
        }
      }

      function findCamera(id) {
        return cameras.find(c => c.camera_id === id) || null;
      }

      function renderButton() {
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
        if (cam.location) parts.push(cam.location);
        if (cam.pi_host) parts.push('Pi ' + cam.pi_host);
        subtitle.textContent = parts.length ? parts.join(' · ') : ('Camera ' + cam.camera_id);
        streamTitleLabel.textContent = (cam.name || ('Camera ' + cam.camera_id)) + ' · Live feed';
      }

      function renderMenu() {
        if (!cameras.length) {
          menu.innerHTML = '<div class="empty">No cameras registered yet.</div>';
          return;
        }
        menu.innerHTML = cameras.map(cam => {
          const live = cam.stream_active;
          const selected = cam.camera_id === selectedId;
          const line2Parts = [];
          if (cam.location) line2Parts.push(cam.location);
          if (cam.pi_host) line2Parts.push('Pi ' + cam.pi_host);
          if (!cam.registered) line2Parts.push('unregistered');
          return (
            '<div class="cam-option' + (selected ? ' selected' : '') + '" role="option" data-cid="' + cam.camera_id + '">' +
              '<span class="dot' + (live ? ' live' : '') + '"></span>' +
              '<div class="cam-text">' +
                '<div class="line1">' + escapeHtml(cam.name || ('Camera ' + cam.camera_id)) + ' <span style="color:var(--text-muted);font-weight:400;">#' + cam.camera_id + '</span></div>' +
                (line2Parts.length ? '<div class="line2">' + escapeHtml(line2Parts.join(' · ')) + '</div>' : '') +
              '</div>' +
              '<span class="badge' + (live ? ' live' : '') + '">' + (live ? 'Live' : 'Idle') + '</span>' +
            '</div>'
          );
        }).join('');
        menu.querySelectorAll('.cam-option').forEach(el => {
          el.addEventListener('click', () => {
            const cid = parseInt(el.getAttribute('data-cid'), 10);
            selectCamera(cid);
            closeMenu();
          });
        });
      }

      function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, c => ({
          '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
        })[c]);
      }

      function openMenu() { picker.classList.add('open'); button.setAttribute('aria-expanded', 'true'); }
      function closeMenu() { picker.classList.remove('open'); button.setAttribute('aria-expanded', 'false'); }
      function toggleMenu() { picker.classList.contains('open') ? closeMenu() : openMenu(); }

      button.addEventListener('click', (e) => { e.stopPropagation(); toggleMenu(); });
      document.addEventListener('click', (e) => {
        if (!picker.contains(e.target)) closeMenu();
      });
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeMenu();
      });

      function selectCamera(id) {
        if (id == null || id === selectedId) {
          renderButton();
          renderMenu();
          return;
        }
        selectedId = id;
        setUrlCamera(id);
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
        // Point stream img at new camera (force reconnect)
        streamImg.style.display = '';
        noCam.style.display = 'none';
        streamImg.src = '/stream/' + id + '?t=' + Date.now();
        renderButton();
        renderMenu();
        refreshMetrics();
        refreshLog();
      }

      async function loadCameras(initial) {
        try {
          const res = await fetch('/cameras');
          const data = await res.json();
          cameras = Array.isArray(data.cameras) ? data.cameras : [];
        } catch (e) {
          cameras = [];
        }
        if (selectedId == null) {
          const urlCam = parseUrlCamera();
          if (urlCam != null && findCamera(urlCam)) {
            selectCamera(urlCam);
          } else if (cameras.length) {
            const live = cameras.find(c => c.stream_active);
            selectCamera((live || cameras[0]).camera_id);
          } else {
            streamImg.style.display = 'none';
            noCam.style.display = 'flex';
            renderButton();
            renderMenu();
          }
        } else {
          renderButton();
          renderMenu();
        }
      }

      function refreshMetrics() {
        if (selectedId == null) return;
        const cid = selectedId;
        fetch('/metrics?camera_id=' + cid)
          .then(r => r.json())
          .then(d => {
            if (cid !== selectedId) return;
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
              renderMenu();
            }
          })
          .catch(() => {});
      }

      function formatLogTime(ts) {
        const d = new Date(ts * 1000);
        return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
      }

      function refreshLog() {
        if (selectedId == null) return;
        const cid = selectedId;
        fetch('/log?camera_id=' + cid)
          .then(r => r.json())
          .then(entries => {
            if (cid !== selectedId) return;
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
        const isAlert = entry.kind === 'alert';
        const rowClass = 'system-entry' + (isAlert ? ' alert' : (level === 'warning' ? ' warning' : ''));
        const label = isAlert ? 'Alert' : (level === 'warning' ? 'Warning' : (entry.kind || 'System'));
        const body = isAlert
          ? '<strong>' + escapeHtml(entry.message || '') + '</strong>'
          : escapeHtml(entry.message || '');
        const frame = entry.frame_url
          ? '<img src="' + escapeHtml(entry.frame_url) + '" alt="Alert frame" loading="lazy">'
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
            const emergency = state.mode === 'emergency';
            systemModeValue.textContent = emergency ? 'Emergency' : 'Free';
            systemModeValue.className = 'value' + (emergency ? ' emergency' : '');
            systemIntentValue.textContent = state.emergency_intent || 'None';
            systemScannerValue.textContent = state.scanner_running ? 'Scanning' : 'Idle';
          })
          .catch(() => {});
      }

      function refreshSystemLog() {
        fetch('/system/log')
          .then(r => r.json())
          .then(entries => {
            const alerts = entries.filter(entry => entry.kind === 'alert');
            const newestAlertId = alerts.reduce((maxId, entry) => Math.max(maxId, entry.id || 0), 0);
            if (systemLogInitialized && newestAlertId > lastAlertEventId) playAlarm();
            lastAlertEventId = Math.max(lastAlertEventId, newestAlertId);
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
        if (selectedId == null) return;
        fetch('/log/clear?camera_id=' + selectedId, { method: 'POST' })
          .then(() => refreshLog())
          .catch(() => {});
      });

      document.getElementById('system-log-clear-btn').addEventListener('click', () => {
        fetch('/system/log/clear', { method: 'POST' })
          .then(() => {
            lastAlertEventId = 0;
            systemLogInitialized = false;
            refreshSystemLog();
          })
          .catch(() => {});
      });

      window.addEventListener('hashchange', () => {
        const id = parseUrlCamera();
        if (id != null && id !== selectedId) selectCamera(id);
      });

      loadCameras(true);
      setInterval(loadCameras, 3000);
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
