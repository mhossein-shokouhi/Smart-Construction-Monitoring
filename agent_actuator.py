# agent_actuator.py (runs on each Raspberry Pi)
#
# Each Pi hosts exactly one IMX500 camera. Its numeric identity is declared
# through the CAMERA_ID environment variable (default 0). The supervisor agent
# on the laptop routes commands to the correct Pi, but every request still
# carries a camera_id so mistakes can be detected instead of silently running
# the wrong camera.
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# This Pi's own camera id. Override with `CAMERA_ID=1 python agent_actuator.py`.
SELF_CAMERA_ID = int(os.environ.get("CAMERA_ID", "0"))
SELF_CAMERA_NAME = os.environ.get("CAMERA_NAME", f"Camera {SELF_CAMERA_ID}")

# URL of the laptop's stream_receiver_server. Children inherit this via env.
STREAM_SERVER_URL = os.environ.get("STREAM_SERVER_URL", "")

# Resolve all project paths relative to this file so the Pi can run
# `uvicorn agent_actuator:app` from any cwd without breaking subprocess launches.
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

# Track running processes per camera_id (normally only SELF_CAMERA_ID)
processes = {}   # {camera_id: Popen}
camera_modes = {}  # {camera_id: str}


class ModeRequest(BaseModel):
    camera_id: int
    mode: str  # "surveillance", "construction", "idle"


class StateRequest(BaseModel):
    camera_id: int


# Each value is a list of command parts (argv). Paths are resolved against
# SCRIPT_DIR so they work regardless of where uvicorn is launched from.
SCRIPTS = {
    "surveillance": [
        sys.executable,
        str(SCRIPT_DIR / "object_detection_demo.py"),
        "--model",
        str(MODELS_DIR / "imx500_network_nanodet_plus_416x416_pp.rpk"),
    ],
    "construction": [
        sys.executable,
        str(SCRIPT_DIR / "segmentation_demo_overlay.py"),
        "--model",
        str(MODELS_DIR / "imx500_network_deeplabv3plus.rpk"),
    ],
}


def _wrong_camera_response(camera_id: int) -> dict:
    return {
        "status": "error",
        "error": (
            f"This Pi serves camera_id={SELF_CAMERA_ID} ({SELF_CAMERA_NAME}); "
            f"received request for camera_id={camera_id}."
        ),
        "served_camera_id": SELF_CAMERA_ID,
        "requested_camera_id": camera_id,
    }


def stop_camera(camera_id: int) -> None:
    proc = processes.get(camera_id)
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    processes[camera_id] = None
    camera_modes[camera_id] = "idle"


def start_mode(camera_id: int, mode: str) -> None:
    stop_camera(camera_id)

    if mode == "idle":
        return

    cmd = SCRIPTS.get(mode)
    if not cmd:
        raise ValueError(f"Unknown mode: {mode}")

    env = os.environ.copy()
    env["STREAM_CAMERA_ID"] = str(camera_id)
    if STREAM_SERVER_URL:
        env["STREAM_SERVER_URL"] = STREAM_SERVER_URL

    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
        cwd=str(SCRIPT_DIR),
    )
    processes[camera_id] = proc
    camera_modes[camera_id] = mode


@app.post("/set_mode")
def set_mode(req: ModeRequest):
    if req.camera_id != SELF_CAMERA_ID:
        return _wrong_camera_response(req.camera_id)
    try:
        start_mode(req.camera_id, req.mode)
        return {
            "status": "ok",
            "camera_id": req.camera_id,
            "camera_name": SELF_CAMERA_NAME,
            "mode": req.mode,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "camera_id": req.camera_id}


@app.post("/get_state")
def get_state(req: StateRequest):
    """
    Return the current operating state of the camera served by this Pi.

    - mode: best-effort current mode ("surveillance", "construction", "idle", or "unknown")
    - last_requested_mode: last mode requested by the supervisor
    - process_running: whether an associated process is currently alive
    """
    if req.camera_id != SELF_CAMERA_ID:
        return _wrong_camera_response(req.camera_id)

    camera_id = req.camera_id
    proc = processes.get(camera_id)
    process_running = proc is not None and proc.poll() is None
    last_requested_mode = camera_modes.get(camera_id, "unknown")

    if not process_running and last_requested_mode in ("surveillance", "construction"):
        mode = "idle"
    else:
        mode = last_requested_mode

    return {
        "status": "ok",
        "camera_id": camera_id,
        "camera_name": SELF_CAMERA_NAME,
        "mode": mode,
        "last_requested_mode": last_requested_mode,
        "process_running": process_running,
    }


@app.get("/cameras")
def list_cameras():
    """Report which cameras this Pi is responsible for (always exactly one)."""
    return {
        "status": "ok",
        "cameras": [
            {"camera_id": SELF_CAMERA_ID, "name": SELF_CAMERA_NAME},
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "camera_id": SELF_CAMERA_ID,
        "camera_name": SELF_CAMERA_NAME,
    }
