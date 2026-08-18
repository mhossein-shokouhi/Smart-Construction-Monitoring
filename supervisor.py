import warnings
warnings.filterwarnings("ignore")

import html
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse, Response
import gradio as gr
import requests
from openai import OpenAI

from safety_vlm import SafetyScanner
from search_vlm import SearchScanner
from reporting import ConstructionReporting

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


CAMERAS_FILE = Path(__file__).with_name("cameras.json")
REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
REALTIME_VOICE = os.environ.get("OPENAI_REALTIME_VOICE", "marin")
REALTIME_CALLS_URL = "https://api.openai.com/v1/realtime/calls"
SUPERVISOR_MODEL = os.environ.get("OPENAI_SUPERVISOR_MODEL", "gpt-5-mini")
STREAM_RECEIVER_URL = os.environ.get("STREAM_RECEIVER_URL", "http://127.0.0.1:9000").rstrip("/")
SEARCH_VLM_MODEL = os.environ.get(
    "OPENAI_SEARCH_VLM_MODEL",
    os.environ.get("OPENAI_EMERGENCY_VLM_MODEL", "gpt-5.5"),
)
SEARCH_VLM_DETAIL = os.environ.get(
    "OPENAI_SEARCH_VLM_DETAIL",
    os.environ.get("OPENAI_EMERGENCY_VLM_DETAIL", "high"),
)
SEARCH_MATCH_THRESHOLD = float(
    os.environ.get(
        "SEARCH_MATCH_THRESHOLD",
        os.environ.get("EMERGENCY_MATCH_THRESHOLD", "0.75"),
    )
)
SAFETY_VLM_MODEL = os.environ.get("OPENAI_SAFETY_VLM_MODEL", SEARCH_VLM_MODEL)
SAFETY_VLM_DETAIL = os.environ.get("OPENAI_SAFETY_VLM_DETAIL", SEARCH_VLM_DETAIL)
SAFETY_MATCH_THRESHOLD = float(os.environ.get("SAFETY_MATCH_THRESHOLD", "0.75"))
SAFETY_SITE_TIMEZONE = os.environ.get("SAFETY_SITE_TIMEZONE", "America/Vancouver")
SAFETY_ACCESS_START_HOUR = int(os.environ.get("SAFETY_ACCESS_START_HOUR", "9"))
SAFETY_ACCESS_END_HOUR = int(os.environ.get("SAFETY_ACCESS_END_HOUR", "17"))
REPORTING_VLM_MODEL = os.environ.get("OPENAI_REPORTING_VLM_MODEL", "gpt-5.6")
REPORTING_VLM_DETAIL = os.environ.get("OPENAI_REPORTING_VLM_DETAIL", "high")
REPORTING_REASONING_EFFORT = os.environ.get("OPENAI_REPORTING_REASONING_EFFORT", "medium")
REPORTING_SITE_TIMEZONE = os.environ.get("REPORTING_SITE_TIMEZONE", SAFETY_SITE_TIMEZONE)
REPORTING_CAPTURE_START_HOUR = int(os.environ.get("REPORTING_CAPTURE_START_HOUR", "9"))
REPORTING_CAPTURE_END_HOUR = int(os.environ.get("REPORTING_CAPTURE_END_HOUR", "17"))
REPORTING_CAPTURE_POLL_SEC = max(
    1.0,
    float(os.environ.get("REPORTING_CAPTURE_POLL_SEC", "30")),
)
REPORTING_MAX_FRAME_AGE_SEC = max(
    1.0,
    float(os.environ.get("REPORTING_MAX_FRAME_AGE_SEC", "10")),
)
REPORTING_MAX_ANALYSIS_WORKERS = max(
    1,
    int(os.environ.get("REPORTING_MAX_ANALYSIS_WORKERS", "2")),
)
REPORTING_SNAPSHOT_DIR = os.environ.get("REPORTING_SNAPSHOT_DIR") or None
REPORTING_OUTPUT_DIR = os.environ.get("REPORTING_OUTPUT_DIR") or None
OPERATIONAL_STATE_PUBLISH_INTERVAL_SEC = max(
    0.5,
    float(os.environ.get("OPERATIONAL_STATE_PUBLISH_INTERVAL_SEC", "2.0")),
)
MAX_TOOL_ROUNDS = 6


def load_cameras() -> dict:
    """Load camera registry. Returns {camera_id: {name, location, pi_host, pi_port}}."""
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
            "pi_port": int(cam.get("pi_port", 8000)),
        }
    return registry


CAMERAS = load_cameras()
OPERATIONAL_MODES = {"free", "safety", "search", "investigation"}
PLACEHOLDER_OPERATIONAL_MODES = {"investigation"}
_orchestration_lock = threading.RLock()
_operational_lock = threading.Lock()
_state_publisher_lock = threading.Lock()
_state_publisher_started = False
_startup_mode_lock = threading.Lock()
_startup_mode_initialized = False
_operational_state = {
    "mode": "free",
    "objective": None,
}
search_scanner = SearchScanner(
    client=client,
    receiver_url=STREAM_RECEIVER_URL,
    model=SEARCH_VLM_MODEL,
    image_detail=SEARCH_VLM_DETAIL,
    match_threshold=SEARCH_MATCH_THRESHOLD,
    max_workers=max(4, len(CAMERAS) or 1),
)
safety_scanner = SafetyScanner(
    client=client,
    receiver_url=STREAM_RECEIVER_URL,
    model=SAFETY_VLM_MODEL,
    image_detail=SAFETY_VLM_DETAIL,
    match_threshold=SAFETY_MATCH_THRESHOLD,
    max_workers=max(4, len(CAMERAS) or 1),
    site_timezone=SAFETY_SITE_TIMEZONE,
    access_start_hour=SAFETY_ACCESS_START_HOUR,
    access_end_hour=SAFETY_ACCESS_END_HOUR,
)


def _pi_base_url(camera_id: int) -> Optional[str]:
    cam = CAMERAS.get(camera_id)
    if not cam or not cam.get("pi_host"):
        return None
    return f"http://{cam['pi_host']}:{cam['pi_port']}"


def call_pi_set_mode(camera_id: int, mode: str) -> dict:
    base = _pi_base_url(camera_id)
    if base is None:
        return {
            "status": "error",
            "error": f"Unknown camera_id {camera_id}. Known ids: {sorted(CAMERAS.keys())}",
        }
    try:
        r = requests.post(
            f"{base}/set_mode",
            json={"camera_id": camera_id, "mode": mode},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def call_pi_get_state(camera_id: int) -> dict:
    base = _pi_base_url(camera_id)
    if base is None:
        return {
            "status": "error",
            "error": f"Unknown camera_id {camera_id}. Known ids: {sorted(CAMERAS.keys())}",
        }
    try:
        r = requests.post(
            f"{base}/get_state",
            json={"camera_id": camera_id},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def list_cameras_tool() -> dict:
    """Return the registered camera roster for the model to reason about."""
    return {
        "status": "ok",
        "cameras": [
            {
                "camera_id": cid,
                "name": info["name"],
                "location": info.get("location", ""),
                "pi_host": info.get("pi_host"),
            }
            for cid, info in sorted(CAMERAS.items())
        ],
    }


def _post_receiver_json(path: str, payload: dict) -> bool:
    try:
        response = requests.post(f"{STREAM_RECEIVER_URL}{path}", json=payload, timeout=2)
        response.raise_for_status()
        return True
    except Exception:
        return False


def _scanner_running_for_mode(mode: str) -> bool:
    if mode == "search":
        return search_scanner.is_running()
    if mode == "safety":
        return safety_scanner.is_running()
    return False


def _publish_operational_state() -> bool:
    with _orchestration_lock:
        with _operational_lock:
            mode = _operational_state["mode"]
            payload = {
                "mode": mode,
                "objective": _operational_state["objective"],
                "scanner_running": _scanner_running_for_mode(mode),
                "placeholder": mode in PLACEHOLDER_OPERATIONAL_MODES,
            }
        return _post_receiver_json("/system/state", payload)


def _reconcile_operational_state() -> bool:
    """Publish desired state and resume a pending scanner after receiver acknowledgement."""
    with _orchestration_lock:
        receiver_synced = _publish_operational_state()
        if not receiver_synced:
            return False

        with _operational_lock:
            mode = _operational_state["mode"]
            objective = _operational_state["objective"]
        if mode == "search" and objective and not search_scanner.is_running():
            search_scanner.start(objective)
            return _publish_operational_state()
        if mode == "safety" and not safety_scanner.is_running():
            safety_scanner.start()
            return _publish_operational_state()
        return True


def _operational_state_publisher_loop() -> None:
    while True:
        _reconcile_operational_state()
        time.sleep(OPERATIONAL_STATE_PUBLISH_INTERVAL_SEC)


def start_operational_state_publisher() -> None:
    """Keep the receiver synchronized if either laptop service restarts."""
    global _state_publisher_started

    with _state_publisher_lock:
        if _state_publisher_started:
            return
        _state_publisher_started = True
        threading.Thread(
            target=_operational_state_publisher_loop,
            name="operational-state-publisher",
            daemon=True,
        ).start()


def _post_system_log(
    *,
    kind: str,
    level: str,
    message: str,
    camera_id: Optional[int] = None,
) -> None:
    payload = {
        "kind": kind,
        "level": level,
        "message": message,
    }
    if camera_id is not None:
        payload["camera_id"] = camera_id
    _post_receiver_json("/system/log", payload)


reporting_service = ConstructionReporting(
    client=client,
    receiver_url=STREAM_RECEIVER_URL,
    cameras=CAMERAS,
    model=REPORTING_VLM_MODEL,
    image_detail=REPORTING_VLM_DETAIL,
    reasoning_effort=REPORTING_REASONING_EFFORT,
    site_timezone=REPORTING_SITE_TIMEZONE,
    capture_start_hour=REPORTING_CAPTURE_START_HOUR,
    capture_end_hour=REPORTING_CAPTURE_END_HOUR,
    capture_poll_sec=REPORTING_CAPTURE_POLL_SEC,
    max_frame_age_sec=REPORTING_MAX_FRAME_AGE_SEC,
    max_analysis_workers=REPORTING_MAX_ANALYSIS_WORKERS,
    snapshot_root=REPORTING_SNAPSHOT_DIR,
    output_dir=REPORTING_OUTPUT_DIR,
    log_callback=_post_system_log,
)


def start_reporting_recorder() -> None:
    """Start passive hourly evidence capture once for the supervisor process."""
    reporting_service.start()


def _set_all_cameras_default() -> list[dict]:
    camera_ids = sorted(CAMERAS.keys())
    if not camera_ids:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(camera_ids))) as executor:
        return list(executor.map(lambda camera_id: call_pi_set_mode(camera_id, "default"), camera_ids))


def get_operational_mode_tool() -> dict:
    with _orchestration_lock:
        with _operational_lock:
            mode = _operational_state["mode"]
            return {
                "status": "ok",
                "mode": mode,
                "objective": _operational_state["objective"],
                "scanner_running": _scanner_running_for_mode(mode),
                "placeholder": mode in PLACEHOLDER_OPERATIONAL_MODES,
            }


def get_safety_state_tool(timeout_sec: float = 2.0) -> dict:
    try:
        response = requests.get(f"{STREAM_RECEIVER_URL}/system/state", timeout=timeout_sec)
        response.raise_for_status()
        state = response.json()
        return {
            "status": "ok",
            "safety_status": state.get("safety_status", "clear"),
            "active_hazards": state.get("active_safety_hazards") or [],
            "updated_at": state.get("safety_updated_at"),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Could not read construction safety state from the receiver: {exc}",
        }


def clear_safety_hazard(reason: Optional[str] = None) -> dict:
    clean_reason = str(reason or "").strip() or "Cleared explicitly by the operator."
    receiver_synced = _post_receiver_json(
        "/system/safety/clear",
        {"reason": clean_reason},
    )
    if not receiver_synced:
        return {
            "status": "error",
            "receiver_synced": False,
            "message": (
                "The construction safety state could not be cleared because the receiver "
                "did not acknowledge the request."
            ),
        }
    return {
        "status": "ok",
        "safety_status": "clear",
        "receiver_synced": True,
        "message": (
            "Construction safety state cleared by the operator. The dashboard is green unless "
            "the Safety scanner detects a new hazard."
        ),
    }


def set_operational_mode(mode: str, objective: Optional[str] = None) -> dict:
    clean_mode = str(mode or "").strip().lower()
    clean_objective = str(objective or "").strip() or None
    if clean_mode not in OPERATIONAL_MODES:
        return {
            "status": "error",
            "error": (
                f"Unknown operational mode '{clean_mode}'. Available modes: "
                f"{', '.join(sorted(OPERATIONAL_MODES))}."
            ),
        }
    if clean_mode == "search" and not clean_objective:
        return {
            "status": "error",
            "error": "Search Mode requires a visual target or object description.",
        }
    if clean_mode in {"free", "safety"}:
        clean_objective = None

    with _orchestration_lock:
        return _set_operational_mode_locked(clean_mode, clean_objective)


def _set_operational_mode_locked(clean_mode: str, clean_objective: Optional[str]) -> dict:
    with _operational_lock:
        previous_mode = _operational_state["mode"]
        _operational_state["mode"] = clean_mode
        _operational_state["objective"] = clean_objective

    if clean_mode == "free":
        search_scanner.stop()
        safety_scanner.stop()
        camera_results = _set_all_cameras_default()
        receiver_synced = _publish_operational_state()
        failed = [
            {
                "camera_id": camera_id,
                "result": result,
            }
            for camera_id, result in zip(sorted(CAMERAS.keys()), camera_results)
            if result.get("status") != "ok"
        ]

        if not CAMERAS:
            message = (
                "Free Mode activated. Automated scanning is stopped; no cameras are registered."
            )
        elif failed:
            failed_ids = ", ".join(str(item["camera_id"]) for item in failed)
            message = (
                "Free Mode is active and automated scanning is stopped, but camera(s) "
                f"{failed_ids} could not be set to default processing mode."
            )
        else:
            message = (
                "Free Mode activated. Automated scanning is stopped and all registered cameras "
                "are in default processing mode for unprocessed live viewing."
            )
        if not receiver_synced:
            message += " The receiver status update is pending."

        _post_system_log(kind="mode", level="info", message=message)
        if failed:
            failed_ids = ", ".join(str(item["camera_id"]) for item in failed)
            _post_system_log(
                kind="mode",
                level="warning",
                message=f"Free Mode could not normalize camera(s): {failed_ids}.",
            )

        return {
            "status": "ok" if not failed and receiver_synced else "partial_error",
            "mode": "free",
            "objective": None,
            "scanner_running": False,
            "placeholder": False,
            "receiver_synced": receiver_synced,
            "camera_results": camera_results,
            "message": message,
        }

    if clean_mode == "safety":
        search_scanner.stop()
        safety_scanner.stop()

        # Publish the selected mode with scanning stopped before normalizing the
        # fleet. Reconciliation only starts the VLM scanner after the receiver
        # has acknowledged the state.
        _publish_operational_state()
        camera_results = _set_all_cameras_default()
        receiver_synced = _reconcile_operational_state()
        scanner_running = safety_scanner.is_running()
        failed = [
            {
                "camera_id": camera_id,
                "result": result,
            }
            for camera_id, result in zip(sorted(CAMERAS.keys()), camera_results)
            if result.get("status") != "ok"
        ]

        hours = (
            f"{SAFETY_ACCESS_START_HOUR:02d}:00-{SAFETY_ACCESS_END_HOUR:02d}:00 "
            f"{SAFETY_SITE_TIMEZONE}"
        )
        message = (
            f"Safety Mode activated. From {hours}, active camera frames are checked for Fire "
            "Hazard and Work-Zone Intrusion in one VLM pass. Outside that window, they are "
            "checked for Fire Hazard and Unauthorized Entry."
        )
        if not CAMERAS:
            message += " No cameras are registered."
        elif failed:
            failed_ids = ", ".join(str(item["camera_id"]) for item in failed)
            message += f" Camera(s) {failed_ids} could not be set to default processing mode."
        if not receiver_synced:
            if scanner_running:
                message += " The Safety scanner is running; its receiver status update is pending."
            else:
                message += " The Safety scanner is waiting for the receiver to acknowledge the mode."

        _post_system_log(kind="mode", level="info", message=message)
        if failed:
            failed_ids = ", ".join(str(item["camera_id"]) for item in failed)
            _post_system_log(
                kind="mode",
                level="warning",
                message=f"Safety Mode could not reach camera(s): {failed_ids}.",
            )
        if not receiver_synced:
            _post_system_log(
                kind="mode",
                level="warning",
                message=(
                    "The Safety scanner is running, but its latest status has not reached the receiver."
                    if scanner_running
                    else (
                        "Safety Mode could not synchronize with the receiver. Scanning will start "
                        "automatically after synchronization recovers."
                    )
                ),
            )

        return {
            "status": "ok" if not failed and receiver_synced else "partial_error",
            "mode": "safety",
            "objective": None,
            "scanner_running": scanner_running,
            "placeholder": False,
            "receiver_synced": receiver_synced,
            "camera_results": camera_results,
            "message": message,
        }

    if clean_mode in PLACEHOLDER_OPERATIONAL_MODES:
        search_scanner.stop()
        safety_scanner.stop()
        receiver_synced = _publish_operational_state()
        label = clean_mode.title()
        message = (
            f"{label} Mode selected. Its camera-orchestration workflow is a placeholder "
            "and will be configured when its requirements are finalized."
        )
        if not receiver_synced:
            message += " The receiver status update is pending."
        _post_system_log(kind="mode", level="info", message=message)
        return {
            "status": "ok" if receiver_synced else "partial_error",
            "mode": clean_mode,
            "objective": clean_objective,
            "scanner_running": False,
            "placeholder": True,
            "receiver_synced": receiver_synced,
            "camera_results": [],
            "message": message,
        }

    assert clean_mode == "search"
    assert clean_objective is not None
    safety_scanner.stop()
    search_was_running = previous_mode == "search" and search_scanner.is_running()
    if search_was_running:
        search_scanner.stop()
        message = f"Search target updated: {clean_objective}"
    else:
        message = f"Search Mode activated: {clean_objective}"

    # Announce the new objective while scanning is stopped, then normalize all
    # cameras again so retargeting also recovers Pis that were offline or reset.
    _publish_operational_state()
    camera_results = _set_all_cameras_default()
    receiver_synced = _reconcile_operational_state()
    scanner_running = search_scanner.is_running()
    if not receiver_synced:
        if scanner_running:
            message += " The scanner is running; its receiver status update is pending."
        else:
            message += " The Search scanner is waiting for the receiver to acknowledge the objective."
    _post_system_log(kind="mode", level="info", message=message)

    failed = [
        {
            "camera_id": camera_id,
            "result": result,
        }
        for camera_id, result in zip(sorted(CAMERAS.keys()), camera_results)
        if result.get("status") != "ok"
    ]
    if failed:
        failed_ids = ", ".join(str(item["camera_id"]) for item in failed)
        _post_system_log(
            kind="mode",
            level="warning",
            message=f"Search Mode could not reach camera(s): {failed_ids}.",
        )

    if not receiver_synced:
        synchronization_detail = (
            "The Search scanner is running, but its latest status has not reached the receiver."
            if scanner_running
            else (
                "Search Mode could not synchronize its objective with the receiver. "
                "Scanning will start automatically after synchronization recovers."
            )
        )
        _post_system_log(
            kind="mode",
            level="warning",
            message=synchronization_detail,
        )

    return {
        "status": "ok" if not failed and receiver_synced else "partial_error",
        "mode": "search",
        "objective": clean_objective,
        "scanner_running": scanner_running,
        "placeholder": False,
        "receiver_synced": receiver_synced,
        "camera_results": camera_results,
        "message": message,
    }


def initialize_startup_mode() -> dict:
    """Normalize the fleet into Free Mode once before the UI accepts commands."""
    global _startup_mode_initialized

    with _startup_mode_lock:
        if _startup_mode_initialized:
            return get_operational_mode_tool()
        result = set_operational_mode("free")
        _startup_mode_initialized = True
        return result


def _coerce_camera_ids(args: dict) -> list[int]:
    """Accept normal and defensive multi-camera argument shapes."""
    if "camera_ids" in args and args["camera_ids"] is not None:
        raw_ids = args["camera_ids"]
    elif "camera_id" in args and args["camera_id"] is not None:
        raw_ids = args["camera_id"]
    else:
        raise KeyError("camera_id")

    if isinstance(raw_ids, str):
        raw_ids = raw_ids.strip()
        if raw_ids.lower() in {"all", "*", "every"}:
            camera_ids = sorted(CAMERAS.keys())
        elif "," in raw_ids:
            camera_ids = [part.strip() for part in raw_ids.split(",") if part.strip()]
        else:
            camera_ids = [raw_ids]
    elif isinstance(raw_ids, (list, tuple)):
        camera_ids = list(raw_ids)
    else:
        camera_ids = [raw_ids]

    if not camera_ids:
        raise ValueError("camera_ids cannot be empty")
    return [int(camera_id) for camera_id in camera_ids]


def _combine_camera_results(action: str, camera_ids: list[int], results: list[dict]) -> dict:
    """Return a single-camera result unchanged; aggregate multi-camera results."""
    if len(results) == 1:
        return results[0]

    ok_count = sum(1 for result in results if result.get("status") == "ok")
    if ok_count == len(results):
        status = "ok"
    elif ok_count > 0:
        status = "partial_error"
    else:
        status = "error"

    return {
        "status": status,
        "action": action,
        "camera_ids": camera_ids,
        "results": results,
    }


def trim_conversation(conversation, max_interactions=5):
    if not conversation:
        return conversation

    system = conversation[0]
    rest = conversation[1:]
    interactions = []
    current = []

    def is_user(msg):
        return isinstance(msg, dict) and msg.get("role") == "user"

    for msg in rest:
        if is_user(msg):
            if current:
                interactions.append(current)
            current = [msg]
        else:
            current.append(msg)

    if current:
        interactions.append(current)

    trimmed = interactions[-max_interactions:]

    flat = [system]
    for block in trimmed:
        flat.extend(block)
    return flat


tools = [
    {
        "type": "function",
        "name": "set_camera_mode",
        "description": (
            "Set the processing mode for a specific camera on its Raspberry Pi. "
            "Use default for raw video streamed to the laptop without object detection "
            "or bounding boxes; surveillance for object detection; construction for "
            "semantic segmentation; idle to stop the camera process. "
            "Each camera has a unique integer id and lives on its own Pi; the "
            "supervisor routes the command to the correct device. For multiple "
            "cameras, call this tool once for each camera id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "camera_id": {"type": "integer", "description": "Unique integer id of the camera."},
                "mode": {
                    "type": "string",
                    "enum": ["default", "surveillance", "construction", "idle"],
                },
            },
            "required": ["camera_id", "mode"],
        },
    },
    {
        "type": "function",
        "name": "get_camera_state",
        "description": (
            "Get the current operating mode and runtime state of a specific camera. "
            "For multiple cameras, call this tool once for each camera id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "camera_id": {"type": "integer", "description": "Unique integer id of the camera."},
            },
            "required": ["camera_id"],
        },
    },
    {
        "type": "function",
        "name": "list_cameras",
        "description": (
            "List every camera known to the supervisor, including its id, name, "
            "location and the Pi host it runs on. Call this when the user asks "
            "what cameras exist, or when a request is ambiguous about which camera to target."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "set_operational_mode",
        "description": (
            "Select the fleet-wide operational mode. Free Mode stops Search and sets every reachable "
            "camera to default processing for unprocessed live viewing. Search Mode configures cameras "
            "for raw streaming and starts laptop-side VLM scanning for any visible target, including "
            "people, animals, vehicles, equipment, or other objects. Safety Mode configures cameras "
            "for raw streaming and scans each frame for construction hazards. Investigation Mode is "
            "currently a selectable placeholder. Daily reporting is a separate background capability, "
            "not an operational mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["free", "safety", "search", "investigation"],
                    "description": "The fleet-wide operational mode to select.",
                },
                "objective": {
                    "type": "string",
                    "description": (
                        "The operator's objective. Required for Search Mode as a concise visual target "
                        "description, for example 'red fire extinguisher' or 'child in a blue jacket'. "
                        "Unused for Free and Safety Modes and optional for Investigation Mode."
                    ),
                },
            },
            "required": ["mode"],
        },
    },
    {
        "type": "function",
        "name": "get_operational_mode",
        "description": (
            "Report the current fleet-wide operational mode, its objective, whether its workflow is "
            "a placeholder, and whether that mode's scanner is running."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "get_safety_state",
        "description": (
            "Report whether the latched construction safety state is clear or in a stop-work hazard "
            "state, including the active hazard causes recorded by the receiver."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "clear_safety_hazard",
        "description": (
            "Explicitly clear the latched red construction safety hazard state after the operator "
            "asks to clear, reset, or acknowledge it. This does not change operational mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Optional short operator-provided reason for clearing the state.",
                }
            },
        },
    },
    {
        "type": "function",
        "name": "get_reporting_status",
        "description": (
            "Check hourly construction-report snapshot coverage for a date without changing the "
            "current operational mode. If date is omitted, use the current site-local date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "report_date": {
                    "type": "string",
                    "description": "Optional date in YYYY-MM-DD format.",
                }
            },
        },
    },
    {
        "type": "function",
        "name": "generate_daily_report",
        "description": (
            "Generate a PDF construction progress report from the requested day's hourly camera "
            "snapshots. This does not switch operational mode. Include the operator's stated daily "
            "goal when provided; it improves progress and completion reasoning. If date is omitted, "
            "use the current site-local date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "report_date": {
                    "type": "string",
                    "description": "Optional report date in YYYY-MM-DD format.",
                },
                "goal": {
                    "type": "string",
                    "description": (
                        "Optional construction goal for that day, copied faithfully from the operator."
                    ),
                },
            },
        },
    },
]


def _camera_roster_text() -> str:
    if not CAMERAS:
        return "No cameras are currently registered."
    lines = []
    for cid, info in sorted(CAMERAS.items()):
        loc = f" — {info['location']}" if info.get("location") else ""
        lines.append(f"  - id {cid}: {info['name']}{loc} (Pi {info.get('pi_host')})")
    return "Registered cameras:\n" + "\n".join(lines)


SYSTEM_PROMPT = f"""
You are a supervisor assistant for a multi-camera operations system.
Each camera has a unique integer id and is connected to its own Raspberry Pi.

{_camera_roster_text()}

Behaviour rules:
- Users give natural language requests (e.g. "switch camera 1 to surveillance mode",
  "put camera 0 in default mode",
  "what is camera 0 doing?", "which cameras do we have?").
- Use `set_camera_mode` to change the processing mode of a specific camera.
- Available modes are `default` for raw camera footage streamed to the laptop
  with no object detection, bounding boxes, or inference overlays;
  `surveillance` for object detection; `construction` for semantic
  segmentation; and `idle` to stop the camera process.
- Use `get_camera_state` to report the current mode / status of a specific camera.
- If a user asks to control or inspect multiple cameras in one request, call
  the relevant tool once per target camera before summarizing the result. Do
  not put multiple ids into a single `camera_id`.
- If a user asks for "all cameras", target every registered camera id listed
  above.
- Use `list_cameras` when the user asks what cameras exist, or when a request
  does not specify which camera_id to act on and you need to list the options.
- Separate from per-camera processing modes, the fleet has four operational
  modes: `free`, `safety`, `search`, and `investigation`. It starts
  in Free Mode.
- Use `set_operational_mode` whenever the operator asks to enter or switch an
  operational mode. Use `get_operational_mode` when asked which operational
  mode is active or what objective it is pursuing.
- Search Mode is the implemented fleet-wide visual-search workflow. When the
  operator explicitly asks for Search Mode, or asks the system to find or
  locate a described target across cameras, call `set_operational_mode` with
  mode `search` and a concise objective copied from the request. The target can
  be any visibly identifiable person, animal, vehicle, equipment, or object.
  Do not ask for confirmation when a target description is already present; if
  it is missing, ask what the system should search for.
- In Search Mode, all reachable cameras remain in `default` processing mode
  while the laptop-side VLM scanner inspects active streams. Reject requests to
  move individual cameras to other processing modes until Search Mode is left.
- Safety Mode is the implemented continuous construction-safety workflow. It
  keeps every reachable camera in `default` processing mode and evaluates each
  sampled frame once against exactly two applicable checks. During the configured
  09:00-17:00 site-local construction window, those checks are `Fire Hazard`
  and `Work-Zone Intrusion`. Outside that window, machinery is considered off,
  so the checks are `Fire Hazard` and `Unauthorized Entry`. A frame can
  trigger more than one hazard. Safety alerts are red, include a visible
  STOP WORK message and cause, and latch the construction safety state red.
- A latched hazard survives operational-mode changes.
  Call `clear_safety_hazard` only when the operator explicitly asks to clear,
  reset, or acknowledge the safety state. Never clear it merely because the
  operator selects another mode. Use `get_safety_state` when asked whether the
  construction site is currently clear or which hazards are active.
- In Safety Mode, cameras are also locked to `default` processing so the Safety
  scanner continues receiving unmodified frames. Reject requests to move an
  individual camera to another processing mode until Safety Mode is left.
- Free Mode is the neutral live-view baseline. Selecting it stops Search,
  stops Safety scanning, clears the Search objective, and sets every reachable
  camera to `default` processing mode so the operator sees unprocessed footage
  with no automated scanner or operational workflow running. Free Mode does
  not clear a latched construction safety hazard.
- Investigation Mode is a placeholder until its workflow is defined. Still
  call `set_operational_mode` when it is requested, then clearly explain that
  no automatic camera reconfiguration is configured for it yet.
- Daily reporting is not an operational mode and never interrupts the active
  operational mode. A passive recorder saves one fresh frame per registered
  camera at each hourly slot from {REPORTING_CAPTURE_START_HOUR:02d}:00 through
  {REPORTING_CAPTURE_END_HOUR:02d}:00 in {REPORTING_SITE_TIMEZONE}. Use
  `get_reporting_status` when asked what evidence is available for a date.
- Use `generate_daily_report` when the operator asks for a daily construction
  report. Pass the requested date in YYYY-MM-DD form; if the operator says
  "today", omit the date so the site-local current date is used. Pass the
  operator's goal exactly when one is provided. Do not ask for a goal because
  it is optional. The tool compares each camera's ordered daily sequence,
  synthesizes the camera observations, and creates a PDF. In text chat, present
  the returned `report_url` as a Markdown link. If someone asks to enter
  Reporting Mode, explain that reporting now runs in the background and ask
  which day's report they want instead of changing operational mode.
- If the operator asks to stop or cancel Search Mode without naming a next
  operational mode, switch to Free Mode.
- If the user targets a camera that is not registered, politely say so and list
  the available camera ids.
- After any camera tool result, explain clearly what happened. On success,
  confirm the camera (by id and name) and processing mode. On error, explain
  the problem in simple terms and suggest what to try.
- After an operational-mode result, confirm the selected mode and objective.
  If the result says it is a placeholder, always say so plainly.
- For casual chat, respond normally without mentioning cameras.
"""


VOICE_SYSTEM_PROMPT = SYSTEM_PROMPT + """

Voice conversation rules:
- You are speaking out loud to an operator, so keep responses concise, natural,
  and easy to understand in one pass.
- Use a clear, polished, neutral English delivery unless the user asks for a
  different language or accent.
- Do not read raw JSON or internal tool names out loud. Summarize the result in
  plain language after a tool call finishes.
- When a command changes camera state, briefly confirm the camera id, name, and
  new processing mode. When it changes the fleet's operational mode, confirm
  that mode and its objective. When a command is ambiguous, ask a short
  clarifying question.
"""


def execute_supervisor_tool(tool_name: str, args: Any) -> dict:
    """Execute a camera or operational-mode tool for text and realtime voice sessions."""
    if isinstance(args, str):
        try:
            args = json.loads(args) if args else {}
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"Invalid JSON arguments: {str(e)}"}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return {"status": "error", "error": "Tool arguments must be an object."}

    if tool_name == "set_camera_mode":
        try:
            camera_ids = _coerce_camera_ids(args)
            mode = args["mode"]
        except (KeyError, ValueError, TypeError) as e:
            return {
                "status": "error",
                "error": f"Missing or invalid argument for set_camera_mode: {str(e)}",
            }
        with _orchestration_lock:
            with _operational_lock:
                current_operational_mode = _operational_state["mode"]
            if current_operational_mode in {"search", "safety"} and mode != "default":
                label = current_operational_mode.title()
                return {
                    "status": "error",
                    "error": (
                        f"{label} Mode is active. Cameras are locked to default processing mode "
                        "until another operational mode is selected."
                    ),
                }
            results = [call_pi_set_mode(camera_id, mode) for camera_id in camera_ids]
            return _combine_camera_results("set_camera_mode", camera_ids, results)

    if tool_name == "get_camera_state":
        try:
            camera_ids = _coerce_camera_ids(args)
        except (KeyError, ValueError, TypeError) as e:
            return {
                "status": "error",
                "error": f"Missing or invalid argument for get_camera_state: {str(e)}",
            }
        results = [call_pi_get_state(camera_id) for camera_id in camera_ids]
        return _combine_camera_results("get_camera_state", camera_ids, results)

    if tool_name == "list_cameras":
        return list_cameras_tool()

    if tool_name == "set_operational_mode":
        return set_operational_mode(
            str(args.get("mode") or ""),
            args.get("objective"),
        )

    if tool_name == "get_operational_mode":
        return get_operational_mode_tool()

    if tool_name == "get_safety_state":
        return get_safety_state_tool()

    if tool_name == "clear_safety_hazard":
        return clear_safety_hazard(args.get("reason"))

    if tool_name == "get_reporting_status":
        try:
            return reporting_service.get_status(args.get("report_date"))
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

    if tool_name == "generate_daily_report":
        try:
            return reporting_service.generate_daily_report(
                args.get("report_date"),
                args.get("goal"),
            )
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

    return {"status": "error", "error": f"Unknown tool call: {tool_name}"}


def _realtime_session_config() -> dict:
    """Session config sent to OpenAI during the WebRTC SDP handshake."""
    return {
        "type": "realtime",
        "model": REALTIME_MODEL,
        "instructions": VOICE_SYSTEM_PROMPT,
        "audio": {
            "input": {
                "noise_reduction": {
                    "type": "far_field",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.72,
                    "prefix_padding_ms": 250,
                    "silence_duration_ms": 650,
                    "create_response": False,
                    "interrupt_response": False,
                },
            },
            "output": {
                "voice": REALTIME_VOICE,
            },
        },
        "tools": tools,
        "tool_choice": "auto",
    }


def attach_voice_routes(app) -> None:
    """Expose small HTTP endpoints used by the browser voice client."""
    if getattr(app.state, "supervisor_voice_routes_attached", False):
        return
    app.state.supervisor_voice_routes_attached = True

    @app.post("/voice/session")
    async def create_voice_session(request: Request):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return JSONResponse(
                {"error": "OPENAI_API_KEY is not set on the supervisor server."},
                status_code=500,
            )

        offer_sdp = (await request.body()).decode("utf-8", errors="replace")
        if not offer_sdp.strip():
            return JSONResponse({"error": "Missing WebRTC offer SDP."}, status_code=400)

        try:
            upstream = requests.post(
                REALTIME_CALLS_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={
                    "sdp": (None, offer_sdp),
                    "session": (
                        None,
                        json.dumps(_realtime_session_config()),
                        "application/json",
                    ),
                },
                timeout=30,
            )
        except requests.RequestException as e:
            return JSONResponse(
                {"error": "Failed to reach OpenAI Realtime API.", "details": str(e)},
                status_code=502,
            )

        if upstream.status_code >= 400:
            details = upstream.text[:1200]
            return JSONResponse(
                {
                    "error": "OpenAI Realtime session creation failed.",
                    "status_code": upstream.status_code,
                    "details": details,
                    "message": (
                        "OpenAI Realtime session creation failed "
                        f"({upstream.status_code}): {details}"
                    ),
                },
                status_code=upstream.status_code,
            )

        return Response(content=upstream.text, media_type="application/sdp")

    @app.post("/voice/tool")
    async def run_voice_tool(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Request body must be JSON."}, status_code=400)

        tool_name = payload.get("name")
        tool_args = payload.get("arguments", {})
        if not tool_name:
            return JSONResponse({"error": "Missing tool name."}, status_code=400)

        result = execute_supervisor_tool(tool_name, tool_args)
        return JSONResponse(
            {
                "call_id": payload.get("call_id"),
                "name": tool_name,
                "result": result,
            }
        )

    @app.get("/operational/state")
    async def get_operational_state():
        state = get_operational_mode_tool()
        safety_state = get_safety_state_tool(timeout_sec=0.5)
        if safety_state.get("status") == "ok":
            state["safety_status"] = safety_state["safety_status"]
            state["active_safety_hazards"] = safety_state["active_hazards"]
        else:
            state["safety_status"] = "unknown"
            state["active_safety_hazards"] = []
        return JSONResponse(state)

    @app.get("/reports/{filename}")
    async def download_daily_report(filename: str):
        report_path = reporting_service.resolve_report_path(filename)
        if report_path is None:
            return JSONResponse({"error": "Report not found."}, status_code=404)
        return FileResponse(
            report_path,
            media_type="application/pdf",
            filename=report_path.name,
        )


def supervisor_step(user_msg, chat_history, conversation):
    """Main logic for each user message."""
    if conversation is None or len(conversation) == 0:
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    conversation = trim_conversation(conversation)
    conversation.append({"role": "user", "content": user_msg})

    response = client.responses.create(
        model=SUPERVISOR_MODEL,
        input=conversation,
        tools=tools,
    )

    for _ in range(MAX_TOOL_ROUNDS):
        tool_calls = [
            item for item in response.output
            if getattr(item, "type", None) == "function_call"
        ]

        if not tool_calls:
            assistant_text = response.output_text
            conversation.append({"role": "assistant", "content": assistant_text})
            chat_history.append((user_msg, assistant_text))
            return chat_history, conversation

        conversation.extend(response.output)
        for tool_call in tool_calls:
            call_id = getattr(tool_call, "call_id", None)
            if not call_id:
                assistant_text = "The model requested a supervisor tool without a call id."
                conversation.append({"role": "assistant", "content": assistant_text})
                chat_history.append((user_msg, assistant_text))
                return chat_history, conversation

            tool_result = execute_supervisor_tool(
                getattr(tool_call, "name", None),
                getattr(tool_call, "arguments", None),
            )
            conversation.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(tool_result),
            })

        response = client.responses.create(
            model=SUPERVISOR_MODEL,
            input=conversation,
            tools=tools,
        )

    assistant_text = (
        "I ran into too many tool-call rounds while trying to finish that "
        "request. Please try again with fewer camera operations at once."
    )
    conversation.append({"role": "assistant", "content": assistant_text})
    chat_history.append((user_msg, assistant_text))
    return chat_history, conversation


def _supervisor_theme() -> gr.themes.Base:
    """Visual theme aligned with stream_receiver_server dashboard (dark, teal accent, DM Sans)."""
    return (
        gr.themes.Base(
            primary_hue=gr.themes.colors.teal,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("DM Sans"),
            font_mono=(gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"),
        )
        .set(
            body_background_fill="linear-gradient(145deg, #0f0f14 0%, #16162a 40%, #1a1a2e 100%)",
            body_background_fill_dark="linear-gradient(145deg, #0f0f14 0%, #16162a 40%, #1a1a2e 100%)",
            body_text_color="#e8e8ed",
            body_text_color_dark="#e8e8ed",
            body_text_color_subdued="#8b8b9a",
            body_text_color_subdued_dark="#8b8b9a",
            background_fill_primary="rgba(255,255,255,0.04)",
            background_fill_primary_dark="rgba(255,255,255,0.04)",
            background_fill_secondary="rgba(22,22,42,0.92)",
            background_fill_secondary_dark="rgba(22,22,42,0.92)",
            border_color_primary="rgba(255,255,255,0.08)",
            border_color_primary_dark="rgba(255,255,255,0.08)",
            color_accent="#00d4aa",
            color_accent_soft="rgba(0, 212, 170, 0.14)",
            color_accent_soft_dark="rgba(0, 212, 170, 0.14)",
            border_color_accent_subdued="rgba(0, 212, 170, 0.35)",
            border_color_accent_subdued_dark="rgba(0, 212, 170, 0.35)",
            input_background_fill="rgba(0,0,0,0.28)",
            input_background_fill_dark="rgba(0,0,0,0.28)",
            input_border_color="rgba(255,255,255,0.1)",
            input_border_color_dark="rgba(255,255,255,0.1)",
            input_placeholder_color="#8b8b9a",
            input_placeholder_color_dark="#8b8b9a",
            shadow_drop="0 8px 32px rgba(0, 0, 0, 0.35)",
            block_radius="16px",
            button_primary_background_fill="#00a884",
            button_primary_background_fill_dark="#00a884",
            button_primary_background_fill_hover="#00d4aa",
            button_primary_background_fill_hover_dark="#00d4aa",
            button_primary_text_color="#0f0f14",
            button_primary_text_color_dark="#0f0f14",
            button_primary_border_color="transparent",
            button_primary_border_color_dark="transparent",
            block_background_fill="rgba(255,255,255,0.04)",
            block_background_fill_dark="rgba(255,255,255,0.04)",
            block_border_color="rgba(255,255,255,0.08)",
            block_border_color_dark="rgba(255,255,255,0.08)",
            block_label_text_color="#8b8b9a",
            block_label_text_color_dark="#8b8b9a",
            block_title_text_color="#e8e8ed",
            block_title_text_color_dark="#e8e8ed",
        )
    )


SUPERVISOR_CSS = """
/* Page shell — match stream_receiver_server feel */
.gradio-container {
  max-width: 920px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding: 24px 20px 40px !important;
}

.sup-page-header {
  width: 100%;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 20px;
  flex-wrap: wrap;
}
.sup-page-header h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
  letter-spacing: 0;
  background: linear-gradient(135deg, #fff 0%, #00d4aa 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.sup-page-header .sup-sub {
  margin: 6px 0 0 0;
  font-size: 0.875rem;
  color: #8b8b9a;
  line-height: 1.45;
}
.sup-mode-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(0, 212, 170, 0.35);
  background: rgba(0, 212, 170, 0.12);
  color: #00d4aa;
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.sup-mode-chip.search {
  border-color: rgba(56, 189, 248, 0.42);
  background: rgba(56, 189, 248, 0.14);
  color: #7dd3fc;
}
.sup-mode-chip.hazard {
  border-color: rgba(248, 113, 113, 0.78);
  background: rgba(185, 28, 28, 0.3);
  color: #fecaca;
  box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.15), 0 8px 24px rgba(127, 29, 29, 0.28);
}
.sup-mode-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 10px currentColor;
}

.sup-roster-wrap {
  margin-bottom: 22px;
}
.sup-roster-label {
  display: block;
  font-size: 0.6875rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #8b8b9a;
  margin-bottom: 10px;
}
.sup-roster-grid {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.sup-cam-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.22);
}
.sup-cam-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  background: #00d4aa;
  box-shadow: 0 0 10px rgba(0, 212, 170, 0.55);
}
.sup-cam-line1 {
  font-size: 0.9375rem;
  font-weight: 500;
  color: #e8e8ed;
}
.sup-cam-id {
  color: #8b8b9a;
  font-weight: 400;
}
.sup-cam-line2 {
  font-size: 0.75rem;
  color: #8b8b9a;
  margin-top: 3px;
}
.sup-roster-empty {
  padding: 14px 16px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px dashed rgba(255, 255, 255, 0.12);
  color: #8b8b9a;
  font-size: 0.875rem;
}
.sup-roster-empty code {
  font-size: 0.8125rem;
  color: #00d4aa;
}

/* Chat panel */
#supervisor-chatbot {
  border-radius: 16px !important;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.04);
}
#supervisor-chatbot .bubble-wrap {
  background: rgba(0, 0, 0, 0.22) !important;
  border-radius: 0 0 14px 14px;
  min-height: 420px;
}
#supervisor-chatbot .block-label {
  font-size: 0.8125rem !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #8b8b9a !important;
  padding: 16px 18px !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, transparent 100%);
}

/* Composer row */
.sup-input-row {
  margin-top: 14px !important;
  align-items: stretch !important;
  gap: 12px !important;
}
.sup-input-row .block {
  box-shadow: none !important;
}
.sup-input-row textarea, .sup-input-row input {
  border-radius: 12px !important;
  font-size: 0.9375rem !important;
}
.sup-input-row button {
  border-radius: 12px !important;
  font-weight: 600 !important;
  min-width: 100px;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
}

/* Realtime voice panel */
.sup-voice-panel {
  margin: 0 0 16px 0;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.28);
  overflow: hidden;
}
.sup-voice-main {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 16px;
  flex-wrap: wrap;
}
.sup-voice-copy {
  min-width: 220px;
  flex: 1;
}
.sup-voice-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0;
  font-size: 0.8125rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #8b8b9a;
}
.sup-voice-led {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #8b8b9a;
  box-shadow: none;
  flex-shrink: 0;
}
.sup-voice-panel.active .sup-voice-led,
.sup-voice-panel.working .sup-voice-led {
  background: #00d4aa;
  box-shadow: 0 0 12px rgba(0, 212, 170, 0.7);
}
.sup-voice-panel.error .sup-voice-led {
  background: #ef4444;
  box-shadow: 0 0 12px rgba(239, 68, 68, 0.55);
}
.sup-voice-status {
  margin-top: 5px;
  font-size: 0.9375rem;
  color: #e8e8ed;
}
.sup-voice-detail {
  margin-top: 3px;
  font-size: 0.75rem;
  color: #8b8b9a;
}
.sup-voice-button {
  display: inline-flex;
  align-items: center;
  gap: 9px;
  min-width: 156px;
  justify-content: center;
  border: 1px solid rgba(0, 212, 170, 0.35);
  border-radius: 12px;
  background: #00a884;
  color: #0f0f14;
  padding: 11px 16px;
  font: inherit;
  font-size: 0.9375rem;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
  transition: background 0.2s, transform 0.2s, border-color 0.2s;
}
.sup-voice-button:hover {
  background: #00d4aa;
  transform: translateY(-1px);
}
.sup-voice-button.active {
  background: rgba(239, 68, 68, 0.9);
  border-color: rgba(239, 68, 68, 0.55);
  color: #ffffff;
}
.sup-voice-button:disabled {
  cursor: wait;
  opacity: 0.75;
  transform: none;
}
.sup-voice-icon {
  width: 16px;
  height: 16px;
  border: 2px solid currentColor;
  border-radius: 9px 9px 11px 11px;
  position: relative;
  display: inline-block;
}
.sup-voice-icon::before {
  content: "";
  position: absolute;
  left: 50%;
  bottom: -7px;
  width: 2px;
  height: 6px;
  background: currentColor;
  transform: translateX(-50%);
}
.sup-voice-icon::after {
  content: "";
  position: absolute;
  left: 50%;
  bottom: -10px;
  width: 12px;
  height: 2px;
  border-radius: 2px;
  background: currentColor;
  transform: translateX(-50%);
}
.sup-voice-log {
  max-height: 145px;
  overflow-y: auto;
  padding: 10px 16px 14px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(0, 0, 0, 0.16);
}
.sup-voice-entry {
  display: flex;
  gap: 10px;
  padding: 5px 0;
  font-size: 0.8125rem;
  line-height: 1.4;
}
.sup-voice-entry .who {
  width: 76px;
  flex-shrink: 0;
  color: #00d4aa;
  font-weight: 600;
}
.sup-voice-entry .text {
  color: #e8e8ed;
  word-break: break-word;
}
.sup-voice-entry.muted .who,
.sup-voice-entry.muted .text {
  color: #8b8b9a;
}
.sup-voice-entry.error .who,
.sup-voice-entry.error .text {
  color: #ef4444;
}
"""


SUPERVISOR_JS = r"""
() => {
  const RETRY_MS = 250;
  const MIN_ASSISTANT_MUTE_MS = 3000;
  const AUDIO_DELTA_MUTE_EXTENSION_MS = 1600;
  const OUTPUT_SILENCE_HOLD_MS = 2200;
  const OUTPUT_MONITOR_INTERVAL_MS = 100;
  const OUTPUT_SILENCE_RMS = 0.012;
  const OUTPUT_FALLBACK_DONE_DELAY_MS = 7000;

  function initVoiceClient() {
    const panel = document.getElementById("sup-voice-panel");
    const button = document.getElementById("sup-voice-toggle");
    const label = document.getElementById("sup-voice-toggle-label");
    const statusText = document.getElementById("sup-voice-status-text");
    const detailText = document.getElementById("sup-voice-detail");
    const log = document.getElementById("sup-voice-log");

    if (!panel || !button || !label || !statusText || !detailText || !log) {
      window.setTimeout(initVoiceClient, RETRY_MS);
      return;
    }
    if (panel.dataset.voiceReady === "true") return;
    panel.dataset.voiceReady = "true";

    const state = {
      active: false,
      starting: false,
      pc: null,
      dc: null,
      micStream: null,
      micTrack: null,
      micSender: null,
      micDetached: false,
      micSwitchPromise: Promise.resolve(),
      audioEl: null,
      audioContext: null,
      outputSource: null,
      outputAnalyser: null,
      outputSamples: null,
      outputMonitorTimer: null,
      handledCalls: new Set(),
      pendingCalls: new Map(),
      lastAssistantText: "",
      currentAssistantTranscript: "",
      assistantSpeaking: false,
      assistantMuteUntil: 0,
      assistantResponseDone: false,
      outputSawAudio: false,
      outputSilentSince: null,
      outputGuardStartedAt: 0,
      resumeMicTimer: null,
    };

    function setStatus(text, mode, detail) {
      panel.classList.remove("active", "working", "error");
      if (mode) panel.classList.add(mode);
      statusText.textContent = text;
      detailText.textContent = detail || "";
    }

    function setButton(isActive, isBusy) {
      button.disabled = !!isBusy;
      button.classList.toggle("active", !!isActive);
      button.setAttribute("aria-pressed", isActive ? "true" : "false");
      label.textContent = isActive ? "End voice chat" : "Start voice chat";
    }

    function appendEntry(who, text, className) {
      if (!text) return;
      const empty = log.querySelector(".sup-voice-entry-empty");
      if (empty) empty.remove();
      const row = document.createElement("div");
      row.className = "sup-voice-entry" + (className ? " " + className : "");
      const whoEl = document.createElement("span");
      whoEl.className = "who";
      whoEl.textContent = who;
      const textEl = document.createElement("span");
      textEl.className = "text";
      textEl.textContent = text;
      row.append(whoEl, textEl);
      log.appendChild(row);
      while (log.children.length > 12) log.removeChild(log.firstElementChild);
      log.scrollTop = log.scrollHeight;
    }

    function parseMaybeJson(value) {
      if (!value) return {};
      if (typeof value === "object") return value;
      try {
        return JSON.parse(value);
      } catch (error) {
        return {};
      }
    }

    function extractAssistantText(item) {
      const parts = Array.isArray(item?.content) ? item.content : [];
      return parts.map((part) => {
        if (part.type === "output_text") return part.text || "";
        if (part.type === "text") return part.text || "";
        if (part.type === "audio") return part.transcript || "";
        if (part.type === "output_audio") return part.transcript || "";
        return "";
      }).join("").trim();
    }

    function appendAssistantText(text) {
      const clean = String(text || "").trim();
      if (!clean || clean === state.lastAssistantText) return;
      state.lastAssistantText = clean;
      appendEntry("Supervisor", clean);
    }

    function sendRealtimeEvent(event) {
      if (!state.dc || state.dc.readyState !== "open") return;
      state.dc.send(JSON.stringify(event));
    }

    function setMicEnabled(enabled) {
      if (!state.micTrack) return;
      state.micTrack.enabled = enabled;
      if (!state.micSender?.replaceTrack) return;

      const shouldDetach = !enabled;
      if (state.micDetached === shouldDetach) return;
      state.micDetached = shouldDetach;
      const nextTrack = enabled ? state.micTrack : null;
      state.micSwitchPromise = state.micSwitchPromise
        .catch(() => {})
        .then(() => state.micSender.replaceTrack(nextTrack))
        .catch((error) => {
          appendEntry("Error", "Microphone routing failed: " + error.message, "error");
        });
    }

    function clearInputBuffer() {
      sendRealtimeEvent({ type: "input_audio_buffer.clear" });
    }

    function isInputBlocked() {
      return state.assistantSpeaking || state.micDetached || Boolean(state.resumeMicTimer);
    }

    function getFallbackResumeDelay() {
      const words = state.currentAssistantTranscript.trim().split(/\s+/).filter(Boolean).length;
      if (!words) return OUTPUT_FALLBACK_DONE_DELAY_MS;
      return Math.min(Math.max((words / 2.6) * 1000 + 1800, 4000), 18000);
    }

    function setupOutputAnalyser(stream) {
      try {
        const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCtor) return;
        if (!state.audioContext) state.audioContext = new AudioContextCtor();
        if (state.audioContext.state === "suspended") {
          state.audioContext.resume().catch(() => {});
        }
        if (state.outputSource) state.outputSource.disconnect();
        state.outputSource = state.audioContext.createMediaStreamSource(stream);
        state.outputAnalyser = state.audioContext.createAnalyser();
        state.outputAnalyser.fftSize = 1024;
        state.outputAnalyser.smoothingTimeConstant = 0.1;
        state.outputSamples = new Uint8Array(state.outputAnalyser.fftSize);
        state.outputSource.connect(state.outputAnalyser);
      } catch (error) {
        state.outputAnalyser = null;
        state.outputSamples = null;
      }
    }

    function outputRms() {
      if (!state.outputAnalyser || !state.outputSamples) return null;
      state.outputAnalyser.getByteTimeDomainData(state.outputSamples);
      let sum = 0;
      for (const value of state.outputSamples) {
        const centered = (value - 128) / 128;
        sum += centered * centered;
      }
      return Math.sqrt(sum / state.outputSamples.length);
    }

    function stopOutputMonitor() {
      if (!state.outputMonitorTimer) return;
      window.clearInterval(state.outputMonitorTimer);
      state.outputMonitorTimer = null;
    }

    function finishAssistantOutputGuard() {
      stopOutputMonitor();
      if (state.resumeMicTimer) {
        window.clearTimeout(state.resumeMicTimer);
        state.resumeMicTimer = null;
      }
      state.assistantSpeaking = false;
      state.assistantResponseDone = false;
      state.outputSawAudio = false;
      state.outputSilentSince = null;
      state.outputGuardStartedAt = 0;
      state.assistantMuteUntil = 0;
      clearInputBuffer();
      if (state.active) {
        setMicEnabled(true);
        setStatus("Listening", "active", "Microphone is live.");
      }
    }

    function startOutputMonitor() {
      if (state.outputMonitorTimer) return;
      state.outputMonitorTimer = window.setInterval(() => {
        if (!state.active || !state.assistantSpeaking) {
          stopOutputMonitor();
          return;
        }

        setMicEnabled(false);
        clearInputBuffer();

        const now = Date.now();
        const rms = outputRms();
        const hasOutputAudio = rms !== null && rms > OUTPUT_SILENCE_RMS;

        if (hasOutputAudio) {
          state.outputSawAudio = true;
          state.outputSilentSince = null;
          state.assistantMuteUntil = Math.max(
            state.assistantMuteUntil,
            now + AUDIO_DELTA_MUTE_EXTENSION_MS,
          );
          return;
        }

        if (!state.assistantResponseDone) return;

        if (!state.outputSawAudio && rms === null) {
          if (now - state.outputGuardStartedAt >= getFallbackResumeDelay()) {
            finishAssistantOutputGuard();
          }
          return;
        }

        if (!state.outputSilentSince) state.outputSilentSince = now;

        const minimumMuteElapsed = now - state.outputGuardStartedAt >= MIN_ASSISTANT_MUTE_MS;
        const heldSilence = now - state.outputSilentSince >= OUTPUT_SILENCE_HOLD_MS;
        const pastMuteTail = now >= state.assistantMuteUntil;
        if (minimumMuteElapsed && heldSilence && pastMuteTail) {
          finishAssistantOutputGuard();
        }
      }, OUTPUT_MONITOR_INTERVAL_MS);
    }

    function muteMicForAssistant() {
      state.assistantSpeaking = true;
      state.assistantMuteUntil = Math.max(
        state.assistantMuteUntil,
        Date.now() + AUDIO_DELTA_MUTE_EXTENSION_MS,
      );
      if (state.resumeMicTimer) {
        window.clearTimeout(state.resumeMicTimer);
        state.resumeMicTimer = null;
      }
      setMicEnabled(false);
      clearInputBuffer();
      startOutputMonitor();
    }

    function markAssistantOutputDone() {
      state.assistantResponseDone = true;
      state.assistantMuteUntil = Math.max(
        state.assistantMuteUntil,
        Date.now() + AUDIO_DELTA_MUTE_EXTENSION_MS,
      );
      startOutputMonitor();
      if (state.resumeMicTimer) window.clearTimeout(state.resumeMicTimer);
      state.resumeMicTimer = window.setTimeout(() => {
        if (state.assistantResponseDone && state.assistantSpeaking && !state.outputAnalyser) {
          finishAssistantOutputGuard();
        }
      }, getFallbackResumeDelay());
    }

    function rememberFunctionCall(item) {
      if (!item || item.type !== "function_call") return;
      const callId = item.call_id || item.id;
      if (!callId) return;
      const existing = state.pendingCalls.get(callId) || {};
      state.pendingCalls.set(callId, { ...existing, ...item, call_id: callId });
    }

    async function runFunctionCall(item) {
      if (!item || item.type !== "function_call") return;
      const callId = item.call_id || item.id;
      const name = item.name;
      if (!callId || !name || state.handledCalls.has(callId)) return;
      if (!state.dc || state.dc.readyState !== "open") return;

      state.handledCalls.add(callId);
      setStatus("Running supervisor command", "working", name);

      let payload;
      try {
        const res = await fetch("/voice/tool", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            call_id: callId,
            name,
            arguments: parseMaybeJson(item.arguments),
          }),
        });
        payload = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(payload.error || "Supervisor tool call failed.");
        }
      } catch (error) {
        payload = { result: { status: "error", error: error.message } };
        appendEntry("Error", error.message, "error");
      }

      const output = JSON.stringify(payload.result || payload);
      sendRealtimeEvent({
        type: "conversation.item.create",
        item: {
          type: "function_call_output",
          call_id: callId,
          output,
        },
      });
      sendRealtimeEvent({ type: "response.create" });
    }

    function handleRealtimeEvent(raw) {
      let event;
      try {
        event = JSON.parse(raw.data);
      } catch (error) {
        return;
      }

      if (event.type === "input_audio_buffer.speech_started") {
        if (isInputBlocked()) {
          clearInputBuffer();
          return;
        }
        setStatus("Listening", "active", "Microphone is live.");
        return;
      }
      if (event.type === "input_audio_buffer.speech_stopped") {
        if (isInputBlocked()) {
          clearInputBuffer();
          return;
        }
        setStatus("Thinking", "working", "Processing your command.");
        return;
      }
      if (event.type === "input_audio_buffer.committed") {
        if (isInputBlocked()) {
          clearInputBuffer();
          return;
        }
        setStatus("Thinking", "working", "Processing your command.");
        sendRealtimeEvent({ type: "response.create" });
        return;
      }
      if (event.type === "conversation.item.input_audio_transcription.completed") {
        if (isInputBlocked()) return;
        appendEntry("You", event.transcript || "");
        return;
      }
      if (event.type === "response.created") {
        state.currentAssistantTranscript = "";
        state.assistantResponseDone = false;
        state.outputSawAudio = false;
        state.outputSilentSince = null;
        state.outputGuardStartedAt = Date.now();
        state.assistantMuteUntil = Math.max(
          state.assistantMuteUntil,
          Date.now() + MIN_ASSISTANT_MUTE_MS,
        );
        muteMicForAssistant();
        setStatus("Supervisor speaking", "working", "Microphone is paused to prevent speaker echo.");
        return;
      }
      if (event.type === "response.output_audio.delta" || event.type === "response.audio.delta") {
        state.assistantMuteUntil = Math.max(
          state.assistantMuteUntil,
          Date.now() + AUDIO_DELTA_MUTE_EXTENSION_MS,
        );
        muteMicForAssistant();
        return;
      }
      if (event.type === "response.output_audio.done" || event.type === "response.audio.done") {
        markAssistantOutputDone();
        return;
      }
      if (event.type === "response.audio_transcript.done" || event.type === "response.output_audio_transcript.done") {
        state.currentAssistantTranscript = event.transcript || state.currentAssistantTranscript;
        appendAssistantText(event.transcript || "");
        return;
      }
      if (event.type === "response.output_item.added") {
        rememberFunctionCall(event.item);
        return;
      }
      if (event.type === "response.function_call_arguments.delta") {
        const existing = state.pendingCalls.get(event.call_id) || { type: "function_call", call_id: event.call_id };
        existing.arguments = (existing.arguments || "") + (event.delta || "");
        state.pendingCalls.set(event.call_id, existing);
        return;
      }
      if (event.type === "response.function_call_arguments.done") {
        const existing = state.pendingCalls.get(event.call_id) || { type: "function_call", call_id: event.call_id };
        runFunctionCall({
          ...existing,
          name: event.name || existing.name,
          arguments: event.arguments || existing.arguments,
        });
        return;
      }
      if (event.type === "response.output_item.done") {
        if (event.item?.type === "function_call") runFunctionCall(event.item);
        if (event.item?.type === "message") appendAssistantText(extractAssistantText(event.item));
        return;
      }
      if (event.type === "response.done") {
        (event.response?.output || []).forEach((item) => {
          if (item.type === "function_call") runFunctionCall(item);
          if (item.type === "message") appendAssistantText(extractAssistantText(item));
        });
        markAssistantOutputDone();
        if (state.active) setStatus("Finishing response", "working", "Waiting for assistant audio to finish.");
        return;
      }
      if (event.type === "error") {
        const message = event.error?.message || "Realtime API error.";
        appendEntry("Error", message, "error");
        setStatus("Voice error", "error", message);
      }
    }

    function cleanup() {
      if (state.dc && state.dc.readyState === "open") {
        try { state.dc.send(JSON.stringify({ type: "response.cancel" })); } catch (error) {}
      }
      if (state.resumeMicTimer) {
        window.clearTimeout(state.resumeMicTimer);
        state.resumeMicTimer = null;
      }
      stopOutputMonitor();
      if (state.outputSource) {
        try { state.outputSource.disconnect(); } catch (error) {}
      }
      if (state.audioContext) {
        try { state.audioContext.close(); } catch (error) {}
      }
      if (state.dc) state.dc.close();
      if (state.pc) state.pc.close();
      if (state.micStream) state.micStream.getTracks().forEach((track) => track.stop());
      if (state.audioEl) state.audioEl.srcObject = null;
      state.active = false;
      state.starting = false;
      state.assistantSpeaking = false;
      state.assistantMuteUntil = 0;
      state.pc = null;
      state.dc = null;
      state.micStream = null;
      state.micTrack = null;
      state.micSender = null;
      state.micDetached = false;
      state.micSwitchPromise = Promise.resolve();
      state.audioContext = null;
      state.outputSource = null;
      state.outputAnalyser = null;
      state.outputSamples = null;
      state.currentAssistantTranscript = "";
      state.assistantResponseDone = false;
      state.outputSawAudio = false;
      state.outputSilentSince = null;
      state.outputGuardStartedAt = 0;
      state.handledCalls.clear();
      state.pendingCalls.clear();
    }

    async function startVoice() {
      if (state.active || state.starting) return;
      if (!navigator.mediaDevices?.getUserMedia) {
        setStatus("Voice unavailable", "error", "This browser cannot access a microphone from this page.");
        return;
      }

      state.starting = true;
      setButton(true, true);
      setStatus("Connecting voice", "working", "Requesting microphone access.");

      try {
        const pc = new RTCPeerConnection();
        const dc = pc.createDataChannel("oai-events");
        const micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });

        let audioEl = state.audioEl;
        if (!audioEl) {
          audioEl = document.createElement("audio");
          audioEl.autoplay = true;
          audioEl.style.display = "none";
          panel.appendChild(audioEl);
          state.audioEl = audioEl;
        }

        const micTrack = micStream.getAudioTracks()[0] || null;
        const micSender = micTrack ? pc.addTrack(micTrack, micStream) : null;
        pc.ontrack = (event) => {
          audioEl.srcObject = event.streams[0];
          setupOutputAnalyser(event.streams[0]);
          audioEl.play().catch(() => {});
        };
        dc.addEventListener("open", () => {
          state.active = true;
          state.starting = false;
          setButton(true, false);
          setStatus("Listening", "active", "Microphone is live.");
          appendEntry("System", "Voice chat connected.", "muted");
        });
        dc.addEventListener("message", handleRealtimeEvent);
        dc.addEventListener("close", () => {
          if (state.active) {
            cleanup();
            setButton(false, false);
            setStatus("Voice idle", "", "AI-generated voice is off.");
          }
        });
        pc.addEventListener("connectionstatechange", () => {
          if (["failed", "disconnected", "closed"].includes(pc.connectionState) && state.active) {
            cleanup();
            setButton(false, false);
            setStatus("Voice disconnected", "error", "Start voice chat again to reconnect.");
          }
        });

        state.pc = pc;
        state.dc = dc;
        state.micStream = micStream;
        state.micTrack = micTrack;
        state.micSender = micSender;
        state.micDetached = false;

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const sdpResponse = await fetch("/voice/session", {
          method: "POST",
          headers: { "Content-Type": "application/sdp" },
          body: offer.sdp,
        });
        const answerSdp = await sdpResponse.text();
        if (!sdpResponse.ok) {
          let message = answerSdp || "Failed to create voice session.";
          try {
            const parsed = JSON.parse(answerSdp);
            message = parsed.message || parsed.details || parsed.error || message;
          } catch (error) {}
          throw new Error(message);
        }

        await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
        setStatus("Finishing connection", "working", "Opening the audio channel.");
      } catch (error) {
        cleanup();
        setButton(false, false);
        appendEntry("Error", error.message, "error");
        setStatus("Voice error", "error", error.message);
      }
    }

    function stopVoice() {
      cleanup();
      setButton(false, false);
      setStatus("Voice idle", "", "AI-generated voice is off.");
      appendEntry("System", "Voice chat ended.", "muted");
    }

    button.addEventListener("click", () => {
      if (state.active || state.starting) stopVoice();
      else startVoice();
    });
  }

  function initOperationalModeChip() {
    const chip = document.getElementById("sup-mode-chip");
    const label = document.getElementById("sup-mode-label");
    if (!chip || !label) {
      window.setTimeout(initOperationalModeChip, RETRY_MS);
      return;
    }
    async function refreshMode() {
      try {
        const res = await fetch("/operational/state");
        const state = await res.json();
        const mode = ["free", "safety", "search", "investigation"].includes(state.mode)
          ? state.mode
          : "free";
        const hazard = state.safety_status === "hazard";
        chip.classList.remove("free", "safety", "search", "investigation", "hazard");
        chip.classList.add(mode);
        if (hazard) chip.classList.add("hazard");
        const modeLabel = mode.charAt(0).toUpperCase() + mode.slice(1);
        label.textContent = hazard ? "Stop work · " + modeLabel : modeLabel;
      } catch (error) {
        label.textContent = "Unknown";
      }
    }
    refreshMode();
    window.setInterval(refreshMode, 2000);
  }

  function initSupervisorUi() {
    initVoiceClient();
    initOperationalModeChip();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initSupervisorUi);
  } else {
    initSupervisorUi();
  }
}
"""


def _supervisor_header_html() -> str:
    return """<header class="sup-page-header">
  <div class="sup-title-block">
    <h1>Supervisor</h1>
    <p class="sup-sub">Natural-language control for your multi-camera fleet — select an operational mode, configure cameras, or read status.</p>
  </div>
  <div class="sup-mode-chip free" id="sup-mode-chip"><span class="sup-mode-dot"></span><span id="sup-mode-label">Free</span></div>
</header>"""


def _camera_roster_html() -> str:
    if not CAMERAS:
        return (
            '<div class="sup-roster-wrap"><span class="sup-roster-label">Registered cameras</span>'
            '<div class="sup-roster-empty">No cameras registered in <code>cameras.json</code>.</div></div>'
        )
    cards = []
    for cid, info in sorted(CAMERAS.items()):
        name = html.escape(str(info.get("name", f"Camera {cid}")))
        loc = info.get("location") or ""
        loc_e = html.escape(str(loc)) if loc else ""
        host = info.get("pi_host")
        host_e = html.escape(str(host)) if host else ""
        line2_parts = []
        if loc_e:
            line2_parts.append(loc_e)
        if host_e:
            line2_parts.append(f"Pi {host_e}")
        line2 = " · ".join(line2_parts)
        line2_html = f'<div class="sup-cam-line2">{line2}</div>' if line2 else ""
        cards.append(
            f'<div class="sup-cam-card"><span class="sup-cam-dot"></span>'
            f'<div><div class="sup-cam-line1">{name} <span class="sup-cam-id">#{cid}</span></div>'
            f"{line2_html}</div></div>"
        )
    inner = '<div class="sup-roster-grid">' + "".join(cards) + "</div>"
    return (
        '<div class="sup-roster-wrap"><span class="sup-roster-label">Registered cameras</span>'
        + inner
        + "</div>"
    )


def _voice_panel_html() -> str:
    return """<section class="sup-voice-panel" id="sup-voice-panel">
  <div class="sup-voice-main">
    <div class="sup-voice-copy">
      <div class="sup-voice-title"><span class="sup-voice-led" id="sup-voice-led"></span><span>Voice chat</span></div>
      <div class="sup-voice-status" id="sup-voice-status-text">Voice idle</div>
      <div class="sup-voice-detail" id="sup-voice-detail">AI-generated voice is off.</div>
    </div>
    <button type="button" class="sup-voice-button" id="sup-voice-toggle" aria-pressed="false">
      <span class="sup-voice-icon" aria-hidden="true"></span>
      <span id="sup-voice-toggle-label">Start voice chat</span>
    </button>
  </div>
  <div class="sup-voice-log" id="sup-voice-log">
    <div class="sup-voice-entry sup-voice-entry-empty muted"><span class="who">System</span><span class="text">Voice events will appear here.</span></div>
  </div>
</section>"""


def build_ui():
    initialize_startup_mode()
    start_operational_state_publisher()
    start_reporting_recorder()
    with gr.Blocks(
        title="Supervisor Agent",
        theme=_supervisor_theme(),
        css=SUPERVISOR_CSS,
        js=SUPERVISOR_JS,
        fill_width=True,
    ) as demo:
        gr.HTML(_supervisor_header_html())
        gr.HTML(_voice_panel_html())

        chatbot = gr.Chatbot(
            label="Supervisor Agent",
            height=600,
            elem_id="supervisor-chatbot",
            placeholder="Ask anything about your cameras, or say hi…",
        )

        with gr.Row(elem_classes=["sup-input-row"]):
            user_in = gr.Textbox(
                placeholder="Give a command or chat…",
                label="Your message",
                show_label=False,
                scale=8,
                lines=1,
                max_lines=4,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        conversation_state = gr.State(
            [{"role": "system", "content": SYSTEM_PROMPT}]
        )
        chat_history_state = gr.State([])

        def on_user_submit(user_msg, chat_history, conversation):
            chat_history, conversation = supervisor_step(user_msg, chat_history, conversation)
            return chat_history, conversation, ""

        user_in.submit(
            fn=on_user_submit,
            inputs=[user_in, chat_history_state, conversation_state],
            outputs=[chatbot, conversation_state, user_in],
        )
        send_btn.click(
            fn=on_user_submit,
            inputs=[user_in, chat_history_state, conversation_state],
            outputs=[chatbot, conversation_state, user_in],
        )

    attach_voice_routes(demo.app)
    return demo


if __name__ == "__main__":
    ui = build_ui()
    app, _, _ = ui.launch(prevent_thread_lock=True)
    attach_voice_routes(app)
    ui.block_thread()
