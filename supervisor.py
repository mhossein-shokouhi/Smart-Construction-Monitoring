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
SAFETY_VLM_MODEL = os.environ.get("OPENAI_SAFETY_VLM_MODEL", "gpt-5.6-sol")
SAFETY_VLM_DETAIL = os.environ.get("OPENAI_SAFETY_VLM_DETAIL", "auto")
SAFETY_REASONING_EFFORT = os.environ.get("OPENAI_SAFETY_REASONING_EFFORT", "medium")
SAFETY_MATCH_THRESHOLD = float(os.environ.get("SAFETY_MATCH_THRESHOLD", "0.75"))
SAFETY_SITE_TIMEZONE = os.environ.get("SAFETY_SITE_TIMEZONE", "America/Vancouver")
SAFETY_ACCESS_START_HOUR = int(os.environ.get("SAFETY_ACCESS_START_HOUR", "9"))
SAFETY_ACCESS_END_HOUR = int(os.environ.get("SAFETY_ACCESS_END_HOUR", "17"))
REPORTING_VLM_MODEL = os.environ.get("OPENAI_REPORTING_VLM_MODEL", "gpt-5.6")
REPORTING_VLM_DETAIL = os.environ.get("OPENAI_REPORTING_VLM_DETAIL", "high")
REPORTING_REASONING_EFFORT = os.environ.get("OPENAI_REPORTING_REASONING_EFFORT", "medium")
REPORTING_SITE_TIMEZONE = os.environ.get("REPORTING_SITE_TIMEZONE", SAFETY_SITE_TIMEZONE)
REPORTING_CAPTURE_POLL_SEC = max(
    1.0,
    float(os.environ.get("REPORTING_CAPTURE_POLL_SEC", "10")),
)
REPORTING_MAX_FRAME_AGE_SEC = max(
    1.0,
    float(os.environ.get("REPORTING_MAX_FRAME_AGE_SEC", "10")),
)
REPORTING_MAX_ANALYSIS_WORKERS = max(
    1,
    int(os.environ.get("REPORTING_MAX_ANALYSIS_WORKERS", "2")),
)
REPORTING_MAX_FRAMES_PER_CAMERA = max(
    2,
    int(os.environ.get("REPORTING_MAX_FRAMES_PER_CAMERA", "24")),
)
REPORTING_SNAPSHOT_DIR = os.environ.get("REPORTING_SNAPSHOT_DIR") or None
REPORTING_OUTPUT_DIR = os.environ.get("REPORTING_OUTPUT_DIR") or None
OPERATIONAL_STATE_PUBLISH_INTERVAL_SEC = max(
    0.5,
    float(os.environ.get("OPERATIONAL_STATE_PUBLISH_INTERVAL_SEC", "2.0")),
)
MAX_TOOL_ROUNDS = 6


def load_cameras() -> dict:
    """Load camera registry, including each camera's operator-facing zone."""
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
            "pi_port": int(cam.get("pi_port", 8000)),
        }
    return registry


CAMERAS = load_cameras()
ZONES: dict[str, list[int]] = {}
for _camera_id, _camera in sorted(CAMERAS.items()):
    ZONES.setdefault(_camera["zone"], []).append(_camera_id)
ZONE_NAMES = tuple(sorted(ZONES))
_ZONE_LOOKUP = {zone.casefold(): zone for zone in ZONE_NAMES}
OPERATIONAL_MODES = {"free", "safety", "search", "investigation"}
PLACEHOLDER_OPERATIONAL_MODES = {"investigation"}
_orchestration_lock = threading.RLock()
_operational_lock = threading.Lock()
_state_publisher_lock = threading.Lock()
_state_publisher_started = False
_startup_mode_lock = threading.Lock()
_startup_mode_initialized = False
_zone_states = {
    zone: {"mode": "free", "objective": None}
    for zone in ZONE_NAMES
}
search_scanners = {
    zone: SearchScanner(
        client=client,
        receiver_url=STREAM_RECEIVER_URL,
        model=SEARCH_VLM_MODEL,
        image_detail=SEARCH_VLM_DETAIL,
        match_threshold=SEARCH_MATCH_THRESHOLD,
        max_workers=max(1, min(4, len(camera_ids))),
        camera_ids=camera_ids,
        scope_label=zone,
    )
    for zone, camera_ids in ZONES.items()
}
safety_scanners = {
    zone: SafetyScanner(
        client=client,
        receiver_url=STREAM_RECEIVER_URL,
        model=SAFETY_VLM_MODEL,
        image_detail=SAFETY_VLM_DETAIL,
        reasoning_effort=SAFETY_REASONING_EFFORT,
        match_threshold=SAFETY_MATCH_THRESHOLD,
        max_workers=max(1, min(4, len(camera_ids))),
        site_timezone=SAFETY_SITE_TIMEZONE,
        access_start_hour=SAFETY_ACCESS_START_HOUR,
        access_end_hour=SAFETY_ACCESS_END_HOUR,
        camera_ids=camera_ids,
        scope_label=zone,
    )
    for zone, camera_ids in ZONES.items()
}


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
                "zone": info["zone"],
                "pi_host": info.get("pi_host"),
            }
            for cid, info in sorted(CAMERAS.items())
        ],
        "zones": [
            {"zone": zone, "camera_ids": list(camera_ids)}
            for zone, camera_ids in sorted(ZONES.items())
        ],
    }


def _post_receiver_json(path: str, payload: dict) -> bool:
    try:
        response = requests.post(f"{STREAM_RECEIVER_URL}{path}", json=payload, timeout=2)
        response.raise_for_status()
        return True
    except Exception:
        return False


def _scanner_running_for_zone(zone: str, mode: str) -> bool:
    if mode == "search":
        return search_scanners[zone].is_running()
    if mode == "safety":
        return safety_scanners[zone].is_running()
    return False


def _zone_state_payload_locked() -> list[dict[str, Any]]:
    return [
        {
            "zone": zone,
            "camera_ids": list(ZONES[zone]),
            "mode": state["mode"],
            "objective": state["objective"],
            "scanner_running": _scanner_running_for_zone(zone, state["mode"]),
            "placeholder": state["mode"] in PLACEHOLDER_OPERATIONAL_MODES,
        }
        for zone, state in sorted(_zone_states.items())
    ]


def _summarize_zone_payload(zone_states: list[dict[str, Any]]) -> dict[str, Any]:
    if not zone_states:
        return {
            "mode": "free",
            "objective": None,
            "scanner_running": False,
            "placeholder": False,
        }
    modes = {state["mode"] for state in zone_states}
    objectives = {state["objective"] for state in zone_states}
    if len(modes) == 1:
        mode = next(iter(modes))
        objective = next(iter(objectives)) if len(objectives) == 1 else None
    else:
        mode = "mixed"
        objective = None
    return {
        "mode": mode,
        "objective": objective,
        "scanner_running": any(state["scanner_running"] for state in zone_states),
        "placeholder": bool(zone_states) and all(state["placeholder"] for state in zone_states),
    }


def _publish_operational_state() -> bool:
    with _orchestration_lock:
        with _operational_lock:
            zone_states = _zone_state_payload_locked()
            payload = {**_summarize_zone_payload(zone_states), "zones": zone_states}
        return _post_receiver_json("/system/state", payload)


def _reconcile_operational_state() -> bool:
    """Publish desired state and resume a pending scanner after receiver acknowledgement."""
    with _orchestration_lock:
        receiver_synced = _publish_operational_state()
        if not receiver_synced:
            return False

        with _operational_lock:
            desired_states = {
                zone: dict(state)
                for zone, state in _zone_states.items()
            }
        scanner_started = False
        for zone, state in desired_states.items():
            mode = state["mode"]
            objective = state["objective"]
            if mode == "search" and objective and not search_scanners[zone].is_running():
                search_scanners[zone].start(objective)
                scanner_started = True
            elif mode == "safety" and not safety_scanners[zone].is_running():
                safety_scanners[zone].start()
                scanner_started = True
        return _publish_operational_state() if scanner_started else True


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
    capture_poll_sec=REPORTING_CAPTURE_POLL_SEC,
    max_frame_age_sec=REPORTING_MAX_FRAME_AGE_SEC,
    max_analysis_workers=REPORTING_MAX_ANALYSIS_WORKERS,
    max_frames_per_camera=REPORTING_MAX_FRAMES_PER_CAMERA,
    snapshot_root=REPORTING_SNAPSHOT_DIR,
    output_dir=REPORTING_OUTPUT_DIR,
    log_callback=_post_system_log,
)


def start_reporting_recorder() -> None:
    """Start passive minute-by-minute evidence capture once for the supervisor process."""
    reporting_service.start()


def _coerce_zone_names(raw_zones: Any = None) -> list[str]:
    if raw_zones is None:
        return list(ZONE_NAMES)
    if isinstance(raw_zones, str):
        clean = raw_zones.strip()
        if clean.casefold() in {"all", "*", "every", "all zones"}:
            return list(ZONE_NAMES)
        raw_values = [part.strip() for part in clean.split(",") if part.strip()]
    elif isinstance(raw_zones, (list, tuple, set)):
        raw_values = list(raw_zones)
    else:
        raw_values = [raw_zones]
    if not raw_values:
        raise ValueError("At least one zone is required.")

    resolved: list[str] = []
    unknown: list[str] = []
    for value in raw_values:
        clean = str(value or "").strip()
        zone = _ZONE_LOOKUP.get(clean.casefold())
        if zone is None:
            unknown.append(clean or "<empty>")
        elif zone not in resolved:
            resolved.append(zone)
    if unknown:
        raise ValueError(
            f"Unknown zone(s): {', '.join(unknown)}. Available zones: "
            + (", ".join(ZONE_NAMES) or "none")
            + "."
        )
    return resolved


def _set_cameras_default(camera_ids: list[int]) -> list[dict[str, Any]]:
    if not camera_ids:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(camera_ids))) as executor:
        raw_results = list(
            executor.map(lambda camera_id: call_pi_set_mode(camera_id, "default"), camera_ids)
        )
    return [
        {"camera_id": camera_id, "zone": CAMERAS[camera_id]["zone"], "result": result}
        for camera_id, result in zip(camera_ids, raw_results)
    ]


def get_operational_mode_tool(zones: Any = None) -> dict:
    try:
        selected_zones = _coerce_zone_names(zones)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}
    with _orchestration_lock:
        with _operational_lock:
            all_states = _zone_state_payload_locked()
            selected_states = [
                state for state in all_states if state["zone"] in selected_zones
            ]
    summary = _summarize_zone_payload(selected_states)
    return {
        "status": "ok",
        **summary,
        "zones": selected_states,
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
            "a Safety scanner detects a new hazard."
        ),
    }


def set_operational_mode(
    mode: str,
    objective: Optional[str] = None,
    zones: Any = None,
) -> dict:
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
    try:
        target_zones = _coerce_zone_names(zones)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}
    if not target_zones:
        return {"status": "error", "error": "No zones are configured."}

    with _orchestration_lock:
        return _set_operational_mode_locked(clean_mode, clean_objective, target_zones)


def _set_operational_mode_locked(
    clean_mode: str,
    clean_objective: Optional[str],
    target_zones: list[str],
) -> dict:
    with _operational_lock:
        previous_states = {
            zone: dict(_zone_states[zone])
            for zone in target_zones
        }
        for zone in target_zones:
            _zone_states[zone] = {
                "mode": clean_mode,
                "objective": clean_objective,
            }

    for zone in target_zones:
        search_scanners[zone].stop()
        safety_scanners[zone].stop()

    # Publish the new desired states with target scanners stopped. Search and
    # Safety only start after the receiver acknowledges every zone objective.
    _publish_operational_state()

    target_camera_ids = sorted(
        camera_id
        for zone in target_zones
        for camera_id in ZONES[zone]
    )
    camera_results = (
        _set_cameras_default(target_camera_ids)
        if clean_mode in {"free", "safety", "search"}
        else []
    )
    receiver_synced = _reconcile_operational_state()
    failed = [
        item for item in camera_results
        if item["result"].get("status") != "ok"
    ]
    with _operational_lock:
        selected_states = [
            state
            for state in _zone_state_payload_locked()
            if state["zone"] in target_zones
        ]
    scanner_running = any(state["scanner_running"] for state in selected_states)
    zone_text = ", ".join(target_zones)

    if clean_mode == "free":
        message = (
            f"Free Mode activated for {zone_text}. Automated scanning is stopped and "
            "the zone cameras are in default processing mode."
        )
    elif clean_mode == "safety":
        hours = (
            f"{SAFETY_ACCESS_START_HOUR:02d}:00-{SAFETY_ACCESS_END_HOUR:02d}:00 "
            f"{SAFETY_SITE_TIMEZONE}"
        )
        message = (
            f"Safety Mode activated for {zone_text}. From {hours}, the selected zone cameras "
            "are checked for Fire Hazard, Work-Zone Intrusion, and Obstacle Hazard. Obstacle "
            "detections are shown as silent dashboard warnings; outside that window the cameras "
            "are checked for Fire Hazard and Unauthorized Entry."
        )
    elif clean_mode == "investigation":
        message = (
            f"Investigation Mode selected for {zone_text}. Its camera-orchestration workflow "
            "is a placeholder."
        )
    else:
        was_search = any(state["mode"] == "search" for state in previous_states.values())
        verb = "Search target updated" if was_search else "Search Mode activated"
        message = f"{verb} for {zone_text}: {clean_objective}"

    if failed:
        failed_ids = ", ".join(str(item["camera_id"]) for item in failed)
        message += f" Camera(s) {failed_ids} could not be set to default processing mode."
        _post_system_log(
            kind="mode",
            level="warning",
            message=f"{clean_mode.title()} Mode could not reach camera(s): {failed_ids}.",
        )
    if not receiver_synced:
        if scanner_running:
            message += " Scanning is running, but its receiver status update is pending."
        elif clean_mode in {"search", "safety"}:
            message += " Scanning is waiting for the receiver to acknowledge the zone state."
        else:
            message += " The receiver status update is pending."

    _post_system_log(kind="mode", level="info", message=message)
    return {
        "status": "ok" if not failed and receiver_synced else "partial_error",
        "mode": clean_mode,
        "objective": clean_objective,
        "target_zones": target_zones,
        "zones": selected_states,
        "scanner_running": scanner_running,
        "placeholder": clean_mode in PLACEHOLDER_OPERATIONAL_MODES,
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
            "location, zone, and the Pi host it runs on. Call this when the user asks "
            "what cameras exist, or when a request is ambiguous about which camera to target."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "set_operational_mode",
        "description": (
            "Select the operational mode for one or more construction zones. If zones are omitted, "
            "the command applies to all zones. Free Mode stops Search and sets cameras in the selected "
            "zones to default processing. Search Mode configures those cameras "
            "for raw streaming and starts laptop-side VLM scanning for any visible target, including "
            "people, animals, vehicles, equipment, or other objects. Safety Mode configures cameras "
            "for raw streaming and scans each frame for construction hazards. Investigation Mode is "
            "currently a selectable placeholder. Progress reporting is a separate background capability, "
            "not an operational mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["free", "safety", "search", "investigation"],
                    "description": "The operational mode to select for the target zones.",
                },
                "objective": {
                    "type": "string",
                    "description": (
                        "The operator's objective. Required for Search Mode as a concise visual target "
                        "description, for example 'red fire extinguisher' or 'child in a blue jacket'. "
                        "Unused for Free and Safety Modes and optional for Investigation Mode."
                    ),
                },
                "zones": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(ZONE_NAMES)},
                    "description": (
                        "Target zone names. Omit only when the operator explicitly wants all zones."
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
            "Report operational mode, objective, placeholder state, and scanner status per zone. "
            "If zones are omitted, report every zone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "zones": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(ZONE_NAMES)},
                }
            },
        },
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
            "Check minute-by-minute construction-report snapshot coverage for one or more zones "
            "over a recent lookback interval without changing operational mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lookback_minutes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10080,
                    "description": "Recent duration to inspect in minutes. Defaults to 60.",
                },
                "zones": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(ZONE_NAMES)},
                    "description": "Optional zones to inspect; omit to inspect all zones.",
                },
            },
        },
    },
    {
        "type": "function",
        "name": "generate_progress_report",
        "description": (
            "Generate a separate PDF construction progress report for each requested zone from that "
            "zone's minute-by-minute snapshots over a requested recent duration. This does not switch "
            "operational mode. Convert hours to minutes, for example two hours is 120 minutes. Include "
            "the operator's stated goal when provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lookback_minutes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10080,
                    "description": "How far back from now the report should cover, in whole minutes.",
                },
                "goal": {
                    "type": "string",
                    "description": (
                        "Optional construction goal for that interval, copied faithfully from the operator."
                    ),
                },
                "zones": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(ZONE_NAMES)},
                    "description": "One or more zones; each receives its own PDF report.",
                },
            },
            "required": ["lookback_minutes", "zones"],
        },
    },
]


def _camera_roster_text() -> str:
    if not CAMERAS:
        return "No cameras are currently registered."
    lines = []
    for cid, info in sorted(CAMERAS.items()):
        loc = f" — {info['location']}" if info.get("location") else ""
        lines.append(
            f"  - id {cid}: {info['name']}{loc}; zone {info['zone']} (Pi {info.get('pi_host')})"
        )
    zone_lines = [
        f"  - {zone}: camera ids {', '.join(str(camera_id) for camera_id in camera_ids)}"
        for zone, camera_ids in sorted(ZONES.items())
    ]
    return (
        "Registered cameras:\n"
        + "\n".join(lines)
        + "\nConfigured zones:\n"
        + "\n".join(zone_lines)
    )


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
- Separate from per-camera processing modes, every construction zone has its
  own operational mode: `free`, `safety`, `search`, or `investigation`. Every
  zone starts in Free Mode, and different zones can run different modes at the
  same time.
- Use `set_operational_mode` whenever the operator asks to enter or switch an
  operational mode. Pass exactly the requested zone names. If the operator
  explicitly says all zones, omit `zones` or pass every configured zone. Never
  change an unmentioned zone. Use `get_operational_mode` when asked which modes
  and objectives are active.
- Search Mode is the implemented zone-wide visual-search workflow. When the
  operator explicitly asks for Search Mode, or asks the system to find or
  locate a described target across cameras, call `set_operational_mode` with
  mode `search`, the requested zones, and a concise objective copied from the request. The target can
  be any visibly identifiable person, animal, vehicle, equipment, or object.
  Do not ask for confirmation when a target description is already present; if
  it is missing, ask what the system should search for.
- In Search Mode, cameras in that zone remain in `default` processing mode
  while the laptop-side VLM scanner inspects active streams. Reject requests to
  move individual cameras to other processing modes until Search Mode is left.
- Safety Mode is the implemented continuous construction-safety workflow for
  the selected zone. It keeps that zone's reachable cameras in `default`
  processing mode and evaluates each
  sampled frame once against every applicable check. During the configured
  09:00-17:00 site-local construction window, those checks are `Fire Hazard`,
  `Work-Zone Intrusion`, and `Obstacle Hazard`. Work-Zone Intrusion requires a
  recognizable white or light-colored leveling machine and a person close enough
  to share its immediate working space or likely path of movement. It becomes a
  hazard only when that nearby person is visibly missing a high-visibility safety
  vest, a protective hard hat or helmet, or both. A nearby person clearly wearing
  both required items is compliant and does not trigger this check. It also excludes
  someone clearly distant or only in the background and the normal operator properly
  seated at the machine. After hours, PPE does not exempt a person from Unauthorized
  Entry.
  Obstacle Hazard requires that same recognizable machine plus a substantial obstacle,
  such as a traffic cone or a large green, blue, or white pipe on the ground, visibly
  close to the machine or directly in its apparent travel or working path. Do not treat
  an object elsewhere in the frame, ordinary terrain, markings, shadows, small debris,
  or machine parts as an Obstacle Hazard.
  Outside that window, machinery is considered off,
  so the checks are `Fire Hazard` and `Unauthorized Entry`. A frame can
  trigger more than one detection. An Obstacle Hazard result is presented as a
  yellow/orange, silent dashboard warning and is recorded with its cause and
  triggering frame; it does not latch the construction safety state. `Fire
  Hazard`, `Work-Zone Intrusion`, and `Unauthorized Entry` remain audible red
  STOP WORK alerts that latch the construction safety state red.
- A latched hazard survives operational-mode changes.
  Call `clear_safety_hazard` only when the operator explicitly asks to clear,
  reset, or acknowledge the safety state. Never clear it merely because the
  operator selects another mode. Use `get_safety_state` when asked whether the
  construction site is currently clear or which hazards are active.
- In Safety Mode, cameras in that zone are locked to `default` processing so the Safety
  scanner continues receiving unmodified frames. Reject requests to move an
  individual camera to another processing mode until Safety Mode is left.
- Free Mode is the neutral live-view baseline for a zone. Selecting it stops
  Search and Safety scanning in the selected zones, clears their Search
  objectives, and sets their reachable cameras to `default` processing mode.
  Free Mode does
  not clear a latched construction safety hazard.
- Investigation Mode is a per-zone placeholder until its workflow is defined.
  Still call `set_operational_mode` when it is requested, then clearly explain that
  no automatic camera reconfiguration is configured for it yet.
- Progress reporting is not an operational mode and never interrupts the active
  operational mode. A passive recorder saves one fresh frame per registered
  camera during every clock minute, all day, in {REPORTING_SITE_TIMEZONE}. Use
  `get_reporting_status` when asked what recent evidence is available. The
  snapshots persist under {reporting_service.snapshot_root}, organized by date,
  zone, camera, and minute; state this path when the operator asks where the
  captured evidence is stored.
- Use `generate_progress_report` when the operator asks for a construction
  report covering a recent duration. A zone and duration are required. Convert
  spoken durations to whole minutes: five minutes is 5, two hours is 120, and
  one day is 1440. Pass every requested zone and create a separate PDF for each.
  Pass the operator's goal exactly when one is provided. Do not ask for a goal
  because it is optional. The tool compares each camera's ordered interval sequence,
  synthesizes the camera observations, and creates a PDF. In text chat, present
  the returned `report_url` as a Markdown link. If someone asks to enter
  Reporting Mode, explain that reporting now runs in the background and ask
  which zone and recent duration they want instead of changing operational mode.
- If the operator asks to stop or cancel Search Mode without naming a next
  operational mode, switch the referenced zones to Free Mode. Ask which zone
  only when it cannot be inferred; do not stop Search in other zones.
- If the user targets a camera that is not registered, politely say so and list
  the available camera ids.
- After any camera tool result, explain clearly what happened. On success,
  confirm the camera (by id and name) and processing mode. On error, explain
  the problem in simple terms and suggest what to try.
- After an operational-mode result, confirm the selected zones, mode, and objective.
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
  new processing mode. When it changes operational mode, confirm the zones,
  mode, and objective. When a command is ambiguous, ask a short
  clarifying question.
"""


def get_reporting_status_for_zones(
    lookback_minutes: int = 60,
    zones: Any = None,
) -> dict:
    selected_zones = _coerce_zone_names(zones)
    statuses = [
        reporting_service.get_status(lookback_minutes, zone)
        for zone in selected_zones
    ]
    return {
        "status": "ok",
        "lookback_minutes": lookback_minutes,
        "zones": statuses,
        "message": f"Reporting coverage returned for {len(statuses)} zone(s).",
    }


def generate_progress_reports_for_zones(
    lookback_minutes: int,
    zones: Any,
    goal: Optional[str] = None,
) -> dict:
    if zones is None:
        raise ValueError("At least one zone is required to generate a progress report.")
    selected_zones = _coerce_zone_names(zones)
    reports = [
        reporting_service.generate_interval_report(lookback_minutes, zone, goal)
        for zone in selected_zones
    ]
    ok_count = sum(report.get("status") == "ok" for report in reports)
    usable_count = sum(report.get("status") in {"ok", "partial_error"} for report in reports)
    if ok_count == len(reports):
        status = "ok"
    elif usable_count:
        status = "partial_error"
    else:
        status = "error"
    result = {
        "status": status,
        "lookback_minutes": lookback_minutes,
        "target_zones": selected_zones,
        "reports": reports,
        "message": (
            f"Generated {usable_count} of {len(reports)} requested zone report(s)."
        ),
    }
    if len(reports) == 1:
        result.update(reports[0])
        result["reports"] = reports
    return result


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
                locked_zones = {
                    CAMERAS[camera_id]["zone"]: _zone_states[CAMERAS[camera_id]["zone"]]["mode"]
                    for camera_id in camera_ids
                    if camera_id in CAMERAS
                    and _zone_states[CAMERAS[camera_id]["zone"]]["mode"] in {"search", "safety"}
                }
            if locked_zones and mode != "default":
                details = ", ".join(
                    f"{zone} ({operational_mode.title()} Mode)"
                    for zone, operational_mode in sorted(locked_zones.items())
                )
                return {
                    "status": "error",
                    "error": (
                        f"Camera processing is locked to default in: {details}. "
                        "Change those zone modes first."
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
            args.get("zones", args.get("zone")),
        )

    if tool_name == "get_operational_mode":
        return get_operational_mode_tool(args.get("zones", args.get("zone")))

    if tool_name == "get_safety_state":
        return get_safety_state_tool()

    if tool_name == "clear_safety_hazard":
        return clear_safety_hazard(args.get("reason"))

    if tool_name == "get_reporting_status":
        try:
            return get_reporting_status_for_zones(
                args.get("lookback_minutes", 60),
                args.get("zones", args.get("zone")),
            )
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

    if tool_name == "generate_progress_report":
        try:
            return generate_progress_reports_for_zones(
                args.get("lookback_minutes"),
                args.get("zones", args.get("zone")),
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
    async def download_progress_report(filename: str):
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
.sup-mode-chip.mixed {
  border-color: rgba(168, 85, 247, 0.48);
  background: rgba(126, 34, 206, 0.16);
  color: #d8b4fe;
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
        const mode = ["free", "safety", "search", "investigation", "mixed"].includes(state.mode)
          ? state.mode
          : "free";
        const hazard = state.safety_status === "hazard";
        chip.classList.remove("free", "safety", "search", "investigation", "mixed", "hazard");
        chip.classList.add(mode);
        if (hazard) chip.classList.add("hazard");
        const modeLabel = mode.charAt(0).toUpperCase() + mode.slice(1);
        label.textContent = hazard ? "Stop work · " + modeLabel : modeLabel;
        const zoneStates = Array.isArray(state.zones) ? state.zones : [];
        chip.title = zoneStates.map((item) => {
          const objective = item.objective ? " - " + item.objective : "";
          return item.zone + ": " + item.mode + objective;
        }).join("\n");
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
    <p class="sup-sub">Natural-language control for your multi-camera zones — select modes, configure cameras, or read status.</p>
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
        line2_parts.append(html.escape(str(info.get("zone") or "Unassigned")))
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
