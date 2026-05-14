import warnings
warnings.filterwarnings("ignore")

import html
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response
import gradio as gr
import requests
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


CAMERAS_FILE = Path(__file__).with_name("cameras.json")
REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
REALTIME_VOICE = os.environ.get("OPENAI_REALTIME_VOICE", "marin")
REALTIME_CALLS_URL = "https://api.openai.com/v1/realtime/calls"


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
            "Each camera has a unique integer id and lives on its own Pi; the "
            "supervisor routes the command to the correct device."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "camera_id": {"type": "integer", "description": "Unique integer id of the camera."},
                "mode": {
                    "type": "string",
                    "enum": ["surveillance", "construction", "idle"],
                },
            },
            "required": ["camera_id", "mode"],
        },
    },
    {
        "type": "function",
        "name": "get_camera_state",
        "description": "Get the current operating mode and runtime state of a specific camera.",
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
You are a supervisor assistant for a multi-camera smart construction monitoring system.
Each camera has a unique integer id and is connected to its own Raspberry Pi.

{_camera_roster_text()}

Behaviour rules:
- Users give natural language requests (e.g. "switch camera 1 to surveillance mode",
  "what is camera 0 doing?", "which cameras do we have?").
- Use `set_camera_mode` to change the processing mode of a specific camera.
- Use `get_camera_state` to report the current mode / status of a specific camera.
- Use `list_cameras` when the user asks what cameras exist, or when a request
  does not specify which camera_id to act on and you need to list the options.
- If the user targets a camera that is not registered, politely say so and list
  the available camera ids.
- After any tool result, explain clearly what happened:
  - On success, confirm the camera (by id and name) and the mode.
  - On error, explain the error in simple terms and suggest what to try.
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
  new mode. When a command is ambiguous, ask a short clarifying question.
"""


def execute_camera_tool(tool_name: str, args) -> dict:
    """Execute one of the camera tools for realtime voice sessions."""
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
            cam_id = int(args["camera_id"])
            mode = args["mode"]
        except (KeyError, ValueError, TypeError) as e:
            return {
                "status": "error",
                "error": f"Missing or invalid argument for set_camera_mode: {str(e)}",
            }
        return call_pi_set_mode(cam_id, mode)

    if tool_name == "get_camera_state":
        try:
            cam_id = int(args["camera_id"])
        except (KeyError, ValueError, TypeError) as e:
            return {
                "status": "error",
                "error": f"Missing or invalid argument for get_camera_state: {str(e)}",
            }
        return call_pi_get_state(cam_id)

    if tool_name == "list_cameras":
        return list_cameras_tool()

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
                    "create_response": True,
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

        result = execute_camera_tool(tool_name, tool_args)
        return JSONResponse(
            {
                "call_id": payload.get("call_id"),
                "name": tool_name,
                "result": result,
            }
        )


def supervisor_step(user_msg, chat_history, conversation):
    """Main logic for each user message."""
    if conversation is None or len(conversation) == 0:
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    conversation = trim_conversation(conversation)
    conversation.append({"role": "user", "content": user_msg})

    response = client.responses.create(
        model="gpt-5-mini",
        input=conversation,
        tools=tools,
    )

    tool_call = None
    for item in response.output:
        if item.type == "function_call":
            tool_call = item
            break

    if not tool_call:
        assistant_text = response.output_text
        conversation.append({"role": "assistant", "content": assistant_text})
        chat_history.append((user_msg, assistant_text))
        return chat_history, conversation

    try:
        args = json.loads(tool_call.arguments) if tool_call.arguments else {}
    except Exception as e:
        assistant_text = f"Error parsing tool arguments: {str(e)}"
        conversation.append({"role": "assistant", "content": assistant_text})
        chat_history.append((user_msg, assistant_text))
        return chat_history, conversation

    tool_name = getattr(tool_call, "name", None)
    tool_result = None

    if tool_name == "set_camera_mode":
        try:
            cam_id = int(args["camera_id"])
            mode = args["mode"]
        except (KeyError, ValueError, TypeError) as e:
            assistant_text = f"Missing or invalid argument for set_camera_mode: {str(e)}"
            conversation.append({"role": "assistant", "content": assistant_text})
            chat_history.append((user_msg, assistant_text))
            return chat_history, conversation
        tool_result = call_pi_set_mode(cam_id, mode)
    elif tool_name == "get_camera_state":
        try:
            cam_id = int(args["camera_id"])
        except (KeyError, ValueError, TypeError) as e:
            assistant_text = f"Missing or invalid argument for get_camera_state: {str(e)}"
            conversation.append({"role": "assistant", "content": assistant_text})
            chat_history.append((user_msg, assistant_text))
            return chat_history, conversation
        tool_result = call_pi_get_state(cam_id)
    elif tool_name == "list_cameras":
        tool_result = list_cameras_tool()
    else:
        assistant_text = f"Unknown tool call: {tool_name}"
        conversation.append({"role": "assistant", "content": assistant_text})
        chat_history.append((user_msg, assistant_text))
        return chat_history, conversation

    conversation.extend(response.output)
    conversation.append({
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": json.dumps(tool_result),
    })

    followup = client.responses.create(
        model="gpt-5-mini",
        input=conversation,
        tools=tools,
    )

    assistant_text = followup.output_text
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
  letter-spacing: -0.02em;
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
      audioEl: null,
      handledCalls: new Set(),
      pendingCalls: new Map(),
      lastAssistantText: "",
      assistantSpeaking: false,
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
    }

    function clearInputBuffer() {
      sendRealtimeEvent({ type: "input_audio_buffer.clear" });
    }

    function muteMicForAssistant() {
      state.assistantSpeaking = true;
      if (state.resumeMicTimer) {
        window.clearTimeout(state.resumeMicTimer);
        state.resumeMicTimer = null;
      }
      setMicEnabled(false);
      clearInputBuffer();
    }

    function resumeMicAfterAssistant() {
      state.assistantSpeaking = false;
      if (state.resumeMicTimer) window.clearTimeout(state.resumeMicTimer);
      state.resumeMicTimer = window.setTimeout(() => {
        clearInputBuffer();
        if (state.active) setMicEnabled(true);
        state.resumeMicTimer = null;
      }, 550);
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
      setStatus("Running camera command", "working", name);

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
          throw new Error(payload.error || "Camera tool call failed.");
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
        if (state.assistantSpeaking) {
          clearInputBuffer();
          return;
        }
        setStatus("Listening", "active", "Microphone is live.");
        return;
      }
      if (event.type === "input_audio_buffer.speech_stopped") {
        if (state.assistantSpeaking) {
          clearInputBuffer();
          return;
        }
        setStatus("Thinking", "working", "Processing your command.");
        return;
      }
      if (event.type === "input_audio_buffer.committed" && state.assistantSpeaking) {
        clearInputBuffer();
        return;
      }
      if (event.type === "conversation.item.input_audio_transcription.completed") {
        if (state.assistantSpeaking) return;
        appendEntry("You", event.transcript || "");
        return;
      }
      if (event.type === "response.created") {
        muteMicForAssistant();
        setStatus("Supervisor speaking", "working", "Microphone is paused to prevent speaker echo.");
        return;
      }
      if (event.type === "response.output_audio.delta" || event.type === "response.audio.delta") {
        muteMicForAssistant();
        return;
      }
      if (event.type === "response.output_audio.done" || event.type === "response.audio.done") {
        resumeMicAfterAssistant();
        return;
      }
      if (event.type === "response.audio_transcript.done" || event.type === "response.output_audio_transcript.done") {
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
        resumeMicAfterAssistant();
        if (state.active) setStatus("Listening", "active", "Microphone is live.");
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
      if (state.dc) state.dc.close();
      if (state.pc) state.pc.close();
      if (state.micStream) state.micStream.getTracks().forEach((track) => track.stop());
      if (state.audioEl) state.audioEl.srcObject = null;
      state.active = false;
      state.starting = false;
      state.assistantSpeaking = false;
      state.pc = null;
      state.dc = null;
      state.micStream = null;
      state.micTrack = null;
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

        micStream.getTracks().forEach((track) => pc.addTrack(track, micStream));
        pc.ontrack = (event) => {
          audioEl.srcObject = event.streams[0];
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
        state.micTrack = micStream.getAudioTracks()[0] || null;

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

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initVoiceClient);
  } else {
    initVoiceClient();
  }
}
"""


def _supervisor_header_html() -> str:
    return """<header class="sup-page-header">
  <div class="sup-title-block">
    <h1>Supervisor</h1>
    <p class="sup-sub">Natural-language control for your multi-camera fleet — switch modes, read status, or list registered devices.</p>
  </div>
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
    with gr.Blocks(
        title="Supervisor Agent",
        theme=_supervisor_theme(),
        css=SUPERVISOR_CSS,
        js=SUPERVISOR_JS,
        fill_width=True,
    ) as demo:
        gr.HTML(_supervisor_header_html())
        gr.HTML(_camera_roster_html())
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
