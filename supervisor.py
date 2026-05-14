import warnings
warnings.filterwarnings("ignore")

import html
import json
import os
from pathlib import Path
from typing import Optional

import gradio as gr
import requests
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


CAMERAS_FILE = Path(__file__).with_name("cameras.json")


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


def build_ui():
    with gr.Blocks(
        title="Supervisor Agent",
        theme=_supervisor_theme(),
        css=SUPERVISOR_CSS,
        fill_width=True,
    ) as demo:
        gr.HTML(_supervisor_header_html())
        gr.HTML(_camera_roster_html())

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

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
