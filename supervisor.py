import warnings
warnings.filterwarnings("ignore")

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


def build_ui():
    with gr.Blocks(title="Supervisor Agent") as demo:
        gr.Markdown("# Supervisor Agent")
        if CAMERAS:
            roster_md = "\n".join(
                f"- **id {cid}** — {info['name']}"
                + (f" ({info['location']})" if info.get("location") else "")
                + f" · Pi `{info.get('pi_host')}`"
                for cid, info in sorted(CAMERAS.items())
            )
            gr.Markdown(f"**Registered cameras**\n\n{roster_md}")
        else:
            gr.Markdown("_No cameras registered in `cameras.json`._")

        chatbot = gr.Chatbot(label="Supervisor Agent", height=600)

        with gr.Row():
            user_in = gr.Textbox(
                placeholder="Give a command or chat...",
                label="Your Message",
                show_label=False,
                scale=8,
            )
            send_btn = gr.Button("Send", scale=1)

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
