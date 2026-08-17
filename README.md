# Smart Construction Monitoring

An AI-agent–based orchestration system for monitoring construction sites with a
network of edge-AI smart cameras. A natural-language **supervisor agent** on the
operator's laptop interprets high-level prompts (e.g. *"switch the front gate
camera to surveillance mode"*) and autonomously reconfigures a fleet of
Raspberry Pi–backed cameras in real time. Video is streamed back to the laptop
and displayed in a single, elegant dashboard.

---

## How it works

```
 ┌────────────────────────────┐         commands          ┌─────────────────────────────┐
 │         Laptop (you)       │  ───────────────────────▶ │     Raspberry Pi + IMX500   │
 │                            │      HTTP (FastAPI)       │        (one per camera)     │
 │  supervisor.py             │                           │                             │
 │   └─ LLM + Gradio UI       │                           │  agent_actuator.py          │
 │                            │                           │   └─ launches on demand:    │
 │  stream_receiver_server.py │  ◀─────────────────────── │      • raw_stream_*         │
 │   └─ Dashboard + MJPEG     │       JPEG frames         │      • object_detection_*   │
 │                            │                           │      • segmentation_*       │
 └────────────────────────────┘                           └─────────────────────────────┘
```

1. The operator types a natural-language command into the **supervisor** UI on
   the laptop.
2. The supervisor (GPT-backed) decides what to do and calls the correct
   Raspberry Pi over HTTP using the camera registry in `cameras.json`.
3. The Pi's **agent actuator** starts the appropriate camera pipeline:
   - **Default mode** → raw camera footage streamed to the laptop (`raw_stream_demo.py`)
   - **Surveillance mode** → on-device **object detection** (`object_detection_demo.py`)
   - **Construction mode** → on-device **semantic segmentation** (`segmentation_demo_overlay.py`)
   - **Idle mode** → stop all inference
4. Frames are streamed back to the laptop's **stream receiver**, which exposes
   a live dashboard at `http://<laptop-ip>:9000` with per-camera views, latency
   / FPS metrics, and an event log.

Running inference at the edge keeps bandwidth low and latency predictable, so
the whole system scales to many cameras over a single wireless link (in our
case, Rogers 5G).

---

## Repository layout

| Path | Role |
| --- | --- |
| `supervisor.py` | LLM supervisor agent + Gradio UI (runs on the laptop) |
| `emergency_vlm.py` | Laptop-side emergency scanner that samples active streams and queries the VLM |
| `stream_receiver_server.py` | Receives streams from all Pis and serves the dashboard (runs on the laptop) |
| `agent_actuator.py` | FastAPI service that runs on each Raspberry Pi |
| `raw_stream_demo.py` | Default-mode pipeline (raw Picamera2 stream, no inference overlays) |
| `object_detection_demo.py` | Surveillance-mode pipeline (IMX500 object detection) |
| `segmentation_demo_overlay.py` | Construction-mode pipeline (IMX500 semantic segmentation) |
| `cameras.json` | Registry of cameras → Pi hosts / ports |
| `models/` | IMX500 `.rpk` network packages used by the demos |
| `assets/` | Label files and colour palette for the demos |
| `performance_metrics/ping_metrics.py` | Helper script for measuring RTT / jitter on the network link |
| `requirements-laptop.txt` | Python deps for the laptop components |
| `requirements-pi.txt` | Python deps for the Raspberry Pi components |

---

## Prerequisites

**Laptop**
- Python 3.10+
- An OpenAI API key (exported as `OPENAI_API_KEY`)
- Network reachability to every Raspberry Pi listed in `cameras.json`

**Each Raspberry Pi** (one per camera)
- Raspberry Pi 5 running Raspberry Pi OS (Bookworm)
- Sony **IMX500** intelligent-vision camera module
- System packages for the IMX500 stack:
  ```bash
  sudo apt update
  sudo apt install -y python3-picamera2 imx500-all
  ```
- Python 3.10+

---

## 1 · Configure the camera registry

Edit `cameras.json` so it describes every camera in your deployment. Each
entry's `pi_host` / `pi_port` must point at the Raspberry Pi that hosts that
camera:

```json
{
  "cameras": [
    { "id": 0, "name": "Camera 0", "location": "Front Gate", "pi_host": "192.168.1.50", "pi_port": 8000 },
    { "id": 1, "name": "Camera 1", "location": "Warehouse",  "pi_host": "192.168.1.51", "pi_port": 8000 }
  ]
}
```

Both the supervisor and the stream receiver on the laptop read this file, so
keep them in sync.

---

## 2 · Set up each Raspberry Pi

Clone the repo onto the Pi and install its dependencies:

```bash
git clone https://github.com/<your-username>/Smart-Construction-Monitoring.git
cd Smart-Construction-Monitoring
pip install -r requirements-pi.txt
```

Start the actuator, telling it which camera id this Pi serves and where to
push frames (replace `<laptop-ip>` with your laptop's LAN address):

```bash
CAMERA_ID=0 \
CAMERA_NAME="Front Gate" \
STREAM_SERVER_URL="http://<laptop-ip>:9000" \
python -m uvicorn agent_actuator:app --host 0.0.0.0 --port 8000
```

Environment variables:

| Variable | Purpose |
| --- | --- |
| `CAMERA_ID` | Integer id this Pi is responsible for (must match `cameras.json`) |
| `CAMERA_NAME` | Human-readable name (optional) |
| `STREAM_SERVER_URL` | URL of the laptop's stream receiver, e.g. `http://192.168.1.100:9000` |

Repeat on every Pi, bumping `CAMERA_ID` each time.

Sanity check from the laptop:

```bash
curl http://<pi-ip>:8000/health
# {"status":"ok","camera_id":0,"camera_name":"Front Gate"}
```

---

## 3 · Start the stream receiver on the laptop

In one terminal:

```bash
pip install -r requirements-laptop.txt
python stream_receiver_server.py --host 0.0.0.0 --port 9000
```

Open the dashboard in a browser:

```
http://localhost:9000
```

You'll see a camera selector, a live MJPEG view, per-camera delay / jitter /
FPS metrics, and an event log. Choose **Grid View** at the top of the camera
selector to monitor every known camera at once in a responsive grid. Inactive
cameras remain visible as black **No stream** tiles and automatically reconnect
when their Pi starts sending frames.

---

## 4 · Start the supervisor agent on the laptop

In a second terminal:

```bash
export OPENAI_API_KEY="sk-..."
python supervisor.py
```

Gradio will open a local web UI (by default `http://127.0.0.1:7860`). Chat with
the supervisor in plain English:

- *"List the cameras."*
- *"Put camera 0 in default mode."*
- *"Put camera 0 in surveillance mode."*
- *"Switch the warehouse camera to construction mode."*
- *"What is camera 1 currently doing?"*
- *"Set all cameras to idle."*
- *"Emergency: find the missing worker in a red hard hat and yellow vest."*
- *"Clear the emergency; the worker has been found."*

The supervisor translates each request into the right HTTP call to the
corresponding Pi, reports back in natural language, and the dashboard updates
automatically as soon as frames start arriving.

### Voice chat mode

The supervisor UI also includes a **Start voice chat** button. Click it, allow
microphone access, and speak commands such as:

- *"Put camera 2 in construction mode."*
- *"Switch the front gate camera to default mode."*
- *"What cameras are available?"*
- *"Set the front gate camera back to idle."*

Voice chat uses OpenAI's Realtime API over WebRTC. The browser sends microphone
audio directly through a peer connection and receives natural spoken audio back;
`supervisor.py` only performs the secure session handshake and executes camera
tool calls, so your standard `OPENAI_API_KEY` is never exposed to the browser.

Optional voice settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_REALTIME_MODEL` | `gpt-realtime` | Realtime speech-to-speech model used by the voice agent. Use `gpt-realtime-2` if it is enabled for your OpenAI project. |
| `OPENAI_REALTIME_VOICE` | `marin` | Spoken voice returned by the Realtime model |
| `OPENAI_EMERGENCY_VLM_MODEL` | `gpt-5.5` | Vision-capable model used by the laptop-side emergency scanner |
| `OPENAI_EMERGENCY_VLM_DETAIL` | `high` | Image detail sent to the emergency VLM (`low`, `high`, or `auto`) |
| `EMERGENCY_MATCH_THRESHOLD` | `0.75` | Minimum VLM confidence required before the dashboard raises an alert |
| `STREAM_RECEIVER_URL` | `http://127.0.0.1:9000` | Receiver URL used by the supervisor to read active streams and publish system logs |

Microphone access works on `localhost` / `127.0.0.1` in modern browsers. If you
open the supervisor from another device, serve it over HTTPS so the browser will
allow microphone capture.

---

## Operational modes

| Mode | What runs on the Pi | Typical use |
| --- | --- | --- |
| `default` | Raw Picamera2 stream, no object detection or bounding boxes | Viewing unprocessed live camera footage on the laptop |
| `surveillance` | Object detection (e.g. NanoDet / MobileNet SSD on the IMX500) | Spotting people, vehicles, and abnormal activity |
| `construction` | Semantic segmentation (DeepLabV3+ on the IMX500) | Extracting machinery / site structure for digital-twin updates |
| `idle` | No inference; camera process stopped | Saving power / bandwidth |

All modes are selectable from the supervisor prompt — you never need to
SSH into a Pi to change them.

## Agentic modes

| Mode | Behaviour |
| --- | --- |
| `free` | Default state. Each camera can be controlled independently through the supervisor. |
| `emergency` | Activated when the operator reports an emergency or asks the supervisor to search for a missing person / person of interest. The supervisor commands every reachable camera into `default` mode and starts `emergency_vlm.py` on the laptop. |

While Emergency mode is active, the scanner asks the receiver which streams are
currently live, samples one latest frame per active camera on each pass, and
sends those frames to the configured VLM with the operator's visual-search
intent. Camera feeds remain visible in the receiver dashboard. The dashboard now
includes a **System Logs** tab; when the VLM reports a match above the configured
threshold, that tab records a bold alert, stores the triggering frame, and plays
an alarm sound in the browser.

---

## Troubleshooting

- **"Unknown camera_id" from the supervisor** – the id isn't in `cameras.json`,
  or the `pi_host` is wrong. Fix the file and restart `supervisor.py`.
- **Dashboard stays blank** – confirm `STREAM_SERVER_URL` on the Pi points to
  the laptop's reachable IP (not `localhost`) and that port `9000` isn't
  blocked by a firewall.
- **Model fails to load on the Pi** – make sure `imx500-all` is installed and
  that the `.rpk` file under `models/` matches the mode you requested.
- **`ValueError: numpy.dtype size changed ... Expected 96 from C header, got 88 from PyObject`**
  on the Pi when switching to surveillance / construction mode – this means
  NumPy 2.x got installed alongside the apt-provided `picamera2` /
  `simplejpeg`, which on Raspberry Pi OS Bookworm are compiled against
  NumPy 1.x. Pin NumPy back:
  ```bash
  pip install "numpy<2" --force-reinstall
  ```
  (`requirements-pi.txt` already pins `numpy<2`, so a fresh
  `pip install -r requirements-pi.txt` will do the right thing.)
- **High latency / jitter** – run `performance_metrics/ping_metrics.py
  <pi-ip>` from the laptop to measure the raw network characteristics of the
  link.

---

## License

Released for research and demonstration purposes. See repository settings for
the current license.
