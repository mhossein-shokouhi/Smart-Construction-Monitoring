# Smart Construction Monitoring

An AI-agent–based orchestration system for monitoring construction sites with a
network of edge-AI smart cameras. A natural-language **supervisor agent** on the
operator's laptop interprets high-level prompts (e.g. *"switch the front gate
camera to surveillance mode"*) and autonomously reconfigures a fleet of
Raspberry Pi–backed cameras in real time. The processed video is streamed back
to the laptop and displayed in a single, elegant dashboard.

---

## How it works

```
 ┌────────────────────────────┐         commands          ┌─────────────────────────────┐
 │         Laptop (you)       │  ───────────────────────▶ │     Raspberry Pi + IMX500   │
 │                            │      HTTP (FastAPI)       │        (one per camera)     │
 │  supervisor.py             │                           │                             │
 │   └─ LLM + Gradio UI       │                           │  agent_actuator.py          │
 │                            │                           │   └─ launches on demand:    │
 │  stream_receiver_server.py │  ◀─────────────────────── │      • object_detection_*   │
 │   └─ Dashboard + MJPEG     │    processed JPEG frames  │      • segmentation_*       │
 └────────────────────────────┘                           └─────────────────────────────┘
```

1. The operator types a natural-language command into the **supervisor** UI on
   the laptop.
2. The supervisor (GPT-backed) decides what to do and calls the correct
   Raspberry Pi over HTTP using the camera registry in `cameras.json`.
3. The Pi's **agent actuator** starts the appropriate edge-AI pipeline on its
   IMX500 camera:
   - **Surveillance mode** → on-device **object detection** (`object_detection_demo.py`)
   - **Construction mode** → on-device **semantic segmentation** (`segmentation_demo_overlay.py`)
   - **Idle mode** → stop all inference
4. Each processed frame is annotated on the Pi and streamed back to the
   laptop's **stream receiver**, which exposes a live dashboard at
   `http://<laptop-ip>:9000` with per-camera views, latency / FPS metrics, and
   an event log.

Running inference at the edge keeps bandwidth low and latency predictable, so
the whole system scales to many cameras over a single wireless link (in our
case, Rogers 5G).

---

## Repository layout

| Path | Role |
| --- | --- |
| `supervisor.py` | LLM supervisor agent + Gradio UI (runs on the laptop) |
| `stream_receiver_server.py` | Receives streams from all Pis and serves the dashboard (runs on the laptop) |
| `agent_actuator.py` | FastAPI service that runs on each Raspberry Pi |
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
FPS metrics, and an event log. Cameras appear here as soon as their Pi starts
streaming frames.

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
- *"Put camera 0 in surveillance mode."*
- *"Switch the warehouse camera to construction mode."*
- *"What is camera 1 currently doing?"*
- *"Set all cameras to idle."*

The supervisor translates each request into the right HTTP call to the
corresponding Pi, reports back in natural language, and the dashboard updates
automatically as soon as frames start arriving.

---

## Operational modes

| Mode | What runs on the Pi | Typical use |
| --- | --- | --- |
| `surveillance` | Object detection (e.g. NanoDet / MobileNet SSD on the IMX500) | Spotting people, vehicles, and abnormal activity |
| `construction` | Semantic segmentation (DeepLabV3+ on the IMX500) | Extracting machinery / site structure for digital-twin updates |
| `idle` | No inference; camera process stopped | Saving power / bandwidth |

All three modes are selectable from the supervisor prompt — you never need to
SSH into a Pi to change them.

---

## Troubleshooting

- **"Unknown camera_id" from the supervisor** – the id isn't in `cameras.json`,
  or the `pi_host` is wrong. Fix the file and restart `supervisor.py`.
- **Dashboard stays blank** – confirm `STREAM_SERVER_URL` on the Pi points to
  the laptop's reachable IP (not `localhost`) and that port `9000` isn't
  blocked by a firewall.
- **Model fails to load on the Pi** – make sure `imx500-all` is installed and
  that the `.rpk` file under `models/` matches the mode you requested.
- **High latency / jitter** – run `performance_metrics/ping_metrics.py
  <pi-ip>` from the laptop to measure the raw network characteristics of the
  link.

---

## License

Released for research and demonstration purposes. See repository settings for
the current license.
