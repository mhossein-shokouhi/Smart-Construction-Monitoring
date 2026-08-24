# Agentic Multi-Camera Operations

An AI-agent–based orchestration system for coordinating a network of edge-AI
smart cameras. A natural-language **supervisor agent** on the operator's laptop
interprets high-level objectives, selects operational modes for one or more
construction zones, and
reconfigures Raspberry Pi–backed cameras in real time. Video is streamed back
to the laptop and displayed in a single dashboard.

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

1. The operator gives the **supervisor** a natural-language objective or camera
   command from the laptop UI.
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
| `search_vlm.py` | Laptop-side Search Mode scanner that samples active streams and queries the VLM |
| `safety_vlm.py` | Laptop-side Safety Mode scanner that evaluates each sampled frame against all applicable construction hazards in one VLM request |
| `reporting.py` | Passive hourly snapshot recorder plus on-demand temporal VLM analysis and PDF report generation |
| `stream_receiver_server.py` | Receives streams from all Pis and serves the dashboard (runs on the laptop) |
| `agent_actuator.py` | FastAPI service that runs on each Raspberry Pi |
| `raw_stream_demo.py` | Default-mode pipeline (raw Picamera2 stream, no inference overlays) |
| `object_detection_demo.py` | Surveillance-mode pipeline (IMX500 object detection) |
| `segmentation_demo_overlay.py` | Construction-mode pipeline (IMX500 semantic segmentation) |
| `cameras.json` | Registry of cameras → zones, Pi hosts, and ports |
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
entry's `zone` is the operator-facing construction zone containing that camera,
and `pi_host` / `pi_port` must point at the Raspberry Pi that hosts it. Several
cameras may share the same zone:

```json
{
  "cameras": [
    { "id": 0, "name": "Camera 0", "location": "North", "zone": "Excavation", "pi_host": "192.168.1.50", "pi_port": 8000 },
    { "id": 1, "name": "Camera 1", "location": "South", "zone": "Excavation", "pi_host": "192.168.1.51", "pi_port": 8000 },
    { "id": 2, "name": "Camera 2", "location": "Gate", "zone": "Entrance", "pi_host": "192.168.1.52", "pi_port": 8000 }
  ]
}
```

Both the supervisor and the stream receiver on the laptop read this file, so
keep them in sync.

---

## 2 · Set up each Raspberry Pi

Clone the repo onto the Pi and install its dependencies:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
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
- *"Put the Excavation zone in Search Mode and find a red fire extinguisher."*
- *"Search all zones for a child wearing a blue jacket."*
- *"Stop Search Mode in the Excavation zone and return it to Free Mode."*
- *"Put the Excavation and Entrance zones in Safety Mode."*
- *"Is the construction site currently clear?"*
- *"Clear the safety hazard; the area has been inspected."*
- *"Generate today's report for the Excavation zone. Our goal was to excavate and level Area B."*
- *"Generate separate reports for the Excavation and Entrance zones for August 18."*
- *"How many reporting snapshots were captured for the Excavation zone on August 18?"*

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
- *"Put the Excavation zone in Search Mode and find the red fire extinguisher."*
- *"Generate today's construction progress report for the Excavation zone."*
- *"Switch the Entrance zone to Investigation Mode."*

Voice chat uses OpenAI's Realtime API over WebRTC. The browser sends microphone
audio directly through a peer connection and receives natural spoken audio back;
`supervisor.py` only performs the secure session handshake and executes
supervisor tool calls, so your standard `OPENAI_API_KEY` is never exposed to
the browser.

Optional voice settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_REALTIME_MODEL` | `gpt-realtime` | Realtime speech-to-speech model used by the voice agent. Use `gpt-realtime-2` if it is enabled for your OpenAI project. |
| `OPENAI_REALTIME_VOICE` | `marin` | Spoken voice returned by the Realtime model |
| `OPENAI_SEARCH_VLM_MODEL` | `gpt-5.5` | Vision-capable model used by the laptop-side Search scanner |
| `OPENAI_SEARCH_VLM_DETAIL` | `high` | Image detail sent to the Search VLM (`low`, `high`, or `auto`) |
| `SEARCH_MATCH_THRESHOLD` | `0.75` | Minimum VLM confidence required before the dashboard raises a Search match alert |
| `OPENAI_SAFETY_VLM_MODEL` | Search model value | Vision-capable model used by the Safety scanner |
| `OPENAI_SAFETY_VLM_DETAIL` | Search detail value | Image detail sent to the Safety VLM (`low`, `high`, or `auto`) |
| `SAFETY_MATCH_THRESHOLD` | `0.75` | Minimum confidence required before a Safety hazard is latched and displayed |
| `SAFETY_SITE_TIMEZONE` | `America/Vancouver` | IANA site timezone used for the after-hours access rule |
| `SAFETY_ACCESS_START_HOUR` | `9` | First permitted construction-hour clock hour, inclusive |
| `SAFETY_ACCESS_END_HOUR` | `17` | End of permitted construction hours; this hour is after-hours |
| `OPENAI_REPORTING_VLM_MODEL` | `gpt-5.6` | Vision-capable model used for per-camera timeline analysis and final report synthesis |
| `OPENAI_REPORTING_VLM_DETAIL` | `high` | Image detail sent for daily report analysis |
| `OPENAI_REPORTING_REASONING_EFFORT` | `medium` | Reasoning effort used by both reporting model stages |
| `REPORTING_SITE_TIMEZONE` | Safety timezone value | IANA timezone used to assign hourly snapshots to a site-local day |
| `REPORTING_CAPTURE_START_HOUR` | `9` | First hourly report snapshot, inclusive |
| `REPORTING_CAPTURE_END_HOUR` | `17` | Last hourly report snapshot, inclusive |
| `REPORTING_CAPTURE_POLL_SEC` | `30` | How often the recorder retries a missing camera within the current hourly slot |
| `REPORTING_MAX_FRAME_AGE_SEC` | `10` | Maximum age of a receiver frame accepted as hourly evidence |
| `REPORTING_MAX_ANALYSIS_WORKERS` | `2` | Maximum camera timelines analyzed concurrently |
| `REPORTING_SNAPSHOT_DIR` | OS temporary directory | Optional directory for timestamped hourly JPEG evidence and metadata |
| `REPORTING_OUTPUT_DIR` | `output/pdf` | Directory for generated daily PDF reports |
| `STREAM_RECEIVER_URL` | `http://127.0.0.1:9000` | Receiver URL used by the supervisor to read active streams and publish system logs |
| `OPERATIONAL_STATE_PUBLISH_INTERVAL_SEC` | `2.0` | How often the supervisor republishes operational state so the receiver can recover after a restart |

Microphone access works on `localhost` / `127.0.0.1` in modern browsers. If you
open the supervisor from another device, serve it over HTTPS so the browser will
allow microphone capture.

---

## Camera processing modes

| Mode | What runs on the Pi | Typical use |
| --- | --- | --- |
| `default` | Raw Picamera2 stream, no object detection or bounding boxes | Viewing unprocessed live camera footage on the laptop |
| `surveillance` | Object detection (e.g. NanoDet / MobileNet SSD on the IMX500) | Spotting people, vehicles, and abnormal activity |
| `construction` | Semantic segmentation (DeepLabV3+ on the IMX500) | Extracting machinery / site structure for digital-twin updates |
| `idle` | No inference; camera process stopped | Saving power / bandwidth |

All modes are selectable from the supervisor prompt — you never need to
SSH into a Pi to change them.

## Zone operational modes

Every camera belongs to exactly one zone according to `cameras.json`. Each zone
has its own mode and objective, so one zone can run Safety while another runs a
Search for a specific object and the remaining zones stay in Free. A command can
target one zone, several named zones, or all zones. Only cameras in the selected
zones are normalized or scanned; unmentioned zones keep their existing state.
The dashboard shows a card for every zone and labels the overall state **Mixed**
when their modes differ.

| Mode | Behaviour |
| --- | --- |
| `free` | Initial live-view mode. Stops Search and Safety scanning in that zone and sets its reachable cameras to raw `default` streaming without an automated scanner or workflow. A previously latched safety hazard remains visible until explicitly cleared. |
| `safety` | Configures that zone's reachable cameras for raw `default` streaming and continuously checks each sampled frame for the applicable construction safety hazards in one VLM request. |
| `search` | Configures that zone's reachable cameras for raw `default` streaming and scans its active feeds for the operator's visual target. Different zones can use different targets. |
| `investigation` | Selectable placeholder. Its investigation workflow will be added when requirements are finalized. |

Search Mode reuses the proven visual-search workflow within each selected zone. Its scanner
asks the receiver which streams are live, samples the latest frame from each
active camera on every pass, and sends those frames to the configured VLM with
the operator's target description. Camera feeds remain visible in the receiver
dashboard. When the VLM reports a match above the configured threshold, the
**System Logs** tab records a highlighted match, stores the triggering frame,
and plays an alarm sound in the browser. Search Mode and match indicators use
neutral blue/teal styling instead of critical-state red.

Safety Mode uses the same zone-filtered stream sampling foundation but evaluates
exactly two applicable safety checks together. During configured construction
hours, each frame is assessed for:

- **Fire Hazard** — visible flame, fire, or smoke.
- **Work-Zone Intrusion** — a person, animal, or other living creature
  visibly present in an active machinery work zone.

Outside the configured 09:00–17:00 site-local window, the laptop performs a
local clock check. Machinery is considered off, so **Work-Zone Intrusion** is
replaced by **Unauthorized Entry**, which checks for a person present at the
site. **Fire Hazard** remains active at all times. This decision is made
in-process before the VLM request, so every frame still uses one model call.
The full camera view is treated as the monitored work zone/site in the current
demo.

A single frame can trigger multiple hazards. Each positive assessment above the
configured confidence threshold produces a red **STOP WORK** alert with its
cause and triggering frame. The receiver separately
latches the construction safety state red, and that state remains red even if
the operator switches modes or clears the System Log. It returns to green only
when the operator explicitly asks the supervisor to clear or acknowledge the
safety state. A new visible hazard can latch it red again.

Every zone starts in Free Mode. Entering Free Mode for selected zones stops their
active Search or Safety scans, clears their objectives, and normalizes only their
registered cameras to `default` processing for an unprocessed footage view. It does not run a VLM
scanner or any other operational workflow, and it does not silently clear a
latched construction hazard.

Investigation is intentionally an honest placeholder: selecting it updates the
operational state and stops any active Search or Safety scan, but does not invent
or apply camera configuration until that mode's detailed requirements are
defined. A per-camera processing mode remains available whenever that camera's
own zone is not in Search or Safety Mode.

## Daily construction reporting

Reporting is a background capability, not an operational mode. It does not stop
Search or Safety, change the current operational state, or reconfigure a Pi. While
the supervisor is running, it reads the receiver's latest fresh frame and saves
one timestamped JPEG per registered camera for every hourly slot from 09:00
through 17:00 in the configured site timezone. If a camera is unavailable or its
frame is stale, the recorder retries during that same hour. Each camera and hour
is stored only once. The evidence is placed under the operating system's temporary
directory by default; use `REPORTING_SNAPSHOT_DIR` to choose another location.

When the operator asks for a report, at least one zone must be named. The system
generates a separate PDF for every requested zone and never mixes another zone's
frames into it. It uses a compact two-stage analysis for each zone:

1. Each camera's ordered snapshots are sent together in one multimodal VLM request,
   with the camera name and capture time immediately before each image. The model
   returns structured activities, visible changes, progress estimates, issues, and
   confidence for that camera.
2. One text-only synthesis request combines that zone's camera observations, removes
   duplicate views of the same work, and creates the zone narrative. The
   laptop then lays out a downloadable PDF with the summary, timeline, issues,
   evidence coverage, and representative first/last frames.

This gives the VLM temporal context without one enormous all-camera image request
and avoids maintaining a separate always-running progress database. The structured
observations are working context for the requested report rather than long-term
agent memory. If the operator includes the day's goal, the report can estimate
progress toward it. Without a goal, the prompt explicitly prevents an invented
completion percentage and reports only visible activity and change.

Generated PDFs use zone-specific filenames, are saved under `output/pdf`, and
are served from the supervisor at `/reports/<filename>`. Missing cameras or
timestamps reduce evidence coverage but do not stop a report as long as at least
one valid snapshot in the requested zone can be analyzed. A report can be
requested while any operational mode is active.

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
