"""Laptop-side Search Mode scanner that evaluates live camera frames with a VLM."""

from __future__ import annotations

import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests


class SearchScanner:
    """Poll active streams and ask a vision-capable model whether each frame matches a target."""

    def __init__(
        self,
        *,
        client,
        receiver_url: str,
        model: str,
        image_detail: str = "high",
        sample_interval_sec: float = 1.0,
        match_threshold: float = 0.75,
        alert_cooldown_sec: float = 10.0,
        max_workers: int = 4,
    ) -> None:
        self.client = client
        self.receiver_url = receiver_url.rstrip("/")
        self.model = model
        self.image_detail = image_detail
        self.sample_interval_sec = sample_interval_sec
        self.match_threshold = match_threshold
        self.alert_cooldown_sec = alert_cooldown_sec
        self.max_workers = max_workers

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._target = ""
        self._generation = 0
        self._inflight_searches: set[tuple[int, int]] = set()
        self._last_frame_time: dict[int, float] = {}
        self._last_alert_time: dict[tuple[int, int], float] = {}
        self._last_vlm_error_time: dict[tuple[int, int], float] = {}
        self._last_receiver_error_time = 0.0
        self._last_active_set: tuple[int, ...] | None = None

    def start(self, target: str) -> None:
        clean_target = target.strip()
        with self._lock:
            self._target = clean_target
            self._generation += 1
            self._last_alert_time.clear()
            self._last_vlm_error_time.clear()
            already_running = self.is_running_locked()
            if already_running:
                return
            stop_event = threading.Event()
            executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="search-vlm",
            )
            thread = threading.Thread(
                target=self._run,
                args=(stop_event, executor),
                name="search-scanner",
                daemon=True,
            )
            self._stop_event = stop_event
            self._executor = executor
            self._thread = thread
            thread.start()

    def update_target(self, target: str) -> None:
        with self._lock:
            self._target = target.strip()
            self._generation += 1
            self._last_alert_time.clear()
            self._last_vlm_error_time.clear()

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            self._stop_event.set()
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def is_running(self) -> bool:
        with self._lock:
            return self.is_running_locked()

    def is_running_locked(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def _run(self, stop_event: threading.Event, executor: ThreadPoolExecutor) -> None:
        try:
            with self._lock:
                if stop_event.is_set():
                    return
                self._post_log(
                    kind="scan",
                    level="info",
                    message=f"Search VLM scanner started using {self.model}.",
                )
            while not stop_event.is_set():
                cycle_started = time.time()
                active_camera_ids = self._active_camera_ids(stop_event)
                if not self._log_active_set_change(active_camera_ids, stop_event):
                    break

                for camera_id in active_camera_ids:
                    if stop_event.is_set():
                        break
                    with self._lock:
                        if stop_event.is_set():
                            break
                        generation = self._generation
                        target = self._target
                        search_key = (camera_id, generation)
                        if search_key in self._inflight_searches:
                            continue
                        self._inflight_searches.add(search_key)
                    try:
                        future = executor.submit(
                            self._scan_camera_once,
                            camera_id,
                            generation,
                            target,
                        )
                        future.add_done_callback(
                            lambda _future, key=search_key: self._discard_inflight(key)
                        )
                    except RuntimeError:
                        self._discard_inflight(search_key)
                        break

                elapsed = time.time() - cycle_started
                stop_event.wait(max(0.0, self.sample_interval_sec - elapsed))
        finally:
            with self._lock:
                owns_current_run = self._thread is threading.current_thread()
                if owns_current_run:
                    stop_event.set()
                    self._generation += 1
                    self._thread = None
                    self._executor = None
                    self._inflight_searches.clear()
                    self._last_active_set = None
            if owns_current_run:
                executor.shutdown(wait=False, cancel_futures=True)
                self._post_log(
                    kind="scan",
                    level="info",
                    message="Search VLM scanner stopped.",
                )

    def _discard_inflight(self, search_key: tuple[int, int]) -> None:
        with self._lock:
            self._inflight_searches.discard(search_key)

    def _active_camera_ids(self, stop_event: threading.Event) -> list[int]:
        try:
            response = requests.get(f"{self.receiver_url}/cameras", timeout=2)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Receiver camera response must be a JSON object.")
            cameras = payload.get("cameras") or []
            if not isinstance(cameras, list):
                raise ValueError("Receiver camera list must be an array.")
        except Exception as exc:
            now = time.time()
            with self._lock:
                if stop_event.is_set():
                    return []
                if now - self._last_receiver_error_time >= 10:
                    self._last_receiver_error_time = now
                    self._post_log(
                        kind="scan",
                        level="warning",
                        message=f"Unable to read active streams from the receiver: {exc}",
                    )
            return []

        active = []
        for camera in cameras:
            if not isinstance(camera, dict):
                continue
            if camera.get("stream_active"):
                try:
                    active.append(int(camera["camera_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
        return sorted(active)

    def _log_active_set_change(
        self,
        camera_ids: list[int],
        stop_event: threading.Event,
    ) -> bool:
        with self._lock:
            if stop_event.is_set():
                return False
            current = tuple(camera_ids)
            if current == self._last_active_set:
                return True
            self._last_active_set = current
            if current:
                message = "Scanning active camera streams: " + ", ".join(str(cid) for cid in current) + "."
            else:
                message = "No active camera streams are currently available for Search Mode scanning."
            self._post_log(kind="scan", level="info", message=message)
            return True

    def _scan_camera_once(self, camera_id: int, generation: int, target: str) -> None:
        try:
            frame_bytes, frame_time = self._latest_frame(camera_id)
            if frame_bytes is None or frame_time is None:
                return
            previous_frame_time = self._last_frame_time.get(camera_id)
            if previous_frame_time is not None and frame_time <= previous_frame_time:
                return
            self._last_frame_time[camera_id] = frame_time

            result = self._analyze_frame(target=target, frame_bytes=frame_bytes)
            is_match = bool(result.get("match"))
            confidence = self._coerce_confidence(result.get("confidence"))
            if not is_match or confidence < self.match_threshold:
                return

            now = time.time()
            alert_key = (camera_id, generation)
            summary = str(result.get("summary") or "The frame appears to match the search target.").strip()
            with self._lock:
                if generation != self._generation or self._stop_event.is_set():
                    return
                last_alert_time = self._last_alert_time.get(alert_key, 0.0)
                if now - last_alert_time < self.alert_cooldown_sec:
                    return
                self._last_alert_time[alert_key] = now
                # Keep mode/target invalidation serialized with alert publication.
                # This ensures a mode switch cannot be published before an old
                # in-flight result has either been discarded or fully logged.
                self._post_log(
                    kind="alert",
                    level="info",
                    message=f'SEARCH MATCH for "{target}" on camera {camera_id}: {summary}',
                    camera_id=camera_id,
                    confidence=confidence,
                    frame_bytes=frame_bytes,
                )
        except Exception as exc:
            now = time.time()
            error_key = (camera_id, generation)
            with self._lock:
                if generation != self._generation or self._stop_event.is_set():
                    return
                if now - self._last_vlm_error_time.get(error_key, 0.0) >= 10:
                    self._last_vlm_error_time[error_key] = now
                    self._post_log(
                        kind="scan",
                        level="warning",
                        message=f"Search scan failed for camera {camera_id}: {exc}",
                        camera_id=camera_id,
                    )


    def _latest_frame(self, camera_id: int) -> tuple[bytes | None, float | None]:
        response = requests.get(
            f"{self.receiver_url}/latest_frame/{camera_id}",
            timeout=2,
        )
        if response.status_code == 404:
            return None, None
        response.raise_for_status()
        try:
            frame_time = float(response.headers.get("X-Receive-Time", ""))
        except ValueError:
            frame_time = time.time()
        return response.content, frame_time

    def _analyze_frame(self, *, target: str, frame_bytes: bytes) -> dict[str, Any]:
        data_url = "data:image/jpeg;base64," + base64.b64encode(frame_bytes).decode("ascii")
        prompt = (
            "You are the visual-search component of a multi-camera operations system. "
            "Decide whether this single camera frame visibly contains or matches the operator's "
            "search target. The target may be a person, object, vehicle, or visible condition. "
            "Use only visible evidence in the image. If the evidence is ambiguous, answer match=false. "
            "Return only compact JSON with keys: match (boolean), confidence (number 0 to 1), "
            "and summary (short sentence describing the visible evidence).\n\n"
            f"Search target: {target}"
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": data_url,
                            "detail": self.image_detail,
                        },
                    ],
                }
            ],
        )
        return self._parse_model_json(getattr(response, "output_text", "") or "")

    @staticmethod
    def _parse_model_json(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"Model returned non-JSON output: {raw[:200]}")
            payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Model JSON output must be an object.")
        return payload

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(confidence, 1.0))

    def _post_log(
        self,
        *,
        kind: str,
        level: str,
        message: str,
        camera_id: int | None = None,
        confidence: float | None = None,
        frame_bytes: bytes | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "kind": kind,
            "level": level,
            "message": message,
        }
        if camera_id is not None:
            payload["camera_id"] = camera_id
        if confidence is not None:
            payload["confidence"] = confidence
        if frame_bytes is not None:
            payload["frame_jpeg_b64"] = base64.b64encode(frame_bytes).decode("ascii")
        try:
            requests.post(f"{self.receiver_url}/system/log", json=payload, timeout=2)
        except Exception:
            pass
