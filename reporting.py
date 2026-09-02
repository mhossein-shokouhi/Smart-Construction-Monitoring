"""Minute-by-minute construction snapshots and on-demand interval PDF reports."""

from __future__ import annotations

import base64
import html
import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import requests


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORTING_SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "reporting_snapshots"


@dataclass(frozen=True)
class ReportingSnapshot:
    report_date: str
    camera_id: int
    camera_name: str
    camera_location: str
    camera_zone: str
    slot_hour: int
    slot_minute: int
    scheduled_at: str
    captured_at: str
    receiver_frame_time: float
    image_path: Path

    def public_dict(self) -> dict[str, Any]:
        return {
            "date": self.report_date,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "camera_location": self.camera_location,
            "zone": self.camera_zone,
            "scheduled_at": self.scheduled_at,
            "captured_at": self.captured_at,
        }


class ConstructionReporting:
    """Record one frame per minute and report a requested recent interval.

    The recorder is deliberately passive: it reads the receiver's latest frame
    and never changes camera or operational modes. Report generation performs
    one temporal vision request per camera, followed by one text-only synthesis
    request across the compact camera observations.
    """

    def __init__(
        self,
        *,
        client: Any,
        receiver_url: str,
        cameras: dict[int, dict[str, Any]],
        model: str = "gpt-5.6",
        image_detail: str = "high",
        reasoning_effort: str = "medium",
        site_timezone: str = "America/Vancouver",
        capture_poll_sec: float = 10.0,
        max_frame_age_sec: float = 10.0,
        max_analysis_workers: int = 2,
        max_frames_per_camera: int = 24,
        snapshot_root: str | Path | None = None,
        output_dir: str | Path | None = None,
        now_provider: Callable[[], datetime] | None = None,
        log_callback: Callable[..., None] | None = None,
    ) -> None:
        if capture_poll_sec <= 0:
            raise ValueError("Reporting capture poll interval must be positive.")
        if max_frame_age_sec <= 0:
            raise ValueError("Reporting maximum frame age must be positive.")
        if max_frames_per_camera < 2:
            raise ValueError("Reporting must allow at least two analysis frames per camera.")

        self.client = client
        self.receiver_url = receiver_url.rstrip("/")
        self.cameras = {int(camera_id): dict(info) for camera_id, info in cameras.items()}
        self.zones: dict[str, list[int]] = {}
        for camera_id, camera in sorted(self.cameras.items()):
            zone = str(camera.get("zone") or "Unassigned").strip() or "Unassigned"
            camera["zone"] = zone
            self.zones.setdefault(zone, []).append(camera_id)
        self._zone_lookup = {zone.casefold(): zone for zone in self.zones}
        self.model = model
        self.image_detail = image_detail
        self.reasoning_effort = reasoning_effort
        self.site_timezone_name = site_timezone
        self.site_timezone = ZoneInfo(site_timezone)
        self.capture_poll_sec = capture_poll_sec
        self.max_frame_age_sec = max_frame_age_sec
        self.max_analysis_workers = max(1, max_analysis_workers)
        self.max_frames_per_camera = int(max_frames_per_camera)
        self.snapshot_root = Path(
            snapshot_root or DEFAULT_REPORTING_SNAPSHOT_ROOT
        ).expanduser().resolve()
        self.output_dir = Path(
            output_dir or Path(__file__).resolve().parent / "output" / "pdf"
        ).expanduser().resolve()
        self.now_provider = now_provider
        self.log_callback = log_callback

        self._lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._report_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._logged_capture_failures: set[tuple[str, int, int, int]] = set()

    def _site_now(self) -> datetime:
        current = self.now_provider() if self.now_provider is not None else datetime.now(self.site_timezone)
        if current.tzinfo is None:
            return current.replace(tzinfo=self.site_timezone)
        return current.astimezone(self.site_timezone)

    def start(self) -> None:
        """Start the once-per-minute capture loop if it is not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run,
                args=(self._stop_event,),
                name="construction-reporting-recorder",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            stop_event = self._stop_event
            thread = self._thread
        stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=min(2.0, self.capture_poll_sec + 0.5))
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def _run(self, stop_event: threading.Event) -> None:
        try:
            self._log(
                kind="reporting",
                level="info",
                message=(
                    "Minute-by-minute reporting capture is active in "
                    f"{self.site_timezone_name}. Snapshots are saved under "
                    f"{self.snapshot_root}."
                ),
            )
            while not stop_event.is_set():
                try:
                    self.capture_due()
                except Exception as exc:
                    self._log(
                        kind="reporting",
                        level="warning",
                        message=f"Minute reporting capture encountered an error: {exc}",
                    )
                stop_event.wait(self.capture_poll_sec)
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None

    def capture_due(self, current: datetime | None = None) -> dict[str, Any]:
        """Serialize manual/report-triggered capture with the background recorder."""
        with self._capture_lock:
            return self._capture_due_unlocked(current)

    def _capture_due_unlocked(self, current: datetime | None = None) -> dict[str, Any]:
        """Capture every still-missing camera for the current clock minute."""
        now = current or self._site_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=self.site_timezone)
        else:
            now = now.astimezone(self.site_timezone)

        report_date = now.date().isoformat()
        current_slot = (report_date, now.hour, now.minute)
        self._logged_capture_failures = {
            key for key in self._logged_capture_failures if key[:3] == current_slot
        }
        captured: list[int] = []
        already_present: list[int] = []
        failed: list[dict[str, Any]] = []
        for camera_id, camera in sorted(self.cameras.items()):
            image_path, metadata_path = self._slot_paths(
                report_date,
                camera_id,
                now.hour,
                now.minute,
            )
            if image_path.is_file() and metadata_path.is_file():
                already_present.append(camera_id)
                continue
            try:
                frame_bytes, receiver_frame_time = self._fetch_latest_frame(camera_id, now)
                captured_at = now.isoformat(timespec="seconds")
                scheduled_at = datetime(
                    now.year,
                    now.month,
                    now.day,
                    now.hour,
                    now.minute,
                    tzinfo=self.site_timezone,
                ).isoformat(timespec="seconds")
                metadata = {
                    "date": report_date,
                    "camera_id": camera_id,
                    "camera_name": str(camera.get("name") or f"Camera {camera_id}"),
                    "camera_location": str(camera.get("location") or ""),
                    "camera_zone": str(camera.get("zone") or "Unassigned"),
                    "slot_hour": now.hour,
                    "slot_minute": now.minute,
                    "scheduled_at": scheduled_at,
                    "captured_at": captured_at,
                    "receiver_frame_time": receiver_frame_time,
                }
                self._atomic_write(image_path, frame_bytes)
                self._atomic_write(
                    metadata_path,
                    json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
                )
                captured.append(camera_id)
                self._logged_capture_failures.discard(
                    (report_date, now.hour, now.minute, camera_id)
                )
            except Exception as exc:
                failed.append({"camera_id": camera_id, "error": str(exc)})

        if captured:
            camera_text = ", ".join(str(camera_id) for camera_id in captured)
            self._log(
                kind="reporting",
                level="info",
                message=(
                    f"Saved {now.hour:02d}:{now.minute:02d} reporting snapshot(s) "
                    f"for camera(s) {camera_text}."
                ),
            )
        for item in failed:
            key = (report_date, now.hour, now.minute, int(item["camera_id"]))
            if key in self._logged_capture_failures:
                continue
            self._logged_capture_failures.add(key)
            self._log(
                kind="reporting",
                level="warning",
                message=(
                    f"Could not save the {now.hour:02d}:{now.minute:02d} reporting snapshot for camera "
                    f"{item['camera_id']}: {item['error']}"
                ),
                camera_id=int(item["camera_id"]),
            )

        if failed and not captured and not already_present:
            status = "error"
        elif failed:
            status = "partial_error"
        else:
            status = "ok"
        return {
            "status": status,
            "date": report_date,
            "scheduled_time": f"{now.hour:02d}:{now.minute:02d}",
            "snapshot_root": str(self.snapshot_root),
            "captured": captured,
            "already_present": already_present,
            "failed": failed,
        }

    def _fetch_latest_frame(self, camera_id: int, now: datetime) -> tuple[bytes, float]:
        response = requests.get(
            f"{self.receiver_url}/latest_frame/{camera_id}",
            timeout=5,
        )
        response.raise_for_status()
        frame_bytes = bytes(response.content)
        if not frame_bytes:
            raise ValueError("receiver returned an empty frame")
        raw_receive_time = response.headers.get("X-Receive-Time")
        if raw_receive_time is None:
            raise ValueError("receiver did not provide a frame timestamp")
        receiver_frame_time = float(raw_receive_time)
        frame_age = max(0.0, now.timestamp() - receiver_frame_time)
        if frame_age > self.max_frame_age_sec:
            raise ValueError(f"latest frame is stale ({frame_age:.1f} seconds old)")
        return frame_bytes, receiver_frame_time

    def _slot_paths(
        self,
        report_date: str,
        camera_id: int,
        hour: int,
        minute: int,
    ) -> tuple[Path, Path]:
        camera = self.cameras.get(camera_id, {})
        zone_name = str(camera.get("zone") or "Unassigned")
        camera_name = str(camera.get("name") or f"Camera {camera_id}")
        zone_dir = self._slugify(zone_name)
        camera_slug = self._slugify(camera_name)
        camera_dir_name = f"camera-{camera_id}"
        if camera_slug not in {"camera", camera_dir_name}:
            camera_dir_name += f"-{camera_slug}"
        camera_dir = self.snapshot_root / report_date / zone_dir / camera_dir_name
        stem = f"{hour:02d}-{minute:02d}"
        return camera_dir / f"{stem}.jpg", camera_dir / f"{stem}.json"

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with open(temporary_path, "wb") as output_file:
                output_file.write(content)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary_path, path)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    def list_snapshots(
        self,
        report_date: str,
        zone: str | None = None,
    ) -> list[ReportingSnapshot]:
        clean_date = self._validate_report_date(report_date)
        clean_zone = self._resolve_zone(zone) if zone is not None else None
        day_dir = self.snapshot_root / clean_date
        snapshots: list[ReportingSnapshot] = []
        if not day_dir.is_dir():
            return snapshots
        for metadata_path in sorted(day_dir.rglob("*.json")):
            image_path = metadata_path.with_suffix(".jpg")
            if not image_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                camera_id = int(metadata["camera_id"])
                camera_zone = str(
                    metadata.get("camera_zone")
                    or self.cameras.get(camera_id, {}).get("zone")
                    or "Unassigned"
                )
                if clean_zone is not None and camera_zone.casefold() != clean_zone.casefold():
                    continue
                snapshots.append(
                    ReportingSnapshot(
                        report_date=clean_date,
                        camera_id=camera_id,
                        camera_name=str(metadata.get("camera_name") or "Camera"),
                        camera_location=str(metadata.get("camera_location") or ""),
                        camera_zone=camera_zone,
                        slot_hour=int(metadata["slot_hour"]),
                        slot_minute=int(metadata.get("slot_minute", 0)),
                        scheduled_at=str(metadata["scheduled_at"]),
                        captured_at=str(metadata["captured_at"]),
                        receiver_frame_time=float(metadata["receiver_frame_time"]),
                        image_path=image_path.resolve(),
                    )
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
        return sorted(snapshots, key=lambda item: (item.camera_id, item.scheduled_at))

    def list_snapshots_between(
        self,
        period_start: datetime,
        period_end: datetime,
        zone: str | None = None,
    ) -> list[ReportingSnapshot]:
        """Return snapshots in the recent interval, using ``start < time <= end``."""
        start = self._normalize_site_time(period_start)
        end = self._normalize_site_time(period_end)
        if end < start:
            raise ValueError("Report period end must not be before its start.")
        clean_zone = self._resolve_zone(zone) if zone is not None else None
        snapshots: list[ReportingSnapshot] = []
        current_date = start.date()
        while current_date <= end.date():
            snapshots.extend(self.list_snapshots(current_date.isoformat(), clean_zone))
            current_date += timedelta(days=1)
        return [
            snapshot
            for snapshot in snapshots
            if start < self._snapshot_time(snapshot) <= end
        ]

    def get_status(
        self,
        lookback_minutes: int = 60,
        zone: str | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        period_start, period_end, duration = self._interval_bounds(
            lookback_minutes,
            end_time,
        )
        clean_zone = self._resolve_zone(zone) if zone is not None else None
        snapshots = self.list_snapshots_between(period_start, period_end, clean_zone)
        grouped = self._group_snapshots(snapshots)
        expected_times = self._expected_minute_slots(period_start, period_end)
        camera_coverage = []
        for camera_id, camera in sorted(self.cameras.items()):
            if clean_zone is not None and camera.get("zone") != clean_zone:
                continue
            records = grouped.get(camera_id, [])
            captured_times = [self._display_snapshot_time(record) for record in records]
            camera_coverage.append(
                {
                    "camera_id": camera_id,
                    "camera_name": camera.get("name") or f"Camera {camera_id}",
                    "captured_times": self._summarize_times(captured_times),
                    "captured_count": len(records),
                    "expected_count": len(expected_times),
                }
            )
        return {
            "status": "ok",
            "lookback_minutes": duration,
            "period_start": period_start.isoformat(timespec="seconds"),
            "period_end": period_end.isoformat(timespec="seconds"),
            "zone": clean_zone,
            "recorder_running": self.is_running(),
            "site_timezone": self.site_timezone_name,
            "snapshot_root": str(self.snapshot_root),
            "expected_minute_count": len(expected_times),
            "snapshot_count": len(snapshots),
            "camera_coverage": camera_coverage,
            "message": (
                f"{len(snapshots)} reporting snapshot(s) are available from the past "
                f"{duration} minute(s)"
                + (f" in {clean_zone}." if clean_zone else " across all zones.")
            ),
        }

    def generate_interval_report(
        self,
        lookback_minutes: int,
        zone: str | None = None,
        goal: str | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Analyze one zone's recent frames and write a zone-specific interval PDF."""
        period_start, period_end, duration = self._interval_bounds(
            lookback_minutes,
            end_time,
        )
        clean_zone = self._resolve_zone(zone)
        clean_goal = str(goal or "").strip() or None

        with self._report_lock:
            site_now = self._site_now()
            if abs((site_now - period_end).total_seconds()) < 60:
                self.capture_due(site_now)
            snapshots = self.list_snapshots_between(period_start, period_end, clean_zone)
            if not snapshots:
                return {
                    "status": "error",
                    "zone": clean_zone,
                    "lookback_minutes": duration,
                    "period_start": period_start.isoformat(timespec="seconds"),
                    "period_end": period_end.isoformat(timespec="seconds"),
                    "error": (
                        f"No minute snapshots are available for {clean_zone} during the requested "
                        f"past {duration} minute(s)."
                    ),
                }

            grouped = self._group_snapshots(snapshots)
            analysis_sequences = {
                camera_id: self._select_analysis_snapshots(records)
                for camera_id, records in grouped.items()
            }
            analyses: list[dict[str, Any]] = []
            analysis_errors: list[dict[str, Any]] = []
            worker_count = min(self.max_analysis_workers, len(grouped))
            with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
                futures = {
                    executor.submit(self._analyze_camera_sequence, records, clean_goal): camera_id
                    for camera_id, records in analysis_sequences.items()
                }
                for future in as_completed(futures):
                    camera_id = futures[future]
                    try:
                        analyses.append(future.result())
                    except Exception as exc:
                        analysis_errors.append({"camera_id": camera_id, "error": str(exc)})

            if not analyses:
                return {
                    "status": "error",
                    "zone": clean_zone,
                    "lookback_minutes": duration,
                    "period_start": period_start.isoformat(timespec="seconds"),
                    "period_end": period_end.isoformat(timespec="seconds"),
                    "snapshot_count": len(snapshots),
                    "errors": analysis_errors,
                    "error": "The VLM could not analyze any camera sequence for this report.",
                }

            analyses.sort(key=lambda item: int(item.get("camera_id", 0)))
            coverage = self._coverage_summary(
                period_start,
                period_end,
                clean_zone,
                grouped,
                analysis_sequences,
                analysis_errors,
            )
            synthesis_error: str | None = None
            try:
                report = self._synthesize_report(
                    period_start,
                    period_end,
                    clean_zone,
                    clean_goal,
                    analyses,
                    coverage,
                )
            except Exception as exc:
                synthesis_error = str(exc)
                report = self._fallback_report(period_start, period_end, clean_goal, analyses, coverage)

            try:
                pdf_path = self._write_pdf(
                    period_start=period_start,
                    period_end=period_end,
                    zone=clean_zone,
                    goal=clean_goal,
                    report=report,
                    analyses=analyses,
                    grouped_snapshots=grouped,
                    coverage=coverage,
                )
            except Exception as exc:
                return {
                    "status": "error",
                    "zone": clean_zone,
                    "lookback_minutes": duration,
                    "period_start": period_start.isoformat(timespec="seconds"),
                    "period_end": period_end.isoformat(timespec="seconds"),
                    "snapshot_count": len(snapshots),
                    "error": f"The report was analyzed but its PDF could not be created: {exc}",
                }

            partial = bool(analysis_errors or synthesis_error)
            message = (
                f"Construction progress report created for {clean_zone} covering the past "
                f"{duration} minute(s)."
            )
            if analysis_errors:
                missing_ids = ", ".join(str(item["camera_id"]) for item in analysis_errors)
                message += f" Camera analysis was unavailable for: {missing_ids}."
            if synthesis_error:
                message += " The PDF uses a local fallback summary because final synthesis was unavailable."
            self._log(kind="reporting", level="info", message=message)
            return {
                "status": "partial_error" if partial else "ok",
                "zone": clean_zone,
                "lookback_minutes": duration,
                "period_start": period_start.isoformat(timespec="seconds"),
                "period_end": period_end.isoformat(timespec="seconds"),
                "goal": clean_goal,
                "snapshot_count": len(snapshots),
                "camera_count": len(grouped),
                "report_filename": pdf_path.name,
                "report_path": str(pdf_path),
                "report_url": f"/reports/{pdf_path.name}",
                "analysis_errors": analysis_errors,
                "synthesis_error": synthesis_error,
                "message": message,
            }

    def _analyze_camera_sequence(
        self,
        snapshots: list[ReportingSnapshot],
        goal: str | None,
    ) -> dict[str, Any]:
        first = snapshots[0]
        goal_instruction = (
            f"The operator's stated goal is: {goal}"
            if goal
            else (
                "The operator did not provide a goal for this reporting interval. Describe observed work and change, "
                "but use -1 for completion estimates instead of inventing a target."
            )
        )
        prompt = (
            "You are the vision observer for a construction progress report. The following images "
            "are chronological snapshots from the same fixed camera. Compare them as a sequence. "
            "Identify visible activity, meaningful changes since the prior snapshot, accomplished "
            "work, stalled periods, visible issues, and remaining work. Do not claim changes that "
            "are not visually supported. Account for lighting, viewpoint, and occlusion. "
            f"{goal_instruction}\n\n"
            f"Reporting interval: {snapshots[0].scheduled_at} through {snapshots[-1].scheduled_at}\n"
            f"Zone: {first.camera_zone}\n"
            f"Camera: {first.camera_name} (id {first.camera_id})\n"
            f"Location: {first.camera_location or 'Not specified'}"
        )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for index, snapshot in enumerate(snapshots, start=1):
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"Snapshot {index} of {len(snapshots)}. Scheduled time "
                        f"{snapshot.slot_hour:02d}:{snapshot.slot_minute:02d}; "
                        f"captured at {snapshot.captured_at}."
                    ),
                }
            )
            encoded = base64.b64encode(snapshot.image_path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded}",
                    "detail": self.image_detail,
                }
            )

        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "camera_progress_observation",
                    "strict": True,
                    "schema": self._camera_analysis_schema(),
                }
            },
            reasoning={"effort": self.reasoning_effort},
        )
        result = self._parse_model_json(getattr(response, "output_text", "") or "")
        result["camera_id"] = first.camera_id
        result["camera_name"] = first.camera_name
        result["camera_location"] = first.camera_location
        result["zone"] = first.camera_zone
        result["snapshot_count"] = len(snapshots)
        return result

    def _synthesize_report(
        self,
        period_start: datetime,
        period_end: datetime,
        zone: str,
        goal: str | None,
        analyses: list[dict[str, Any]],
        coverage: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "You are the progress agent for a construction report covering a specific time interval. "
            "Synthesize the structured "
            "observations from multiple cameras into one concise, evidence-grounded report. Merge "
            "duplicate observations of the same work, preserve uncertainty, and never treat camera "
            "silence as proof that work stopped. If no operator goal was supplied, set overall "
            "completion to -1 and focus on observed change rather than invented goal completion. "
            "Use plain professional language suitable for a site operator.\n\n"
            + json.dumps(
                {
                    "period_start": period_start.isoformat(timespec="seconds"),
                    "period_end": period_end.isoformat(timespec="seconds"),
                    "zone": zone,
                    "operator_goal": goal,
                    "coverage": self._coverage_for_model(coverage),
                    "camera_observations": analyses,
                },
                ensure_ascii=True,
            )
        )
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "construction_progress_interval_report",
                    "strict": True,
                    "schema": self._report_schema(),
                }
            },
            reasoning={"effort": self.reasoning_effort},
        )
        return self._parse_model_json(getattr(response, "output_text", "") or "")

    @staticmethod
    def _camera_analysis_schema() -> dict[str, Any]:
        observation = {
            "type": "object",
            "properties": {
                "time": {"type": "string"},
                "activity": {"type": "string"},
                "visible_change_since_previous": {"type": "string"},
                "progress_delta": {
                    "type": "string",
                    "enum": ["none", "small", "moderate", "large", "unknown"],
                },
                "estimated_completion_percent": {"type": "integer"},
                "issues": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
            },
            "required": [
                "time",
                "activity",
                "visible_change_since_previous",
                "progress_delta",
                "estimated_completion_percent",
                "issues",
                "confidence",
            ],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "camera_id": {"type": "integer"},
                "camera_name": {"type": "string"},
                "camera_location": {"type": "string"},
                "summary": {"type": "string"},
                "start_state": {"type": "string"},
                "end_state": {"type": "string"},
                "estimated_completion_percent": {"type": "integer"},
                "confidence": {"type": "number"},
                "completed_work": {"type": "array", "items": {"type": "string"}},
                "remaining_work": {"type": "array", "items": {"type": "string"}},
                "issues": {"type": "array", "items": {"type": "string"}},
                "data_quality_notes": {"type": "array", "items": {"type": "string"}},
                "observations": {"type": "array", "items": observation},
            },
            "required": [
                "camera_id",
                "camera_name",
                "camera_location",
                "summary",
                "start_state",
                "end_state",
                "estimated_completion_percent",
                "confidence",
                "completed_work",
                "remaining_work",
                "issues",
                "data_quality_notes",
                "observations",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _report_schema() -> dict[str, Any]:
        timeline_item = {
            "type": "object",
            "properties": {
                "time": {"type": "string"},
                "activity": {"type": "string"},
                "progress_note": {"type": "string"},
                "cameras": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["time", "activity", "progress_note", "cameras"],
            "additionalProperties": False,
        }
        issue_item = {
            "type": "object",
            "properties": {
                "issue": {"type": "string"},
                "evidence": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
            },
            "required": ["issue", "evidence", "priority"],
            "additionalProperties": False,
        }
        camera_item = {
            "type": "object",
            "properties": {
                "camera_name": {"type": "string"},
                "summary": {"type": "string"},
                "estimated_completion_percent": {"type": "integer"},
                "confidence": {"type": "number"},
            },
            "required": ["camera_name", "summary", "estimated_completion_percent", "confidence"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "executive_summary": {"type": "string"},
                "overall_completion_percent": {"type": "integer"},
                "overall_confidence": {"type": "number"},
                "completed_work": {"type": "array", "items": {"type": "string"}},
                "remaining_work": {"type": "array", "items": {"type": "string"}},
                "issues": {"type": "array", "items": issue_item},
                "timeline": {"type": "array", "items": timeline_item},
                "camera_summaries": {"type": "array", "items": camera_item},
                "data_notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "title",
                "executive_summary",
                "overall_completion_percent",
                "overall_confidence",
                "completed_work",
                "remaining_work",
                "issues",
                "timeline",
                "camera_summaries",
                "data_notes",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _parse_model_json(raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Reporting model output must be a JSON object.")
        return parsed

    def _coverage_summary(
        self,
        period_start: datetime,
        period_end: datetime,
        zone: str,
        grouped: dict[int, list[ReportingSnapshot]],
        analysis_sequences: dict[int, list[ReportingSnapshot]],
        analysis_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        expected_times = self._expected_minute_slots(period_start, period_end)
        cameras = []
        for camera_id, camera in sorted(self.cameras.items()):
            if camera.get("zone") != zone:
                continue
            records = grouped.get(camera_id, [])
            cameras.append(
                {
                    "camera_id": camera_id,
                    "camera_name": camera.get("name") or f"Camera {camera_id}",
                    "captured_times": [self._display_snapshot_time(record) for record in records],
                    "captured_count": len(records),
                    "expected_count": len(expected_times),
                    "analyzed_count": len(analysis_sequences.get(camera_id, [])),
                }
            )
        return {
            "period_start": period_start.isoformat(timespec="seconds"),
            "period_end": period_end.isoformat(timespec="seconds"),
            "zone": zone,
            "site_timezone": self.site_timezone_name,
            "expected_minute_count": len(expected_times),
            "cameras": cameras,
            "analysis_errors": analysis_errors,
        }

    @staticmethod
    def _coverage_for_model(coverage: dict[str, Any]) -> dict[str, Any]:
        """Keep report synthesis context useful without repeating every minute timestamp."""
        compact_cameras = []
        for camera in coverage.get("cameras") or []:
            captured_times = camera.get("captured_times") or []
            compact_cameras.append(
                {
                    "camera_id": camera.get("camera_id"),
                    "camera_name": camera.get("camera_name"),
                    "captured_count": camera.get("captured_count", 0),
                    "expected_count": camera.get("expected_count", 0),
                    "analyzed_count": camera.get("analyzed_count", 0),
                    "first_captured_time": captured_times[0] if captured_times else None,
                    "last_captured_time": captured_times[-1] if captured_times else None,
                }
            )
        return {
            "period_start": coverage.get("period_start"),
            "period_end": coverage.get("period_end"),
            "zone": coverage.get("zone"),
            "site_timezone": coverage.get("site_timezone"),
            "expected_minute_count": coverage.get("expected_minute_count", 0),
            "cameras": compact_cameras,
            "analysis_errors": coverage.get("analysis_errors") or [],
        }

    def _select_analysis_snapshots(
        self,
        records: list[ReportingSnapshot],
    ) -> list[ReportingSnapshot]:
        """Keep short intervals intact and evenly sample long ones for efficient VLM use."""
        if len(records) <= self.max_frames_per_camera:
            return list(records)
        last_index = len(records) - 1
        indexes = {
            round(position * last_index / (self.max_frames_per_camera - 1))
            for position in range(self.max_frames_per_camera)
        }
        return [records[index] for index in sorted(indexes)]

    @staticmethod
    def _group_snapshots(
        snapshots: list[ReportingSnapshot],
    ) -> dict[int, list[ReportingSnapshot]]:
        grouped: dict[int, list[ReportingSnapshot]] = {}
        for snapshot in snapshots:
            grouped.setdefault(snapshot.camera_id, []).append(snapshot)
        for records in grouped.values():
            records.sort(key=lambda item: item.scheduled_at)
        return grouped

    @staticmethod
    def _fallback_report(
        period_start: datetime,
        period_end: datetime,
        goal: str | None,
        analyses: list[dict[str, Any]],
        coverage: dict[str, Any],
    ) -> dict[str, Any]:
        completed: list[str] = []
        remaining: list[str] = []
        issues: list[dict[str, str]] = []
        timeline: list[dict[str, Any]] = []
        camera_summaries = []
        for analysis in analyses:
            completed.extend(str(item) for item in analysis.get("completed_work") or [])
            remaining.extend(str(item) for item in analysis.get("remaining_work") or [])
            issues.extend(
                {"issue": str(item), "evidence": "Reported by camera analysis.", "priority": "unknown"}
                for item in analysis.get("issues") or []
            )
            for observation in analysis.get("observations") or []:
                timeline.append(
                    {
                        "time": str(observation.get("time") or "Unknown"),
                        "activity": str(observation.get("activity") or "No activity described."),
                        "progress_note": str(
                            observation.get("visible_change_since_previous") or "No change described."
                        ),
                        "cameras": [str(analysis.get("camera_name") or "Camera")],
                    }
                )
            camera_summaries.append(
                {
                    "camera_name": str(analysis.get("camera_name") or "Camera"),
                    "summary": str(analysis.get("summary") or "No summary available."),
                    "estimated_completion_percent": int(
                        analysis.get("estimated_completion_percent", -1)
                    ),
                    "confidence": float(analysis.get("confidence", 0.0)),
                }
            )
        summary = " ".join(str(item.get("summary") or "") for item in analyses).strip()
        return {
            "title": "Construction Progress Report",
            "executive_summary": summary or (
                "Visible construction activity was reviewed from "
                f"{period_start.strftime('%Y-%m-%d %H:%M')} through "
                f"{period_end.strftime('%Y-%m-%d %H:%M')}."
            ),
            "overall_completion_percent": -1,
            "overall_confidence": 0.0,
            "completed_work": list(dict.fromkeys(completed)),
            "remaining_work": list(dict.fromkeys(remaining)),
            "issues": issues,
            "timeline": timeline,
            "camera_summaries": camera_summaries,
            "data_notes": [
                "The cross-camera synthesis model was unavailable; this report uses camera-level observations.",
                f"Coverage: {sum(item['captured_count'] for item in coverage['cameras'])} snapshots.",
            ],
        }

    def _write_pdf(
        self,
        *,
        period_start: datetime,
        period_end: datetime,
        zone: str,
        goal: str | None,
        report: dict[str, Any],
        analyses: list[dict[str, Any]],
        grouped_snapshots: dict[int, list[ReportingSnapshot]],
        coverage: dict[str, Any],
    ) -> Path:
        try:
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                Image,
                KeepTogether,
                PageBreak,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError as exc:
            raise RuntimeError(
                "PDF support is not installed. Install requirements-laptop.txt and try again."
            ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)
        generated_at = self._site_now()
        zone_slug = self._slugify(zone)
        interval_slug = (
            f"{period_start.strftime('%Y%m%d-%H%M')}-to-"
            f"{period_end.strftime('%Y%m%d-%H%M')}"
        )
        pdf_path = self.output_dir / f"construction-progress-{zone_slug}-{interval_slug}.pdf"
        counter = 2
        while pdf_path.exists():
            pdf_path = self.output_dir / (
                f"construction-progress-{zone_slug}-{interval_slug}-{counter}.pdf"
            )
            counter += 1

        navy = colors.HexColor("#102A43")
        teal = colors.HexColor("#0F766E")
        pale_teal = colors.HexColor("#E6FFFA")
        slate = colors.HexColor("#486581")
        pale = colors.HexColor("#F4F7FA")
        line = colors.HexColor("#D9E2EC")
        red = colors.HexColor("#B42318")

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            textColor=navy,
            alignment=TA_LEFT,
            spaceAfter=8,
        )
        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=slate,
            spaceAfter=16,
        )
        heading_style = ParagraphStyle(
            "Heading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=navy,
            spaceBefore=12,
            spaceAfter=7,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#243B53"),
            spaceAfter=6,
        )
        small_style = ParagraphStyle(
            "Small",
            parent=body_style,
            fontSize=7.5,
            leading=10,
            textColor=slate,
        )
        table_header_style = ParagraphStyle(
            "TableHeader",
            parent=small_style,
            fontName="Helvetica-Bold",
            textColor=colors.white,
            alignment=TA_LEFT,
        )

        def clean(value: Any) -> str:
            printable = (
                str(value or "")
                .replace("\u2013", "-")
                .replace("\u2014", "-")
                .replace("\u2212", "-")
            )
            normalized = unicodedata.normalize("NFKD", printable)
            ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
            return html.escape(ascii_text).replace("\n", "<br/>")

        def paragraph(value: Any, style=body_style) -> Any:
            return Paragraph(clean(value), style)

        def bullet_items(items: list[Any]) -> list[Any]:
            if not items:
                return [paragraph("None identified from the available visual evidence.")]
            flowables = []
            for item in items:
                flowables.append(Paragraph(f"- {clean(item)}", body_style))
            return flowables

        def on_page(canvas, document) -> None:
            canvas.saveState()
            width, height = letter
            canvas.setStrokeColor(line)
            canvas.setLineWidth(0.6)
            canvas.line(document.leftMargin, height - 42, width - document.rightMargin, height - 42)
            canvas.setFont("Helvetica-Bold", 7.5)
            canvas.setFillColor(teal)
            canvas.drawString(document.leftMargin, height - 32, "SITE VISION OPERATIONS")
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(slate)
            canvas.drawRightString(
                width - document.rightMargin,
                28,
                f"Progress report  |  Page {document.page}",
            )
            canvas.restoreState()

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=0.62 * inch,
            leftMargin=0.62 * inch,
            topMargin=0.72 * inch,
            bottomMargin=0.58 * inch,
            title=(
                f"Construction Progress Report - {zone} - "
                f"{period_start.isoformat(timespec='minutes')} to "
                f"{period_end.isoformat(timespec='minutes')}"
            ),
            author="Agentic Multi-Camera Operations",
        )
        story: list[Any] = []
        story.append(paragraph(report.get("title") or "Construction Progress Report", title_style))
        story.append(
            Paragraph(
                f"ZONE&nbsp;&nbsp; <b>{clean(zone)}</b>&nbsp;&nbsp;&nbsp;&nbsp; "
                f"PERIOD&nbsp;&nbsp; <b>{clean(period_start.strftime('%Y-%m-%d %H:%M %Z'))}</b> "
                f"to <b>{clean(period_end.strftime('%Y-%m-%d %H:%M %Z'))}</b><br/>"
                f"GENERATED&nbsp;&nbsp; <b>{clean(generated_at.strftime('%Y-%m-%d %H:%M %Z'))}</b>",
                subtitle_style,
            )
        )

        goal_text = goal or "No explicit interval goal was provided. Progress is described from visible change only."
        story.append(
            Table(
                [[paragraph("OPERATOR GOAL", table_header_style)], [paragraph(goal_text)]],
                colWidths=[doc.width],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), teal),
                        ("BACKGROUND", (0, 1), (-1, -1), pale_teal),
                        ("BOX", (0, 0), (-1, -1), 0.7, teal),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            )
        )
        story.append(Spacer(1, 10))

        completion = int(report.get("overall_completion_percent", -1))
        confidence = float(report.get("overall_confidence", 0.0))
        metric_value = f"{max(0, min(100, completion))}%" if completion >= 0 else "Not estimated"
        confidence_value = f"{max(0.0, min(1.0, confidence)) * 100:.0f}%"
        snapshot_count = sum(item["captured_count"] for item in coverage["cameras"])
        metrics = [
            [paragraph("ESTIMATED COMPLETION", table_header_style), paragraph("MODEL CONFIDENCE", table_header_style), paragraph("VISUAL EVIDENCE", table_header_style)],
            [paragraph(metric_value, heading_style), paragraph(confidence_value, heading_style), paragraph(f"{snapshot_count} snapshots", heading_style)],
        ]
        story.append(
            Table(
                metrics,
                colWidths=[doc.width / 3] * 3,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), navy),
                        ("BACKGROUND", (0, 1), (-1, 1), pale),
                        ("GRID", (0, 0), (-1, -1), 0.5, line),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 9),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                ),
            )
        )

        story.append(paragraph("Executive summary", heading_style))
        story.append(paragraph(report.get("executive_summary") or "No summary was produced."))

        story.append(paragraph("Work completed", heading_style))
        story.extend(bullet_items(report.get("completed_work") or []))
        story.append(paragraph("Remaining work", heading_style))
        story.extend(bullet_items(report.get("remaining_work") or []))

        story.append(paragraph("Issues and watch items", heading_style))
        issues = report.get("issues") or []
        if issues:
            issue_rows = [[paragraph("Priority", table_header_style), paragraph("Issue", table_header_style), paragraph("Visual evidence", table_header_style)]]
            for issue in issues:
                issue_rows.append(
                    [
                        paragraph(str(issue.get("priority") or "unknown").upper(), small_style),
                        paragraph(issue.get("issue") or "", small_style),
                        paragraph(issue.get("evidence") or "", small_style),
                    ]
                )
            story.append(
                Table(
                    issue_rows,
                    colWidths=[0.8 * inch, 2.05 * inch, doc.width - 2.85 * inch],
                    repeatRows=1,
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), red),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                            ("GRID", (0, 0), (-1, -1), 0.4, line),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ]
                    ),
                )
            )
        else:
            story.append(paragraph("No specific issues were identified from the available visual evidence."))

        timeline = report.get("timeline") or []
        if timeline:
            story.append(PageBreak())
            story.append(paragraph("Progress timeline", heading_style))
            timeline_rows = [[paragraph("Time", table_header_style), paragraph("Activity", table_header_style), paragraph("Visible progress", table_header_style), paragraph("Camera", table_header_style)]]
            for item in timeline:
                timeline_rows.append(
                    [
                        paragraph(item.get("time") or "", small_style),
                        paragraph(item.get("activity") or "", small_style),
                        paragraph(item.get("progress_note") or "", small_style),
                        paragraph(", ".join(str(name) for name in item.get("cameras") or []), small_style),
                    ]
                )
            story.append(
                Table(
                    timeline_rows,
                    colWidths=[0.9 * inch, 1.8 * inch, 2.65 * inch, doc.width - 5.35 * inch],
                    repeatRows=1,
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), navy),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                            ("GRID", (0, 0), (-1, -1), 0.4, line),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 5),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ]
                    ),
                )
            )

        story.append(paragraph("Camera summaries", heading_style))
        for camera in report.get("camera_summaries") or []:
            completion_value = int(camera.get("estimated_completion_percent", -1))
            completion_text = (
                f"Estimated completion: {max(0, min(100, completion_value))}%"
                if completion_value >= 0
                else "Completion not estimated"
            )
            story.append(
                KeepTogether(
                    [
                        Paragraph(f"<b>{clean(camera.get('camera_name') or 'Camera')}</b>", body_style),
                        paragraph(camera.get("summary") or "No summary available."),
                        paragraph(
                            f"{completion_text} | Confidence: {float(camera.get('confidence', 0.0)) * 100:.0f}%",
                            small_style,
                        ),
                        Spacer(1, 5),
                    ]
                )
            )

        story.append(PageBreak())
        story.append(paragraph("Representative visual evidence", heading_style))
        for camera_id, records in sorted(grouped_snapshots.items()):
            evidence = [records[0]] if len(records) == 1 else [records[0], records[-1]]
            cells = []
            for snapshot in evidence:
                image = Image(str(snapshot.image_path))
                image._restrictSize(3.1 * inch, 2.15 * inch)
                caption = Paragraph(
                    f"<b>{clean(snapshot.camera_name)}</b><br/>"
                    f"{snapshot.slot_hour:02d}:{snapshot.slot_minute:02d} scheduled | "
                    f"{clean(snapshot.captured_at)}",
                    small_style,
                )
                cells.append([image, caption])
            image_row = [cell[0] for cell in cells]
            caption_row = [cell[1] for cell in cells]
            column_width = doc.width / len(cells)
            story.append(
                KeepTogether(
                    [
                        Table(
                            [image_row, caption_row],
                            colWidths=[column_width] * len(cells),
                            style=TableStyle(
                                [
                                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                    ("BOX", (0, 0), (-1, -1), 0.5, line),
                                ]
                            ),
                        ),
                        Spacer(1, 10),
                    ]
                )
            )

        story.append(paragraph("Evidence coverage and limitations", heading_style))
        coverage_rows = [[
            paragraph("Camera", table_header_style),
            paragraph("Captured", table_header_style),
            paragraph("VLM frames", table_header_style),
            paragraph("Available minutes", table_header_style),
        ]]
        for camera in coverage["cameras"]:
            captured_times = camera["captured_times"]
            if len(captured_times) > 12:
                displayed_times = (
                    captured_times[:5]
                    + [f"... {len(captured_times) - 10} more ..."]
                    + captured_times[-5:]
                )
            else:
                displayed_times = captured_times
            coverage_rows.append(
                [
                    paragraph(camera["camera_name"], small_style),
                    paragraph(f"{camera['captured_count']} of {camera['expected_count']}", small_style),
                    paragraph(str(camera.get("analyzed_count", camera["captured_count"])), small_style),
                    paragraph(", ".join(displayed_times) or "None", small_style),
                ]
            )
        story.append(
            Table(
                coverage_rows,
                colWidths=[1.55 * inch, 0.75 * inch, 0.75 * inch, doc.width - 3.05 * inch],
                repeatRows=1,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), navy),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                        ("GRID", (0, 0), (-1, -1), 0.4, line),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                ),
            )
        )
        story.append(Spacer(1, 7))
        for note in report.get("data_notes") or []:
            story.append(Paragraph(f"- {clean(note)}", small_style))
        story.append(
            paragraph(
                "This report is based on periodic camera snapshots. It summarizes visible evidence and may not capture work that occurred between snapshots or outside camera views.",
                small_style,
            )
        )

        doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
        return pdf_path.resolve()

    def _normalize_site_time(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=self.site_timezone)
        return value.astimezone(self.site_timezone)

    def _interval_bounds(
        self,
        lookback_minutes: int,
        end_time: datetime | None,
    ) -> tuple[datetime, datetime, int]:
        if isinstance(lookback_minutes, bool):
            raise ValueError("Report duration must be a whole number of minutes.")
        try:
            duration = int(lookback_minutes)
        except (TypeError, ValueError) as exc:
            raise ValueError("Report duration must be a whole number of minutes.") from exc
        if str(lookback_minutes).strip() != str(duration) and not (
            isinstance(lookback_minutes, float) and lookback_minutes.is_integer()
        ):
            raise ValueError("Report duration must be a whole number of minutes.")
        if duration < 1:
            raise ValueError("Report duration must be at least 1 minute.")
        if duration > 10080:
            raise ValueError("Report duration cannot exceed 10,080 minutes (7 days).")
        period_end = self._normalize_site_time(end_time or self._site_now())
        return period_end - timedelta(minutes=duration), period_end, duration

    @staticmethod
    def _expected_minute_slots(
        period_start: datetime,
        period_end: datetime,
    ) -> list[datetime]:
        cursor = period_start.replace(second=0, microsecond=0)
        if cursor <= period_start:
            cursor += timedelta(minutes=1)
        last_slot = period_end.replace(second=0, microsecond=0)
        slots = []
        while cursor <= last_slot:
            slots.append(cursor)
            cursor += timedelta(minutes=1)
        return slots

    def _snapshot_time(self, snapshot: ReportingSnapshot) -> datetime:
        parsed = datetime.fromisoformat(snapshot.scheduled_at)
        return self._normalize_site_time(parsed)

    def _display_snapshot_time(self, snapshot: ReportingSnapshot) -> str:
        return self._snapshot_time(snapshot).strftime("%m-%d %H:%M")

    @staticmethod
    def _summarize_times(times: list[str], limit: int = 12) -> list[str]:
        if len(times) <= limit:
            return times
        edge_count = max(1, (limit - 1) // 2)
        return times[:edge_count] + [f"... {len(times) - (2 * edge_count)} more ..."] + times[-edge_count:]

    def resolve_report_path(self, filename: str) -> Path | None:
        clean_name = Path(str(filename or "")).name
        if clean_name != filename or not clean_name.lower().endswith(".pdf"):
            return None
        candidate = (self.output_dir / clean_name).resolve()
        if candidate.parent != self.output_dir or not candidate.is_file():
            return None
        return candidate

    def _resolve_zone(self, value: str | None) -> str:
        clean_value = str(value or "").strip()
        if not clean_value:
            raise ValueError(
                "A zone is required for a construction progress report. Available zones: "
                + (", ".join(sorted(self.zones)) or "none")
                + "."
            )
        zone = self._zone_lookup.get(clean_value.casefold())
        if zone is None:
            raise ValueError(
                f"Unknown zone '{clean_value}'. Available zones: "
                + (", ".join(sorted(self.zones)) or "none")
                + "."
            )
        return zone

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
        return slug or "zone"

    @staticmethod
    def _validate_report_date(value: str) -> str:
        try:
            return date.fromisoformat(str(value)).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError("Report date must use YYYY-MM-DD format.") from exc

    def _log(self, **payload: Any) -> None:
        if self.log_callback is None:
            return
        try:
            self.log_callback(**payload)
        except Exception:
            pass
