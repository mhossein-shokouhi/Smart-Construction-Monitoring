"""Laptop-side Safety Mode scanner for construction-site visual hazards."""

from __future__ import annotations

import base64
from datetime import datetime
import time
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import requests

from search_vlm import SearchScanner


SAFETY_HAZARDS: dict[str, dict[str, str]] = {
    "fire_smoke": {
        "name": "Fire Hazard",
        "description": (
            "Visible flames, an active fire, a smoke plume, or accumulating smoke. "
            "Do not treat ordinary dust, fog, or steam as smoke unless the visual evidence "
            "clearly supports fire or smoke."
        ),
    },
    "work_zone_encroachment": {
        "name": "Work-Zone Intrusion",
        "description": (
            "A white or predominantly light-colored leveling or grading machine is recognizable "
            "in the frame, allowing for ordinary dirt, shadow, or a partial view, and a person is "
            "visibly close enough to share the machine's immediate working space or likely path of "
            "movement. Treat a person directly in front of the machine, beside its active working "
            "area, or otherwise near enough for a plausible contact or struck-by risk as an "
            "intrusion candidate. After confirming that risky machine-person relationship, assess "
            "the same nearby person's required PPE. Set detected=true only when that person is "
            "visibly missing one or both required items: a recognizable high-visibility safety vest "
            "and a protective hard hat or safety helmet. A nearby person clearly wearing both items "
            "is compliant and must not trigger this check; wearing only one still triggers when the "
            "other item is visibly absent. Do not guess that a cropped, blurred, small, or occluded "
            "head or torso lacks PPE; when no missing item can be established reliably, set "
            "detected=false. When several people are present, evaluate each person separately and "
            "trigger if any nearby non-operator person visibly lacks either item. Do not require an "
            "exact measured distance, proof that the machine is moving, or one precise body "
            "position. Do not trigger for someone clearly far away or only in the background, or "
            "for the normal operator properly seated on the machine or at its normal rear operating "
            "position."
        ),
    },
    "machine_obstacle_proximity": {
        "name": "Obstacle Hazard",
        "description": (
            "A white or predominantly light-colored leveling or grading machine is recognizable "
            "in the frame, allowing for ordinary dirt, shadow, or a partial view, and a substantial "
            "ground obstacle is visibly close to the machine or directly in its apparent near-term "
            "travel or working path. The placement must create a plausible contact, collision, or "
            "equipment-damage risk if operation continues. Typical obstacles include traffic cones "
            "and large or wide rigid pipe or conduit sections lying on or crossing the ground, "
            "including green, blue, or white pipes, as well as similarly substantial freestanding "
            "objects. Require both the machine and obstacle plus a clear risky spatial relationship. "
            "Do not trigger merely because an obstacle appears somewhere in the same frame, when it "
            "is clearly distant, outside the machine's apparent path, or safely separated. Do not "
            "misclassify machine parts or attachments, painted markings, shadows, narrow hoses or "
            "cables, small debris, or ordinary soil texture as this hazard. Do not require proof that "
            "the machine is moving or an exact measured distance in a single frame."
        ),
    },
    "after_hours_intrusion": {
        "name": "Unauthorized Entry",
        "description": (
            "A person is visibly present in or entering the monitored construction site "
            "outside configured construction hours. The server has already checked the "
            "site-local time whenever this assessment is included."
        ),
    },
}


class SafetyScanner(SearchScanner):
    """Evaluate each sampled frame once against all currently applicable hazards."""

    def __init__(
        self,
        *,
        client,
        receiver_url: str,
        model: str,
        image_detail: str = "auto",
        reasoning_effort: str = "medium",
        sample_interval_sec: float = 1.0,
        match_threshold: float = 0.75,
        alert_cooldown_sec: float = 15.0,
        max_workers: int = 4,
        site_timezone: str = "America/Vancouver",
        access_start_hour: int = 9,
        access_end_hour: int = 17,
        now_provider: Callable[[], datetime] | None = None,
        camera_ids: Iterable[int] | None = None,
        scope_label: str | None = None,
    ) -> None:
        super().__init__(
            client=client,
            receiver_url=receiver_url,
            model=model,
            image_detail=image_detail,
            sample_interval_sec=sample_interval_sec,
            match_threshold=match_threshold,
            alert_cooldown_sec=alert_cooldown_sec,
            max_workers=max_workers,
            camera_ids=camera_ids,
            scope_label=scope_label,
        )
        if not 0 <= access_start_hour <= 23 or not 0 <= access_end_hour <= 23:
            raise ValueError("Safety access hours must be integers from 0 through 23.")
        if access_start_hour == access_end_hour:
            raise ValueError("Safety access start and end hours must differ.")
        if reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported Safety VLM reasoning effort.")

        self.reasoning_effort = reasoning_effort
        self.site_timezone_name = site_timezone
        self.site_timezone = ZoneInfo(site_timezone)
        self.access_start_hour = access_start_hour
        self.access_end_hour = access_end_hour
        self.now_provider = now_provider
        self._last_hazard_alert_time: dict[tuple[int, int, str], float] = {}

    def start(self) -> None:
        with self._lock:
            self._last_hazard_alert_time.clear()
        super().start("construction safety hazards")

    def _post_log(self, **kwargs: Any) -> None:
        message = str(kwargs.get("message") or "")
        kwargs["message"] = message.replace("Search VLM", "Safety VLM").replace(
            "Search Mode scanning",
            "Safety Mode scanning",
        )
        super()._post_log(**kwargs)

    def _site_now(self) -> datetime:
        current = self.now_provider() if self.now_provider is not None else datetime.now(self.site_timezone)
        if current.tzinfo is None:
            return current.replace(tzinfo=self.site_timezone)
        return current.astimezone(self.site_timezone)

    def is_after_hours(self, current: datetime | None = None) -> bool:
        local_time = current or self._site_now()
        if local_time.tzinfo is not None:
            local_time = local_time.astimezone(self.site_timezone)
        hour = local_time.hour
        if self.access_start_hour < self.access_end_hour:
            return hour < self.access_start_hour or hour >= self.access_end_hour
        return self.access_end_hour <= hour < self.access_start_hour

    def active_hazard_keys(self, current: datetime | None = None) -> tuple[str, ...]:
        keys = ["fire_smoke"]
        if self.is_after_hours(current):
            keys.append("after_hours_intrusion")
        else:
            keys.append("work_zone_encroachment")
            keys.append("machine_obstacle_proximity")
        return tuple(keys)

    def _scan_camera_once(self, camera_id: int, generation: int, _target: str) -> None:
        try:
            frame_bytes, frame_time = self._latest_frame(camera_id)
            if frame_bytes is None or frame_time is None:
                return
            previous_frame_time = self._last_frame_time.get(camera_id)
            if previous_frame_time is not None and frame_time <= previous_frame_time:
                return
            self._last_frame_time[camera_id] = frame_time

            active_keys = self.active_hazard_keys()
            result = self._analyze_frame(hazard_keys=active_keys, frame_bytes=frame_bytes)
            assessments = result.get("assessments")
            if not isinstance(assessments, dict):
                raise ValueError("Safety model output must contain an assessments object.")

            for hazard_key in active_keys:
                assessment = assessments.get(hazard_key)
                if not isinstance(assessment, dict) or not bool(assessment.get("detected")):
                    continue
                confidence = self._coerce_confidence(assessment.get("confidence"))
                if confidence < self.match_threshold:
                    continue

                cause = str(
                    assessment.get("cause")
                    or "The current frame contains visible evidence of this safety hazard."
                ).strip()
                now = time.time()
                alert_key = (camera_id, generation, hazard_key)
                with self._lock:
                    if generation != self._generation or self._stop_event.is_set():
                        return
                    last_alert_time = self._last_hazard_alert_time.get(alert_key, 0.0)
                    if now - last_alert_time < self.alert_cooldown_sec:
                        continue
                    if self._post_hazard(
                        hazard_key=hazard_key,
                        camera_id=camera_id,
                        confidence=confidence,
                        cause=cause,
                        frame_bytes=frame_bytes,
                    ):
                        self._last_hazard_alert_time[alert_key] = now
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
                        message=f"Safety scan failed for camera {camera_id}: {exc}",
                        camera_id=camera_id,
                    )

    def _analyze_frame(self, *, hazard_keys: tuple[str, ...], frame_bytes: bytes) -> dict[str, Any]:
        if not hazard_keys:
            return {"assessments": {}}

        data_url = "data:image/jpeg;base64," + base64.b64encode(frame_bytes).decode("ascii")
        hazard_lines = [
            f'- {key} ({SAFETY_HAZARDS[key]["name"]}): {SAFETY_HAZARDS[key]["description"]}'
            for key in hazard_keys
        ]
        proximity_guidance = ""
        if "work_zone_encroachment" in hazard_keys:
            proximity_guidance += (
                " For Work-Zone Intrusion, use practical scene-level judgment about proximity. "
                "Account for camera perspective, relative scale, overlap, and apparent ground "
                "position. Favor detection when a reasonable observer would consider the person "
                "inside the machine's immediate operating area; do not demand exact geometric "
                "boundaries. Decide machine-person proximity first, then assess the required vest "
                "and helmet on that same person. A normal bright shirt is not automatically a "
                "safety vest, and a cap or hood is not a hard hat or safety helmet. Do not combine "
                "PPE worn by different people. A positive cause must state whether the nearby "
                "person visibly lacks the vest, the helmet, or both."
            )
        if "machine_obstacle_proximity" in hazard_keys:
            proximity_guidance += (
                " For Obstacle Hazard, judge the machine and obstacle as a pair using perspective, "
                "relative scale, overlap, apparent ground position, and the machine's visible "
                "orientation or working area. Detect when a traffic cone, wide rigid pipe, or "
                "similarly substantial object is close enough or placed directly enough in the "
                "apparent path that continued operation could reasonably contact it. Object type "
                "or color alone is not sufficient, and mere co-occurrence in the frame is negative. "
                "If the object is ambiguous, clearly distant, off-path, or part of the machine, set "
                "detected=false."
            )
        prompt = (
            "You are the visual safety monitor for a construction-site camera system. "
            "Evaluate this single frame independently against every hazard check listed below. "
            "One frame may contain multiple hazards, so never stop after the first positive. "
            "Use only visible evidence in the image; if evidence is ambiguous, set detected=false. "
            "A positive assessment must satisfy the applicable hazard description as a whole."
            + proximity_guidance
            + " "
            "For each positive assessment, give a concise cause that states exactly what is visible. "
            "For each negative assessment, use an empty cause. The application—not you—handles "
            "time rules and alert latching.\n\nApplicable checks:\n"
            + "\n".join(hazard_lines)
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
            text={
                "format": {
                    "type": "json_schema",
                    "name": "construction_safety_assessment",
                    "strict": True,
                    "schema": self._response_schema(hazard_keys),
                }
            },
            reasoning={"effort": self.reasoning_effort},
        )
        return self._parse_model_json(getattr(response, "output_text", "") or "")

    @staticmethod
    def _response_schema(hazard_keys: tuple[str, ...]) -> dict[str, Any]:
        assessment_schema = {
            "type": "object",
            "properties": {
                "detected": {"type": "boolean"},
                "confidence": {"type": "number"},
                "cause": {"type": "string"},
            },
            "required": ["detected", "confidence", "cause"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "assessments": {
                    "type": "object",
                    "properties": {key: assessment_schema for key in hazard_keys},
                    "required": list(hazard_keys),
                    "additionalProperties": False,
                }
            },
            "required": ["assessments"],
            "additionalProperties": False,
        }

    def _post_hazard(
        self,
        *,
        hazard_key: str,
        camera_id: int,
        confidence: float,
        cause: str,
        frame_bytes: bytes,
    ) -> bool:
        payload = {
            "hazard_key": hazard_key,
            "camera_id": camera_id,
            "confidence": confidence,
            "cause": cause,
            "frame_jpeg_b64": base64.b64encode(frame_bytes).decode("ascii"),
        }
        try:
            response = requests.post(
                f"{self.receiver_url}/system/safety/hazard",
                json=payload,
                timeout=2,
            )
            response.raise_for_status()
            return True
        except Exception:
            return False
