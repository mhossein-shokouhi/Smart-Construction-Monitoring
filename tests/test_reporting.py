import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import reporting
import supervisor
from reporting import ConstructionReporting


CAMERA_ANALYSIS = {
    "camera_id": 0,
    "camera_name": "Camera",
    "camera_location": "",
    "summary": "Excavation expanded and grading began.",
    "start_state": "Area was largely unexcavated.",
    "end_state": "Most soil was removed and grading was visible.",
    "estimated_completion_percent": 70,
    "confidence": 0.84,
    "completed_work": ["Bulk soil removal progressed."],
    "remaining_work": ["Finish grading the eastern edge."],
    "issues": ["Eastern edge remains uneven."],
    "data_quality_notes": [],
    "observations": [
        {
            "time": "09:00",
            "activity": "Excavator removing soil.",
            "visible_change_since_previous": "Initial reference frame.",
            "progress_delta": "unknown",
            "estimated_completion_percent": 20,
            "issues": [],
            "confidence": 0.82,
        },
        {
            "time": "10:00",
            "activity": "Excavator continuing removal.",
            "visible_change_since_previous": "Excavated region is larger.",
            "progress_delta": "moderate",
            "estimated_completion_percent": 35,
            "issues": [],
            "confidence": 0.86,
        },
    ],
}

FINAL_REPORT = {
    "title": "Construction Progress Report",
    "executive_summary": "Excavation progressed and grading began.",
    "overall_completion_percent": 72,
    "overall_confidence": 0.83,
    "completed_work": ["Most bulk soil was removed."],
    "remaining_work": ["Finish grading the eastern edge."],
    "issues": [
        {
            "issue": "Uneven eastern edge",
            "evidence": "Visible in the final snapshots.",
            "priority": "medium",
        }
    ],
    "timeline": [
        {
            "time": "09:00-10:00",
            "activity": "Bulk excavation",
            "progress_note": "Excavated region expanded.",
            "cameras": ["North Camera", "South Camera"],
        }
    ],
    "camera_summaries": [
        {
            "camera_name": "North Camera",
            "summary": "Excavation expanded.",
            "estimated_completion_percent": 70,
            "confidence": 0.84,
        }
    ],
    "data_notes": ["Minute snapshots were available for each camera."],
}


class FakeResponse:
    def __init__(self, output_text):
        self.output_text = output_text


class RoutingResponses:
    def __init__(self):
        self.requests = []
        self.lock = threading.Lock()

    def create(self, **kwargs):
        with self.lock:
            self.requests.append(kwargs)
        schema_name = kwargs["text"]["format"]["name"]
        payload = CAMERA_ANALYSIS if schema_name == "camera_progress_observation" else FINAL_REPORT
        return FakeResponse(json.dumps(payload))


class FakeClient:
    def __init__(self):
        self.responses = RoutingResponses()


class FakeFrameResponse:
    def __init__(self, content, receive_time):
        self.content = content
        self.headers = {"X-Receive-Time": str(receive_time)}

    def raise_for_status(self):
        return None


class ReportingTests(unittest.TestCase):
    def _service(self, root, *, cameras=None, client=None, max_frames_per_camera=24):
        return ConstructionReporting(
            client=client or FakeClient(),
            receiver_url="http://receiver",
            cameras=cameras
            or {
                0: {"name": "North Camera", "location": "Area B", "zone": "Zone A"},
                1: {"name": "South Camera", "location": "Area B", "zone": "Zone A"},
            },
            site_timezone="America/Vancouver",
            snapshot_root=Path(root) / "snapshots",
            output_dir=Path(root) / "output" / "pdf",
            max_frame_age_sec=10,
            max_frames_per_camera=max_frames_per_camera,
        )

    def test_default_snapshot_root_is_persistent_and_repository_local(self):
        service = ConstructionReporting(
            client=FakeClient(),
            receiver_url="http://receiver",
            cameras={0: {"name": "Camera 0", "zone": "Zone 0"}},
        )

        self.assertEqual(
            service.snapshot_root,
            Path(reporting.__file__).resolve().parent / "data" / "reporting_snapshots",
        )

    def test_minute_capture_saves_once_per_camera_then_advances_next_minute(self):
        current = datetime(2026, 8, 18, 9, 12, 3, tzinfo=ZoneInfo("America/Vancouver"))
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(temporary_dir)
            response = FakeFrameResponse(b"jpeg-bytes", current.timestamp())
            with patch.object(reporting.requests, "get", return_value=response) as get_frame:
                first = service.capture_due(current)
                second = service.capture_due(current)
                response.headers["X-Receive-Time"] = str((current + timedelta(minutes=1)).timestamp())
                third = service.capture_due(current + timedelta(minutes=1))

            self.assertEqual(first["status"], "ok")
            self.assertEqual(first["captured"], [0, 1])
            self.assertEqual(second["already_present"], [0, 1])
            self.assertEqual(third["captured"], [0, 1])
            self.assertEqual(get_frame.call_count, 4)
            snapshots = service.list_snapshots("2026-08-18")
            self.assertEqual(len(snapshots), 4)
            self.assertTrue(all(item.slot_hour == 9 for item in snapshots))
            self.assertEqual({item.slot_minute for item in snapshots}, {12, 13})
            self.assertTrue(all(item.image_path.read_bytes() == b"jpeg-bytes" for item in snapshots))
            relative_paths = {
                item.image_path.relative_to(service.snapshot_root).as_posix()
                for item in snapshots
            }
            self.assertEqual(
                relative_paths,
                {
                    "2026-08-18/zone-a/camera-0-north-camera/09-12.jpg",
                    "2026-08-18/zone-a/camera-0-north-camera/09-13.jpg",
                    "2026-08-18/zone-a/camera-1-south-camera/09-12.jpg",
                    "2026-08-18/zone-a/camera-1-south-camera/09-13.jpg",
                },
            )
            self.assertEqual(first["snapshot_root"], str(service.snapshot_root))

    def test_capture_rejects_stale_frames_and_retries_same_minute(self):
        current = datetime(2026, 8, 18, 10, 15, tzinfo=ZoneInfo("America/Vancouver"))
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(
                temporary_dir,
                cameras={0: {"name": "North Camera", "location": "Area B", "zone": "Zone A"}},
            )
            stale = FakeFrameResponse(b"old", current.timestamp() - 60)
            fresh = FakeFrameResponse(b"new", current.timestamp())
            with patch.object(reporting.requests, "get", side_effect=[stale, fresh]) as get_frame:
                first = service.capture_due(current)
                second = service.capture_due(current)

            self.assertEqual(first["status"], "error")
            self.assertIn("stale", first["failed"][0]["error"])
            self.assertEqual(second["status"], "ok")
            self.assertEqual(second["captured"], [0])
            self.assertEqual(get_frame.call_count, 2)

    def test_capture_runs_all_day_instead_of_only_working_hours(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(
                temporary_dir,
                cameras={0: {"name": "North", "location": "A", "zone": "Zone A"}},
            )
            timezone = ZoneInfo("America/Vancouver")
            before_time = datetime(2026, 8, 18, 8, 59, tzinfo=timezone)
            after_time = datetime(2026, 8, 18, 18, 0, tzinfo=timezone)
            with patch.object(
                reporting.requests,
                "get",
                side_effect=[
                    FakeFrameResponse(b"early", before_time.timestamp()),
                    FakeFrameResponse(b"late", after_time.timestamp()),
                ],
            ):
                before = service.capture_due(before_time)
                after = service.capture_due(after_time)

            self.assertEqual(before["status"], "ok")
            self.assertEqual(after["status"], "ok")

    def test_background_recorder_starts_once_and_stops_cleanly(self):
        outside_window = datetime(
            2026,
            8,
            18,
            8,
            0,
            tzinfo=ZoneInfo("America/Vancouver"),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(temporary_dir)
            service.now_provider = lambda: outside_window
            service.start()
            first_thread = service._thread
            service.start()

            self.assertIs(service._thread, first_thread)
            self.assertTrue(service.is_running())
            service.stop()
            self.assertFalse(service.is_running())

    def test_five_minute_report_uses_each_minute_per_camera_then_text_synthesis(self):
        current = datetime(2026, 8, 18, 10, 5, 30, tzinfo=ZoneInfo("America/Vancouver"))
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(temporary_dir, client=client)
            for minute in range(1, 6):
                slot = current.replace(minute=minute, second=0)
                with patch.object(
                    reporting.requests,
                    "get",
                    return_value=FakeFrameResponse(b"image", slot.timestamp()),
                ):
                    service.capture_due(slot)

            pdf_path = Path(temporary_dir) / "output" / "pdf" / "report.pdf"
            with patch.object(service, "_site_now", return_value=current), patch.object(
                service,
                "_write_pdf",
                return_value=pdf_path,
            ):
                result = service.generate_interval_report(
                    5,
                    "Zone A",
                    "Excavate and level Area B.",
                    end_time=current,
                )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["lookback_minutes"], 5)
            self.assertEqual(result["snapshot_count"], 10)
            self.assertEqual(result["camera_count"], 2)
            self.assertEqual(len(client.responses.requests), 3)
            camera_requests = [
                request
                for request in client.responses.requests
                if request["text"]["format"]["name"] == "camera_progress_observation"
            ]
            self.assertEqual(len(camera_requests), 2)
            for request in camera_requests:
                content = request["input"][0]["content"]
                self.assertEqual(
                    sum(item["type"] == "input_image" for item in content),
                    5,
                )
                prompt_text = " ".join(
                    item.get("text", "") for item in content if item["type"] == "input_text"
                )
                self.assertIn("10:01", prompt_text)
                self.assertIn("10:05", prompt_text)
                self.assertIn("Excavate and level Area B", prompt_text)
                self.assertIn("Zone: Zone A", prompt_text)
            synthesis = next(
                request
                for request in client.responses.requests
                if request["text"]["format"]["name"] == "construction_progress_interval_report"
            )
            self.assertEqual(
                [item["type"] for item in synthesis["input"][0]["content"]],
                ["input_text"],
            )

    def test_missing_goal_instructs_model_not_to_invent_completion(self):
        current = datetime(2026, 8, 18, 9, 0, tzinfo=ZoneInfo("America/Vancouver"))
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(
                temporary_dir,
                cameras={0: {"name": "North Camera", "location": "Area B", "zone": "Zone A"}},
                client=client,
            )
            with patch.object(
                reporting.requests,
                "get",
                return_value=FakeFrameResponse(b"image", current.timestamp()),
            ):
                service.capture_due(current)

            with patch.object(service, "_site_now", return_value=current), patch.object(
                service,
                "_write_pdf",
                return_value=Path(temporary_dir) / "report.pdf",
            ):
                result = service.generate_interval_report(1, "Zone A", end_time=current)

            self.assertEqual(result["status"], "ok")
            camera_request = client.responses.requests[0]
            prompt = camera_request["input"][0]["content"][0]["text"]
            self.assertIn("use -1", prompt)

    def test_zone_filter_keeps_other_zone_frames_out_of_report(self):
        current = datetime(2026, 8, 18, 9, 0, tzinfo=ZoneInfo("America/Vancouver"))
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(
                temporary_dir,
                cameras={
                    0: {"name": "North", "location": "A", "zone": "Zone A"},
                    1: {"name": "South", "location": "B", "zone": "Zone B"},
                },
                client=client,
            )
            with patch.object(
                reporting.requests,
                "get",
                return_value=FakeFrameResponse(b"image", current.timestamp()),
            ):
                service.capture_due(current)

            self.assertEqual(len(service.list_snapshots("2026-08-18", "Zone A")), 1)
            self.assertEqual(len(service.list_snapshots("2026-08-18", "Zone B")), 1)
            with patch.object(service, "_site_now", return_value=current), patch.object(
                service,
                "_write_pdf",
                return_value=Path(temporary_dir) / "zone-a.pdf",
            ):
                result = service.generate_interval_report(1, "Zone A", end_time=current)

            self.assertEqual(result["zone"], "Zone A")
            self.assertEqual(result["camera_count"], 1)
            vision_requests = [
                request for request in client.responses.requests
                if request["text"]["format"]["name"] == "camera_progress_observation"
            ]
            self.assertEqual(len(vision_requests), 1)
            self.assertIn("Camera: North", vision_requests[0]["input"][0]["content"][0]["text"])

    def test_report_requires_a_known_zone(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(temporary_dir)
            with self.assertRaisesRegex(ValueError, "zone is required"):
                service.generate_interval_report(5)
            with self.assertRaisesRegex(ValueError, "Unknown zone"):
                service.generate_interval_report(5, "Moon Base")

    def test_long_interval_evenly_limits_vlm_frames_but_keeps_full_coverage(self):
        timezone = ZoneInfo("America/Vancouver")
        end_time = datetime(2026, 8, 18, 12, 0, 30, tzinfo=timezone)
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(
                temporary_dir,
                cameras={0: {"name": "North", "location": "A", "zone": "Zone A"}},
                client=client,
                max_frames_per_camera=12,
            )
            for offset in range(120):
                slot = end_time.replace(second=0) - timedelta(minutes=119 - offset)
                with patch.object(
                    reporting.requests,
                    "get",
                    return_value=FakeFrameResponse(b"image", slot.timestamp()),
                ):
                    service.capture_due(slot)
            with patch.object(service, "_site_now", return_value=end_time), patch.object(
                service,
                "_write_pdf",
                return_value=Path(temporary_dir) / "report.pdf",
            ):
                result = service.generate_interval_report(120, "Zone A", end_time=end_time)

            request = next(
                item for item in client.responses.requests
                if item["text"]["format"]["name"] == "camera_progress_observation"
            )
            content = request["input"][0]["content"]
            self.assertEqual(sum(item["type"] == "input_image" for item in content), 12)
            self.assertEqual(result["snapshot_count"], 120)
            synthesis = next(
                item for item in client.responses.requests
                if item["text"]["format"]["name"] == "construction_progress_interval_report"
            )
            synthesis_prompt = synthesis["input"][0]["content"][0]["text"]
            self.assertNotIn('"captured_times"', synthesis_prompt)
            self.assertIn('"first_captured_time"', synthesis_prompt)
            self.assertIn('"last_captured_time"', synthesis_prompt)
            coverage = service.get_status(120, "Zone A", end_time=end_time)
            self.assertEqual(coverage["camera_coverage"][0]["captured_count"], 120)
            self.assertEqual(coverage["camera_coverage"][0]["expected_count"], 120)
            self.assertLessEqual(len(coverage["camera_coverage"][0]["captured_times"]), 12)

    def test_missing_minutes_reduce_coverage_without_preventing_report(self):
        timezone = ZoneInfo("America/Vancouver")
        end_time = datetime(2026, 8, 18, 10, 5, 30, tzinfo=timezone)
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(
                temporary_dir,
                cameras={0: {"name": "North", "location": "A", "zone": "Zone A"}},
            )
            for minute in (1, 3, 5):
                slot = end_time.replace(minute=minute, second=0)
                with patch.object(
                    reporting.requests,
                    "get",
                    return_value=FakeFrameResponse(b"image", slot.timestamp()),
                ):
                    service.capture_due(slot)
            with patch.object(service, "_site_now", return_value=end_time), patch.object(
                service,
                "_write_pdf",
                return_value=Path(temporary_dir) / "report.pdf",
            ):
                result = service.generate_interval_report(5, "Zone A", end_time=end_time)

            self.assertEqual(result["status"], "ok")
            status = service.get_status(5, "Zone A", end_time=end_time)
            self.assertEqual(status["camera_coverage"][0]["captured_count"], 3)
            self.assertEqual(status["camera_coverage"][0]["expected_count"], 5)

    def test_report_duration_validation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(temporary_dir)
            for duration in (0, -1, 10081, 2.5, None):
                with self.subTest(duration=duration), self.assertRaises(ValueError):
                    service.generate_interval_report(duration, "Zone A")

    def test_exact_minute_boundary_still_selects_exact_requested_slot_count(self):
        timezone = ZoneInfo("America/Vancouver")
        end_time = datetime(2026, 8, 18, 10, 5, 0, tzinfo=timezone)
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(
                temporary_dir,
                cameras={0: {"name": "North", "location": "A", "zone": "Zone A"}},
            )
            for minute in range(6):
                slot = end_time.replace(minute=minute)
                with patch.object(
                    reporting.requests,
                    "get",
                    return_value=FakeFrameResponse(b"image", slot.timestamp()),
                ):
                    service.capture_due(slot)

            status = service.get_status(5, "Zone A", end_time=end_time)

            self.assertEqual(status["expected_minute_count"], 5)
            self.assertEqual(status["snapshot_count"], 5)
            self.assertEqual(status["camera_coverage"][0]["expected_count"], 5)

    def test_two_hour_interval_can_cross_midnight(self):
        timezone = ZoneInfo("America/Vancouver")
        end_time = datetime(2026, 8, 19, 0, 30, 30, tzinfo=timezone)
        slots = [
            datetime(2026, 8, 18, 23, 0, 0, tzinfo=timezone),
            datetime(2026, 8, 19, 0, 30, 0, tzinfo=timezone),
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(
                temporary_dir,
                cameras={0: {"name": "North", "location": "A", "zone": "Zone A"}},
            )
            for slot in slots:
                with patch.object(
                    reporting.requests,
                    "get",
                    return_value=FakeFrameResponse(b"image", slot.timestamp()),
                ):
                    service.capture_due(slot)

            status = service.get_status(120, "Zone A", end_time=end_time)

            self.assertEqual(status["snapshot_count"], 2)
            self.assertEqual(status["camera_coverage"][0]["expected_count"], 120)
            self.assertEqual(
                status["camera_coverage"][0]["captured_times"],
                ["08-18 23:00", "08-19 00:30"],
            )

    def test_supervisor_generates_one_report_per_zone_without_changing_modes(self):
        zones = list(supervisor.ZONE_NAMES[:2])
        original_states = supervisor._zone_states
        supervisor._zone_states = {
            zone: {"mode": "safety" if zone == zones[0] else "free", "objective": None}
            for zone in supervisor.ZONE_NAMES
        }
        with supervisor._operational_lock:
            before = {zone: dict(state) for zone, state in supervisor._zone_states.items()}
        try:
            with patch.object(
                supervisor.reporting_service,
                "generate_interval_report",
                side_effect=lambda lookback_minutes, zone, goal: {
                    "status": "ok",
                    "zone": zone,
                    "report_url": f"/reports/{zone.lower().replace(' ', '-')}.pdf",
                },
            ) as generate:
                result = supervisor.execute_supervisor_tool(
                    "generate_progress_report",
                    {
                        "lookback_minutes": 120,
                        "zones": zones,
                        "goal": "Level the work area",
                    },
                )

            self.assertEqual(result["status"], "ok")
            self.assertEqual([report["zone"] for report in result["reports"]], zones)
            self.assertEqual(generate.call_count, 2)
            for zone in zones:
                generate.assert_any_call(120, zone, "Level the work area")
            with supervisor._operational_lock:
                self.assertEqual(supervisor._zone_states, before)
        finally:
            supervisor._zone_states = original_states


if __name__ == "__main__":
    unittest.main()
