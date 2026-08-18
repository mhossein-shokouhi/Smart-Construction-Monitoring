import json
import os
import tempfile
import threading
import unittest
from datetime import datetime
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
    "title": "Daily Construction Progress",
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
    "data_notes": ["Two hourly slots were available per camera."],
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
    def _service(self, root, *, cameras=None, client=None):
        return ConstructionReporting(
            client=client or FakeClient(),
            receiver_url="http://receiver",
            cameras=cameras
            or {
                0: {"name": "North Camera", "location": "Area B"},
                1: {"name": "South Camera", "location": "Area B"},
            },
            site_timezone="America/Vancouver",
            snapshot_root=Path(root) / "snapshots",
            output_dir=Path(root) / "output" / "pdf",
            max_frame_age_sec=10,
        )

    def test_hourly_capture_saves_one_fresh_frame_per_camera_and_deduplicates(self):
        current = datetime(2026, 8, 18, 9, 0, 3, tzinfo=ZoneInfo("America/Vancouver"))
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(temporary_dir)
            response = FakeFrameResponse(b"jpeg-bytes", current.timestamp())
            with patch.object(reporting.requests, "get", return_value=response) as get_frame:
                first = service.capture_due(current)
                second = service.capture_due(current)

            self.assertEqual(first["status"], "ok")
            self.assertEqual(first["captured"], [0, 1])
            self.assertEqual(second["already_present"], [0, 1])
            self.assertEqual(get_frame.call_count, 2)
            snapshots = service.list_snapshots("2026-08-18")
            self.assertEqual(len(snapshots), 2)
            self.assertTrue(all(item.slot_hour == 9 for item in snapshots))
            self.assertTrue(all(item.image_path.read_bytes() == b"jpeg-bytes" for item in snapshots))

    def test_capture_rejects_stale_frames_and_retries_same_hour(self):
        current = datetime(2026, 8, 18, 10, 15, tzinfo=ZoneInfo("America/Vancouver"))
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(
                temporary_dir,
                cameras={0: {"name": "North Camera", "location": "Area B"}},
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

    def test_capture_window_includes_0900_and_1700(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(temporary_dir)
            timezone = ZoneInfo("America/Vancouver")
            before = service.capture_due(datetime(2026, 8, 18, 8, 59, tzinfo=timezone))
            after = service.capture_due(datetime(2026, 8, 18, 18, 0, tzinfo=timezone))

            self.assertEqual(before["status"], "not_due")
            self.assertEqual(after["status"], "not_due")
            self.assertIn(9, service.scheduled_hours)
            self.assertIn(17, service.scheduled_hours)

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

    def test_report_uses_one_temporal_vision_call_per_camera_then_text_synthesis(self):
        current = datetime(2026, 8, 18, 10, 0, tzinfo=ZoneInfo("America/Vancouver"))
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = self._service(temporary_dir, client=client)
            for hour in (9, 10):
                frame_time = current.replace(hour=hour).timestamp()
                with patch.object(
                    reporting.requests,
                    "get",
                    return_value=FakeFrameResponse(b"image", frame_time),
                ):
                    service.capture_due(current.replace(hour=hour))

            pdf_path = Path(temporary_dir) / "output" / "pdf" / "report.pdf"
            with patch.object(service, "_site_now", return_value=current), patch.object(
                service,
                "_write_pdf",
                return_value=pdf_path,
            ):
                result = service.generate_daily_report(
                    "2026-08-18",
                    "Excavate and level Area B by the end of the day.",
                )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["snapshot_count"], 4)
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
                    2,
                )
                prompt_text = " ".join(
                    item.get("text", "") for item in content if item["type"] == "input_text"
                )
                self.assertIn("09:00", prompt_text)
                self.assertIn("10:00", prompt_text)
                self.assertIn("Excavate and level Area B", prompt_text)
            synthesis = next(
                request
                for request in client.responses.requests
                if request["text"]["format"]["name"] == "daily_construction_progress_report"
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
                cameras={0: {"name": "North Camera", "location": "Area B"}},
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
                result = service.generate_daily_report("2026-08-18")

            self.assertEqual(result["status"], "ok")
            camera_request = client.responses.requests[0]
            prompt = camera_request["input"][0]["content"][0]["text"]
            self.assertIn("use -1", prompt)

    def test_supervisor_reporting_tool_does_not_change_operational_mode(self):
        with supervisor._operational_lock:
            supervisor._operational_state.update(mode="safety", objective=None)
        expected = {
            "status": "ok",
            "report_url": "/reports/construction-progress-2026-08-18.pdf",
        }
        with patch.object(
            supervisor.reporting_service,
            "generate_daily_report",
            return_value=expected,
        ) as generate:
            result = supervisor.execute_supervisor_tool(
                "generate_daily_report",
                {"report_date": "2026-08-18", "goal": "Level Area B"},
            )

        self.assertEqual(result, expected)
        self.assertEqual(supervisor.get_operational_mode_tool()["mode"], "safety")
        generate.assert_called_once_with("2026-08-18", "Level Area B")
        with supervisor._operational_lock:
            supervisor._operational_state.update(mode="free", objective=None)


if __name__ == "__main__":
    unittest.main()
