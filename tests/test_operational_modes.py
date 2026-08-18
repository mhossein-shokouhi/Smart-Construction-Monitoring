import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import stream_receiver_server
import supervisor
from safety_vlm import SafetyScanner
from search_vlm import SearchScanner


class FakeSearchScanner:
    def __init__(self):
        self.running = False
        self.started_targets = []
        self.updated_targets = []
        self.stop_count = 0

    def start(self, target):
        self.started_targets.append(target)
        self.running = True

    def update_target(self, target):
        self.updated_targets.append(target)

    def stop(self):
        self.stop_count += 1
        self.running = False

    def is_running(self):
        return self.running


class FakeSafetyScanner:
    def __init__(self):
        self.running = False
        self.start_count = 0
        self.stop_count = 0

    def start(self):
        self.start_count += 1
        self.running = True

    def stop(self):
        self.stop_count += 1
        self.running = False

    def is_running(self):
        return self.running


class FakeResponses:
    def __init__(self, output_text=None):
        self.last_request = None
        self.create_count = 0
        self.output_text = output_text or (
            '{"match": true, "confidence": 0.93, "summary": "Target visible."}'
        )

    def create(self, **kwargs):
        self.create_count += 1
        self.last_request = kwargs
        return type(
            "FakeResponse",
            (),
            {"output_text": self.output_text},
        )()


class FakeClient:
    def __init__(self, output_text=None):
        self.responses = FakeResponses(output_text)


class OperationalModeTests(unittest.TestCase):
    def setUp(self):
        self.scanner = FakeSearchScanner()
        self.safety_scanner = FakeSafetyScanner()
        self.scanner_patch = patch.object(supervisor, "search_scanner", self.scanner)
        self.safety_scanner_patch = patch.object(
            supervisor,
            "safety_scanner",
            self.safety_scanner,
        )
        self.scanner_patch.start()
        self.safety_scanner_patch.start()
        with supervisor._operational_lock:
            supervisor._operational_state.update(mode="free", objective=None)

    def tearDown(self):
        self.safety_scanner_patch.stop()
        self.scanner_patch.stop()

    def test_search_requires_a_target_without_changing_mode(self):
        result = supervisor.set_operational_mode("search")

        self.assertEqual(result["status"], "error")
        self.assertIn("requires", result["error"])
        self.assertEqual(supervisor.get_operational_mode_tool()["mode"], "free")

    def test_reporting_is_not_an_operational_mode(self):
        result = supervisor.set_operational_mode("reporting")

        self.assertEqual(result["status"], "error")
        self.assertNotIn("reporting", supervisor.OPERATIONAL_MODES)
        self.assertNotIn("reporting", supervisor.PLACEHOLDER_OPERATIONAL_MODES)
        self.assertEqual(supervisor.get_operational_mode_tool()["mode"], "free")

    @patch.object(supervisor, "_post_receiver_json", return_value=True)
    @patch.object(supervisor, "call_pi_set_mode")
    def test_free_stops_search_and_normalizes_every_camera(self, set_mode, post_json):
        set_mode.side_effect = lambda camera_id, mode: {
            "status": "ok",
            "camera_id": camera_id,
            "mode": mode,
        }
        self.scanner.running = True
        with supervisor._operational_lock:
            supervisor._operational_state.update(
                mode="search",
                objective="red fire extinguisher",
            )

        result = supervisor.set_operational_mode("free", "ignored objective")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "free")
        self.assertIsNone(result["objective"])
        self.assertFalse(result["scanner_running"])
        self.assertFalse(result["placeholder"])
        self.assertTrue(result["receiver_synced"])
        self.assertEqual(self.scanner.stop_count, 1)
        self.assertCountEqual(
            [call.args for call in set_mode.call_args_list],
            [(camera_id, "default") for camera_id in supervisor.CAMERAS],
        )
        state_payloads = [
            call.args[1]
            for call in post_json.call_args_list
            if call.args[0] == "/system/state"
        ]
        self.assertEqual(state_payloads[-1]["mode"], "free")
        self.assertIsNone(state_payloads[-1]["objective"])
        self.assertFalse(state_payloads[-1]["scanner_running"])
        self.assertFalse(state_payloads[-1]["placeholder"])

    @patch.object(supervisor, "_post_receiver_json", return_value=True)
    @patch.object(supervisor, "call_pi_set_mode")
    def test_free_reports_partial_camera_normalization(self, set_mode, _post):
        failed_camera_id = min(supervisor.CAMERAS)

        def set_default(camera_id, mode):
            if camera_id == failed_camera_id:
                return {"status": "error", "error": "offline"}
            return {"status": "ok", "camera_id": camera_id, "mode": mode}

        set_mode.side_effect = set_default
        result = supervisor.set_operational_mode("free")

        self.assertEqual(result["status"], "partial_error")
        self.assertEqual(result["mode"], "free")
        self.assertFalse(result["scanner_running"])
        self.assertIn(str(failed_camera_id), result["message"])
        self.assertEqual(len(result["camera_results"]), len(supervisor.CAMERAS))

    @patch.object(supervisor, "call_pi_set_mode", return_value={"status": "ok"})
    def test_free_receiver_recovery_never_restarts_search(self, _set_mode):
        self.scanner.running = True
        with supervisor._operational_lock:
            supervisor._operational_state.update(
                mode="search",
                objective="blue backpack",
            )

        with patch.object(supervisor, "_post_receiver_json", return_value=False):
            result = supervisor.set_operational_mode("free")

        self.assertEqual(result["status"], "partial_error")
        self.assertFalse(result["receiver_synced"])
        self.assertFalse(self.scanner.running)

        with patch.object(supervisor, "_post_receiver_json", return_value=True):
            self.assertTrue(supervisor._reconcile_operational_state())

        self.assertFalse(self.scanner.running)
        self.assertEqual(self.scanner.started_targets, [])

    @patch.object(supervisor, "_post_receiver_json")
    @patch.object(supervisor, "call_pi_set_mode")
    def test_search_configures_every_camera_and_starts_scanner(self, set_mode, post_json):
        set_mode.side_effect = lambda camera_id, mode: {
            "status": "ok",
            "camera_id": camera_id,
            "mode": mode,
        }

        result = supervisor.set_operational_mode("search", "red fire extinguisher")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "search")
        self.assertFalse(result["placeholder"])
        self.assertEqual(self.scanner.started_targets, ["red fire extinguisher"])
        self.assertCountEqual(
            [call.args for call in set_mode.call_args_list],
            [(camera_id, "default") for camera_id in supervisor.CAMERAS],
        )
        state_payloads = [
            call.args[1]
            for call in post_json.call_args_list
            if call.args[0] == "/system/state"
        ]
        self.assertEqual(state_payloads[-1]["mode"], "search")
        self.assertEqual(state_payloads[-1]["objective"], "red fire extinguisher")
        self.assertTrue(state_payloads[-1]["scanner_running"])

    @patch.object(supervisor, "_post_receiver_json")
    @patch.object(supervisor, "call_pi_set_mode", return_value={"status": "ok"})
    def test_active_search_retargets_after_publishing_new_objective(self, set_mode, post_json):
        supervisor.set_operational_mode("search", "blue backpack")
        set_mode.reset_mock()
        state_events = []

        original_start = self.scanner.start
        original_stop = self.scanner.stop

        def record_start(target):
            state_events.append(("start", target))
            original_start(target)

        def record_stop():
            state_events.append(("stop", None))
            original_stop()

        def record_post(path, payload, **_kwargs):
            if path == "/system/state":
                state_events.append(
                    ("publish", payload["objective"], payload["scanner_running"])
                )
            return {"status": "ok"}

        self.scanner.start = record_start
        self.scanner.stop = record_stop
        post_json.side_effect = record_post
        supervisor.set_operational_mode("search", "yellow hard hat")

        self.assertEqual(
            state_events,
            [
                ("stop", None),
                ("publish", "yellow hard hat", False),
                ("publish", "yellow hard hat", False),
                ("start", "yellow hard hat"),
                ("publish", "yellow hard hat", True),
            ],
        )
        self.assertCountEqual(
            [call.args for call in set_mode.call_args_list],
            [(camera_id, "default") for camera_id in supervisor.CAMERAS],
        )
        self.assertEqual(
            supervisor.get_operational_mode_tool()["objective"],
            "yellow hard hat",
        )

    @patch.object(supervisor, "call_pi_set_mode", return_value={"status": "ok"})
    def test_search_waits_for_receiver_sync_and_recovers_automatically(self, _set_mode):
        with patch.object(supervisor, "_post_receiver_json", return_value=False):
            result = supervisor.set_operational_mode("search", "blue backpack")

        self.assertEqual(result["status"], "partial_error")
        self.assertFalse(result["receiver_synced"])
        self.assertFalse(self.scanner.running)

        with patch.object(supervisor, "_post_receiver_json", return_value=True):
            self.assertTrue(supervisor._reconcile_operational_state())

        self.assertTrue(self.scanner.running)
        self.assertEqual(self.scanner.started_targets, ["blue backpack"])

    @patch.object(supervisor, "call_pi_set_mode", return_value={"status": "ok"})
    def test_search_reports_pending_running_status_without_stopping_scanner(self, _set_mode):
        state_results = iter([True, True, False])

        def post_with_failed_running_state(path, _payload):
            if path == "/system/state":
                return next(state_results)
            return True

        with patch.object(
            supervisor,
            "_post_receiver_json",
            side_effect=post_with_failed_running_state,
        ):
            result = supervisor.set_operational_mode("search", "blue backpack")

        self.assertEqual(result["status"], "partial_error")
        self.assertFalse(result["receiver_synced"])
        self.assertTrue(result["scanner_running"])
        self.assertTrue(self.scanner.running)
        self.assertIn("status update is pending", result["message"])

    @patch.object(supervisor, "_post_receiver_json", return_value=True)
    @patch.object(supervisor, "call_pi_set_mode")
    def test_safety_configures_every_camera_and_starts_one_pass_scanner(
        self,
        set_mode,
        post_json,
    ):
        set_mode.side_effect = lambda camera_id, mode: {
            "status": "ok",
            "camera_id": camera_id,
            "mode": mode,
        }

        result = supervisor.set_operational_mode("safety", "ignored objective")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "safety")
        self.assertIsNone(result["objective"])
        self.assertFalse(result["placeholder"])
        self.assertTrue(result["scanner_running"])
        self.assertEqual(self.safety_scanner.start_count, 1)
        self.assertEqual(self.scanner.stop_count, 1)
        self.assertCountEqual(
            [call.args for call in set_mode.call_args_list],
            [(camera_id, "default") for camera_id in supervisor.CAMERAS],
        )
        state_payloads = [
            call.args[1]
            for call in post_json.call_args_list
            if call.args[0] == "/system/state"
        ]
        self.assertFalse(state_payloads[0]["scanner_running"])
        self.assertTrue(state_payloads[-1]["scanner_running"])

    @patch.object(supervisor, "call_pi_set_mode", return_value={"status": "ok"})
    def test_safety_waits_for_receiver_sync_and_recovers_automatically(self, _set_mode):
        with patch.object(supervisor, "_post_receiver_json", return_value=False):
            result = supervisor.set_operational_mode("safety")

        self.assertEqual(result["status"], "partial_error")
        self.assertFalse(result["receiver_synced"])
        self.assertFalse(self.safety_scanner.running)

        with patch.object(supervisor, "_post_receiver_json", return_value=True):
            self.assertTrue(supervisor._reconcile_operational_state())

        self.assertTrue(self.safety_scanner.running)
        self.assertEqual(self.safety_scanner.start_count, 1)

    def test_operator_clear_tool_does_not_change_operational_mode(self):
        with supervisor._operational_lock:
            supervisor._operational_state.update(mode="safety", objective=None)

        with patch.object(supervisor, "_post_receiver_json", return_value=True) as post_json:
            result = supervisor.execute_supervisor_tool(
                "clear_safety_hazard",
                {"reason": "Area inspected by operator."},
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["safety_status"], "clear")
        self.assertEqual(supervisor.get_operational_mode_tool()["mode"], "safety")
        post_json.assert_called_once_with(
            "/system/safety/clear",
            {"reason": "Area inspected by operator."},
        )

    @patch.object(supervisor, "_post_receiver_json")
    @patch.object(supervisor, "call_pi_set_mode")
    def test_placeholder_modes_stop_scanners_without_reconfiguring_cameras(
        self,
        set_mode,
        _post,
    ):
        self.scanner.running = True

        for mode in ("investigation",):
            result = supervisor.set_operational_mode(mode)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["mode"], mode)
            self.assertTrue(result["placeholder"])
            self.assertEqual(result["camera_results"], [])

        self.assertEqual(self.scanner.stop_count, 1)
        self.assertEqual(self.safety_scanner.stop_count, 1)
        set_mode.assert_not_called()
        self.assertNotIn("free", supervisor.PLACEHOLDER_OPERATIONAL_MODES)
        self.assertNotIn("safety", supervisor.PLACEHOLDER_OPERATIONAL_MODES)

    @patch.object(supervisor, "call_pi_set_mode")
    def test_placeholder_mode_reports_pending_receiver_sync(self, set_mode):
        self.scanner.running = True
        with supervisor._operational_lock:
            supervisor._operational_state.update(
                mode="search",
                objective="blue backpack",
            )

        with patch.object(supervisor, "_post_receiver_json", return_value=False):
            result = supervisor.set_operational_mode("investigation")

        self.assertEqual(result["status"], "partial_error")
        self.assertFalse(result["receiver_synced"])
        self.assertTrue(result["placeholder"])
        self.assertFalse(result["scanner_running"])
        self.assertIn("status update is pending", result["message"])
        set_mode.assert_not_called()

    @patch.object(supervisor, "call_pi_set_mode", return_value={"status": "ok"})
    def test_camera_processing_mode_is_locked_during_visual_scanning(self, set_mode):
        for operational_mode in ("search", "safety"):
            with self.subTest(mode=operational_mode):
                with supervisor._operational_lock:
                    supervisor._operational_state["mode"] = operational_mode

                blocked = supervisor.execute_supervisor_tool(
                    "set_camera_mode",
                    {"camera_id": 0, "mode": "surveillance"},
                )
                allowed = supervisor.execute_supervisor_tool(
                    "set_camera_mode",
                    {"camera_id": 0, "mode": "default"},
                )

                self.assertEqual(blocked["status"], "error")
                self.assertIn(f"{operational_mode.title()} Mode", blocked["error"])
                self.assertEqual(allowed["status"], "ok")
        self.assertEqual(set_mode.call_count, 2)

    @patch.object(supervisor, "call_pi_set_mode", return_value={"status": "ok"})
    def test_free_allows_later_explicit_camera_commands(self, set_mode):
        with supervisor._operational_lock:
            supervisor._operational_state["mode"] = "free"

        result = supervisor.execute_supervisor_tool(
            "set_camera_mode",
            {"camera_id": 0, "mode": "surveillance"},
        )

        self.assertEqual(result["status"], "ok")
        set_mode.assert_called_once_with(0, "surveillance")

    def test_text_and_voice_share_the_new_operational_tool_schema(self):
        tool_names = {tool["name"] for tool in supervisor.tools}
        realtime_config = supervisor._realtime_session_config()

        self.assertIn("set_operational_mode", tool_names)
        self.assertIn("get_operational_mode", tool_names)
        self.assertIn("get_safety_state", tool_names)
        self.assertIn("clear_safety_hazard", tool_names)
        self.assertIn("get_reporting_status", tool_names)
        self.assertIn("generate_daily_report", tool_names)
        self.assertNotIn("activate_emergency_mode", tool_names)
        self.assertEqual(realtime_config["tools"], supervisor.tools)
        for mode in ("free", "safety", "search", "investigation"):
            self.assertIn(mode, supervisor.SYSTEM_PROMPT.lower())
        operational_tool = next(
            tool for tool in supervisor.tools if tool["name"] == "set_operational_mode"
        )
        self.assertEqual(
            operational_tool["parameters"]["properties"]["mode"]["enum"],
            ["free", "safety", "search", "investigation"],
        )
        self.assertIn("not an operational mode", supervisor.SYSTEM_PROMPT)
        self.assertIn("starts\n  in Free Mode", supervisor.SYSTEM_PROMPT)
        self.assertIn("switch to Free Mode", supervisor.SYSTEM_PROMPT)
        self.assertIn("Fire Hazard", supervisor.SYSTEM_PROMPT)
        self.assertIn("Work-Zone Intrusion", supervisor.SYSTEM_PROMPT)
        self.assertIn("Unauthorized Entry", supervisor.SYSTEM_PROMPT)
        self.assertIn("only when the operator explicitly asks", supervisor.SYSTEM_PROMPT)
        self.assertIn(".sup-mode-chip.hazard", supervisor.SUPERVISOR_CSS)
        self.assertIn('"Stop work · " + modeLabel', supervisor.SUPERVISOR_JS)

    def test_startup_initializes_free_mode_only_once(self):
        original_initialized = supervisor._startup_mode_initialized
        try:
            supervisor._startup_mode_initialized = False
            with patch.object(
                supervisor,
                "set_operational_mode",
                return_value={"status": "ok", "mode": "free"},
            ) as set_mode:
                first = supervisor.initialize_startup_mode()
                second = supervisor.initialize_startup_mode()

            self.assertEqual(first["mode"], "free")
            self.assertEqual(second["mode"], "free")
            set_mode.assert_called_once_with("free")
        finally:
            supervisor._startup_mode_initialized = original_initialized

    @patch.object(supervisor, "_post_receiver_json")
    def test_overlapping_search_then_free_finishes_consistently(self, _post):
        camera_call_started = threading.Event()
        release_camera_calls = threading.Event()
        free_attempted = threading.Event()
        free_done = threading.Event()

        def blocked_set_mode(camera_id, mode):
            camera_call_started.set()
            release_camera_calls.wait(timeout=2)
            return {"status": "ok", "camera_id": camera_id, "mode": mode}

        with patch.object(supervisor, "call_pi_set_mode", side_effect=blocked_set_mode):
            search_thread = threading.Thread(
                target=supervisor.set_operational_mode,
                args=("search", "red fire extinguisher"),
            )

            def select_free():
                free_attempted.set()
                supervisor.set_operational_mode("free")
                free_done.set()

            free_thread = threading.Thread(target=select_free)
            search_thread.start()
            self.assertTrue(camera_call_started.wait(timeout=1))
            free_thread.start()
            self.assertTrue(free_attempted.wait(timeout=1))
            self.assertFalse(free_done.wait(timeout=0.05))
            release_camera_calls.set()
            search_thread.join(timeout=2)
            free_thread.join(timeout=2)
            self.assertFalse(search_thread.is_alive())
            self.assertFalse(free_thread.is_alive())

        state = supervisor.get_operational_mode_tool()
        self.assertEqual(state["mode"], "free")
        self.assertFalse(state["scanner_running"])
        self.assertFalse(state["placeholder"])

    @patch.object(supervisor, "_post_receiver_json")
    def test_camera_command_cannot_finish_after_search_configuration(self, _post):
        camera_call_started = threading.Event()
        release_camera_call = threading.Event()
        search_attempted = threading.Event()
        search_done = threading.Event()
        completions = []
        completions_lock = threading.Lock()

        def ordered_set_mode(camera_id, mode):
            if mode == "surveillance":
                camera_call_started.set()
                release_camera_call.wait(timeout=2)
            with completions_lock:
                completions.append((camera_id, mode))
            return {"status": "ok", "camera_id": camera_id, "mode": mode}

        with patch.object(supervisor, "call_pi_set_mode", side_effect=ordered_set_mode):
            camera_thread = threading.Thread(
                target=supervisor.execute_supervisor_tool,
                args=("set_camera_mode", {"camera_id": 0, "mode": "surveillance"}),
            )

            def select_search():
                search_attempted.set()
                supervisor.set_operational_mode("search", "red fire extinguisher")
                search_done.set()

            search_thread = threading.Thread(target=select_search)
            camera_thread.start()
            self.assertTrue(camera_call_started.wait(timeout=1))
            search_thread.start()
            self.assertTrue(search_attempted.wait(timeout=1))
            self.assertFalse(search_done.wait(timeout=0.05))
            release_camera_call.set()
            camera_thread.join(timeout=2)
            search_thread.join(timeout=2)
            self.assertFalse(camera_thread.is_alive())
            self.assertFalse(search_thread.is_alive())

        self.assertEqual(completions[0], (0, "surveillance"))
        self.assertTrue(all(mode == "default" for _, mode in completions[1:]))
        self.assertEqual(supervisor.get_operational_mode_tool()["mode"], "search")


class SearchScannerTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.scanner = SearchScanner(
            client=self.client,
            receiver_url="http://receiver.test",
            model="vision-test",
        )

    def test_vlm_prompt_accepts_generic_objects(self):
        result = self.scanner._analyze_frame(
            target="red fire extinguisher beside a doorway",
            frame_bytes=b"jpeg",
        )

        prompt = self.client.responses.last_request["input"][0]["content"][0]["text"]
        self.assertTrue(result["match"])
        self.assertIn("red fire extinguisher", prompt)
        self.assertIn("person, object, vehicle", prompt)
        self.assertNotIn("emergency intent", prompt.lower())

    def test_match_posts_a_noncritical_search_alert(self):
        self.scanner._target = "red fire extinguisher"
        with patch.object(
            self.scanner,
            "_latest_frame",
            return_value=(b"jpeg", 1.0),
        ), patch.object(
            self.scanner,
            "_analyze_frame",
            return_value={
                "match": True,
                "confidence": 0.94,
                "summary": "A red fire extinguisher is visible.",
            },
        ), patch.object(self.scanner, "_post_log") as post_log:
            self.scanner._scan_camera_once(
                2,
                self.scanner._generation,
                self.scanner._target,
            )

        post_log.assert_called_once()
        alert = post_log.call_args.kwargs
        self.assertEqual(alert["kind"], "alert")
        self.assertEqual(alert["level"], "info")
        self.assertIn("SEARCH MATCH", alert["message"])
        self.assertIn("red fire extinguisher", alert["message"])

    def test_stopped_or_retargeted_scan_cannot_publish_stale_match(self):
        for invalidate in (
            self.scanner.stop,
            lambda: self.scanner.update_target("yellow hard hat"),
        ):
            with self.subTest(invalidate=invalidate):
                self.scanner._stop_event.clear()
                self.scanner._target = "red fire extinguisher"
                generation = self.scanner._generation

                def analyze_then_invalidate(**_kwargs):
                    invalidate()
                    return {
                        "match": True,
                        "confidence": 0.94,
                        "summary": "Old target visible.",
                    }

                with patch.object(
                    self.scanner,
                    "_latest_frame",
                    return_value=(b"jpeg", float(generation + 1)),
                ), patch.object(
                    self.scanner,
                    "_analyze_frame",
                    side_effect=analyze_then_invalidate,
                ), patch.object(self.scanner, "_post_log") as post_log:
                    self.scanner._scan_camera_once(
                        2,
                        generation,
                        "red fire extinguisher",
                    )

                post_log.assert_not_called()

    def test_new_target_has_its_own_alert_cooldown(self):
        self.scanner._target = "red fire extinguisher"
        with patch.object(
            self.scanner,
            "_latest_frame",
            side_effect=[(b"jpeg-1", 1.0), (b"jpeg-2", 2.0)],
        ), patch.object(
            self.scanner,
            "_analyze_frame",
            return_value={"match": True, "confidence": 0.94, "summary": "Visible."},
        ), patch.object(self.scanner, "_post_log") as post_log:
            first_generation = self.scanner._generation
            self.scanner._scan_camera_once(2, first_generation, self.scanner._target)
            self.scanner.update_target("yellow hard hat")
            self.scanner._scan_camera_once(
                2,
                self.scanner._generation,
                self.scanner._target,
            )

        self.assertEqual(post_log.call_count, 2)

    def test_stop_during_camera_discovery_does_not_log_active_streams(self):
        discovery_started = threading.Event()
        release_discovery = threading.Event()
        stop_event = threading.Event()

        def discover_after_stop(_stop_event):
            discovery_started.set()
            release_discovery.wait(timeout=2)
            return [0, 1]

        with patch.object(
            self.scanner,
            "_active_camera_ids",
            side_effect=discover_after_stop,
        ), patch.object(self.scanner, "_post_log") as post_log:
            run_thread = threading.Thread(
                target=self.scanner._run,
                args=(stop_event, object()),
            )
            run_thread.start()
            self.assertTrue(discovery_started.wait(timeout=1))
            stop_event.set()
            release_discovery.set()
            run_thread.join(timeout=2)

        self.assertFalse(run_thread.is_alive())
        messages = [call.kwargs["message"] for call in post_log.call_args_list]
        self.assertFalse(
            any(message.startswith("Scanning active camera streams") for message in messages)
        )

    def test_stop_during_failed_camera_discovery_does_not_log_warning(self):
        discovery_started = threading.Event()
        release_discovery = threading.Event()
        stop_event = threading.Event()

        def fail_after_stop(*_args, **_kwargs):
            discovery_started.set()
            release_discovery.wait(timeout=2)
            raise ConnectionError("receiver unavailable")

        with patch("search_vlm.requests.get", side_effect=fail_after_stop), patch.object(
            self.scanner,
            "_post_log",
        ) as post_log:
            discovery_thread = threading.Thread(
                target=self.scanner._active_camera_ids,
                args=(stop_event,),
            )
            discovery_thread.start()
            self.assertTrue(discovery_started.wait(timeout=1))
            stop_event.set()
            release_discovery.set()
            discovery_thread.join(timeout=2)

        self.assertFalse(discovery_thread.is_alive())
        post_log.assert_not_called()

    def test_run_loop_failure_invalidates_inflight_match(self):
        analysis_started = threading.Event()
        release_analysis = threading.Event()
        stop_event = threading.Event()
        executor = ThreadPoolExecutor(max_workers=1)
        initial_generation = self.scanner._generation
        self.scanner._target = "red fire extinguisher"
        self.scanner.sample_interval_sec = 0.01

        def analyze_after_loop_failure(**_kwargs):
            analysis_started.set()
            release_analysis.wait(timeout=2)
            return {
                "match": True,
                "confidence": 0.94,
                "summary": "Target visible.",
            }

        def run_and_absorb_expected_failure():
            with self.assertRaisesRegex(RuntimeError, "forced run-loop failure"):
                self.scanner._run(stop_event, executor)

        with patch.object(
            self.scanner,
            "_active_camera_ids",
            side_effect=[[2], RuntimeError("forced run-loop failure")],
        ), patch.object(
            self.scanner,
            "_latest_frame",
            return_value=(b"jpeg", 1.0),
        ), patch.object(
            self.scanner,
            "_analyze_frame",
            side_effect=analyze_after_loop_failure,
        ), patch.object(self.scanner, "_post_log") as post_log:
            run_thread = threading.Thread(target=run_and_absorb_expected_failure)
            self.scanner._stop_event = stop_event
            self.scanner._executor = executor
            self.scanner._thread = run_thread
            run_thread.start()
            self.assertTrue(analysis_started.wait(timeout=1))
            run_thread.join(timeout=2)
            self.assertFalse(run_thread.is_alive())
            release_analysis.set()
            executor.shutdown(wait=True)

        self.assertTrue(stop_event.is_set())
        self.assertGreater(self.scanner._generation, initial_generation)
        self.assertFalse(self.scanner.is_running())
        self.assertFalse(
            any(call.kwargs.get("kind") == "alert" for call in post_log.call_args_list)
        )

class SafetyScannerTests(unittest.TestCase):
    timezone = ZoneInfo("America/Vancouver")

    def _scanner(self, output_text=None, now=None):
        return SafetyScanner(
            client=FakeClient(output_text),
            receiver_url="http://receiver.test",
            model="vision-test",
            site_timezone="America/Vancouver",
            access_start_hour=9,
            access_end_hour=17,
            now_provider=(lambda: now) if now is not None else None,
        )

    def test_safety_checks_switch_locally_at_working_hour_boundaries(self):
        scanner = self._scanner()

        self.assertEqual(
            scanner.active_hazard_keys(datetime(2026, 8, 17, 8, 59, tzinfo=self.timezone)),
            ("fire_smoke", "after_hours_intrusion"),
        )
        self.assertEqual(
            scanner.active_hazard_keys(datetime(2026, 8, 17, 9, 0, tzinfo=self.timezone)),
            ("fire_smoke", "work_zone_encroachment"),
        )
        self.assertEqual(
            scanner.active_hazard_keys(datetime(2026, 8, 17, 16, 59, tzinfo=self.timezone)),
            ("fire_smoke", "work_zone_encroachment"),
        )
        self.assertEqual(
            scanner.active_hazard_keys(datetime(2026, 8, 17, 17, 0, tzinfo=self.timezone)),
            ("fire_smoke", "after_hours_intrusion"),
        )

    def test_daytime_frame_uses_one_structured_call_for_two_hazards(self):
        output = (
            '{"assessments":{'
            '"fire_smoke":{"detected":false,"confidence":0.05,"cause":""},'
            '"work_zone_encroachment":{"detected":false,"confidence":0.1,"cause":""}'
            '}}'
        )
        scanner = self._scanner(
            output,
            datetime(2026, 8, 17, 12, 0, tzinfo=self.timezone),
        )
        keys = scanner.active_hazard_keys()

        result = scanner._analyze_frame(hazard_keys=keys, frame_bytes=b"jpeg")

        request = scanner.client.responses.last_request
        prompt = request["input"][0]["content"][0]["text"]
        schema = request["text"]["format"]["schema"]
        self.assertEqual(scanner.client.responses.create_count, 1)
        self.assertEqual(set(result["assessments"]), {"fire_smoke", "work_zone_encroachment"})
        self.assertIn("Fire Hazard", prompt)
        self.assertIn("Work-Zone Intrusion", prompt)
        self.assertNotIn("Unauthorized Entry", prompt)
        self.assertEqual(
            schema["properties"]["assessments"]["required"],
            ["fire_smoke", "work_zone_encroachment"],
        )
        self.assertTrue(request["text"]["format"]["strict"])

    def test_one_frame_can_emit_multiple_safety_hazards_from_one_vlm_call(self):
        output = (
            '{"assessments":{'
            '"fire_smoke":{"detected":true,"confidence":0.96,"cause":"Flames and dark smoke are visible."},'
            '"work_zone_encroachment":{"detected":true,"confidence":0.91,"cause":"A person is inside the work zone."}'
            '}}'
        )
        scanner = self._scanner(
            output,
            datetime(2026, 8, 17, 12, 0, tzinfo=self.timezone),
        )

        with patch.object(
            scanner,
            "_latest_frame",
            return_value=(b"jpeg", 1.0),
        ), patch.object(scanner, "_post_hazard", return_value=True) as post_hazard:
            scanner._scan_camera_once(2, scanner._generation, "ignored")

        self.assertEqual(scanner.client.responses.create_count, 1)
        self.assertEqual(post_hazard.call_count, 2)
        self.assertEqual(
            {call.kwargs["hazard_key"] for call in post_hazard.call_args_list},
            {"fire_smoke", "work_zone_encroachment"},
        )

    def test_after_hours_frame_replaces_work_zone_check_with_unauthorized_entry(self):
        output = (
            '{"assessments":{'
            '"fire_smoke":{"detected":false,"confidence":0.0,"cause":""},'
            '"after_hours_intrusion":{"detected":true,"confidence":0.93,"cause":"A person is at the site."}'
            '}}'
        )
        scanner = self._scanner(
            output,
            datetime(2026, 8, 17, 20, 0, tzinfo=self.timezone),
        )
        keys = scanner.active_hazard_keys()

        scanner._analyze_frame(hazard_keys=keys, frame_bytes=b"jpeg")

        request = scanner.client.responses.last_request
        prompt = request["input"][0]["content"][0]["text"]
        required = request["text"]["format"]["schema"]["properties"]["assessments"]["required"]
        self.assertEqual(scanner.client.responses.create_count, 1)
        self.assertIn("Unauthorized Entry", prompt)
        self.assertNotIn("Work-Zone Intrusion", prompt)
        self.assertEqual(required, ["fire_smoke", "after_hours_intrusion"])

    def test_stopped_safety_scan_cannot_publish_stale_hazard(self):
        scanner = self._scanner(
            now=datetime(2026, 8, 17, 12, 0, tzinfo=self.timezone),
        )
        generation = scanner._generation

        def analyze_then_stop(**_kwargs):
            scanner.stop()
            return {
                "assessments": {
                    "fire_smoke": {
                        "detected": True,
                        "confidence": 0.99,
                        "cause": "Flames are visible.",
                    },
                    "work_zone_encroachment": {
                        "detected": False,
                        "confidence": 0.0,
                        "cause": "",
                    },
                }
            }

        with patch.object(
            scanner,
            "_latest_frame",
            return_value=(b"jpeg", 1.0),
        ), patch.object(
            scanner,
            "_analyze_frame",
            side_effect=analyze_then_stop,
        ), patch.object(scanner, "_post_hazard") as post_hazard:
            scanner._scan_camera_once(2, generation, "ignored")

        post_hazard.assert_not_called()


class ReceiverOperationalStateTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(stream_receiver_server.app)
        with stream_receiver_server._lock:
            stream_receiver_server._system_state.update(
                mode="free",
                objective=None,
                scanner_running=False,
                placeholder=False,
                updated_at=None,
                safety_status="clear",
                active_safety_hazards=[],
                safety_updated_at=None,
            )
            stream_receiver_server._system_log.clear()
            stream_receiver_server._alert_frames.clear()

    def test_all_operational_modes_round_trip_through_receiver_api(self):
        for mode in ("free", "safety", "search", "investigation"):
            objective = (
                "red fire extinguisher"
                if mode == "search"
                else ("ignored objective" if mode == "free" else None)
            )
            response = self.client.post(
                "/system/state",
                json={
                    "mode": mode,
                    "objective": objective,
                    "scanner_running": mode in {"safety", "search"},
                },
            )

            self.assertEqual(response.status_code, 200)
            state = self.client.get("/system/state").json()
            self.assertEqual(state["mode"], mode)
            self.assertEqual(
                state["objective"],
                None if mode in {"free", "safety"} else objective,
            )
            self.assertEqual(state["scanner_running"], mode in {"safety", "search"})
            self.assertEqual(
                state["placeholder"],
                mode == "investigation",
            )

    def test_safety_hazard_latches_until_explicit_operator_clear(self):
        hazard_response = self.client.post(
            "/system/safety/hazard",
            json={
                "hazard_key": "fire_smoke",
                "camera_id": 2,
                "confidence": 0.96,
                "cause": "Flames and dark smoke are visible.",
                "frame_jpeg_b64": "anBlZw==",
            },
        )

        self.assertEqual(hazard_response.status_code, 200)
        state = self.client.get("/system/state").json()
        self.assertEqual(state["safety_status"], "hazard")
        self.assertEqual(len(state["active_safety_hazards"]), 1)
        self.assertEqual(
            state["active_safety_hazards"][0]["hazard_name"],
            "Fire Hazard",
        )
        alert = self.client.get("/system/log").json()[-1]
        self.assertEqual(alert["kind"], "safety_alert")
        self.assertEqual(alert["level"], "critical")
        self.assertIn("STOP WORK", alert["message"])
        self.assertNotIn("stop_note", alert)

        self.client.post("/system/state", json={"mode": "free"})
        self.assertEqual(self.client.get("/system/state").json()["safety_status"], "hazard")

        clear_response = self.client.post(
            "/system/safety/clear",
            json={"reason": "Area inspected."},
        )
        self.assertEqual(clear_response.status_code, 200)
        cleared = self.client.get("/system/state").json()
        self.assertEqual(cleared["safety_status"], "clear")
        self.assertEqual(cleared["active_safety_hazards"], [])
        self.assertEqual(self.client.get("/system/log").json()[-1]["kind"], "safety_clear")

    def test_receiver_rejects_unknown_safety_hazard(self):
        response = self.client.post(
            "/system/safety/hazard",
            json={"hazard_key": "unknown", "camera_id": 0},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/system/state").json()["safety_status"], "clear")

    def test_receiver_rejects_unknown_operational_mode(self):
        response = self.client.post("/system/state", json={"mode": "emergency"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/system/state").json()["mode"], "free")

        reporting_response = self.client.post("/system/state", json={"mode": "reporting"})
        self.assertEqual(reporting_response.status_code, 400)
        self.assertEqual(self.client.get("/system/state").json()["mode"], "free")

    def test_dashboard_renders_all_four_operational_modes(self):
        html = stream_receiver_server.INDEX_HTML

        for label in ("Free", "Safety", "Search", "Investigation"):
            self.assertIn(label.lower(), html.lower())
        self.assertNotIn("'reporting'", html.lower())
        self.assertIn("Operational mode", html)
        self.assertIn("Mode objective", html)
        self.assertIn("No workflow", html)
        self.assertIn("Construction safety", html)
        self.assertIn("Clear for construction", html)
        self.assertIn("STOP WORK", html)
        self.assertIn("Safety hazard frame", html)
        self.assertIn("safety-alert", html)
        self.assertIn("Search match frame", html)
        self.assertNotIn("Emergency intent", html)
        self.assertNotIn("value.emergency", html)


if __name__ == "__main__":
    unittest.main()
