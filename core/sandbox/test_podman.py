"""Regression tests for the Podman Python sandbox plumbing.

The live containment checks are intentionally kept out of this suite: they
need a real Podman machine and are recorded in docs/handoff-podman.md.  These
tests cover the command contract, lifecycle ownership, non-blocking health
path, and explicit fallback reporting without starting a VM.
"""

import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from core import config
from core.portprobe import CachedProbe
from core.sandbox import podman
from core.sandbox import python_exec


class PodmanCommandTests(unittest.TestCase):
    def test_container_command_has_exact_boundary_and_one_mount(self):
        argv = podman._container_argv(
            "podman", r"C:\work space\locally", r".locally-python-123",
            "locally-python-deadbeef")

        required = (
            "--rm", "--network=none", "--read-only", "--memory=512m",
            "--pids-limit=64", "--cap-drop=ALL",
        )
        for flag in required:
            self.assertIn(flag, argv)
        security = argv.index("--security-opt")
        self.assertEqual(argv[security + 1], "no-new-privileges")
        self.assertEqual(argv.count("--volume"), 1)
        volume = argv.index("--volume")
        self.assertEqual(
            argv[volume + 1], r"C:\work space\locally:/workspace:rw")
        workdir = argv.index("--workdir")
        self.assertEqual(
            argv[workdir + 1], "/workspace/.locally-python-123")
        self.assertNotIn("--env", argv)
        self.assertEqual(
            argv[-5:], [podman.IMAGE, "python", "-I", "-B", "run.py"])

    def test_health_status_never_waits_for_machine_probe(self):
        entered = threading.Event()
        release = threading.Event()

        def slow_probe():
            entered.set()
            release.wait(1.0)
            return True

        probe = CachedProbe(slow_probe, ttl=5.0)
        with mock.patch.object(podman, "_machine_probe", probe), \
                mock.patch.object(podman, "podman_available", return_value=True):
            started = time.perf_counter()
            status = podman.sandbox_status()
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 0.1)
            self.assertFalse(status["machine_running"])
            self.assertTrue(entered.wait(0.5))
            release.set()

    def test_run_starts_and_stops_only_the_machine_it_started(self):
        class FinishedProcess:
            pid = 123
            returncode = 0

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory(dir=config.SCRIPT_DIR) as root:
            stdout_path = os.path.join(root, "stdout.bin")
            stderr_path = os.path.join(root, "stderr.bin")
            with mock.patch.object(
                    podman, "podman_command", return_value="podman"), \
                    mock.patch.object(
                        podman, "refresh_machine_running", return_value=False), \
                    mock.patch.object(
                        podman, "_start_machine", return_value=(True, "")), \
                    mock.patch.object(
                        podman, "_stop_machine_if_idle", return_value=True
                    ) as stop, \
                    mock.patch.object(
                        podman.subprocess, "Popen", return_value=FinishedProcess()):
                result = podman.run_python(
                    root, stdout_path, stderr_path, timeout=1.0)

        self.assertIsNone(result["error"])
        self.assertTrue(result["machine_started"])
        self.assertTrue(result["machine_stopped"])
        stop.assert_called_once_with("podman")

    def test_run_leaves_an_already_running_machine_alone(self):
        class FinishedProcess:
            pid = 123
            returncode = 0

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory(dir=config.SCRIPT_DIR) as root:
            stdout_path = os.path.join(root, "stdout.bin")
            stderr_path = os.path.join(root, "stderr.bin")
            with mock.patch.object(
                    podman, "podman_command", return_value="podman"), \
                    mock.patch.object(
                        podman, "refresh_machine_running", return_value=True), \
                    mock.patch.object(podman, "_start_machine") as start, \
                    mock.patch.object(podman, "_stop_machine_if_idle") as stop, \
                    mock.patch.object(
                        podman.subprocess, "Popen", return_value=FinishedProcess()):
                result = podman.run_python(
                    root, stdout_path, stderr_path, timeout=1.0)

        self.assertIsNone(result["error"])
        self.assertFalse(result["machine_started"])
        self.assertIsNone(result["machine_stopped"])
        start.assert_not_called()
        stop.assert_not_called()


class BoundarySelectionTests(unittest.TestCase):
    def setUp(self):
        self.had_setting = hasattr(config, "PYTHON_SANDBOX")
        self.old_setting = getattr(config, "PYTHON_SANDBOX", None)

    def tearDown(self):
        if self.had_setting:
            config.PYTHON_SANDBOX = self.old_setting
        else:
            delattr(config, "PYTHON_SANDBOX")

    @staticmethod
    def _result():
        return {
            "code": "print(1)", "stdout": "1\n", "stderr": "",
            "exit_status": 0, "elapsed_ms": 1, "truncated": False,
            "timed_out": False,
        }

    def test_auto_fallback_is_named_and_warned_in_payload(self):
        config.PYTHON_SANDBOX = "auto"
        with mock.patch.object(
                podman, "podman_available", return_value=False), \
                mock.patch.object(
                    python_exec, "_execute_subprocess",
                    return_value=self._result()), \
                mock.patch.object(python_exec, "_warn_fallback") as warning:
            result = python_exec.execute_python("print(1)")

        self.assertEqual(result["sandbox"], "subprocess")
        self.assertIn("not a security boundary", result["boundary"])
        self.assertIn("Podman is unavailable", result["sandbox_warning"])
        warning.assert_called_once()

    def test_forced_podman_never_silently_falls_back(self):
        config.PYTHON_SANDBOX = "podman"
        with mock.patch.object(
                podman, "podman_available", return_value=False), \
                mock.patch.object(python_exec, "_execute_subprocess") as fallback:
            result = python_exec.execute_python("print(1)")

        self.assertEqual(result["sandbox"], "podman")
        self.assertIn("was requested", result["stderr"])
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
