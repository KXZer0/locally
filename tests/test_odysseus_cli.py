import io
import json
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest import mock

from core import odysseus_cli
from core import odysseus


class OdysseusCliTests(unittest.TestCase):
    def test_start_returns_running_state_as_json(self):
        output = io.StringIO()
        with (mock.patch.object(odysseus_cli.odysseus, "start",
                                return_value=(True, "started")) as start,
              mock.patch.object(odysseus_cli.odysseus, "status",
                                return_value={"running": True}),
              redirect_stdout(output)):
            code = odysseus_cli.main(["start", "--port", "7123"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output.getvalue())["running"])
        start.assert_called_once_with(7123, None, 240, 8000)

    def test_probe_returns_nonzero_when_container_cannot_reach_api(self):
        output = io.StringIO()
        with (mock.patch.object(odysseus_cli.odysseus, "probe_api",
                                return_value={"reachable": False,
                                              "models": [],
                                              "detail": "timeout"}),
              redirect_stdout(output)):
            code = odysseus_cli.main(["probe", "--api-port", "8000"])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(output.getvalue())["reachable"])

    def test_start_failure_has_a_nonzero_exit_and_detail(self):
        output = io.StringIO()
        with (mock.patch.object(odysseus_cli.odysseus, "start",
                                return_value=(False, "Podman is unavailable")),
              mock.patch.object(odysseus_cli.odysseus, "status",
                                return_value={"running": False}),
              redirect_stdout(output)):
            code = odysseus_cli.main(["start"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())["detail"],
                         "Podman is unavailable")


class OdysseusEndpointSyncTests(unittest.TestCase):
    def test_existing_stack_refreshes_a_stale_endpoint(self):
        previous = odysseus._started_dir
        try:
            completed = subprocess.CompletedProcess(
                ["podman", "compose", "up", "-d", "odysseus"], 0,
                stdout="", stderr="")
            with (mock.patch.object(odysseus, "_serves", return_value=True),
                  mock.patch.object(odysseus, "find_dir", return_value="C:/ody"),
                  mock.patch.object(odysseus, "command", return_value="podman"),
                  mock.patch.object(odysseus, "api_base_url",
                                    return_value="http://new-gateway:8000/v1"),
                  mock.patch.object(odysseus, "_compose_env",
                                    return_value={"OLLAMA_BASE_URL": "http://new-gateway:8000/v1"}),
                  mock.patch.object(odysseus, "_run", return_value=completed) as run):
                ok, detail = odysseus.start(7000, "C:/ody", 15, 8000)
            self.assertTrue(ok)
            self.assertEqual(detail, "adopted; endpoint synchronized")
            run.assert_called_once_with(
                ["podman", "compose", "up", "-d", "odysseus"],
                cwd="C:/ody", timeout=30,
                env={"OLLAMA_BASE_URL": "http://new-gateway:8000/v1"})
        finally:
            odysseus._started_dir = previous

if __name__ == "__main__":
    unittest.main()
