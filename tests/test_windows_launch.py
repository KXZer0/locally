import json
import os
import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KEY = ROOT / "scripts" / "locally-key.ps1"


@unittest.skipUnless(os.name == "nt" and shutil.which("powershell"),
                     "Windows PowerShell launch tests")
class WindowsLaunchTests(unittest.TestCase):
    def _capture_wrapper(self, wrapper, wrapper_args, variable):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake = directory / "capture.ps1"
            output = directory / "args.json"
            fake.write_text(
                "$args | ConvertTo-Json -Compress | Set-Content "
                "-LiteralPath $env:LOCALLY_CAPTURE -Encoding UTF8\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment[variable] = str(fake)
            environment["LOCALLY_CAPTURE"] = str(output)
            subprocess.run(
                ["powershell", "-NoProfile", "-File", str(ROOT / wrapper),
                 *wrapper_args],
                cwd=ROOT, env=environment, capture_output=True, text=True,
                check=True,
            )
            return json.loads(output.read_text(encoding="utf-8-sig"))

    def _decision(self, port, mode=None):
        command = ["powershell", "-NoProfile", "-File", str(KEY),
                   "-Port", str(port), "-WhatIf"]
        if mode:
            command.extend(["-Mode", mode])
        result = subprocess.run(
            command,
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout.strip())

    def test_chat_attaches_to_an_existing_api(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            decision = self._decision(listener.getsockname()[1], "chat")
        self.assertEqual(decision["Action"], "attach-chat")
        self.assertNotIn("--start", decision["Command"])

    def test_chat_owns_a_server_when_port_is_closed(self):
        with socket.socket() as unused:
            unused.bind(("127.0.0.1", 0))
            port = unused.getsockname()[1]
        decision = self._decision(port, "chat")
        self.assertEqual(decision["Action"], "start-owned-chat")
        self.assertIn("--start", decision["Command"])

    def test_api_mode_starts_launcher_only_when_needed(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            running = self._decision(listener.getsockname()[1], "api")
        self.assertEqual(running["Action"], "show-api-status")

        with socket.socket() as unused:
            unused.bind(("127.0.0.1", 0))
            port = unused.getsockname()[1]
        stopped = self._decision(port, "api")
        self.assertEqual(stopped["Action"], "start-api")
        self.assertIn("api.ps1", stopped["Command"])

    def test_hardware_key_defaults_to_api_mode(self):
        with socket.socket() as unused:
            unused.bind(("127.0.0.1", 0))
            port = unused.getsockname()[1]
        self.assertEqual(self._decision(port)["Action"], "start-api")

    def test_api_wrapper_uses_configured_start_and_forwards_arguments(self):
        captured = self._capture_wrapper(
            "api.ps1", ["-Port", "8123", "--idle-timeout", "0"],
            "LOCALLY_START_SCRIPT",
        )
        self.assertEqual(captured, ["--port", "8123", "--idle-timeout", "0"])

    def test_chat_remote_url_never_adds_owned_start(self):
        captured = self._capture_wrapper(
            "chat.ps1", ["-Url", "http://example.test:8123", "--plain"],
            "LOCALLY_PYTHON",
        )
        self.assertEqual(
            captured[1:],
            ["chat", "--url", "http://example.test:8123", "--plain"],
        )
        self.assertNotIn("--start", captured)


if __name__ == "__main__":
    unittest.main()
