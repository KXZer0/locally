import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

from core import odysseus_cli


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
        start.assert_called_once_with(7123, None, 240)

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


if __name__ == "__main__":
    unittest.main()
