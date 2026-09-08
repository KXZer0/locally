"""Which model answers when the request names one.

The interesting cases are the two that used to be indistinguishable: a model id
the server has never heard of (keep serving, it is usually a client's
unconfigured default) and a model id that names something sitting on disk
(load it, the caller means it).
"""
import types
import unittest
from unittest import mock

from core import runtime
from core.slots import route


def slot(name, device="NPU", status="ready", model_type="llm"):
    return types.SimpleNamespace(status=status, model_name=name,
                                 device_name=device, model_type=model_type,
                                 model_dir=None)


class RouteRequestTests(unittest.TestCase):
    def route(self, requested, primary, available=(), has_images=False):
        """_route_request with the slot table and the disk stubbed out.

        Returns (slot, names swap_model was called with).
        """
        calls = []

        def fake_swap(name, device=""):
            calls.append(name)
            return {"status": "ok"}

        with (mock.patch.object(runtime, "primary", primary),
              mock.patch.object(runtime, "secondary", None),
              mock.patch("core.models.discovery._available_models",
                         return_value=list(available)),
              mock.patch("core.models.manage.swap_model", side_effect=fake_swap)):
            return route._route_request(has_images, requested), calls

    def test_resident_model_is_used_without_a_swap(self):
        resident = slot("qwen3-8b")
        for asked in ("qwen3-8b", "qwen3-8b@NPU"):
            picked, calls = self.route(asked, resident,
                                       available=[{"name": "qwen3-8b", "path": "/m/q"}])
            self.assertIs(picked, resident, asked)
            self.assertEqual(calls, [], asked)

    def test_model_on_disk_is_loaded_on_demand(self):
        """Asking for gemma used to be answered by Qwen, under gemma's name."""
        resident = slot("qwen3-8b")
        picked, calls = self.route("gemma-4-26b", resident, available=[
            {"name": "qwen3-8b", "path": "/m/q"},
            {"name": "gemma-4-26b", "path": "/m/g"},
        ])
        self.assertEqual(calls, ["gemma-4-26b"])
        self.assertIsNotNone(picked)

    def test_unknown_model_still_falls_through(self):
        """Clients send ids we have never heard of; refusing breaks working turns."""
        resident = slot("qwen3-8b")
        picked, calls = self.route("gpt-4", resident,
                                   available=[{"name": "qwen3-8b", "path": "/m/q"}])
        self.assertIs(picked, resident)
        self.assertEqual(calls, [])

    def test_unloadable_model_is_not_loaded_on_demand(self):
        """It is listed with a reason, but loading it can only fail."""
        resident = slot("qwen3-8b")
        picked, calls = self.route("broken", resident, available=[
            {"name": "broken", "path": "/m/b", "loadable": False,
             "reason": "unsupported architecture"},
        ])
        self.assertIs(picked, resident)
        self.assertEqual(calls, [])

    def test_empty_model_id_never_triggers_a_swap(self):
        resident = slot("qwen3-8b")
        for asked in ("", "   ", "@NPU"):
            picked, calls = self.route(asked, resident,
                                       available=[{"name": "qwen3-8b", "path": "/m/q"}])
            self.assertIs(picked, resident, repr(asked))
            self.assertEqual(calls, [], repr(asked))


if __name__ == "__main__":
    unittest.main()
