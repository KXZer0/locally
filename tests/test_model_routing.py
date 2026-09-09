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
from core.app import create_app
from core.slots import route


def slot(name, device="NPU", status="ready", model_type="llm"):
    return types.SimpleNamespace(status=status, model_name=name,
                                 device_name=device, model_type=model_type,
                                 model_dir=None)


class RouteRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # _load_on_demand's refusal builds a Flask JSON response via
        # openai_error(); jsonify() needs an application context, and these
        # tests call _route_request directly rather than through a client.
        cls._ctx = create_app().app_context()
        cls._ctx.push()

    @classmethod
    def tearDownClass(cls):
        cls._ctx.pop()

    def route(self, requested, primary, available=(), has_images=False,
              secondary=None, lands=None):
        """_route_request with the slot table and the disk stubbed out.

        Returns (slot, names swap_model was called with). `lands`, if given,
        is a slot that swap_model "loads" -- it is dropped into
        runtime.secondary so the post-swap _match_loaded can find a model that
        the real swap would have placed in a slot other than the one asked for.
        """
        calls = []

        def fake_swap(name, device=""):
            calls.append(name)
            if lands is not None:
                runtime.secondary = lands
            return {"status": "ok"}

        with (mock.patch.object(runtime, "primary", primary),
              mock.patch.object(runtime, "secondary", secondary),
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

    def test_unloadable_model_is_refused_not_served_silently(self):
        """It is listed with a reason by /v1/models/available, so naming it is
        deliberate -- the failure belongs to the caller to see, not to be
        buried under an answer from whatever was already resident."""
        resident = slot("qwen3-8b")
        with self.assertRaises(route._TurnError) as ctx:
            self.route("broken", resident, available=[
                {"name": "broken", "path": "/m/b", "loadable": False,
                 "reason": "unsupported architecture"},
            ])
        body, status = ctx.exception.response
        self.assertEqual(status, 400)
        self.assertIn("unsupported architecture",
                      body.get_json()["error"]["message"])

    def test_empty_model_id_never_triggers_a_swap(self):
        resident = slot("qwen3-8b")
        for asked in ("", "   ", "@NPU"):
            picked, calls = self.route(asked, resident,
                                       available=[{"name": "qwen3-8b", "path": "/m/q"}])
            self.assertIs(picked, resident, repr(asked))
            self.assertEqual(calls, [], repr(asked))

    def test_stale_device_suffix_reuses_the_resident_model(self):
        """The model fell back NPU->GPU; a client still holds the '@NPU' id.

        Discovery now advertises a bare id, but an id cached before a fallback
        swap -- or from an older build that suffixed everything -- must still
        resolve to the model sitting there, not trigger a reload onto NPU.
        """
        resident = slot("qwen3-8b", device="GPU")
        for asked in ("qwen3-8b@NPU", "QWEN3-8B@npu", "  qwen3-8b  "):
            picked, calls = self.route(asked, resident,
                                       available=[{"name": "qwen3-8b", "path": "/m/q"}])
            self.assertIs(picked, resident, asked)
            self.assertEqual(calls, [], asked)

    def test_case_folded_name_matches_the_resident_model(self):
        resident = slot("Qwen3-8B-int4-cw-ov")
        picked, calls = self.route("qwen3-8b-int4-cw-ov", resident, available=[
            {"name": "Qwen3-8B-int4-cw-ov", "path": "/m/q"}])
        self.assertIs(picked, resident)
        self.assertEqual(calls, [])

    def test_on_demand_swap_is_found_in_whatever_slot_it_landed_in(self):
        """swap_model may place the model on a different device/slot than the
        request named (device fallback, or the secondary slot). Routing must
        return that slot, not the stale primary."""
        resident = slot("qwen3-8b", device="NPU")
        landed = slot("gemma-4-26b", device="GPU")
        picked, calls = self.route("gemma-4-26b@NPU", resident, lands=landed,
                                   available=[
                                       {"name": "qwen3-8b", "path": "/m/q"},
                                       {"name": "gemma-4-26b", "path": "/m/g"},
                                   ])
        self.assertEqual(calls, ["gemma-4-26b@NPU"])
        self.assertIs(picked, landed)

    def test_text_falls_back_off_a_dead_primary_to_a_serviceable_vlm(self):
        """Primary (NPU) is mid-swap; a VLM is resident on the secondary.

        The old dual-mode branch returned runtime.primary unconditionally for
        text when the secondary was a VLM -- handing back a slot that could
        not serve. A VLM answers text fine.
        """
        dead = slot("qwen3-8b", status="loading")
        vlm = slot("gemma-4-vision", device="GPU", model_type="vlm")
        picked, calls = self.route("", dead, secondary=vlm)
        self.assertIs(picked, vlm)
        self.assertEqual(calls, [])

    def test_text_prefers_a_secondary_llm_over_the_primary(self):
        npu = slot("qwen3-8b", device="NPU")
        gpu_coder = slot("qwen3-coder-30b", device="GPU", model_type="llm")
        picked, _ = self.route("", npu, secondary=gpu_coder)
        self.assertIs(picked, gpu_coder)


if __name__ == "__main__":
    unittest.main()
