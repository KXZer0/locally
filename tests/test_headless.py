import threading
import types
import unittest
from unittest import mock

from core import runtime
from core.app import create_app
from core.slots.locks import SlotLock, _device_lock


class HeadlessTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_api_is_available_without_browser_surface(self):
        self.assertEqual(self.client.get("/").json["mode"], "headless")
        self.assertIsNone(self.app.static_folder)
        self.assertIsNone(self.app.template_folder)
        for path in ("/static/js/main.js", "/v1/ui/bootstrap", "/v1/setup",
                     "/v1/open", "/v1/code/launch", "/v1/opencode/web"):
            self.assertEqual(self.client.get(path).status_code, 404, path)
        for path in ("/v1/models", "/v1/metrics", "/health"):
            self.assertEqual(self.client.get(path).status_code, 200, path)
        rules = {rule.rule for rule in self.app.url_map.iter_rules()}
        self.assertTrue({"/v1/chat/completions", "/v1/messages",
                         "/v1/audio/speech", "/v1/models/load"} <= rules)

    def test_invalid_model_controls_return_json_errors(self):
        for body in ([1], {"model": 7}, {"model": "x", "device": 42}):
            response = self.client.post("/v1/models/load", json=body)
            self.assertEqual(response.status_code, 400)
            self.assertIn("error", response.json)

    @staticmethod
    def _slot(status, name="fixture-8b", device="NPU"):
        return types.SimpleNamespace(status=status, model_name=name,
                                     device_name=device, model_dir=None)

    def _models(self, primary, available=()):
        """GET /v1/models with the slot table and the disk stubbed out."""
        from core.models import discovery
        with (mock.patch.object(runtime, "primary", primary),
              mock.patch.object(runtime, "secondary", None),
              mock.patch.object(runtime, "whisper_slot", None),
              mock.patch.object(discovery, "_available_models",
                                return_value=list(available))):
            return self.client.get("/v1/models").json["data"]

    def test_idle_unloaded_model_is_still_advertised(self):
        """An idle-unloaded slot serves on demand, so it must stay listed.

        Listing only "ready" emptied /v1/models after the idle timeout while
        the server went on answering chat requests. A client that discovers
        models by polling this endpoint reads that as "backend offline" --
        which is how a working NPU looked offline to Odysseus.
        """
        data = self._models(self._slot("idle_unloaded"))
        self.assertEqual([m["id"] for m in data], ["fixture-8b@NPU"])

    def test_dead_slot_is_not_advertised(self):
        """The other half of the same rule: errored/unconfigured stay hidden."""
        for status in ("error", "not_configured", "loading"):
            self.assertEqual(self._models(self._slot(status)), [], status)

    def test_models_on_disk_are_offered_alongside_the_resident_one(self):
        """A picker built from this endpoint needs more than one choice.

        The resident model keeps its @DEVICE id; an unloaded one is named
        alone, because placement is decided at load time. The resident model
        must not also appear as an available one -- that is the same model
        twice under two ids.
        """
        data = self._models(self._slot("ready"), available=[
            {"name": "fixture-8b", "path": "/m/fixture-8b", "type": "llm"},
            {"name": "gemma-4-26b", "path": "/m/gemma", "type": "vlm"},
        ])
        self.assertEqual([m["id"] for m in data], ["fixture-8b@NPU", "gemma-4-26b"])
        self.assertEqual(data[1]["owned_by"], "local-available")

    def test_unloadable_model_on_disk_is_not_offered(self):
        """A GGUF the reader cannot open is listed by /v1/models/available with
        its reason, but offering it here would invite a load that fails."""
        data = self._models(self._slot("ready"), available=[
            {"name": "broken", "path": "/m/broken", "type": "llm",
             "loadable": False, "reason": "unsupported architecture"},
        ])
        self.assertEqual([m["id"] for m in data], ["fixture-8b@NPU"])

    def test_no_slot_is_a_recoverable_api_error(self):
        from core.models import manage
        with (mock.patch.object(runtime, "primary", None),
              mock.patch.object(manage, "_available_models", return_value=[
                  {"name": "fixture", "path": "/fixture"}])):
            response = self.client.post("/v1/models/load", json={"model": "fixture"})
        self.assertEqual(response.status_code, 503)

    def _resident_slot(self, tmpdir, name="fixture-8b", device="NPU"):
        import os
        model_dir = os.path.join(tmpdir, name)
        os.makedirs(model_dir, exist_ok=True)
        loads = []
        return model_dir, loads, types.SimpleNamespace(
            status="ready", model_name=name, device_name=device,
            model_type="llm", model_dir=model_dir,
            load=lambda target, *_a: loads.append(target))

    def test_loading_the_resident_model_does_not_reload_it(self):
        """`/load` on the current model -- a keep-alive, a re-pick, a stale
        `@DEVICE` suffix -- must be a no-op, not a 9-65 s reload."""
        import tempfile
        from core.models import manage
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, loads, s = self._resident_slot(tmp)
            with (mock.patch.object(runtime, "primary", s),
                  mock.patch.object(runtime, "secondary", None),
                  mock.patch.object(manage, "_available_models", return_value=[
                      {"name": "fixture-8b", "path": model_dir}])):
                for asked in ("fixture-8b", "fixture-8b@NPU", "FIXTURE-8B@npu",
                              "fixture-8b@AUTO"):
                    response = self.client.post("/v1/models/load",
                                                json={"model": asked})
                    self.assertEqual(response.status_code, 200, asked)
                    self.assertEqual(response.json["placement"],
                                     "already resident", asked)
                    self.assertEqual(response.json["device"], "NPU", asked)
            self.assertEqual(loads, [])

    def test_explicit_other_device_still_moves_a_resident_model(self):
        """Asking for the resident model @GPU when it sits on NPU is a real
        move request -- it must NOT be short-circuited as already resident."""
        import tempfile
        from core.models import manage
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, loads, s = self._resident_slot(tmp)
            with (mock.patch.object(runtime, "primary", s),
                  mock.patch.object(runtime, "secondary", None),
                  mock.patch.object(manage, "_available_models", return_value=[
                      {"name": "fixture-8b", "path": model_dir}]),
                  mock.patch.object(runtime, "DEVICES", {}),
                  mock.patch("core.models.manage._choose_device",
                             return_value=(None, None, "no GPU on this machine"))):
                response = self.client.post("/v1/models/load",
                                            json={"model": "fixture-8b@GPU"})
            self.assertEqual(response.status_code, 400)
            self.assertIn("no GPU on this machine",
                          response.json["error"]["message"])
            self.assertEqual(loads, [])


class SlotLockTests(unittest.TestCase):
    def available_on_other_thread(self, lock):
        result = []
        def attempt():
            acquired = lock.acquire(blocking=False)
            result.append(acquired)
            if acquired:
                lock.release()
        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        return result[0]

    def test_gpu_to_npu_reserves_target_during_and_after_migration(self):
        device = ["GPU"]
        lock = SlotLock(lambda: device[0])
        npu = _device_lock("NPU")
        with lock.moving_to("NPU"):
            self.assertFalse(self.available_on_other_thread(npu))
            device[0] = "NPU"
        with lock:
            self.assertFalse(self.available_on_other_thread(npu))
        self.assertTrue(self.available_on_other_thread(npu))

    def test_leaving_npu_releases_shared_guard_and_keeps_stable_mutex(self):
        device = ["NPU"]
        lock = SlotLock(lambda: device[0])
        npu = _device_lock("NPU")
        with lock.moving_to("GPU"):
            device[0] = "GPU"
            self.assertFalse(self.available_on_other_thread(npu))
            self.assertFalse(self.available_on_other_thread(lock))
        with lock:
            self.assertTrue(self.available_on_other_thread(npu))
            self.assertFalse(self.available_on_other_thread(lock))

    def test_failed_nonblocking_acquire_releases_slot(self):
        lock = SlotLock(lambda: "NPU")
        npu = _device_lock("NPU")
        with npu:
            self.assertFalse(self.available_on_other_thread(lock))
        self.assertTrue(self.available_on_other_thread(lock))


if __name__ == "__main__":
    unittest.main()
