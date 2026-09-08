import threading
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

    def test_no_slot_is_a_recoverable_api_error(self):
        from core.models import manage
        with (mock.patch.object(runtime, "primary", None),
              mock.patch.object(manage, "_available_models", return_value=[
                  {"name": "fixture", "path": "/fixture"}])):
            response = self.client.post("/v1/models/load", json={"model": "fixture"})
        self.assertEqual(response.status_code, 503)


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
