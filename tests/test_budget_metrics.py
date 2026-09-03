import tempfile
import unittest
import sys
import types
from pathlib import Path
from unittest import mock

# These tests exercise arithmetic only. Keep them runnable on development
# machines without the native OpenVINO wheels; every device call is mocked.
try:
    import openvino  # noqa: F401
except (ImportError, OSError):
    openvino = types.ModuleType("openvino")
    openvino.Core = object
    sys.modules["openvino"] = openvino

from core import config
from core.hardware import devices
from core.metrics import (clear_metrics, metrics_snapshot, record_turn,
                          register_metrics_route)
from core.models import availability
from core.models import geometry


GIB = 2 ** 30


class GpuBudgetTests(unittest.TestCase):
    def setUp(self):
        self.reserve = config.GPU_RESERVE_BYTES

    def tearDown(self):
        config.GPU_RESERVE_BYTES = self.reserve

    def test_reserve_changes_shared_gpu_budget_exactly(self):
        with (mock.patch.object(devices, "_device_mem_bytes", return_value=24 * GIB),
              mock.patch.object(devices, "_gpu_shares_system_ram", return_value=True),
              mock.patch.object(devices, "_mem_status") as memory):
            config.GPU_RESERVE_BYTES = 3 * GIB
            default, ceiling = devices._usable_gpu_bytes("GPU", "GPU", 20 * GIB)
            config.GPU_RESERVE_BYTES = 1 * GIB
            lowered, _ = devices._usable_gpu_bytes("GPU", "GPU", 20 * GIB)

        self.assertEqual(ceiling, 24 * GIB)
        self.assertEqual(default, 17 * GIB)
        self.assertEqual(lowered, 19 * GIB)
        self.assertEqual(lowered - default, 2 * GIB)
        memory.assert_not_called()


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.settings = (config.GPU_RESERVE_BYTES, config.PROMPT_CACHE,
                         config.CONTEXT_TOKENS, config.OFFLOAD_RATIO,
                         config.KV_PRECISION)
        config.PROMPT_CACHE = False
        config.CONTEXT_TOKENS = None
        config.OFFLOAD_RATIO = "auto"
        config.KV_PRECISION = "u8"

    def tearDown(self):
        (config.GPU_RESERVE_BYTES, config.PROMPT_CACHE,
         config.CONTEXT_TOKENS, config.OFFLOAD_RATIO,
         config.KV_PRECISION) = self.settings

    def test_one_memory_read_and_live_reserve_changes_verdict(self):
        with tempfile.TemporaryDirectory() as temp:
            model_path = str(Path(temp) / "model")
            Path(model_path).mkdir()
            model = {"name": "model", "path": model_path, "type": "llm"}

            def usable(_name, _device_id, free):
                return min(24 * GIB, free - config.GPU_RESERVE_BYTES), 24 * GIB

            with (mock.patch.object(availability, "_mem_status",
                                    return_value=(32 * GIB, 10 * GIB)) as memory,
                  mock.patch.object(availability, "_usable_gpu_bytes",
                                    side_effect=usable),
                  mock.patch.object(availability, "_gpu_shares_system_ram",
                                    return_value=True),
                  mock.patch.object(availability, "_gpu_has_xmx", return_value=True),
                  mock.patch.object(availability, "_dir_size_bytes",
                                    return_value=7 * GIB),
                  mock.patch.object(availability, "_moe_expert_fraction",
                                    return_value=0.9),
                  mock.patch.object(availability,
                                    "_effective_kv_bytes_per_token",
                                    return_value=48 * 1024)):
                config.GPU_RESERVE_BYTES = 3 * GIB
                constrained = availability.add_live_fit_data(
                    [model], {"GPU": {"id": "GPU.0"}})[0]["fit"]
                config.GPU_RESERVE_BYTES = 1 * GIB
                roomy = availability.add_live_fit_data(
                    [model], {"GPU": {"id": "GPU.0"}})[0]["fit"]

        self.assertEqual(memory.call_count, 2)  # once in each catalog request
        self.assertEqual(constrained["verdict"], "would_offload_now")
        self.assertFalse(constrained["fits_now"])
        self.assertGreater(constrained["close_to_fit_mb"], 0)
        self.assertEqual(roomy["verdict"], "fits_now")
        self.assertTrue(roomy["fits_now"])
        self.assertEqual(roomy["kv_precision"], "u8")
        self.assertEqual(roomy["system_available_mb"], 10 * 1024)
        self.assertEqual(roomy["driver_ceiling_mb"], 24 * 1024)

    def test_missing_shared_memory_read_never_claims_a_live_fit(self):
        model = {"name": "model", "path": "unused", "type": "llm"}
        with (mock.patch.object(availability, "_mem_status",
                                return_value=(None, None)) as memory,
              mock.patch.object(availability, "_usable_gpu_bytes",
                                return_value=(24 * GIB, 24 * GIB)),
              mock.patch.object(availability, "_gpu_shares_system_ram",
                                return_value=True),
              mock.patch.object(availability, "_gpu_has_xmx", return_value=True),
              mock.patch.object(availability, "_dir_size_bytes",
                                return_value=7 * GIB),
              mock.patch.object(availability, "_moe_expert_fraction",
                                return_value=0.9)):
            fit = availability.add_live_fit_data(
                [model], {"GPU": {"id": "GPU.0"}})[0]["fit"]

        memory.assert_called_once_with()
        self.assertEqual(fit["verdict"], "unknown")
        self.assertIsNone(fit["budget_mb"])
        self.assertEqual(fit["driver_ceiling_mb"], 24 * 1024)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.precision = config.KV_PRECISION

    def tearDown(self):
        config.KV_PRECISION = self.precision

    def test_effective_kv_geometry_tracks_runtime_precision(self):
        with mock.patch.object(geometry, "_kv_bytes_per_token",
                               return_value=96 * 1024):
            config.KV_PRECISION = None
            self.assertEqual(geometry._effective_kv_bytes_per_token("model"),
                             96 * 1024)
            config.KV_PRECISION = "f16"
            self.assertEqual(geometry._effective_kv_bytes_per_token("model"),
                             96 * 1024)
            config.KV_PRECISION = "u8"
            self.assertEqual(geometry._effective_kv_bytes_per_token("model"),
                             48 * 1024)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        clear_metrics()

    def tearDown(self):
        clear_metrics()

    def test_fresh_snapshot_is_empty_and_history_is_bounded(self):
        self.assertEqual(metrics_snapshot(), [])
        for index in range(55):
            record_turn(device="GPU", model="test", prompt_tokens=1000 + index,
                        completion_tokens=20, ttft_ms=500, total_ms=1500)
        data = metrics_snapshot()
        self.assertEqual(len(data), 50)
        self.assertEqual(data[0]["prompt_tokens"], 1005)
        self.assertEqual(data[-1]["prompt_tokens"], 1054)
        self.assertEqual(data[-1]["prefill_tokens_per_second"], 2108.0)
        self.assertEqual(data[-1]["decode_tokens_per_second"], 20.0)

    def test_fresh_metrics_route_returns_an_empty_json_list(self):
        from flask import Flask

        app = Flask(__name__)
        register_metrics_route(app)
        response = app.test_client().get("/v1/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), [])


if __name__ == "__main__":
    unittest.main()
