"""The unset prompt-cache default must mean "per device", never "off".

config.PROMPT_CACHE is tri-state (None = CPU on, GPU off). Readers that tested
it for truth treated None as off, so on a default CPU start the KV pool came
out as 0 GB and --context-tokens / --cache-size-gb were ignored.
"""
import unittest
from unittest.mock import patch

from core import config
from core.slots.planning import MemoryPlanning, prompt_cache_on


class _Slot(MemoryPlanning):
    def __init__(self, device):
        self.device_name = device
        self.device_id = device
        self.model_dir = "/nonexistent"


class PromptCacheDefaultTests(unittest.TestCase):
    def setUp(self):
        self._saved = (config.PROMPT_CACHE, config.PROMPT_CACHE_GB,
                       config.CONTEXT_TOKENS)
        config.PROMPT_CACHE, config.PROMPT_CACHE_GB = None, 3
        config.CONTEXT_TOKENS = None

    def tearDown(self):
        (config.PROMPT_CACHE, config.PROMPT_CACHE_GB,
         config.CONTEXT_TOKENS) = self._saved

    def test_unset_is_per_device(self):
        self.assertTrue(prompt_cache_on("CPU"))
        self.assertFalse(prompt_cache_on("GPU"))
        self.assertFalse(prompt_cache_on("NPU"))

    def test_explicit_flag_wins(self):
        config.PROMPT_CACHE = False
        self.assertFalse(prompt_cache_on("CPU"))
        config.PROMPT_CACHE = True
        self.assertTrue(prompt_cache_on("GPU"))

    def test_default_cpu_slot_gets_its_pool(self):
        with patch("core.slots.planning._effective_kv_bytes_per_token",
                   return_value=0):
            gb, _ = _Slot("CPU")._resolve_kv_pool_gb(vlm=False)
        self.assertEqual(gb, 3)

    def test_default_gpu_slot_has_no_pool(self):
        gb, _ = _Slot("GPU")._resolve_kv_pool_gb(vlm=False)
        self.assertEqual(gb, 0)


if __name__ == "__main__":
    unittest.main()
