"""Unload must release every model-owned native reference without loading models."""

import tempfile
import threading
import unittest
from unittest import mock

from core.slots.device import DeviceSlot
from core.slots.speaker import SpeakerSlot
from core.slots.tts import TtsSlot
from core.slots.vad import VadSlot
from core import startup


class NativeReferenceReleaseTests(unittest.TestCase):
    def test_device_slot_releases_its_lazy_tokenizer(self):
        slot = DeviceSlot("CPU")
        slot.pipe = object()
        slot.tokenizer = object()

        slot.unload()

        self.assertIsNone(slot.pipe)
        self.assertIsNone(slot.tokenizer)

    def test_tts_slot_releases_cached_native_voice_tensors(self):
        slot = TtsSlot("CPU")
        cached_tensor = object()
        slot.pipe = object()
        slot._emb_cache = {"voice": cached_tensor}

        slot.unload()

        self.assertIsNone(slot.pipe)
        self.assertEqual(slot._emb_cache, {})

    def test_vad_slot_releases_compiled_input_ports(self):
        slot = VadSlot("CPU")
        port = object()
        slot.compiled = object()
        slot._audio_in = "input"
        slot._sr_in = "sr"
        slot._state_ins = [("state", port)]

        slot.unload()

        self.assertIsNone(slot.compiled)
        self.assertIsNone(slot._audio_in)
        self.assertIsNone(slot._sr_in)
        self.assertEqual(slot._state_ins, [])

    def test_speaker_slot_releases_input_ports_but_keeps_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            slot = SpeakerSlot("CPU", profile_path=f"{temp}/profile.json")
            audio_port, length_port, profile = object(), object(), object()
            slot.compiled = object()
            slot._audio_in = audio_port
            slot._len_in = length_port
            slot._needs_fbank = True
            slot._n_mels = 128
            slot._window = object()
            slot._melbank = object()
            slot.profile = profile

            slot.unload()

        self.assertIsNone(slot.compiled)
        self.assertIsNone(slot._audio_in)
        self.assertIsNone(slot._len_in)
        self.assertFalse(slot._needs_fbank)
        self.assertEqual(slot._n_mels, 80)
        self.assertIsNone(slot._window)
        self.assertIsNone(slot._melbank)
        self.assertIs(slot.profile, profile)


class StartupFailureCleanupTests(unittest.TestCase):
    class _Slot:
        def __init__(self):
            self.device_name = "CPU"
            self.device_full = ""
            self.model_name = "fixture"
            self.model_type = "llm"
            self.status = "not_configured"
            self.lock = threading.Lock()
            self.resource = None
            self.unload_called = False

        def load(self, _model_dir):
            self.resource = object()

        def warmup(self):
            raise RuntimeError("warmup failed after load")

        def unload(self):
            self.assert_lock_held = self.lock.locked()
            self.unload_called = True
            self.resource = None
            self.status = "idle_unloaded"

    def test_startup_failure_releases_a_partially_loaded_slot(self):
        slot = self._Slot()
        with mock.patch.object(startup, "_banner_printed", False):
            startup._load_in_background(slot, "fixture", {"CPU": {"name": "CPU"}},
                                        1234, 0, [slot])

        self.assertTrue(slot.unload_called)
        self.assertTrue(slot.assert_lock_held)
        self.assertIsNone(slot.resource)
        self.assertEqual(slot.status, "error")


if __name__ == "__main__":
    unittest.main()
