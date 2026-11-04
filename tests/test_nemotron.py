"""parakeet-unified / Nemotron streaming: geometry, decode carry, the slot on
a tiny real export in both streaming modes, streaming == offline, and the
voice session feeding a turn to the model while it is still being spoken.

The fixture (tests/parakeet_fixture.py, build_nemotron) uses eddy-audio's
exact port names and metadata fields; its encoder answers tones, so the
transcript depends on the audio and on where the units cut it.
"""

import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import parakeet_fixture as fx   # noqa: E402

from core import config, runtime   # noqa: E402
from core.asr.geometry import Geometry   # noqa: E402
from core.asr.layout import asr_engine, describe_unknown, parakeet_files   # noqa: E402
from core.asr.mel import log_mel   # noqa: E402
from core.asr.tdt import greedy_decode   # noqa: E402
from core.slots.nemotron import NemotronSlot   # noqa: E402

SPEECH = [(None, .2), (1, .3), (None, .2), (2, .3), (None, .2), (3, .3),
          (None, .2), (1, .6), (None, .3)]
TEXT = "hello world! hello"
WIN_META = {"streaming": True, "context_encoder_frames": {"left": 2, "chunk": 4, "right": 2},
            "window_mel_frames": 65}


def frames_of(audio, n=512):
    return [audio[i:i + n] for i in range(0, len(audio), n)]


class GeometryTests(unittest.TestCase):
    def test_window_unit_waits_for_its_right_context(self):
        g = Geometry(WIN_META, has_cache=False, has_prompt=False)
        self.assertEqual((g.mode, g.step, g.left_s, g.right_s, g.win_s),
                         ("window", 5120, 2560, 2560, 10240))
        self.assertFalse(g.ready(0, 5120 + 2559, False))
        self.assertTrue(g.ready(0, 5120 + 2560, False))
        self.assertTrue(g.ready(0, 10, True))       # the tail runs at finalize
        self.assertFalse(g.ready(10, 10, True))
        audio = np.arange(1, 6001, dtype=np.float32)
        win, valid = g.unit(audio, 0, 6000)
        self.assertEqual((win[:2560].any(), win[2560], valid), (False, 1.0, 6000))
        win, valid = g.unit(audio, 5120, 6000)       # left context + tail
        self.assertEqual((win[0], valid), (2561.0, 6000 - 2560))

    def test_cache_unit_is_a_fixed_chunk_with_full_valid_length(self):
        g = Geometry({**fx.CACHE, **fx.CACHE_SHAPES}, has_cache=True, has_prompt=False)
        self.assertEqual((g.mode, g.step, g.latency_seconds), ("cache", 2560, 0.16))
        self.assertTrue(g.ready(2560, 5120, False))
        chunk, valid = g.unit(np.ones(3000, np.float32), 2560, 3000)
        self.assertEqual((chunk[:440].sum(), chunk[440:].any(), valid), (440, False, 2560))

    def test_offline_variant_and_bad_geometry_are_refused_by_name(self):
        with self.assertRaisesRegex(RuntimeError, "offline variant.*streaming/fp16"):
            Geometry({}, has_cache=False, has_prompt=False)
        with self.assertRaisesRegex(RuntimeError, "window_mel_frames is 64"):
            Geometry({**WIN_META, "window_mel_frames": 64}, False, False)
        with self.assertRaisesRegex(RuntimeError, "cache_channel_shape"):
            Geometry({}, has_cache=True, has_prompt=False)

    def test_prompt_id_resolves_like_eddy(self):
        g = Geometry({**fx.CACHE, **fx.CACHE_SHAPES, "prompt_dictionary": fx.PROMPTS,
                      "default_prompt_id": 101}, has_cache=True, has_prompt=True)
        self.assertEqual([g.prompt_id(x) for x in ("fr-FR", "fr", "FR", "xx", None)],
                         [7, 7, 7, 101, 101])


class FeatureAndDecodeTests(unittest.TestCase):
    def test_valid_samples_masks_and_excludes_padding_from_normalisation(self):
        a = fx.tone_audio([(1, .1)])
        full, _ = log_mel(np.concatenate([a, np.zeros(800, np.float32)]), 16,
                          valid_samples=len(a))
        alone, _ = log_mel(a, 16, valid_samples=len(a))
        v = len(a) // 160
        self.assertFalse(full[:, v:].any())
        np.testing.assert_allclose(full[:, :v], alone[:, :v], atol=1e-5)
        raw, _ = log_mel(a, 16, normalize=False, guard=6e-8)
        self.assertLess(raw.mean(), 0)                 # raw log-mel, not z-scores

    def test_carry_makes_chunked_decode_equal_one_pass(self):
        def predict(tok, state):
            return np.array([tok], np.float32), (state or 0) + 1

        def joint(frame, pred):           # emit frame's token unless just emitted
            v = np.full(5, -9.0, np.float32)
            v[4] = 0.0
            if frame and int(pred[0]) != frame:
                v[int(frame)] = 9.0
            return v
        frames = [1, 1, 0, 2, 2, 2, 3, 0, 3, 3]
        whole = [t for t, _ in greedy_decode(frames, 10, predict, joint, 4, ())]
        carry, parts = {}, []
        for i in range(0, 10, 3):
            chunk = frames[i:i + 3]
            parts += [t for t, _ in greedy_decode(chunk, len(chunk), predict, joint,
                                                  4, (), carry=carry)]
        self.assertEqual(parts, whole)
        self.assertEqual(whole, [1, 2, 3])
        self.assertEqual(carry["last"], 3)


class LayoutTests(unittest.TestCase):
    def test_nemotron_folder_wins_over_the_tdt_role_match(self):
        with tempfile.TemporaryDirectory() as root:
            d = fx.build_nemotron(Path(root) / "streaming" / "fp16")
            self.assertIsNotNone(parakeet_files(d))     # the trap: *encoder*.xml
            self.assertEqual(asr_engine(d), "nemotron")
            self.assertIsNone(asr_engine(root))
            self.assertIn("streaming/fp16", describe_unknown(root))

    def test_not_offered_as_a_chat_model(self):
        from core.models.identity import _is_generative_dir
        with tempfile.TemporaryDirectory() as root:
            self.assertFalse(_is_generative_dir(str(fx.build_nemotron(Path(root) / "n"))))


class SlotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved_cache = config.MODEL_CACHE_DIR
        config.MODEL_CACHE_DIR = None
        self.audio = fx.tone_audio(SPEECH)

    def tearDown(self):
        config.MODEL_CACHE_DIR = self.saved_cache
        self.tmp.cleanup()

    def slot(self, device="CPU", **build):
        d = fx.build_nemotron(Path(self.tmp.name) / build.get("mode", "w"), **build)
        s = NemotronSlot(device, "CPU")
        s.load(str(d))
        s.warmup()
        return s

    def streamed(self, s, language=None):
        st = s.open_stream(language)
        for f in frames_of(self.audio):
            st.feed(f)
        return st.finalize(), st

    def test_window_mode_transcribes_and_streaming_equals_offline(self):
        s = self.slot()
        self.assertEqual(s.geometry.mode, "window")
        self.assertEqual(s.rewritten, 1)                # BitwiseNot -> LogicalNot
        self.assertEqual(s.transcribe(self.audio), TEXT)
        text, st = self.streamed(s)
        self.assertEqual(text, TEXT)
        self.assertEqual(st.units, -(-len(self.audio) // s.geometry.step))
        self.assertEqual(s.info["streaming"], "window")
        self.assertEqual(s.info["language"], "auto")

    def test_cache_mode_carries_encoder_caches_across_chunks(self):
        s = self.slot(mode="cache")
        self.assertEqual(s.transcribe(self.audio), TEXT)
        text, st = self.streamed(s)
        self.assertEqual(text, TEXT)
        units = -(-len(self.audio) // s.geometry.step)
        self.assertEqual(int(st.carry["cache"][2][0]), units)          # cache_len
        self.assertTrue((st.carry["cache"][0] == units).all())         # cache_channel
        self.assertEqual(st.carry["mel_tail"].shape, (fx.NM_MELS, 3))

    def test_predictor_state_crosses_unit_boundaries(self):
        # The final 0.6 s tone spans several units. Spelled once only because
        # the last token is carried; reset it per unit and it repeats.
        s = self.slot()
        g, carry, pos = s.geometry, s.new_carry(), 0
        while g.ready(pos, len(self.audio), True):
            samples, valid = g.unit(self.audio, pos, len(self.audio))
            carry.update(state=None, last=g.blank_id)
            s.run_unit(carry, samples, valid)
            pos += g.step
        self.assertGreater(s.text(carry).count("hello"), 2)

    def test_language_prompt_is_fed_and_reported(self):
        s = self.slot(mode="cache", prompt=True)
        self.assertTrue(s.applies_language)
        self.assertEqual(s.new_carry("fr")["prompt"], 7)
        self.assertEqual(s.transcribe(self.audio, language="fr"), TEXT)
        self.assertEqual(s.info["language"], "prompt")

    def test_npu_path_reshapes_to_static_shapes(self):
        for mode, width in (("window", 65), ("cache", 19)):
            s = self.slot(device="NPU", mode=mode)
            self.assertEqual(list(s.enc.input("mel").get_shape()), [1, fx.NM_MELS, width])
            self.assertFalse(s.dec.inputs[0].get_partial_shape().is_dynamic)
            self.assertEqual(s.transcribe(self.audio), TEXT)

    def test_worker_decodes_during_the_turn_and_cancel_stops_it(self):
        s = self.slot()
        st = s.open_stream()
        st.feed(self.audio)
        ready = (len(self.audio) - s.geometry.right_s) // s.geometry.step
        deadline = time.time() + 5
        while st.units < ready and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(st.units, ready)               # before finalize()
        self.assertEqual(st.finalize(), TEXT)
        c = s.open_stream()
        c.feed(self.audio[:1000])
        c.cancel()
        with self.assertRaises(RuntimeError):
            c.feed(self.audio[:10])

    def test_missing_part_is_named(self):
        d = fx.build_nemotron(Path(self.tmp.name) / "x")
        (d / "nemotron_joint.xml").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing its joint"):
            NemotronSlot("CPU").load(str(d))


class FakeVad:
    def new_state(self):
        return None

    def probability(self, frame, state):
        return 1.0 if float(np.sqrt(np.mean(frame ** 2))) > 0.05 else 0.0


class SessionTests(unittest.TestCase):
    """VoiceStreamSession feeding the turn to the streaming slot as it arrives."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        config.MODEL_CACHE_DIR, self.saved = None, config.MODEL_CACHE_DIR
        d = fx.build_nemotron(Path(self.tmp.name) / "w")
        self.slot = NemotronSlot("CPU", "CPU")
        self.slot.load(str(d))
        self.slot.warmup()
        self.patches = [mock.patch.object(runtime, "_vad_slot", FakeVad()),
                        mock.patch.object(runtime, "_speaker_slot", None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        config.MODEL_CACHE_DIR = self.saved
        self.tmp.cleanup()

    def run_turn(self, slot, spec):
        from core.voice.session import VoiceStreamSession
        with mock.patch.object(runtime, "whisper_slot", slot):
            s = VoiceStreamSession()
            events, stream, units_at_end = [], None, None
            for f in frames_of(fx.tone_audio(spec)):
                if len(f) < 512:
                    break
                before = stream.units if stream is not None else None
                for ev in s.feed(f):
                    events.append(ev)
                    if ev["type"] == "speech_start":
                        stream = s.stream
                    if ev["type"] == "speech_end":
                        units_at_end = before
                # A microphone sends a frame every 32 ms; the worker keeps up
                # with that. Pace the test the same way, not faster.
                deadline = time.time() + 2
                while (s.stream is not None and time.time() < deadline
                       and s.stream.geo.ready(s.stream.pos, s.stream.n, False)):
                    time.sleep(0.001)
            out, deadline = [], time.time() + 5
            while not out and time.time() < deadline and s.awaiting is not None:
                out = s.drain()
                time.sleep(0.01)
            return s, events, out, units_at_end

    def test_turn_is_transcribed_while_spoken(self):
        spec = [(None, .3), (1, .4), (None, .2), (2, .4), (None, 1.3)]
        with mock.patch("core.voice.session.VoiceStreamSession._transcribe",
                        side_effect=AssertionError("batch path used")):
            s, events, out, units = self.run_turn(self.slot, spec)
        self.assertEqual([e["type"] for e in events], ["speech_start", "speech_end"])
        self.assertEqual([o["text"] for o in out], ["hello world"])
        self.assertGreater(units, 3)            # decoded before the pause ended
        self.assertIsNone(s.stream)

    def test_too_short_turn_cancels_its_stream(self):
        s, events, out, _ = self.run_turn(self.slot, [(None, .3), (1, .25), (None, 1.2)])
        self.assertTrue(events[-1].get("tooShort"))
        self.assertEqual(out, [])

    def test_batch_engine_keeps_the_old_path(self):
        calls = []
        batch = types.SimpleNamespace(
            status="ready", device_name="CPU", ensure_loaded=lambda: None,
            transcribe=lambda a, language=None: calls.append(len(a)) or "batch text")
        spec = [(None, .3), (1, .4), (None, .2), (2, .4), (None, 1.3)]
        with mock.patch("core.voice.asr_runs._slot_serviceable", return_value=True):
            _, _, out, _ = self.run_turn(batch, spec)
        self.assertEqual([o["text"] for o in out], ["batch text"])
        self.assertTrue(calls)

    def test_verbose_json_reports_an_applied_language_prompt(self):
        from core.app import create_app
        try:
            import soundfile   # noqa: F401
        except ImportError:
            self.skipTest("soundfile not installed")
        from test_parakeet import _wav
        slot = types.SimpleNamespace(engine="nemotron", status="ready", device_name="CPU",
                                     model_name="m", applies_language=True,
                                     last_detected_language="fr-FR",
                                     ensure_loaded=lambda: None,
                                     transcribe=lambda a, language=None: "salut")
        with mock.patch.object(runtime, "whisper_slot", slot):
            v = create_app().test_client().post(
                "/v1/audio/transcriptions",
                data={"file": (_wav(), "a.wav"), "language": "fr",
                      "response_format": "verbose_json"},
                content_type="multipart/form-data").json
        self.assertEqual((v["language_applied"], v["detected_language"]), (True, "fr-FR"))


class BuildSlotTests(unittest.TestCase):
    def test_folder_picks_the_streaming_engine_and_npu_is_allowed(self):
        from core.launch.slots import _asr_slot
        with tempfile.TemporaryDirectory() as root:
            d = fx.build_nemotron(Path(root) / "u")
            a = types.SimpleNamespace(asr_dir=str(d), whisper_dir=None,
                                      asr_device="npu", whisper_device="CPU")
            s = _asr_slot(a, {"CPU": {}, "NPU": {}}, lambda k: k)
            self.assertIsInstance(s, NemotronSlot)
            self.assertEqual(s.device_name, "NPU")


if __name__ == "__main__":
    unittest.main()
