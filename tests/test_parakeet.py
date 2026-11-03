"""Parakeet TDT: decoding, vocabulary, layout detection, slot and routing.

The decode tests drive core/asr/tdt.py with hand-built logits, so each TDT
rule (duration skip, blank, zero-duration token, runaway cut) is pinned by a
case that fails if that one rule changes. The slot tests compile a tiny real
OpenVINO export with Parakeet's port contract (tests/parakeet_fixture.py).
"""

import io
import json
import os
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import parakeet_fixture as fx   # noqa: E402

from core import config, runtime   # noqa: E402
from core.asr.layout import asr_engine, describe_unknown, parakeet_files   # noqa: E402
from core.asr.tdt import greedy_decode, merge_chunks, resolve_durations   # noqa: E402
from core.asr.vocab import load_vocab   # noqa: E402
from core.slots.parakeet import ParakeetSlot   # noqa: E402

BLANK = 4
DURS = (0, 1, 2, 3, 4)


def logits(token, dur_index):
    v = np.full(BLANK + 1 + len(DURS), -9.0, dtype=np.float32)
    v[token] = 9.0
    v[BLANK + 1 + dur_index] = 9.0
    return v


class Scripted:
    """predict/joint pair that plays back (token, duration-index) per call and
    records which frame each joint call saw and how often the predictor ran."""

    def __init__(self, script):
        self.script = list(script)
        self.frames = []
        self.predicts = 0

    def predict(self, token, state):
        self.predicts += 1
        return np.array([token], dtype=np.float32), (state or 0) + 1

    def joint(self, frame, pred):
        self.frames.append(int(frame))
        tok, d = self.script.pop(0) if self.script else (BLANK, 1)
        return logits(tok, d)


class GreedyTdtTests(unittest.TestCase):
    def decode(self, script, n, durations=DURS, **kw):
        s = Scripted(script)
        out = greedy_decode(np.arange(n), n, s.predict, s.joint, BLANK,
                            durations, **kw)
        return out, s

    def test_duration_head_skips_frames_on_blank_and_token(self):
        # blank skip 3 -> token at frame 3 skip 2 -> token at 5 skip 4 -> end.
        out, s = self.decode([(BLANK, 3), (1, 2), (2, 4)], n=9)
        self.assertEqual(out, [(1, 3), (2, 5)])
        self.assertEqual(s.frames, [0, 3, 5])

    def test_blank_with_zero_duration_still_advances_one_frame(self):
        out, s = self.decode([(BLANK, 0), (BLANK, 0), (1, 1)], n=3)
        self.assertEqual(out, [(1, 2)])
        self.assertEqual(s.frames, [0, 1, 2])

    def test_zero_duration_token_stays_on_its_frame(self):
        # Two pieces emitted on frame 0, then a skip.
        out, s = self.decode([(1, 0), (2, 0), (BLANK, 4)], n=4)
        self.assertEqual(out, [(1, 0), (2, 0)])
        self.assertEqual(s.frames, [0, 0, 0])

    def test_runaway_frame_is_cut_at_max_symbols(self):
        out, _ = self.decode([(1, 0)] * 10 + [(BLANK, 4)], n=2, max_symbols=3)
        self.assertEqual([f for _, f in out], [0, 0, 0, 1, 1, 1])

    def test_predictor_runs_once_per_emission_not_per_blank(self):
        _, s = self.decode([(BLANK, 1)] * 5 + [(1, 1)] + [(BLANK, 1)] * 5, n=11)
        self.assertEqual(s.predicts, 2)     # start + after the one token

    def test_plain_rnnt_has_no_duration_head(self):
        s = Scripted([])
        def joint(frame, pred):
            v = np.full(BLANK + 1, -9.0, dtype=np.float32)
            v[1 if frame == 1 and pred[0] == BLANK else BLANK] = 9.0
            return v
        out = greedy_decode(np.arange(3), 3, s.predict, joint, BLANK, ())
        self.assertEqual(out, [(1, 1)])

    def test_padding_frames_are_never_decoded(self):
        _, s = self.decode([(BLANK, 1)] * 10, n=4)
        self.assertEqual(s.frames, [0, 1, 2, 3])

    def test_short_joint_output_is_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, "duration scores"):
            greedy_decode([0], 1, lambda t, s: (0, s), lambda f, p: np.zeros(3),
                          BLANK, DURS)


class MergeTests(unittest.TestCase):
    def test_overlap_split_at_midpoint_keeps_words_whole(self):
        word = {1: True, 2: False, 3: True, 4: True}.get     # 2 continues 1
        a = [(1, 0.0), (3, 4.9), (1, 9.8), (2, 10.1)]      # ...▁un|der at the seam
        b = [(1, 9.8), (2, 10.1), (4, 10.6), (3, 13.0)]
        merged = merge_chunks([(a, None, 10.0), (b, 10.0, None)], word)
        self.assertEqual([t for t, _ in merged], [1, 3, 1, 2, 4, 3])


class VocabTests(unittest.TestCase):
    def test_v3_nested_json_blank_is_one_past_the_end(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "parakeet_vocab.json"
            p.write_text(json.dumps({"id_to_token": {"0": "<unk>", "1": "▁Hi",
                                                     "2": ",", "3": "▁there"},
                                     "vocab_size": 4}), encoding="utf-8")
            v = load_vocab(p)
        self.assertEqual(v.blank_id, 4)
        self.assertEqual(v.decode([0, 1, 2, 3, 4]), "Hi, there")
        self.assertTrue(v.is_word_start(1))
        self.assertFalse(v.is_word_start(2))

    def test_v2_flat_json_and_vocab_txt_with_explicit_blank(self):
        with tempfile.TemporaryDirectory() as d:
            flat = Path(d) / "v.json"
            flat.write_text(json.dumps({"0": "▁a", "1": "b"}), encoding="utf-8")
            txt = Path(d) / "vocab.txt"
            txt.write_text("▁a 0\nb 1\n<blk> 2\n", encoding="utf-8")
            self.assertEqual(load_vocab(flat).blank_id, 2)
            v = load_vocab(txt)
        self.assertEqual(v.blank_id, 2)
        self.assertEqual(v.decode([0, 1, 0]), "ab a")

    def test_special_pieces_never_reach_the_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.json"
            p.write_text(json.dumps(["<|nospeech|>", "▁ok", "<unk>"]),
                         encoding="utf-8")
            self.assertEqual(load_vocab(p).decode([0, 1, 2]), "ok")


class DurationTests(unittest.TestCase):
    def test_bins_from_config_then_joint_width(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "config.json"
            cfg.write_text(json.dumps({"tdt_durations": [0, 1, 2]}))
            self.assertEqual(resolve_durations(cfg, None, 8192), (0, 1, 2))
        self.assertEqual(resolve_durations(None, 8198, 8192), (0, 1, 2, 3, 4))
        self.assertEqual(resolve_durations(None, 8193, 8192), ())
        with self.assertRaisesRegex(RuntimeError, "tdt_durations"):
            resolve_durations(None, 8196, 8192)


class LayoutTests(unittest.TestCase):
    def test_engine_is_read_from_the_folder(self):
        with tempfile.TemporaryDirectory() as root:
            pk = fx.build(Path(root) / "parakeet")
            wh = Path(root) / "whisper"
            wh.mkdir()
            (wh / "config.json").write_text(json.dumps({"model_type": "whisper"}))
            fused = Path(root) / "fused"
            fused.mkdir()
            (fused / "decoder_joint-model.onnx").write_bytes(b"")
            self.assertEqual(asr_engine(pk), "parakeet")
            self.assertEqual(asr_engine(wh), "whisper")
            self.assertIsNone(asr_engine(fused))
            self.assertIn("fused decoder_joint", describe_unknown(fused))
            self.assertIsNone(parakeet_files(pk)["preprocessor"])

    def test_parakeet_is_not_offered_as_a_chat_model(self):
        from core.models.identity import _is_generative_dir
        with tempfile.TemporaryDirectory() as root:
            self.assertFalse(_is_generative_dir(str(fx.build(Path(root) / "p"))))


class SlotTests(unittest.TestCase):
    """The slot against a real compiled (tiny) export."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = (config.ASR_CHUNK_SECONDS, config.ASR_NPU_WINDOW_SECONDS,
                      config.MODEL_CACHE_DIR)
        config.MODEL_CACHE_DIR = None

    def tearDown(self):
        (config.ASR_CHUNK_SECONDS, config.ASR_NPU_WINDOW_SECONDS,
         config.MODEL_CACHE_DIR) = self.saved
        self.tmp.cleanup()

    def slot(self, device="CPU", **build):
        d = fx.build(Path(self.tmp.name) / "m", **build)
        s = ParakeetSlot(device, "CPU")
        s.load(str(d))
        s.warmup()
        return s

    def test_transcribes_through_numpy_mel_and_ir_preprocessor(self):
        self.assertEqual(self.slot().transcribe(np.zeros(16000, np.float32)),
                         "hello world!")
        s = self.slot(preprocessor=True)
        self.assertIsNotNone(s.pre)
        self.assertEqual(s.transcribe(np.zeros(12345, np.float32)), "hello world!")

    def test_long_audio_is_windowed_and_stitched_once(self):
        config.ASR_CHUNK_SECONDS = 5
        s = self.slot()
        # Every window starts "hello world!"; only the first one owns it.
        self.assertEqual(s.transcribe(np.zeros(16000 * 17, np.float32)),
                         "hello world!")

    def test_npu_path_reshapes_the_encoder_to_a_static_window(self):
        config.ASR_NPU_WINDOW_SECONDS = 2
        s = self.slot(device="NPU")
        self.assertEqual(s.window_frames, 201)
        self.assertEqual(list(s.enc.inputs[0].get_shape()), [1, fx.N_MELS, 201])
        self.assertEqual(s.transcribe(np.zeros(16000 * 5, np.float32)),
                         "hello world!")

    def test_language_hint_is_accepted_and_recorded_not_applied(self):
        s = self.slot()
        self.assertEqual(s.transcribe(np.zeros(8000, np.float32), language="en"),
                         "hello world!")
        self.assertEqual(s.last_language_hint, "en")
        self.assertEqual(s.info["language"], "auto")

    def test_idle_unload_releases_graphs_and_reloads_on_demand(self):
        s = self.slot()
        with s.lock:
            s.unload()
        self.assertIsNone(s.enc)
        self.assertEqual((s._reqs, s._io), ({}, {}))
        self.assertEqual(s.status, "idle_unloaded")
        s.ensure_loaded()
        self.assertEqual(s.status, "ready")
        self.assertEqual(s.transcribe(np.zeros(8000, np.float32)), "hello world!")

    def test_a_folder_without_a_vocabulary_fails_plainly(self):
        d = fx.build(Path(self.tmp.name) / "nv")
        (d / "parakeet_vocab.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "no vocabulary"):
            ParakeetSlot("CPU").load(str(d))


def _wav(seconds=1.0):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(int(16000 * seconds), "<i2").tobytes())
    buf.seek(0)
    return buf


class RoutingTests(unittest.TestCase):
    def setUp(self):
        from core.app import create_app
        self.client = create_app().test_client()
        self.slot = types.SimpleNamespace(
            engine="parakeet", status="ready", device_name="NPU",
            model_name="parakeet-tdt-0.6b-v3-ov", calls=[],
            info={"engine": "parakeet", "status": "ready"})
        self.slot.ensure_loaded = lambda: None
        self.slot.transcribe = (lambda a, language=None:
                                self.slot.calls.append((len(a), language)) or "hi there")

    def post(self, **form):
        try:
            import soundfile   # noqa: F401
        except ImportError:
            self.skipTest("soundfile not installed")
        with mock.patch.object(runtime, "whisper_slot", self.slot):
            return self.client.post("/v1/audio/transcriptions",
                                    data={"file": (_wav(), "a.wav"), **form},
                                    content_type="multipart/form-data")

    def test_transcription_routes_to_the_parakeet_slot(self):
        r = self.post(language="en")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json, {"text": "hi there"})
        self.assertEqual(r.headers["X-Device"], "NPU")
        self.assertEqual(self.slot.calls, [(16000, "en")])

    def test_text_and_verbose_json_formats(self):
        self.assertEqual(self.post(response_format="text").data, b"hi there")
        v = self.post(response_format="verbose_json", language="en").json
        self.assertEqual((v["engine"], v["language"], v["language_applied"]),
                         ("parakeet", "en", False))
        self.assertAlmostEqual(v["duration"], 1.0)
        self.assertEqual(self.post(response_format="srt").status_code, 400)

    def test_health_and_warm_name_the_engine(self):
        with mock.patch.object(runtime, "whisper_slot", self.slot):
            health = self.client.get("/health").json
            warm = self.client.post("/v1/audio/warm", json={}).json
            models = self.client.get("/v1/models").json["data"]
        self.assertEqual(health["asr"]["engine"], "parakeet")
        self.assertEqual(health["whisper"], health["asr"])
        self.assertTrue(warm["asr"]["ready"])
        self.assertIn("parakeet@NPU", [m["id"] for m in models])


class BuildSlotTests(unittest.TestCase):
    def args(self, **kw):
        base = dict(asr_dir=None, whisper_dir=None, asr_device=None,
                    whisper_device="CPU")
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_asr_dir_picks_the_engine_from_the_folder(self):
        from core.launch.slots import _asr_slot
        from core.slots.whisper import WhisperSlot
        devices = {"CPU": {"id": "CPU"}, "NPU": {"id": "NPU"}}
        with tempfile.TemporaryDirectory() as root:
            pk = fx.build(Path(root) / "p")
            wh = Path(root) / "w"
            wh.mkdir()
            (wh / "config.json").write_text(json.dumps({"model_type": "whisper"}))
            a = self.args(asr_dir=str(pk), asr_device="npu")
            s = _asr_slot(a, devices, lambda k: k)
            self.assertIsInstance(s, ParakeetSlot)
            self.assertEqual((s.device_name, a.asr_dir), ("NPU", str(pk)))
            # --whisper-dir on a Parakeet folder means "this folder".
            self.assertIsInstance(_asr_slot(self.args(whisper_dir=str(pk)),
                                            devices, lambda k: k), ParakeetSlot)
            w = _asr_slot(self.args(asr_dir=str(wh), asr_device="NPU"),
                          devices, lambda k: k)
            self.assertIsInstance(w, WhisperSlot)
            self.assertEqual(w.device_name, "CPU")    # Whisper never on NPU
            with self.assertRaises(SystemExit):
                _asr_slot(self.args(asr_dir=root), devices, lambda k: k)


if __name__ == "__main__":
    unittest.main()
