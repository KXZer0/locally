"""Speech to text with NVIDIA Parakeet TDT, on the plain OpenVINO runtime.

There is no Parakeet pipeline in openvino_genai, so this drives the four
graphs FluidInference exports (core/asr/layout.py) directly, the way VadSlot
drives Silero: mel preprocessor -> FastConformer encoder -> LSTM predictor +
joint, decoded greedily with TDT duration skips (core/asr/tdt.py).

Device placement follows FluidInference's own runtime (eddy-audio): the
preprocessor always runs on the CPU, and the encoder, predictor and joint go
to the requested device. On the NPU the encoder is reshaped to a static
window first (config.ASR_NPU_WINDOW_SECONDS), because the NPU compiles static
shapes only; the predictor and joint fall back to the CPU if the NPU will not
take them, and /health says where they ended up.
"""

import gc
import time

import numpy as np
import openvino as ov

from core import config
from core.asr import ports as P
from core.asr.layout import parakeet_files
from core.asr.mel import HOP, log_mel
from core.asr.tdt import greedy_decode, merge_chunks, resolve_durations
from core.asr.vocab import load_vocab
from core.models.identity import model_display_name
from core.slots.locks import _device_lock

SR = 16000
PREPROC_ROUND = 1000   # eddy-audio: the v3 preprocessor fails on some odd lengths


class ParakeetSlot:
    """Holds a compiled Parakeet export. Same surface as WhisperSlot."""

    engine = "parakeet"

    def __init__(self, device_name, device_id=None):
        self.device_name = device_name
        self.device_id = device_id or device_name
        self.device_full = ""
        self.model_name = ""
        self.model_type = "stt"
        self.status = "not_configured"
        self.lock = _device_lock(self.device_name)
        self.last_used = time.time()
        self.model_dir = None
        self.decoder_device = None
        self.last_language_hint = None
        self._clear()

    def _clear(self):
        self.pre = self.enc = self.dec = self.joint = None
        self._reqs = {}
        self._io = {}
        self.vocab = None
        self.durations = ()
        self.window_frames = None    # static encoder width, or None (dynamic)

    # --- loading ---------------------------------------------------------

    def load(self, model_dir):
        self.status = "loading"
        self.last_used = time.time()
        self.model_dir = model_dir
        self.model_name = model_display_name(model_dir)
        files = parakeet_files(model_dir)
        if not files:
            raise RuntimeError(f"{model_dir} is not a Parakeet OpenVINO export "
                               "(no encoder/decoder/joint .xml found).")
        if not files["vocab"]:
            raise RuntimeError(f"{model_dir} has no vocabulary "
                               "(parakeet_vocab.json or vocab.txt).")
        print(f"  [{self.device_name}] Loading Parakeet ({self.model_name})...",
              flush=True)
        self.vocab = load_vocab(files["vocab"])
        core = ov.Core()
        cache = {"CACHE_DIR": config.MODEL_CACHE_DIR} if config.MODEL_CACHE_DIR else {}
        npu = self.device_name == "NPU"

        enc = core.read_model(str(files["encoder"]))
        mel_in, len_in = P.signal_and_length(enc.inputs, ("mel", "audio_signal"),
                                             "mel input", "the encoder")
        n_mels = P.template(mel_in)[1] if P.rank(mel_in) == 3 else None
        if not n_mels:
            raise RuntimeError("The Parakeet encoder's mel input has no fixed "
                               f"bin count ({P.describe(enc.inputs)}).")
        width = P.template(mel_in)[2]
        if npu and width is None:
            width = int(config.ASR_NPU_WINDOW_SECONDS * 100) + 1
            shapes = {mel_in.get_any_name(): [1, n_mels, width]}
            if len_in is not None:
                shapes[len_in.get_any_name()] = P.static_shape(len_in)
            enc.reshape(shapes)
        self.window_frames = width
        self.enc = core.compile_model(enc, self.device_id, cache)

        dec = core.read_model(str(files["decoder"]))
        joint = core.read_model(str(files["joint"]))
        if npu:
            for m in (dec, joint):
                if not m.is_dynamic():
                    continue
                m.reshape({p.get_any_name(): P.static_shape(p) for p in m.inputs})
        self.decoder_device = self.device_name
        try:
            self.dec = core.compile_model(dec, self.device_id, cache)
            self.joint = core.compile_model(joint, self.device_id, cache)
        except Exception as e:
            if not npu:
                raise
            # Tiny per-step graphs; eddy-audio makes the same fallback.
            print(f"  [{self.device_name}] predictor/joint refused by the NPU "
                  f"({str(e).splitlines()[0][:120]}); running them on CPU",
                  flush=True)
            self.decoder_device = "CPU"
            self.dec = core.compile_model(core.read_model(str(files["decoder"])),
                                          "CPU", cache)
            self.joint = core.compile_model(core.read_model(str(files["joint"])),
                                            "CPU", cache)
        if files["preprocessor"]:
            self.pre = core.compile_model(str(files["preprocessor"]), "CPU", cache)
        self._bind(files, mel_in, len_in, n_mels)

    def _bind(self, files, mel_in, len_in, n_mels):
        """Resolve every port and the joint's head layout, once."""
        io = {"n_mels": n_mels, "mel_in": mel_in.get_any_name(),
              "len_in": len_in.get_any_name() if len_in is not None else None,
              "len_dtype": (len_in.get_element_type().to_dtype()
                            if len_in is not None else np.int64)}
        io["enc_out"], io["enc_len"] = P.signal_and_length(
            self.enc.outputs, ("encoder_output", "outputs"), "output", "the encoder")
        (io["tok"], io["tok_len"], io["states"], io["dec_out"],
         io["state_outs"]) = P.decoder_roles(self.dec.inputs, self.dec.outputs)
        io["j_enc"], io["j_dec"] = P.joint_roles(self.joint.inputs)
        io["j_enc_t"] = P.template(io["j_enc"])
        io["j_dec_t"] = P.template(io["j_dec"])
        if self.pre is not None:
            io["pre_in"], io["pre_len"] = P.signal_and_length(
                self.pre.inputs, ("audio", "signal", "wave"), "audio input",
                "the preprocessor")
            io["pre_out"], io["pre_out_len"] = P.signal_and_length(
                self.pre.outputs, ("mel", "spec", "feat"), "output", "the preprocessor")
        self._io = io
        self._reqs = {k: m.create_infer_request() for k, m in
                      (("pre", self.pre), ("enc", self.enc), ("dec", self.dec),
                       ("joint", self.joint)) if m is not None}
        out_dim = P.template(self.joint.outputs[0])[-1]
        self.durations = resolve_durations(files["config"], out_dim,
                                           self.vocab.blank_id)

    def warmup(self):
        # A real inference, so a model that cannot run fails at startup. The
        # unlocked path: ensure_loaded() calls this while holding the lock.
        noise = (np.random.default_rng(0).standard_normal(SR) * 1e-3).astype(np.float32)
        self._transcribe(noise)
        self.status = "ready"
        print(f"  [{self.device_name}] Parakeet ready "
              f"({len(self.vocab)} pieces, {len(self.durations)} durations"
              + (f", predictor on {self.decoder_device}"
                 if self.decoder_device != self.device_name else "") + ")",
              flush=True)

    def unload(self):
        """Release the compiled graphs. Caller must hold self.lock."""
        if self.enc is None and not self._reqs and not self._io:
            return
        print(f"  [{self.device_name}] Idle — unloading {self.model_name}", flush=True)
        # Ports and requests are native objects that point back at the graph.
        self._clear()
        self.status = "idle_unloaded"
        gc.collect()

    def ensure_loaded(self):
        if self.enc is not None and self.status == "ready":
            return
        with self.lock:
            if self.enc is not None and self.status == "ready":
                return
            if self.model_dir is None:
                raise RuntimeError(f"Slot {self.device_name} has no model_dir")
            print(f"  [{self.device_name}] Reloading {self.model_name}...", flush=True)
            self.load(self.model_dir)
            self.warmup()

    # --- inference -------------------------------------------------------

    def transcribe(self, audio_samples, language=None):
        """float32 16 kHz mono -> text. `language` is a hint Parakeet cannot
        take: the model has no language token and identifies the language
        itself, so the hint is recorded, not applied."""
        self.last_language_hint = language or None
        with self.lock:
            text = self._transcribe(audio_samples)
            self.last_used = time.time()
        return text

    def _window_samples(self):
        if self.window_frames:
            return (self.window_frames - 1) * HOP
        return int(config.ASR_CHUNK_SECONDS * SR)

    def _transcribe(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""
        win = self._window_samples()
        if audio.size <= win:
            return self.vocab.decode([t for t, _ in self._run_window(audio, 0.0)])
        overlap = min(2 * SR, win // 8)
        starts = list(range(0, audio.size - overlap, win - overlap))
        chunks = []
        for i, s in enumerate(starts):
            seg = audio[s:s + win]
            toks = self._run_window(seg, s / SR)
            keep_from = None if i == 0 else (s + overlap / 2) / SR
            last = i == len(starts) - 1 or s + win >= audio.size
            keep_until = None if last else (s + win - overlap / 2) / SR
            chunks.append((toks, keep_from, keep_until))
            if last:
                break
        merged = merge_chunks(chunks, self.vocab.is_word_start)
        return self.vocab.decode([t for t, _ in merged])

    def _features(self, samples):
        io = self._io
        if self.pre is None:
            return log_mel(samples, io["n_mels"])
        n = -(-samples.size // PREPROC_ROUND) * PREPROC_ROUND
        padded = np.zeros(n, dtype=np.float32)
        padded[:samples.size] = samples
        feed = {io["pre_in"]: padded[None, :]}
        if io["pre_len"] is not None:
            feed[io["pre_len"]] = np.array(
                [n], dtype=io["pre_len"].get_element_type().to_dtype())
        res = self._reqs["pre"].infer(feed)
        mel = np.array(res[io["pre_out"]])[0]
        valid = (int(np.array(res[io["pre_out_len"]]).reshape(-1)[0])
                 if io["pre_out_len"] is not None else mel.shape[-1])
        return mel, valid

    def _run_window(self, samples, offset_s):
        """One encoder pass + decode -> [(token, global_seconds), ...]."""
        io = self._io
        mel, valid = self._features(samples)
        if self.window_frames:
            fixed = np.zeros((io["n_mels"], self.window_frames), dtype=np.float32)
            valid = min(valid, self.window_frames)
            fixed[:, :valid] = mel[:, :valid]
            mel = fixed
        feed = {io["mel_in"]: mel[None].astype(np.float32)}
        if io["len_in"] is not None:
            feed[io["len_in"]] = np.array([valid], dtype=io["len_dtype"])
        res = self._reqs["enc"].infer(feed)
        enc = np.array(res[io["enc_out"]])[0]
        hidden = io["j_enc_t"][-1]
        # NeMo and the FluidInference export are channels-first [H, T].
        frames = enc.T if (hidden is None or enc.shape[0] == hidden) else enc
        n = (int(np.array(res[io["enc_len"]]).reshape(-1)[0])
             if io["enc_len"] is not None else frames.shape[0])
        n = min(n, frames.shape[0])
        if n <= 0:
            return []
        frame_s = HOP / SR * max(1, round(valid / n))
        toks = greedy_decode(frames, n, self._predict, self._joint,
                             self.vocab.blank_id, self.durations)
        return [(t, offset_s + f * frame_s) for t, f in toks]

    def _predict(self, token, state):
        io = self._io
        feed = {io["tok"]: np.array([[token]], dtype=io["tok"].get_element_type().to_dtype())}
        if io["tok_len"] is not None:
            feed[io["tok_len"]] = np.array(
                [1], dtype=io["tok_len"].get_element_type().to_dtype())
        for i, port in enumerate(io["states"]):
            feed[port] = P.zeros(port) if state is None else state[i]
        res = self._reqs["dec"].infer(feed)
        return (np.array(res[io["dec_out"]]),
                tuple(np.array(res[p]) for p in io["state_outs"]))

    def _joint(self, enc_vec, pred):
        io = self._io
        res = self._reqs["joint"].infer({
            io["j_enc"]: P.fit(enc_vec, io["j_enc_t"]),
            io["j_dec"]: P.fit(pred.reshape(-1), io["j_dec_t"])})
        return np.array(res[0]).reshape(-1)

    @property
    def info(self):
        return {
            "status": self.status,
            "model": self.model_name,
            "type": self.model_type,
            "engine": self.engine,
            "device_name": self.device_name,
            "device": self.device_full,
            "decoder_device": self.decoder_device,
            "window_s": round(self._window_samples() / SR, 2) if self._io else None,
            "language": "auto",
        }
