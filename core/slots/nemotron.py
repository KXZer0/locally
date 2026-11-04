"""Streaming speech to text: parakeet-unified-en-0.6b and the Nemotron models.

Same slot surface as ParakeetSlot (which it extends for the lifecycle), a
different model underneath: a streaming FastConformer encoder and a plain
RNN-T joint (no duration head), in FluidInference's `nemotron_*` layout.
Everything here was read from eddy-audio's `eddy::nemotron` backend
(src/models/nemotron/nemotron_openvino.cpp), which serves all three models
through one code path; the port names below are its names.

    encoder  in:  mel [1, bins, F] f32, mel_length [1] i32
                  + cache_channel, cache_time (metadata shapes), cache_len [1]
                    on the cache-aware Nemotron encoders
                  + prompt_id [1] i32 on the multilingual one
             out: encoded [1, D, T] (+ cache_*_out, same shapes as the inputs)
    decoder  in:  token [1, 1] i32|i64, token_length [1], h_in, c_in
                  [layers, 1, hidden]
             out: decoder_out [1, H, 1], h_out, c_out
    joint    in:  encoder [1, D, 1], decoder [1, H, 1]
             out: logits, blank at metadata blank_idx

The audio is cut by core/asr/geometry.py and every transcription -- an
upload or a live voice turn -- goes through core/asr/stream.py, so the
offline and streaming results are the same computation.
"""

import time

import numpy as np
import openvino as ov

from core import config
from core.asr import ports as P
from core.asr.geometry import Geometry, read_metadata
from core.asr.layout import nemotron_files
from core.asr.mel import log_mel
from core.asr.nemotron_io import (bind, npu_safe, static_encoder_shapes,
                                  static_small_shapes)
from core.asr.stream import StreamingTranscriber
from core.asr.tdt import greedy_decode
from core.asr.vocab import Vocab, load_vocab
from core.models.identity import model_display_name
from core.slots.parakeet import SR, ParakeetSlot

LOG_GUARD = 6e-8      # eddy's MelFeaturizer: the value stored in the exported IR


class NemotronSlot(ParakeetSlot):
    engine = "nemotron"

    def _clear(self):
        super()._clear()
        self.geometry = None
        self.applies_language = False
        self.last_detected_language = None

    def load(self, model_dir):
        self.status = "loading"
        self.last_used = time.time()
        self.model_dir = model_dir
        self.model_name = model_display_name(model_dir)
        files = nemotron_files(model_dir)
        if not files:
            raise RuntimeError(f"{model_dir} is not a Nemotron-family export "
                               "(no nemotron_encoder.xml + metadata.json).")
        missing = [n for n, p in files.items() if p is None]
        if missing:
            raise RuntimeError(f"{model_dir} is missing its {', '.join(missing)} "
                               "file(s) (expected nemotron_encoder/decoder/joint"
                               ".xml, nemotron_vocab.json, metadata.json).")
        print(f"  [{self.device_name}] Loading {self.model_name} "
              "(streaming RNN-T)...", flush=True)
        core = ov.Core()
        cache = {"CACHE_DIR": config.MODEL_CACHE_DIR} if config.MODEL_CACHE_DIR else {}
        npu = self.device_name == "NPU"

        enc = core.read_model(str(files["encoder"]))
        names = set().union(*(p.get_names() for p in enc.inputs))
        geo = Geometry(read_metadata(files["metadata"]),
                       has_cache="cache_channel" in names,
                       has_prompt="prompt_id" in names,
                       where=str(files["metadata"]))
        self.rewritten = npu_safe(enc)
        if npu and enc.is_dynamic():
            enc.reshape(static_encoder_shapes(enc, geo))
        self.enc = core.compile_model(enc, self.device_id, cache)
        dec = core.read_model(str(files["decoder"]))
        joint = core.read_model(str(files["joint"]))
        if npu:
            for m in (dec, joint):
                if m.is_dynamic():
                    m.reshape(static_small_shapes(m, geo))
        self.decoder_device = self.device_name
        try:
            self.dec = core.compile_model(dec, self.device_id, cache)
            self.joint = core.compile_model(joint, self.device_id, cache)
        except Exception as e:
            if not npu:
                raise
            print(f"  [{self.device_name}] predictor/joint refused by the NPU "
                  f"({str(e).splitlines()[0][:120]}); running them on CPU",
                  flush=True)
            self.decoder_device = "CPU"
            self.dec = core.compile_model(str(files["decoder"]), "CPU", cache)
            self.joint = core.compile_model(str(files["joint"]), "CPU", cache)
        pieces = load_vocab(files["vocab"]).pieces
        self.vocab = Vocab(pieces, geo.blank_id)
        self.geometry = geo
        self.applies_language = geo.has_prompt
        self._io = bind(self.enc, self.dec, self.joint, geo)
        self._reqs = {k: m.create_infer_request() for k, m in
                      (("enc", self.enc), ("dec", self.dec), ("joint", self.joint))}

    def warmup(self):
        noise = (np.random.default_rng(0).standard_normal(SR) * 1e-3).astype(np.float32)
        self._transcribe(noise)
        self.status = "ready"
        g = self.geometry
        print(f"  [{self.device_name}] {self.model_name} ready: {g.mode} streaming, "
              f"{g.unit_seconds:.2f} s units, {g.latency_seconds:.2f} s to first "
              f"decode, {len(self.vocab)} pieces"
              + (", language prompt" if g.has_prompt else "")
              + (f", predictor on {self.decoder_device}"
                 if self.decoder_device != self.device_name else ""), flush=True)

    # --- the four things core/asr/stream.py needs ---------------------------

    def new_carry(self, language=None):
        g = self.geometry
        carry = {"state": None, "last": g.blank_id, "tokens": [], "mel_tail": None,
                 "prompt": g.prompt_id(language) if g.has_prompt else None}
        if g.mode == "cache":
            carry["cache"] = [np.zeros(g.cache_channel, np.float32),
                              np.zeros(g.cache_time, np.float32),
                              np.zeros([1], self._int(self._io["cache_len"]))]
        return carry

    def run_unit(self, carry, samples, valid):
        g, io = self.geometry, self._io
        mel, _ = log_mel(samples, g.n_mels, normalize=g.normalize,
                         valid_samples=valid, guard=LOG_GUARD)
        if g.mode == "window":
            width = g.window_mel
            fixed = np.zeros((g.n_mels, width), np.float32)
            fixed[:, :min(width, mel.shape[1])] = mel[:, :width]
        else:
            width = g.total_mel
            lead = (carry["mel_tail"] if carry["mel_tail"] is not None
                    else np.zeros((g.n_mels, g.pre_cache), np.float32))
            joined = np.concatenate([lead, mel], axis=1)[:, :width]
            fixed = np.zeros((g.n_mels, width), np.float32)
            fixed[:, :joined.shape[1]] = joined
            carry["mel_tail"] = mel[:, mel.shape[1] - min(g.pre_cache, mel.shape[1]):]
        feed = {io["mel"]: fixed[None]}
        if io["mel_len"] is not None:
            feed[io["mel_len"]] = np.array([width], self._int(io["mel_len"]))
        if g.mode == "cache":
            for k, v in zip(("cache_channel", "cache_time", "cache_len"), carry["cache"]):
                feed[io[k]] = v
        if io["prompt"] is not None:
            feed[io["prompt"]] = np.array([carry["prompt"] or 0], self._int(io["prompt"]))
        res = self._reqs["enc"].infer(feed)
        if g.mode == "cache":
            carry["cache"] = [np.array(res[io[k + "_out"]]).reshape(v.shape).astype(v.dtype)
                              for k, v in zip(("cache_channel", "cache_time", "cache_len"),
                                              carry["cache"])]
        enc = np.array(res[io["encoded"]])[0]
        hidden = io["j_enc_t"][1] if len(io["j_enc_t"]) > 1 else None
        frames = enc if (hidden is not None and enc.shape[-1] == hidden
                         and enc.shape[0] != hidden) else enc.T    # eddy: [D, T]
        if g.mode == "window":
            frames = frames[g.left:g.left + g.chunk]
        toks = greedy_decode(frames, frames.shape[0], self._predict, self._joint,
                             g.blank_id, (), carry=carry)
        carry["tokens"].extend(t for t, _ in toks)

    def text(self, carry):
        g = self.geometry
        keep, detected = [], None
        for t in carry["tokens"]:
            if t in g.lang_tags:
                detected = detected or self.vocab.piece(t).replace("▁", "").strip("<> ")
            elif t != g.blank_id and t < g.vocab_size:
                keep.append(t)
        carry["detected_language"] = detected
        self.last_detected_language = detected
        return self.vocab.decode(keep)

    # --- inference --------------------------------------------------------

    def open_stream(self, language=None):
        """A StreamingTranscriber for one live turn (core/voice/session.py)."""
        self.ensure_loaded()
        self.last_used = time.time()
        return StreamingTranscriber(self, language)

    def _transcribe(self, audio, language=None):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""
        s = StreamingTranscriber(self, language, background=False)
        s.feed(audio)
        return s.finalize()

    def transcribe(self, audio_samples, language=None):
        self.last_language_hint = language or None
        with self.lock:
            text = self._transcribe(audio_samples, language)
            self.last_used = time.time()
        return text

    @staticmethod
    def _int(port):
        return port.get_element_type().to_dtype()

    def _predict(self, token, state):
        io = self._io
        feed = {io["tok"]: np.array([[token]], self._int(io["tok"]))}
        if io["tok_len"] is not None:
            feed[io["tok_len"]] = np.array([1], self._int(io["tok_len"]))
        g = self.geometry
        for i, port in enumerate(io["states"]):
            feed[port] = (np.zeros([g.decoder_layers, 1, g.decoder_hidden], np.float32)
                          if state is None else state[i])
        res = self._reqs["dec"].infer(feed)
        return (np.array(res[io["dec_out"]]),
                tuple(np.array(res[p]) for p in io["state_outs"]))

    def _joint(self, enc_vec, pred):
        io = self._io
        res = self._reqs["joint"].infer({
            io["j_enc"]: P.fit(enc_vec, io["j_enc_t"]),
            io["j_dec"]: P.fit(pred.reshape(-1), io["j_dec_t"])})
        return np.array(res[io["logits"]]).reshape(-1)

    @property
    def info(self):
        g = self.geometry
        return {
            "status": self.status, "model": self.model_name, "type": self.model_type,
            "engine": self.engine, "device_name": self.device_name,
            "device": self.device_full, "decoder_device": self.decoder_device,
            "streaming": g.mode if g else None,
            "unit_s": round(g.unit_seconds, 3) if g else None,
            "latency_s": round(g.latency_seconds, 3) if g else None,
            "language": ("prompt" if g and g.has_prompt else "auto"),
        }
