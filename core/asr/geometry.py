"""How a Nemotron-family export wants its audio cut, read from metadata.json.

Three NVIDIA models ship in this layout (FluidInference's OpenVINO exports,
all driven by eddy-audio's `eddy::nemotron` backend, which is where every
field name and default below was read from):

  * parakeet-unified-en-0.6b (`streaming/fp16`): a STATELESS encoder run over
    a sliding [left | chunk | right] window of audio. Only the chunk's
    encoder frames are decoded; the window overlaps, nothing else carries.
    metadata: "streaming": true, "context_encoder_frames": {left, chunk,
    right}, "window_mel_frames", "subsampling_factor", "feature_normalize".
  * Nemotron-3.5-ASR-Streaming-Multilingual-0.6B and its English sibling
    nemotron-speech-streaming-en-0.6b: a CACHE-AWARE encoder fed fixed chunks
    of chunk_mel_frames, prefixed with the previous chunk's last
    pre_encode_cache mel frames, carrying cache_channel / cache_time /
    cache_len across chunks. metadata: "chunk_mel_frames",
    "pre_encode_cache", "total_mel_frames", "cache_channel_shape",
    "cache_time_shape", and for the multilingual one "prompt_dictionary",
    "default_prompt_id", "lang_tag_token_ids".

Both modes are causal at a fixed granularity, which is the whole point:
audio can be fed as it arrives and a unit decoded as soon as its last sample
(plus the right context, in window mode) is here. `ready()` and `unit()`
are the only places that know the arithmetic, and offline transcription uses
them too, so streaming and offline cannot disagree.

Which mode is decided the way eddy decides it: window when the metadata says
so, cache-aware when the encoder has a `cache_channel` input. An encoder that
is neither (the dynamic `fp16/` offline variant of parakeet-unified) is
refused by name rather than driven by a guess.
"""

import json

import numpy as np

from core.asr.mel import HOP

SR = 16000


class Geometry:
    def __init__(self, meta, has_cache, has_prompt, where="metadata.json"):
        g = lambda k, d: meta.get(k, d)    # noqa: E731  (eddy's m.value(k, d))
        self.sample_rate = int(g("sample_rate", SR))
        if self.sample_rate != SR:
            raise RuntimeError(f"{where} says sample_rate {self.sample_rate}; "
                               "this engine feeds 16 kHz audio only.")
        self.n_mels = int(g("mel_features", 128))
        self.blank_id = int(g("blank_idx", 13087))
        self.vocab_size = int(g("vocab_size", self.blank_id))
        self.decoder_hidden = int(g("decoder_hidden", 640))
        self.decoder_layers = int(g("decoder_layers", 2))
        self.normalize = bool(g("feature_normalize", False))
        self.prompts = {str(k): int(v) for k, v in
                        (g("prompt_dictionary", {}) or {}).items()}
        self.default_prompt = int(g("default_prompt_id", 101))
        self.lang_tags = {int(t) for t in (g("lang_tag_token_ids", []) or [])}
        self.has_prompt = has_prompt
        ctx = g("context_encoder_frames", None)
        if g("streaming", False) and isinstance(ctx, dict):
            self.mode = "window"
            sub = int(g("subsampling_factor", 8))
            self.left = int(ctx.get("left", 0))
            self.chunk = int(ctx.get("chunk", 0))
            self.right = int(ctx.get("right", 0))
            self.window_mel = int(g("window_mel_frames", 0))
            if self.chunk <= 0 or self.window_mel <= 0:
                raise RuntimeError(f"{where} declares window streaming but its "
                                   "context_encoder_frames.chunk or "
                                   "window_mel_frames is missing or zero.")
            efs = sub * HOP                    # samples per encoder frame
            self.step = self.chunk * efs       # samples decoded per unit
            self.left_s = self.left * efs
            self.right_s = self.right * efs
            self.win_s = (self.left + self.chunk + self.right) * efs
            if 1 + self.win_s // HOP != self.window_mel:
                raise RuntimeError(
                    f"{where}: a {self.win_s}-sample window makes "
                    f"{1 + self.win_s // HOP} mel frames, but window_mel_frames "
                    f"is {self.window_mel}; the 25 ms / 10 ms framing does not fit.")
        elif has_cache:
            self.mode = "cache"
            self.chunk_mel = int(g("chunk_mel_frames", 112))
            self.pre_cache = int(g("pre_encode_cache", 9))
            self.total_mel = int(g("total_mel_frames", 121))
            self.step = self.chunk_mel * self.sample_rate // 100
            self.cache_channel = [int(d) for d in g("cache_channel_shape", []) or []]
            self.cache_time = [int(d) for d in g("cache_time_shape", []) or []]
            if not self.cache_channel or not self.cache_time:
                raise RuntimeError(f"The encoder carries caches but {where} has no "
                                   "cache_channel_shape / cache_time_shape.")
        else:
            raise RuntimeError(
                f"This export's encoder has no cache inputs and {where} does not "
                "declare window streaming, so it is the offline variant. Point "
                "--asr-dir at the streaming build instead (for parakeet-unified, "
                "the streaming/fp16 folder of FluidInference/parakeet-unified-en-0.6b-ov).")

    # --- the arithmetic both offline and streaming use ---------------------

    def ready(self, pos, n, final):
        """Can the unit starting at sample `pos` run with `n` samples in hand?"""
        if pos >= n:
            return False
        if final:
            return True
        lookahead = self.right_s if self.mode == "window" else 0
        return pos + self.step + lookahead <= n

    def unit(self, audio, pos, n):
        """(samples, valid_samples) the featurizer sees for the unit at `pos`.

        Window: [pos - left, pos + chunk + right), zero outside the audio, and
        valid = the real samples inside it (eddy: real_hi - real_lo). Cache:
        [pos, pos + chunk), zero-padded, and valid = the FULL padded length on
        every chunk, which is what eddy's WER-validated reference passes.
        """
        if self.mode == "window":
            start = pos - self.left_s
            out = np.zeros(self.win_s, dtype=np.float32)
            lo, hi = max(0, start), min(n, start + self.win_s)
            if hi > lo:
                out[lo - start:hi - start] = audio[lo:hi]
            return out, max(0, hi - lo)
        out = np.zeros(self.step, dtype=np.float32)
        hi = min(n, pos + self.step)
        out[:hi - pos] = audio[pos:hi]
        return out, self.step

    @property
    def unit_seconds(self):
        return self.step / SR

    @property
    def latency_seconds(self):
        """Audio that must exist past a unit before it can be decoded."""
        return (self.step + (self.right_s if self.mode == "window" else 0)) / SR

    def prompt_id(self, language):
        """eddy's resolve_prompt_id: exact key, then a two-letter code matching
        the first "xx-*" key, else the default ("auto") prompt."""
        if not language:
            return self.default_prompt
        if language in self.prompts:
            return self.prompts[language]
        if len(language) == 2:
            pre = language.lower() + "-"
            for k in sorted(self.prompts):
                if k.lower().startswith(pre):
                    return self.prompts[k]
        return self.default_prompt


def read_metadata(path):
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"Could not read {path}: {e}")
    if not isinstance(meta, dict):
        raise RuntimeError(f"{path} is not a JSON object.")
    return meta
