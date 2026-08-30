"""Whose voice is it? A speaker embedding, and the profile it compares to.

Silero answers 'is this speech', never 'is this MY user's speech' -- and only
the second question can end a turn correctly in a room with other people in
it. Three verdicts, not two: abstain is what makes this safe to ship, because
a system that cannot tell must let the turn through rather than swallow the
user's speech."""

import gc
import json
import numpy as np
import openvino as ov
import os
import time
from pathlib import Path
from core import config
from core.slots.locks import _device_lock


class SpeakerSlot:
    """Holds a speaker-embedding model, and the enrolled profile it compares to.

    Why this exists: Silero answers "is this speech?", and nothing in the voice
    path ever answered "is this MY user's speech?". Those are different
    questions, and only the second one can end a turn correctly in a room with
    other people in it. Without this, someone else talking opens a turn, gets
    transcribed, and is sent as the user's own message; and 400 ms of anyone's
    voice cuts the assistant off mid-sentence. Both were reported as bugs in
    the VAD. The VAD was right; it was being asked the wrong question.

    Structurally a sibling of VadSlot: a small model on the plain OpenVINO
    runtime, not a genai pipeline, with input names and shapes discovered from
    the compiled model rather than hardcoded, because the published speaker
    models disagree about all of them.

    The profile is a running mean of L2-normalised embeddings, which is the
    standard way to combine enrolment utterances and is why enrolling twice in
    different moods beats enrolling once carefully. It is stamped with the
    model that produced it: embeddings from two different models are not
    comparable, and silently scoring against a stale profile would reject the
    real user with no visible cause.
    """

    SR = 16000
    MIN_ENROLL_S = 6.0     # total audio before a profile is usable
    MIN_VERIFY_S = 0.55    # below this, abstain rather than guess
    PROFILE_VERSION = 1

    # Kaldi fbank geometry, matching torchaudio.compliance.kaldi.fbank's
    # defaults — which is what WeSpeaker, SpeechBrain and 3D-Speaker all train
    # against. These are not free parameters: features that disagree with the
    # ones a model was trained on still produce confident-looking embeddings,
    # they just stop discriminating, which is the worst possible failure here.
    FRAME_LEN = 400        # 25 ms
    FRAME_SHIFT = 160      # 10 ms
    NFFT = 512             # next power of two above FRAME_LEN
    PREEMPH = 0.97
    LOW_FREQ = 20.0

    def __init__(self, device_name, device_id=None, profile_path=None,
                 threshold=0.35):
        self.device_name = device_name
        self.device_id = device_id or device_name
        self.device_full = ""
        self.compiled = None
        self.model_name = ""
        self.model_type = "speaker"
        self.status = "not_configured"
        self.lock = _device_lock(self.device_name)
        self.last_used = time.time()
        self.model_dir = None
        self.threshold = threshold
        self.profile_path = profile_path or os.path.join(config.SCRIPT_DIR,
                                                         "speaker-profile.json")
        self._audio_in = None
        self._len_in = None
        self._needs_fbank = False  # rank-3 input: the model wants features
        self._n_mels = 80
        self._window = None        # cached Povey window
        self._melbank = None       # cached filterbank matrix
        self.profile = None        # np.ndarray, L2-normalised
        self.profile_meta = {}     # {"model", "seconds", "utterances", "version"}
        self.last_score = None     # most recent cosine, for /health and tuning
        self._load_profile()

    # -- model ---------------------------------------------------------------

    @staticmethod
    def _model_file(model_dir):
        p = Path(model_dir)
        if p.is_file():
            return p
        for pattern in ("*.xml", "*.onnx"):
            hits = sorted(p.glob(pattern))
            if hits:
                return hits[0]
        raise RuntimeError(
            f"No .onnx or .xml model found in {p}. Point --speaker-dir at a "
            f"speaker-embedding model (e.g. ECAPA-TDNN or WeSpeaker) or its folder.")

    def load(self, model_dir):
        self.status = "loading"
        self.last_used = time.time()
        self.model_dir = model_dir
        path = self._model_file(model_dir)
        self.model_name = path.stem
        print(f"  [{self.device_name}] Loading speaker ID ({self.model_name})...",
              flush=True)
        core = ov.Core()
        if config.MODEL_CACHE_DIR:
            core.set_property({"CACHE_DIR": config.MODEL_CACHE_DIR})
        model = core.read_model(str(path))
        self.compiled = core.compile_model(model, self.device_id)

        # Two input conventions exist and both are common, so both are handled
        # rather than one being declared unsupported: rank 2 is a raw waveform
        # [batch, samples], rank 3 is precomputed features [batch, frames, mels]
        # — which is what the ONNX exports people can actually download
        # (WeSpeaker, SpeechBrain, 3D-Speaker) all take. Rejecting rank 3 would
        # have meant a slot that loads almost nothing real.
        self._audio_in, self._len_in, self._needs_fbank = None, None, False
        for port in self.compiled.inputs:
            rank = len(port.partial_shape)
            if rank == 3 and self._audio_in is None:
                self._audio_in = port
                self._needs_fbank = True
                dim = port.partial_shape[2]
                if dim.is_static:
                    self._n_mels = int(dim.get_length())
            elif rank == 2 and self._audio_in is None:
                self._audio_in = port
            elif rank <= 1:
                self._len_in = port        # some exports want a lengths input
        if self._audio_in is None:
            raise RuntimeError(
                f"{self.model_name} has no waveform or feature input; inputs "
                f"are {[str(p.partial_shape) for p in self.compiled.inputs]}")
        if self._needs_fbank:
            print(f"  [{self.device_name}] {self.model_name} takes "
                  f"{self._n_mels}-bin fbank features; computing them here.",
                  flush=True)
        self.status = "warming_up"

    def warmup(self):
        try:
            # Noise, not silence. Digital silence is the one input guaranteed
            # to produce the degenerate embedding embed() refuses, so warming
            # up on zeros made warmup fail for every model — turning the whole
            # feature off at startup with an error that looked like a bad
            # model file. Found by the first run of the test suite.
            rng = np.random.default_rng(0)
            self.embed(rng.standard_normal(int(self.SR * 1.0)).astype(np.float32) * 0.05)
            self.status = "ready"
            state = ("no profile enrolled" if self.profile is None
                     else f"profile: {self.profile_meta.get('seconds', 0):.0f}s")
            print(f"  [{self.device_name}] Speaker ID ready ({state})", flush=True)
        except Exception as e:
            self.status = "error"
            print(f"  [{self.device_name}] Speaker ID failed: {e}", flush=True)

    def unload(self):
        if self.compiled is None:
            return
        print(f"  [{self.device_name}] Idle — unloading {self.model_name}", flush=True)
        self.compiled = None
        self.status = "idle_unloaded"
        import gc
        gc.collect()

    def ensure_loaded(self):
        if self.compiled is not None and self.status == "ready":
            return
        with self.lock:
            if self.compiled is not None and self.status == "ready":
                return
            if self.model_dir is None:
                raise RuntimeError("Speaker slot has no model_dir")
            self.load(self.model_dir)
            self.warmup()

    # -- embedding -----------------------------------------------------------

    @staticmethod
    def _mel_points(n_bins, sr, nfft, low_freq):
        """Kaldi triangular mel filterbank, as a [nfft//2+1, n_bins] matrix."""
        def to_mel(f):
            return 1127.0 * np.log(1.0 + f / 700.0)

        def from_mel(m):
            return 700.0 * (np.exp(m / 1127.0) - 1.0)

        high_freq = sr / 2.0
        edges = from_mel(np.linspace(to_mel(low_freq), to_mel(high_freq),
                                     n_bins + 2))
        bins = np.linspace(0, sr / 2.0, nfft // 2 + 1)
        fb = np.zeros((bins.size, n_bins), dtype=np.float32)
        for i in range(n_bins):
            left, centre, right = edges[i], edges[i + 1], edges[i + 2]
            rise = (bins - left) / max(centre - left, 1e-10)
            fall = (right - bins) / max(right - centre, 1e-10)
            fb[:, i] = np.maximum(0.0, np.minimum(rise, fall))
        return fb

    def _fbank(self, samples):
        """Log-mel filterbank features, [frames, n_mels], Kaldi conventions.

        Written out rather than pulled from torchaudio because this project
        does not depend on torch and is not going to start for 40 lines of
        arithmetic. The conventions that matter, all of which change the
        output enough to matter: per-frame DC removal, then pre-emphasis, a
        Povey window (hann^0.85), a power spectrum, and a log with a floor.
        Cepstral mean normalisation at the end is what makes the embedding
        robust to the microphone rather than to the speaker.
        """
        n = self._n_mels
        if samples.size < self.FRAME_LEN:
            samples = np.pad(samples, (0, self.FRAME_LEN - samples.size))
        # snip_edges=True: only whole frames, no padding at the tail.
        count = 1 + (samples.size - self.FRAME_LEN) // self.FRAME_SHIFT
        idx = (np.arange(self.FRAME_LEN)[None, :]
               + self.FRAME_SHIFT * np.arange(count)[:, None])
        frames = samples[idx].astype(np.float32)
        frames = frames - frames.mean(axis=1, keepdims=True)   # remove_dc_offset
        pre = np.empty_like(frames)
        pre[:, 0] = frames[:, 0] - self.PREEMPH * frames[:, 0]
        pre[:, 1:] = frames[:, 1:] - self.PREEMPH * frames[:, :-1]
        if self._window is None:
            hann = 0.5 - 0.5 * np.cos(2 * np.pi
                                      * np.arange(self.FRAME_LEN) / self.FRAME_LEN)
            self._window = np.power(hann, 0.85).astype(np.float32)   # Povey
        spec = np.abs(np.fft.rfft(pre * self._window, n=self.NFFT)) ** 2
        if self._melbank is None or self._melbank.shape[1] != n:
            self._melbank = self._mel_points(n, self.SR, self.NFFT, self.LOW_FREQ)
        mel = spec.astype(np.float32) @ self._melbank
        feats = np.log(np.maximum(mel, 1.1920929e-7))            # Kaldi's floor
        return (feats - feats.mean(axis=0, keepdims=True)).astype(np.float32)  # CMN

    def embed(self, samples):
        """L2-normalised speaker embedding for a mono 16 kHz float32 waveform."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        payload = (self._fbank(samples)[None, :, :] if self._needs_fbank
                   else samples.reshape(1, -1))
        with self.lock:
            req = self.compiled.create_infer_request()
            feed = {self._audio_in: payload}
            if self._len_in is not None:
                feed[self._len_in] = np.array([samples.size], dtype=np.int64)
            req.infer(feed)
            vec = np.array(req.get_tensor(self.compiled.outputs[0]).data,
                           dtype=np.float32).reshape(-1)
            self.last_used = time.time()
        norm = float(np.linalg.norm(vec))
        # A zero vector cannot be normalised, and silently returning zeros would
        # score 0.0 against everything — indistinguishable from a stranger.
        if not np.isfinite(norm) or norm < 1e-8:
            raise RuntimeError("speaker model returned a degenerate embedding")
        return vec / norm

    # -- profile -------------------------------------------------------------

    def _load_profile(self):
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.profile = np.array(data["embedding"], dtype=np.float32)
            self.profile_meta = {k: data.get(k) for k in
                                 ("model", "seconds", "utterances", "version")}
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  WARNING: speaker profile at {self.profile_path} "
                  f"is unreadable ({e}); enrol again.", flush=True)

    def _save_profile(self):
        tmp = self.profile_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": self.PROFILE_VERSION,
                       "model": self.model_name,
                       "seconds": self.profile_meta.get("seconds", 0.0),
                       "utterances": self.profile_meta.get("utterances", 0),
                       "embedding": [float(x) for x in self.profile]}, f)
        os.replace(tmp, self.profile_path)   # atomic: never a half-written profile

    @property
    def profile_ready(self):
        """Enrolled, with enough audio, by THIS model."""
        if self.profile is None:
            return False
        if self.profile_meta.get("model") not in (None, self.model_name):
            return False
        return (self.profile_meta.get("seconds") or 0) >= self.MIN_ENROLL_S

    @property
    def profile_stale(self):
        return (self.profile is not None
                and self.profile_meta.get("model") not in (None, self.model_name))

    def enroll(self, samples):
        """Fold one utterance into the profile. Returns the new profile state."""
        seconds = len(samples) / self.SR
        if seconds < 1.0:
            raise ValueError(f"need at least 1 second of speech, got {seconds:.1f}s")
        vec = self.embed(samples)
        if self.profile is None or self.profile_stale:
            # A stale profile is replaced, not averaged into: mixing embeddings
            # from two models produces a vector that matches neither voice.
            total, count, acc = 0.0, 0, vec
        else:
            total = self.profile_meta.get("seconds") or 0.0
            count = self.profile_meta.get("utterances") or 0
            # Weight by duration: a 10 s enrolment says more than a 1 s one.
            acc = self.profile * total + vec * seconds
            acc = acc / max(float(np.linalg.norm(acc)), 1e-8)
        self.profile = acc
        self.profile_meta = {"model": self.model_name,
                             "seconds": total + seconds,
                             "utterances": count + 1,
                             "version": self.PROFILE_VERSION}
        self._save_profile()
        return self.profile_state

    def clear_profile(self):
        self.profile = None
        self.profile_meta = {}
        try:
            os.remove(self.profile_path)
        except FileNotFoundError:
            pass

    @property
    def profile_state(self):
        return {
            "enrolled": self.profile is not None,
            "ready": self.profile_ready,
            "stale": self.profile_stale,
            "seconds": round(self.profile_meta.get("seconds") or 0.0, 1),
            "utterances": self.profile_meta.get("utterances") or 0,
            "needed_seconds": self.MIN_ENROLL_S,
            "model": self.profile_meta.get("model"),
            "threshold": self.threshold,
        }

    # -- verification --------------------------------------------------------

    def verify(self, samples):
        """(verdict, score) for a candidate utterance.

        Three outcomes, not two. "abstain" is what makes this safe to ship:
        too little audio, no profile, or a model that failed to load all mean
        "cannot tell", and a system that cannot tell must let the turn through
        rather than silently swallow the user's speech. A voice assistant that
        ignores its owner is a worse failure than one that occasionally listens
        to the television.
        """
        if not self.profile_ready or self.status != "ready":
            return "abstain", None
        n = len(samples)
        if n < self.SR * self.MIN_VERIFY_S:
            return "abstain", None
        try:
            score = float(np.dot(self.embed(samples), self.profile))
        except Exception as e:
            print(f"  [{self.device_name}] speaker verify failed: {e}", flush=True)
            return "abstain", None
        self.last_score = score
        return ("match" if score >= self.threshold else "reject"), score

    @property
    def info(self):
        return {
            "status": self.status,
            "model": self.model_name,
            "type": self.model_type,
            "device_name": self.device_name,
            "device": self.device_full,
            "profile": self.profile_state,
            "last_score": (round(self.last_score, 3)
                           if self.last_score is not None else None),
        }
