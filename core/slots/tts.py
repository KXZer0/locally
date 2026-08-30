"""Text to speech. Backend-agnostic: genai drives SpeechT5 and Kokoro through
the same class, so the model directory decides. There is no streaming --
generate() returns the whole utterance -- so every caller must be written to
wait for a complete waveform."""

import gc
import numpy as np
import openvino as ov
import openvino_genai as ovg
import time
from pathlib import Path
from core.models.identity import model_display_name
from core.slots.locks import _device_lock


class TtsSlot:
    """Holds a Text2SpeechPipeline for text-to-speech.

    Backend-agnostic on purpose: openvino-genai 2026.3 drives both SpeechT5
    and Kokoro through this one class, and the model directory decides which.
    There is no streaming — generate() returns the whole utterance — so every
    caller must be written to wait for a complete waveform.
    """

    def __init__(self, device_name, device_id=None):
        self.device_name = device_name
        self.device_id = device_id or device_name
        self.device_full = ""
        self.pipe = None
        self.model_name = ""
        self.model_type = "tts"
        self.status = "not_configured"
        self.lock = _device_lock(self.device_name)
        self.last_used = time.time()
        self.model_dir = None
        self.last_synth_ms = None
        self.default_voice = None     # name of the voice used when none is asked for
        self._emb_cache = {}          # voice name -> ov.Tensor

    def voices(self):
        """Available voice names, from a `voices/*.bin` dir if the model ships one.

        Kokoro does (54 of them); SpeechT5 does not — it falls back to a
        built-in default embedding, so an empty list here is not an error.
        """
        if not self.model_dir:
            return []
        vdir = Path(self.model_dir) / "voices"
        if not vdir.is_dir():
            return []
        return sorted(p.stem for p in vdir.glob("*.bin"))

    def _speaker_embedding(self, voice):
        """Load a voice .bin as a tensor of the shape this backend expects.

        SpeechT5 wants {1,512}, Kokoro {510,1,256} — ask the pipeline rather
        than hardcoding either, so one code path serves both.
        """
        if voice in self._emb_cache:
            return self._emb_cache[voice]
        path = Path(self.model_dir) / "voices" / f"{voice}.bin"
        if not path.is_file():
            raise RuntimeError(f"Unknown voice '{voice}'. Available: "
                               f"{', '.join(self.voices()) or '(none)'}")
        shape = list(self.pipe.get_speaker_embedding_shape())
        expected = int(np.prod(shape))
        data = np.fromfile(str(path), dtype=np.float32)
        if data.size != expected:
            raise RuntimeError(
                f"Voice '{voice}' has {data.size} floats but this model expects "
                f"{expected} (shape {shape}) — voice file and model don't match."
            )
        tensor = ov.Tensor(data.reshape(shape))
        self._emb_cache[voice] = tensor
        return tensor

    def load(self, model_dir):
        self.status = "loading"
        # Same reason as DeviceSlot.load: a fresh load must reset the idle
        # clock, or the watchdog unloads it on its next pass.
        self.last_used = time.time()
        self.model_dir = model_dir
        self.model_name = model_display_name(model_dir)
        print(f"  [{self.device_name}] Loading TTS ({self.model_name})...", flush=True)
        TtsPipe = getattr(ovg, "Text2SpeechPipeline", None)
        if TtsPipe is None:
            raise RuntimeError(
                "No Text2SpeechPipeline in this openvino_genai build. "
                "Upgrade to openvino-genai >= 2025.3."
            )
        self.pipe = TtsPipe(str(model_dir), self.device_id)
        self._emb_cache = {}

        # Kokoro *requires* the caller to supply a speaker embedding; SpeechT5
        # has a built-in fallback. Pick a sensible default so the endpoint
        # works out of the box either way.
        names = self.voices()
        if names:
            preferred = ("af_heart", "af_bella", "am_michael")
            self.default_voice = next((v for v in preferred if v in names), names[0])
            print(f"  [{self.device_name}] {len(names)} voices, "
                  f"default '{self.default_voice}'", flush=True)

    def warmup(self):
        self.status = "ready"
        print(f"  [{self.device_name}] TTS ready", flush=True)

    def unload(self):
        """Release the loaded pipeline. Caller must hold self.lock."""
        if self.pipe is None:
            return
        print(f"  [{self.device_name}] Idle — unloading {self.model_name}", flush=True)
        self.pipe = None
        self.status = "idle_unloaded"
        import gc
        gc.collect()

    def ensure_loaded(self):
        """Reload pipeline if it was unloaded. Blocks until ready."""
        if self.pipe is not None and self.status == "ready":
            return
        with self.lock:
            if self.pipe is not None and self.status == "ready":
                return
            if self.model_dir is None:
                raise RuntimeError(f"Slot {self.device_name} has no model_dir")
            print(f"  [{self.device_name}] Reloading {self.model_name}...", flush=True)
            self.load(self.model_dir)
            self.warmup()

    def synthesize(self, text, voice=None):
        """Synthesize text. Returns (float32 samples, sample_rate).

        With no voice file on disk the embedding is left to the pipeline —
        SpeechT5 falls back to the cmu-arctic-xvectors 7306-th vector, so
        nothing has to ship with locally for the default voice to work.
        """
        name = voice or self.default_voice
        emb = self._speaker_embedding(name) if name else None

        t0 = time.perf_counter()
        with self.lock:
            result = (self.pipe.generate(text, emb) if emb is not None
                      else self.pipe.generate(text))
            self.last_used = time.time()
        self.last_synth_ms = int((time.perf_counter() - t0) * 1000)

        speeches = getattr(result, "speeches", None)
        if not speeches:
            raise RuntimeError("TTS returned no audio")
        audio = np.asarray(speeches[0].data, dtype=np.float32).reshape(-1)
        rate = int(getattr(result, "output_sample_rate", 16000))
        return audio, rate

    @property
    def info(self):
        return {
            "status": self.status,
            "model": self.model_name,
            "type": self.model_type,
            "device_name": self.device_name,
            "device": self.device_full,
            "last_synth_ms": self.last_synth_ms,
            "voice": self.default_voice,
            "voices": self.voices(),
        }
