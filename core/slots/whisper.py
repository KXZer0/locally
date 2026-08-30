"""Speech to text."""

import gc
import openvino_genai as ovg
import time
from core.models.identity import model_display_name
from core.slots.locks import _device_lock


class WhisperSlot:
    """Holds a WhisperPipeline for speech-to-text."""

    def __init__(self, device_name, device_id=None):
        self.device_name = device_name
        self.device_id = device_id or device_name
        self.device_full = ""
        self.pipe = None
        self.model_name = ""
        self.model_type = "stt"
        self.status = "not_configured"
        self.lock = _device_lock(self.device_name)
        # The idle watchdog walks every slot and reads last_used / calls
        # unload(); without these an audio slot crashed the watchdog thread on
        # its first pass, which silently disabled idle-unload for ALL slots.
        self.last_used = time.time()
        self.model_dir = None

    def load(self, model_dir):
        self.status = "loading"
        # Same reason as DeviceSlot.load: a fresh load must reset the idle
        # clock, or the watchdog unloads it on its next pass.
        self.last_used = time.time()
        self.model_dir = model_dir
        self.model_name = model_display_name(model_dir)
        print(f"  [{self.device_name}] Loading Whisper ({self.model_name})...",
              flush=True)
        WhisperPipe = getattr(ovg, "WhisperPipeline", None)
        if WhisperPipe is None:
            raise RuntimeError(
                "No WhisperPipeline in this openvino_genai build. "
                "Upgrade to openvino-genai >= 2025.1."
            )
        self.pipe = WhisperPipe(str(model_dir), self.device_id)

    def warmup(self):
        self.status = "ready"
        print(f"  [{self.device_name}] Whisper ready", flush=True)

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

    def transcribe(self, audio_samples, language=None):
        """Transcribe float32 audio at 16 kHz. Returns text."""
        kwargs = {}
        if language:
            kwargs["language"] = f"<|{language}|>"
            kwargs["task"] = "transcribe"
        with self.lock:
            result = self.pipe.generate(audio_samples, **kwargs)
            self.last_used = time.time()
        if hasattr(result, "texts") and result.texts:
            return result.texts[0].strip()
        return str(result).strip()

    @property
    def info(self):
        return {
            "status": self.status,
            "model": self.model_name,
            "type": self.model_type,
            # device_name is the canonical "NPU"/"GPU"/"CPU" the UI colour-codes
            # by; device_full is the marketing name. The generative slots report
            # both, so this one does too.
            "device_name": self.device_name,
            "device": self.device_full,
        }
