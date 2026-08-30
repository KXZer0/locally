"""The utility models: OCR, matting, upscale, detect, layout, embed, rerank."""

import gc
import time
from core import config
from core.slots.locks import _device_lock

try:
    # Imported for the disabled-task reasons only -- the engines themselves stay
    # lazy (UtilSlot.load), so a missing utility_pipeline keeps the server usable
    # for chat instead of failing at import.
    from utility_pipeline import _DISABLED_TASKS as _UTIL_DISABLED_TASKS
except ImportError:
    _UTIL_DISABLED_TASKS = {}


class UtilSlot:
    """Holds the utility (OCR/vision) models for ONE engine.

    Follows the same informal slot protocol as WhisperSlot/TtsSlot — device_*,
    pipe, status, lock, last_used, model_dir, load/warmup/unload/ensure_loaded/
    info — so the idle watchdog and /v1/models/unload pick it up generically.
    A slot missing last_used or unload() kills the watchdog thread on its first
    pass and silently disables idle-unload for EVERY slot (fixed 2026-08-09 for
    WhisperSlot; don't reintroduce it here).

    Unlike every other slot in this file it holds no genai pipeline: the models
    are plain OpenVINO IRs driven through ov.Core().compile_model, which is the
    first non-genai inference path in locally. The actual work lives in
    ocr_pipeline.py because scripts/npu-probe.py — the gate that decides whether
    this may run at all — has to exercise the same code.

    One slot per engine ("NPU" or "GPU"), so the UI's engine toggle is slot
    selection rather than a reload. Non-OCR utility models compile lazily.
    """

    def __init__(self, device_name, device_id=None):
        self.device_name = device_name
        self.device_id = device_id or device_name
        self.device_full = ""
        self.pipe = None                  # the OcrEngine; named for the protocol
        self.utility = None               # lazy non-OCR UtilityEngine models
        self.model_name = "Local utilities"
        self.model_type = "util"
        self.status = "not_configured"
        self.lock = _device_lock(self.device_name)
        self.last_used = time.time()
        self.model_dir = None
        self.last_ms = None
        self.tasks = {}

    def load(self, model_dir):
        self.status = "loading"
        self.last_used = time.time()
        self.model_dir = model_dir
        try:
            from ocr_pipeline import OcrEngine
            from utility_pipeline import UtilityEngine
        except ImportError as e:
            raise RuntimeError(f"utility pipeline not importable: {e}")
        print(f"  [{self.device_name}] Loading utilities ({self.model_name})...",
              flush=True)
        engine = OcrEngine(model_dir, device=self.device_id,
                           cache_dir=config.MODEL_CACHE_DIR)
        engine.load()
        self.pipe = engine
        # The remaining models compile on first use. Loading all of them at
        # startup adds tens of seconds and memory for utilities the user may
        # never open; availability is still reported immediately from disk.
        self.utility = UtilityEngine(model_dir, device=self.device_id,
                                     cache_dir=config.MODEL_CACHE_DIR)
        self.tasks = {"read": True, **self.utility.availability}

    def warmup(self):
        self.status = "ready"
        print(f"  [{self.device_name}] Utilities ready", flush=True)

    def unload(self):
        """Release the compiled models. Caller must hold self.lock."""
        if self.pipe is None and self.utility is None:
            return
        print(f"  [{self.device_name}] Idle — unloading utilities", flush=True)
        if self.pipe is not None:
            self.pipe.unload()
        if self.utility is not None:
            self.utility.unload()
        self.pipe = None
        self.utility = None
        self.status = "idle_unloaded"
        import gc
        gc.collect()

    def ensure_loaded(self):
        """Reload if unloaded. Blocks until ready."""
        if self.pipe is not None and self.status == "ready":
            return
        with self.lock:
            if self.pipe is not None and self.status == "ready":
                return
            if self.model_dir is None:
                raise RuntimeError(f"Util slot {self.device_name} has no model_dir")
            print(f"  [{self.device_name}] Reloading utilities...", flush=True)
            self.load(self.model_dir)
            self.warmup()

    def read(self, img):
        """OCR one PIL image. Returns the ocr_pipeline result dict."""
        with self.lock:
            t0 = time.perf_counter()
            result = self.pipe.read(img)
            self.last_ms = round((time.perf_counter() - t0) * 1000, 1)
            self.last_used = time.time()
        return result

    def run_utility(self, name, *args, **kwargs):
        """Run one lazy utility under the device-wide inference lock."""
        with self.lock:
            if self.utility is None:
                raise RuntimeError("Utility models are not loaded")
            result = getattr(self.utility, name)(*args, **kwargs)
            self.last_used = time.time()
        return result

    @property
    def info(self):
        # Preserve installed capabilities across idle-unload. The next request
        # can wake the slot, so hiding its buttons while unloaded would make
        # that reload path unreachable from the UI.
        tasks = dict(self.tasks)
        loaded = []
        # Start from the global disables, then let the engine add the ones that
        # are specific to *this* device — a task the GPU serves and the NPU
        # cannot is a different sentence from one nobody can serve.
        reasons = dict(_UTIL_DISABLED_TASKS)
        if self.utility is not None:
            tasks.update(self.utility.availability)
            reasons.update(self.utility.unavailable_reasons)
            loaded = self.utility.loaded
        return {
            "status": self.status,
            "model": self.model_name,
            "type": self.model_type,
            "device_name": self.device_name,
            "device": self.device_full,
            "last_ms": self.last_ms,
            "tasks": tasks,
            "loaded_tasks": loaded,
            # Why a task is off, when the answer isn't "download the model".
            # Without this the UI can only say "model not installed", which
            # sends people to fetch a model that would not help. Now covers
            # three causes: globally disabled, not installed, and not supported
            # on this engine — each needing a different action from the user.
            "disabled": reasons,
        }
