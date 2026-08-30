"""Is anyone speaking? Silero, on the plain OpenVINO runtime.

Turn-taking used to be a loudness gate in the browser, which cannot work: a
level says how loud the room is, not whether anyone is talking, so any steady
noise held the gate open permanently. Measured separation: real speech 0.801
mean probability against 0.007 for loud white noise."""

import gc
import numpy as np
import openvino as ov
import time
from pathlib import Path
from core import config
from core.slots.locks import _device_lock


class VadSlot:
    """Holds a Silero voice-activity model.

    Turn-taking used to be a loudness gate in the browser, which cannot work:
    a level tells you how loud the room is, not whether anyone is talking, so
    any steady background noise held the gate open permanently. This is a small
    recurrent model (~1M params) that answers the question directly.

    It is not a genai pipeline — there is no VAD pipeline in openvino_genai — so
    it drives the runtime API directly. OpenVINO's ONNX frontend reads
    silero_vad.onnx as-is, so no conversion step stands between downloading the
    model and using it.

    Input/output names and widths differ across the published variants, so they
    are discovered from the compiled model rather than hardcoded. Two real
    differences this has to absorb, both found by trying it:

      * `silero_vad.onnx` (the stock v5) carries an If node that switches on
        sample rate, and OpenVINO's ONNX frontend cannot convert it ("the input
        data tensor's rank has to be known"). `silero_vad_openvino_16k.onnx`
        from the same repo is the 16 kHz-only build without that branch.
      * That build takes 576 samples, not 512: v5 prepends 64 samples of
        context from the previous frame. The context is ours to carry, so the
        model's audio width tells us how much of it to keep.
    """

    FRAME = 512          # new samples per inference at 16 kHz
    SR = 16000
    _CTX_KEY = "__context"   # our carried context, not one of the model's inputs

    def __init__(self, device_name, device_id=None):
        self.device_name = device_name
        self.device_id = device_id or device_name
        self.device_full = ""
        self.compiled = None
        self.model_name = ""
        self.model_type = "vad"
        self.status = "not_configured"
        self.lock = _device_lock(self.device_name)
        self.last_used = time.time()
        self.model_dir = None
        self._audio_in = None
        self._sr_in = None
        self._state_ins = []

    @staticmethod
    def _model_file(model_dir):
        """Accept either the .onnx/.xml itself or a directory holding one."""
        p = Path(model_dir)
        if p.is_file():
            return p
        for pattern in ("*.xml", "*openvino*.onnx", "*.onnx"):
            # The openvino-specific build is preferred where both are present:
            # the stock v5 ONNX fails to convert, and picking it alphabetically
            # would turn a working folder into a startup error.
            hits = sorted(p.glob(pattern))
            if hits:
                return hits[0]
        raise RuntimeError(
            f"No .onnx or .xml model found in {p}. Point --vad-dir at the "
            f"Silero VAD model (silero_vad_openvino_16k.onnx) or its folder."
        )

    def load(self, model_dir):
        self.status = "loading"
        self.last_used = time.time()
        self.model_dir = model_dir
        path = self._model_file(model_dir)
        self.model_name = path.stem
        print(f"  [{self.device_name}] Loading VAD ({self.model_name})...", flush=True)
        core = ov.Core()
        model = core.read_model(str(path))
        ov_config = ({"CACHE_DIR": config.MODEL_CACHE_DIR}
                     if config.MODEL_CACHE_DIR else {})
        self.compiled = core.compile_model(model, self.device_id, ov_config)

        # Sort the inputs into audio / sample-rate / recurrent state.
        self._audio_in = None
        self._sr_in = None
        self._state_ins = []
        for port in self.compiled.inputs:
            name = port.get_any_name()
            shape = list(port.get_partial_shape())
            if name == "sr":
                self._sr_in = name
            elif self._audio_in is None and len(shape) <= 2:
                self._audio_in = name
                last = shape[-1]
                self._window = last.get_length() if last.is_static else self.FRAME
            else:
                self._state_ins.append((name, port))
        if self._audio_in is None:
            raise RuntimeError(
                f"Could not identify the audio input of {path.name} "
                f"(inputs: {[p.get_any_name() for p in self.compiled.inputs]})."
            )
        # Anything the window wants beyond a fresh frame is context we carry.
        self._context = max(0, self._window - self.FRAME)

    def warmup(self):
        # A real inference, not a status flip: a model that cannot run should
        # fail at startup, not on the first word someone says. Deliberately the
        # UNLOCKED path — ensure_loaded() calls this while already holding the
        # device lock, and the lock is not reentrant, so going through
        # probability() here deadlocks the reload after an idle unload. It
        # cannot race: nothing can be served by a slot that is not ready yet.
        state = self.new_state()
        self._infer(np.zeros(self.FRAME, dtype=np.float32), state)
        self.status = "ready"
        print(f"  [{self.device_name}] VAD ready", flush=True)

    def new_state(self):
        """Fresh recurrent state — one per conversation, carried across frames."""
        out = {self._CTX_KEY: np.zeros(self._context, dtype=np.float32)}
        for name, port in self._state_ins:
            shape = [d.get_length() if d.is_static else 1
                     for d in port.get_partial_shape()]
            out[name] = np.zeros(shape, dtype=np.float32)
        return out

    def probability(self, frame, state):
        """Speech probability for one 512-sample frame. Mutates `state`."""
        with self.lock:
            prob = self._infer(frame, state)
            self.last_used = time.time()
        return prob

    def _infer(self, frame, state):
        """The inference itself. Caller holds the lock (or knows it is alone)."""
        req = self.compiled.create_infer_request()
        frame = frame.astype(np.float32)
        ctx = state.get(self._CTX_KEY)
        window = np.concatenate([ctx, frame]) if self._context else frame
        feed = {self._audio_in: window.reshape(1, -1)}
        if self._sr_in is not None:
            feed[self._sr_in] = np.array(self.SR, dtype=np.int64)
        feed.update({k: v for k, v in state.items() if k != self._CTX_KEY})
        req.infer(feed)
        if self._context:
            state[self._CTX_KEY] = frame[-self._context:].copy()
        outputs = list(self.compiled.outputs)
        prob = float(np.array(req.get_tensor(outputs[0]).data).flatten()[0])
        # Every output after the first is the next recurrent state, in the same
        # order as the state inputs.
        for (name, _), port in zip(self._state_ins, outputs[1:]):
            state[name] = np.array(req.get_tensor(port).data, dtype=np.float32)
        return prob

    def unload(self):
        """Release the compiled model. Caller must hold self.lock."""
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
                raise RuntimeError(f"Slot {self.device_name} has no model_dir")
            print(f"  [{self.device_name}] Reloading {self.model_name}...", flush=True)
            self.load(self.model_dir)
            self.warmup()

    @property
    def info(self):
        return {
            "status": self.status,
            "model": self.model_name,
            "type": self.model_type,
            "device_name": self.device_name,
            "device": self.device_full,
            "frame_ms": round(self.FRAME / self.SR * 1000),
        }
