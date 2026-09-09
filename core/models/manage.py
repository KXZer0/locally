"""Load, swap, and unload runtime model slots."""

import gc
import os
import time
from datetime import datetime

from flask import jsonify, request

from core import runtime
from core.errors import _TurnError, openai_error
from core.genai.results import explain_genai_error
from core.hardware.memory import _memory_snapshot, _settle_memory
from core.models.discovery import _available_models, _choose_device
from core.models.gguf import unsupported_reason as gguf_unsupported_reason
from core.models.identity import _is_model_dir, model_display_name
from core.models.integrity import _dir_size_bytes, _verify_weights_integrity

def _is_transient_load_error(e):
    """Whether loading again could plausibly succeed.

    Only the tokenizers-extension race is known to pass on a second
    try; everything else is a property of the model or the machine and
    reproduces exactly.
    """
    msg = str(e).lower()
    return ("openvino_tokenizers" in msg
            or "failed to load shared object" in msg)


def _refuse(*args, **kwargs):
    """Build the _TurnError for a refusal, so callers can `raise _refuse(...)`.

    swap_model() is reached both by its endpoint and by the chat path loading
    a model on demand, and the two need the same refusals in different shapes:
    an HTTP response for one, an exception for the other. Building the response
    here keeps a single wording for both.
    """
    return _TurnError(openai_error(*args, **kwargs))


def load_model():
    """POST /v1/models/load — the HTTP face of swap_model()."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return openai_error("Request body must be a JSON object")
    if any(key in body and not isinstance(body[key], str) for key in ("model", "name", "device")):
        return openai_error("'model', 'name', and 'device' must be strings")
    name = (body.get("model") or body.get("name") or "").strip()
    if not name:
        return openai_error("'model' is required (name or directory path)")
    try:
        return jsonify(swap_model(name, body.get("device") or ""))
    except _TurnError as e:
        return e.response


def swap_model(name, device=""):
    """Swap the model in a slot without restarting the server.

    Ollama-style one-at-a-time: the slot's current model is unloaded before
    the new one is loaded, so peak memory is one model, not two. Synchronous
    — it returns when the model is ready to serve, which for a big IR can be
    a minute; that's honest about what's happening rather than reporting
    success on a model that can't answer yet.

    Raises _TurnError carrying an API error response rather than returning
    one, so the chat path can load on demand and surface exactly the refusals
    this endpoint gives instead of reimplementing them.
    """
    # Accept "name@DEVICE" like the rest of the API, an exact directory, or a
    # display name from /v1/models/available.
    device = (device or "").upper()
    if "@" in name:
        name, _, dev = name.partition("@")
        device = device or dev.upper()

    target = None
    if os.path.isdir(name) and _is_model_dir(name):
        target = os.path.abspath(name)
    else:
        for m in _available_models():
            if m["name"].lower() == name.lower() or \
                    os.path.basename(m["path"]).lower() == name.lower():
                target = m["path"]
                break
    if not target:
        known = ", ".join(m["name"] for m in _available_models()) or "(none found)"
        raise _refuse(f"Unknown model '{name}'. Available: {known}")

    # Already resident? Naming the model that is loaded -- a client re-picking
    # the current model, `/load` used as a keep-alive, a chat request routed
    # here on demand for a model a stale suffix hid -- must not pay the whole
    # 9-65 s reload only to land back where it started. An explicit device that
    # differs from where the model sits is a real "move it there" request and
    # is not short-circuited; a matching or absent device is a no-op.
    real_target = os.path.realpath(target)
    for s in (runtime.primary, runtime.secondary):
        if (s and s.status == "ready" and s.model_dir
                and os.path.realpath(s.model_dir) == real_target
                and (not device or device in ("AUTO", "ANY")
                     or s.device_name == device)):
            return {"status": "ok", "model": s.model_name,
                    "device": s.device_name, "type": s.model_type,
                    "placement": "already resident", "load_seconds": 0.0}

    # Pick the slot automatically. An explicit device is honoured when that
    # device can actually host the model, and otherwise treated as a
    # preference rather than an order — a request that would OOM or hit the
    # NPU's vision/quantization limits gets placed where it can run instead
    # of failing.
    if device in ("AUTO", "ANY"):
        device = ""
    if device and device not in ("NPU", "GPU", "CPU"):
        raise _refuse("Device must be NPU, GPU, CPU, or AUTO")
    if runtime.primary is None:
        raise _refuse("Server has not initialized its model slots", "server_error", 503)
    if runtime.primary.device_name == "REMOTE":
        raise _refuse("Local model swapping requires a local server, not --proxy-url")
    dev_name, dev_id, why = _choose_device(target, device or None)
    if dev_name is None:
        raise _refuse(f"No device can host '{name}' ({why}).")
    placement = why

    # Prefer a slot already on that device; otherwise move a slot there. With
    # one slot ("one model at a time") this is what lets the NPU be used at
    # all — the slot follows the model to the right silicon.
    slot = next((s for s in (runtime.primary, runtime.secondary)
                 if s and s.device_name == dev_name), None)
    moved_from = None
    if slot is None:
        slot = runtime.primary
        moved_from = slot.device_name

    if slot.status == "loading":
        raise _refuse("That slot is already loading a model.",
                            "server_error", 409)

    err = _verify_weights_integrity(target)
    if err:
        raise _refuse(err)

    # Refuse before touching the running model. The swap unloads first so peak
    # memory stays at one model, which means a load that was never going to
    # succeed costs the user the model they had: a GGUF whose architecture the
    # reader does not implement is refused in ~90 ms, and doing that AFTER the
    # unload turned a working NPU slot into an error state for a file the
    # server already knew it could not open. Anything knowable from the file
    # alone belongs on this side of the unload.
    refusal = gguf_unsupported_reason(target)
    if refusal:
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
              f"refused before unload: {refusal}", flush=True)
        raise _refuse(refusal, "invalid_request_error", 400)

    # Hold the lock across unload+load so a concurrent request can't reach a
    # half-swapped slot. A generation in flight keeps the lock, so we wait
    # for it rather than yanking the pipeline out from under it.
    t0 = time.perf_counter()
    print(f"\n{datetime.now():%H:%M:%S} <> [{slot.device_name}] Swap "
          f"{slot.model_name or '(empty)'} -> {model_display_name(target)}"
          + (f" (moving {moved_from} -> {dev_name})" if moved_from else ""), flush=True)
    with slot.lock.moving_to(dev_name):
        slot.unload()
        # Wait for the release to actually land before loading. The offload
        # ratio is resolved from live free RAM, and the GPU driver hands the
        # outgoing model's pages back over ~2 s (_settle_memory) — so reading
        # availability the instant unload() returns saw a machine that still
        # looked full and pinned the incoming model at ratio 90. Measured:
        # swapping a 20.4 GB model out and a 14.3 GB one in took gemma from
        # ratio 0 to ratio 90, i.e. ~5 tok/s, for no reason but timing. This
        # is the one moment the machine is guaranteed to be mid-release, so
        # it is the one place the settle has to happen.
        gc.collect()
        _settle_memory()
        # Re-point the slot only after the old pipeline is gone, so the two
        # models are never resident at once — the whole point of one-at-a-time.
        if moved_from:
            slot.device_name = dev_name
            slot.device_id = dev_id
            slot.device_full = runtime.DEVICES.get(dev_name, {}).get("name", dev_name)
        # The outgoing pipeline's native resources are released on the GC's
        # schedule, and loading a new pipeline before that finishes can fail
        # re-registering the tokenizers extension ("Failed to load shared
        # object: openvino_tokenizers.dll") — observed once, then the identical
        # request succeeded. Collect again and retry rather than surfacing a
        # flake the user can only fix by clicking Load twice.
        last_err = None
        for attempt in (1, 2, 3):
            try:
                slot.load(target)
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"  [{slot.device_name}] load attempt {attempt} failed: {e}",
                      flush=True)
                slot.pipe = None
                gc.collect()
                # The retry exists for one specific flake: the tokenizers
                # extension failing to re-register while the previous pipeline
                # is still being collected. A model this build cannot open
                # fails identically three times, so retrying only makes the
                # user wait to read the same sentence twice more.
                if not _is_transient_load_error(e):
                    break
                time.sleep(1.5 * attempt)
        if last_err is not None:
            slot.status = "error"
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
                  f"Swap failed: {last_err}", flush=True)
            raise _refuse(
                f"Failed to load '{name}': {explain_genai_error(last_err)}",
                "server_error", 500)
        slot.warmup()
        if slot.status == "error":
            raise _refuse("Model loaded but warmup failed; check server logs", "server_error", 500)
        slot.prewarmed = False
        slot.last_ttft_ms = None
    elapsed = time.perf_counter() - t0
    print(f"{datetime.now():%H:%M:%S} <> [{slot.device_name}] Swap done "
          f"({elapsed:.1f}s)", flush=True)

    return {"status": "ok", "model": slot.model_name,
            "device": slot.device_name, "type": slot.model_type,
            "placement": placement, "load_seconds": round(elapsed, 1)}


def unload_model():
    """Free a slot's memory now, without stopping the server.

    The model is remembered, so the next request to that slot reloads it
    (ensure_loaded). This is the manual version of the idle watchdog for
    when you want the RAM back immediately.

    Measured before and after, and the numbers are returned: the caller is
    about to launch something heavy and needs to know what it actually got,
    not that the call returned 200. See the memory notes in TODONT.md for
    why the answer is system-available RAM and not process RSS.

    Two different numbers, deliberately kept apart. `weights_mb` is ours and
    auditable — the on-disk size of what was just dropped. `system_available`
    is an observation about the whole machine: on a box under pressure it can
    move by far more than we released (measured: +22.8 GB after unloading
    4.8 GB of models at 368 MB free, as Windows dumped standby pages at the
    same time). Attributing that to locally would be a lie, so the caller
    gets both and the UI reports them as what they are.
    """
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict) or not isinstance(body.get("device", ""), str):
        return openai_error("Request body must be an object with an optional device string")
    want = (body.get("device") or "").upper()
    # "device" cannot express "the OCR engines but not the model answering on
    # the same device" -- and on a one-model box the chat model and the
    # utilities are usually on the SAME device, so freeing the utilities by
    # device took the conversation down with them. `scope` says which KIND of
    # slot is meant; the two compose (scope util + device NPU is legal).
    scope = str(body.get("scope") or "all").lower()
    kinds = {
        "all": (runtime.primary, runtime.secondary, runtime.whisper_slot,
                runtime.tts_slot, runtime.util_npu, runtime.util_gpu, runtime.util_cpu),
        "chat": (runtime.primary, runtime.secondary),
        "audio": (runtime.whisper_slot, runtime.tts_slot),
        "util": (runtime.util_npu, runtime.util_gpu, runtime.util_cpu),
    }
    if scope not in kinds:
        return openai_error("'scope' must be one of: all, chat, audio, util")

    targets = [s for s in kinds[scope]
               if s and s.status not in ("not_configured", "idle_unloaded")]
    if not targets and scope != "all":
        # Nothing of that kind is loaded, which is the state the caller wanted.
        return jsonify({"status": "ok", "unloaded": [],
                        "memory": {"before": _memory_snapshot(),
                                   "after": _memory_snapshot(),
                                   "weights_mb": 0}})
    if want:
        targets = [s for s in targets if s.device_name == want]
        if not targets:
            return openai_error(f"No loaded slot on device '{want}'.")

    if not targets:
        # Everything is already unloaded — that's the desired state, not an
        # error, and it must not be reported as "busy" (which is what the
        # 409 below would say if we fell through to it).
        return jsonify({"status": "ok", "unloaded": [],
                        "memory": {"before": _memory_snapshot(),
                                   "after": _memory_snapshot(),
                                   "returned_mb": 0, "process_freed_mb": 0,
                                   "weights_mb": 0, "settle_seconds": 0.0}})

    before = _memory_snapshot()
    freed = []
    for slot in targets:
        # Don't yank a pipeline out from under a running generation.
        if not slot.lock.acquire(blocking=False):
            continue
        try:
            name = slot.model_name
            weights = _dir_size_bytes(slot.model_dir) if slot.model_dir else None
            slot.unload()
            freed.append({"device": slot.device_name, "model": name,
                          "weights_mb": weights and round(weights / 2 ** 20)})
        finally:
            slot.lock.release()

    if not freed:
        return openai_error("All matching slots are busy generating.",
                            "server_error", 409)

    gc.collect()
    after, settled = _settle_memory()

    def _delta(key):
        a, b = before.get(key), after.get(key)
        return None if a is None or b is None else b - a

    # returned_mb is machine-wide movement, NOT a claim about this process
    # (see the docstring). process_freed_mb is what python.exe itself gave
    # up — a lower bound, since NPU/GPU driver allocations are never fully
    # charged to it (measured: 1.6 GB of process drop for 6.7 GB of system
    # movement on the same unload).
    returned = _delta("system_available_mb")
    dropped = _delta("process_private_mb")
    memory = {"before": before, "after": after,
              "settle_seconds": settled, "returned_mb": returned,
              "process_freed_mb": None if dropped is None else -dropped,
              "weights_mb": sum(f["weights_mb"] or 0 for f in freed) or None}

    print(f"{datetime.now():%H:%M:%S} <> Unloaded on request: "
          + ", ".join(f"{f['model']}@{f['device']}" for f in freed)
          # ASCII arrow on purpose: this line goes to a cp1252 console.
          + (f" -- system available {before['system_available_mb']} -> "
             f"{after['system_available_mb']} MB "
             f"({returned:+d}, {settled}s)" if returned is not None else ""),
          flush=True)
    return jsonify({"status": "ok", "unloaded": freed, "memory": memory})

