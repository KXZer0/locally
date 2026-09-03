"""Root page, diagnostics, health, and UI bootstrap payloads."""

import atexit
import json
import time
from datetime import datetime

from flask import jsonify, render_template, request

from core import config, odysseus, opencode_web, runtime
from core.chat.common import overall_status
from core.chat.turn import python_tool_status
from core.hardware.devices import (_gpu_shares_system_ram,
                                   _usable_gpu_bytes)
from core.hardware.memory import _mem_status
from core.models.discovery import _available_models_data, _models_data
from core.models.integrity import _dir_size_bytes
from core.routes import audio as audio_routes
from core.slots.select import _slot_serviceable
from core.system.coding import _coding_mode_state
from core.tools.builtin import BUILTIN_TOOLS, _builtin_tools_supported
from core.web.search import _searx, web_search_status

# Debug logging
# ---------------------------------------------------------------------------

def _log_request(api_label):
    if not config.DEBUG_REQUESTS:
        return
    body_raw = request.get_data(as_text=True)
    try:
        body_str = json.dumps(json.loads(body_raw), indent=2) if body_raw else ""
    except Exception:
        body_str = body_raw
    ua = request.headers.get("User-Agent", "")
    print(f"{datetime.now():%H:%M:%S} [DEBUG/{api_label}] {request.method} {request.path}"
          f"  UA={ua!r}", flush=True)
    if body_str:
        for line in body_str.splitlines():
            print(f"  {line}", flush=True)


def _debug_openai():
    _log_request("OpenAI")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def gui():
    return render_template("index.html")


def _health_data():
    """One snapshot of every serveable slot, without binding it to HTTP.

    The UI bootstrap endpoint needs the exact same truth as /health. Keeping
    the payload in one function prevents the fast path and the public endpoint
    from drifting into subtly different ideas of what is ready.
    """
    devices = {}
    if runtime.primary and runtime.primary.status != "not_configured":
        devices[runtime.primary.device_name.lower()] = runtime.primary.info
    if runtime.secondary and runtime.secondary.status != "not_configured":
        devices[runtime.secondary.device_name.lower()] = runtime.secondary.info
    # prompt_cache stays a bare bool — start-openclaw.ps1's health check
    # truth-tests it; the details live in prompt_cache_info (per-slot TTFT
    # and prewarm state are in each device's info block).
    result = {"status": overall_status(), "devices": devices,
              "prompt_cache": config.PROMPT_CACHE,
              "web_search": web_search_status(),
              "python_tool": python_tool_status(),
              "coding_mode": _coding_mode_state(),
              # Which server-owned tools the serving slot will be offered. The
              # web UI reads this to decide whether to ask for a streamed turn:
              # the tool loop has to see a whole tool-call block before it can
              # run it, so a turn that may call one is buffered.
              "builtin_tools": [
                  t["spec"]["function"]["name"]
                  for t in BUILTIN_TOOLS.values() if t["enabled"]()
              ] if _builtin_tools_supported(
                  next((x for x in (runtime.primary, runtime.secondary)
                        if _slot_serviceable(x)), None)) else [],
              "prompt_cache_info": {
                  "enabled": config.PROMPT_CACHE,
                  "pool_gb": config.PROMPT_CACHE_GB,
                  "prewarm_file": config.PREWARM_FILE,
              },
              # The OpenCode web server's state belongs here for the same
              # reason every other subsystem's does: it is a process this one
              # may be holding open, and nothing else reports it.
              "opencode_web": opencode_web.status(),
              # Odysseus is a separate program, but locally may be holding its
              # containers up, so its state belongs in the same snapshot for
              # the same reason OpenCode's does.
              "odysseus": odysseus.status()}
    if runtime.whisper_slot and runtime.whisper_slot.status != "not_configured":
        result["whisper"] = runtime.whisper_slot.info
    if runtime.tts_slot and runtime.tts_slot.status != "not_configured":
        result["tts"] = runtime.tts_slot.info
    # The Voice tab enables auto turn-taking from this. Reported only when the
    # socket can actually carry it, so the UI never offers a control that is
    # wired to nothing.
    if (runtime._vad_slot and runtime._vad_slot.status != "not_configured"
            and audio_routes.stream_available):
        result["vad"] = runtime._vad_slot.info
    # Reported whenever the slot exists, even with no profile: the Voice tab
    # needs to know the difference between "not installed" (hide enrolment)
    # and "installed but not enrolled" (offer it), and those look identical
    # from the client's side otherwise.
    if runtime._speaker_slot and runtime._speaker_slot.status != "not_configured":
        result["speaker"] = runtime._speaker_slot.info
    # The UI enables each utility's engine toggle from this, the same way the
    # Voice tab gates on whisper/tts.
    util = {name: slot.info
            for name, slot in (("npu", runtime.util_npu), ("gpu", runtime.util_gpu),
                               ("cpu", runtime.util_cpu))
            if slot and slot.status != "not_configured"}
    if util:
        result["util"] = util
    return result


def health():
    return jsonify(_health_data())


def memory_status():
    return jsonify(_memory_data())


def _memory_data():
    """What the machine can actually back right now — for the UI's memory HUD.

    Every number here already existed server-side and none of it was ever shown,
    which is how a slot pinned at 90% offload read as "the model got slow" and
    how an 18.3 GB model paging at 0.5 tok/s looked like a model problem.

    The one thing this must communicate, because nothing else in the UI does:
    on an integrated GPU the driver's advertised ceiling is a *policy share of
    system RAM*, not memory anybody set aside. Reporting `ceiling` alone is what
    makes "I have 24 GB of VRAM" a reasonable conclusion on a 31.5 GB machine.
    So ceiling and genuinely-free are always reported together.

    Thresholds come from `_usable_gpu_bytes()` — the same function the offload
    decision uses. A HUD that disagreed with the placement it is describing
    would be worse than no HUD.
    """
    total, available = _mem_status()
    usable, ceiling = _usable_gpu_bytes("GPU", "GPU") if "GPU" in runtime.DEVICES \
        else (None, None)
    mib = 2 ** 20

    slots = []
    for slot in (runtime.primary, runtime.secondary):
        if not slot or slot.status == "not_configured":
            continue
        weights = _dir_size_bytes(slot.model_dir) if slot.model_dir else None
        slots.append({
            "model": slot.model_name,
            "device": slot.device_name,
            "status": slot.status,
            "weights_mb": weights and round(weights / mib),
            "offload_ratio": slot.offload_ratio,
            "offload_note": slot.offload_note.strip() or None,
        })

    # Worst state across slots wins: one offloading model is the headline even
    # if another is comfortable.
    offloading = [s for s in slots if s["offload_ratio"]]
    if available is not None and available < config.GPU_RESERVE_BYTES:
        state = "critical"
        message = (f"{available / 2 ** 30:.1f} GB free — the OS is close to "
                   f"paging, which collapses generation speed.")
        action = "Close other applications, or load a smaller model."
    elif offloading:
        worst = max(offloading, key=lambda s: s["offload_ratio"])
        state = "offloading"
        message = (f"{worst['model']} is streaming {worst['offload_ratio']}% of "
                   f"its expert weights from disk.")
        action = ("Free RAM, or raise the iGPU Shared GPU Memory Override, to "
                  "keep it fully resident.")
    else:
        state = "resident"
        message = "Models are fully resident."
        action = None

    return {
        "state": state,
        "message": message,
        "action": action,
        "system": {
            "total_mb": total and round(total / mib),
            "available_mb": available and round(available / mib),
        },
        "gpu": {
            # Named to keep the distinction unmissable: the ceiling is what the
            # driver advertises, usable is what the machine can currently back.
            "driver_ceiling_mb": ceiling and round(ceiling / mib),
            "usable_mb": usable and round(usable / mib),
            "shares_system_ram": _gpu_shares_system_ram("GPU")
            if "GPU" in runtime.DEVICES else None,
            "reserve_mb": round(config.GPU_RESERVE_BYTES / mib),
        },
        "slots": slots,
    }


def ui_bootstrap():
    """Everything the first frame needs, captured in one HTTP round trip.

    The old browser startup made six serial requests: /health triggered model,
    available-model and memory refreshes, then init() repeated both model
    requests. Localhost latency is small, but every `await` yielded a separate
    parse, Flask dispatch and render opportunity, so the controls visibly
    filled in one after another. Keep the public endpoints for API clients and
    polling; this is only the web UI's coherent initial snapshot.
    """
    started = time.perf_counter()
    payload = {
        "health": _health_data(),
        "models": _models_data(),
        "available_models": _available_models_data(),
        "memory": _memory_data(),
    }
    response = jsonify(payload)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["Server-Timing"] = f"bootstrap;dur={elapsed_ms:.1f}"
    return response


# ---------------------------------------------------------------------------
# Utilities (OCR, native documents, images, and local semantic search)
# ---------------------------------------------------------------------------

# On-demand SearXNG
#
# Same contract as every model slot here: load when needed, unload when idle.
# A search backend sitting resident all day costs ~80 MB and a process for a
# feature used a few times an hour, and this machine is already choosing
# between an 8B model and an offload ratio.
#
# The one rule that matters: we only ever stop a process WE started. If SearXNG
# was already listening when we first looked -- because the user ran
# scripts/searxng.ps1 themselves -- it is theirs, and killing it on an idle
# timer would be a genuinely infuriating bug.


atexit.register(_searx.stop)
atexit.register(opencode_web.stop)
# Only ever stops a stack this process started, and `stop` never `down`, so
# nothing the user has in Odysseus can be lost by locally exiting.
atexit.register(odysseus.stop)
