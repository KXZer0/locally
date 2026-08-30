#!/usr/bin/env python3
"""locally — OpenAI-compatible API server for Intel NPU / ARC GPU.

Auto-detects available devices (NPU, GPU, CPU) and model type (VLM/LLM).
NPU-first: works on any Intel Core Ultra laptop. ARC GPU optional.
Dual mode: NPU for chat + GPU for vision, simultaneously.

Usage:
    python locally.py                                        # auto-detect device
    python locally.py --device NPU                           # force NPU
    python locally.py --device GPU                           # force GPU
    python locally.py --gpu-model-dir gpu-model              # dual: NPU chat + GPU vision
    python locally.py --model-dir ~/models/qwen3-14b-int4-ov --device GPU  # big LLM on GPU
    python locally.py --whisper-dir whisper-model             # add speech-to-text
    python locally.py --scan                                 # what models do I have?
"""

import argparse
import base64
import gc
import hashlib
import concurrent.futures
import ctypes
import urllib.error
import urllib.parse
import urllib.request
import atexit
import subprocess
import io
import ipaddress
import itertools
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
import threading
import uuid
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from urllib.parse import unquote

# Silence OpenVINO's verbose property dump on model load (Model: OV Tokenizer
# + ~25 lines of NETWORK_NAME / NUM_STREAMS / INFERENCE_NUM_THREADS / ...).
# Must be set BEFORE openvino is imported.
os.environ.setdefault("OPENVINO_LOG_LEVEL", "0")

import numpy as np
import openvino as ov
import openvino_genai as ovg
from flask import Flask, Response, jsonify, request, render_template
from PIL import Image
try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    # Imported at module level only for the exception type and the
    # disabled-task reasons — the engines themselves stay lazy (UtilSlot.load),
    # so a missing utility_pipeline keeps the server usable for chat instead of
    # failing at import.
    from utility_pipeline import UtilityUnavailable
    from utility_pipeline import _DISABLED_TASKS as _UTIL_DISABLED_TASKS
except ImportError:
    class UtilityUnavailable(RuntimeError):
        """Stand-in so the util endpoints' except clauses always resolve."""

    _UTIL_DISABLED_TASKS = {}

# ---------------------------------------------------------------------------
# core/ — the parts that are worth reading on their own.
#
# Imported by name rather than with * so that every symbol below has one
# obvious home, and so `grep "def foo"` still finds exactly one definition.
# ---------------------------------------------------------------------------
from core import config, runtime
from core.genai.tokens import _count_tokens
from core.slots.capability import (_npu_tools_affordable, _tool_capable,
                                   _tools_refused_note, _tools_supported)
from core.slots.locks import _device_lock
from core.slots.device import DeviceSlot
from core.slots.proxy import ProxySlot, _gen_to_openai
from core.slots.speaker import SpeakerSlot
from core.slots.tts import TtsSlot
from core.slots.util import UtilSlot
from core.slots.vad import VadSlot
from core.slots.whisper import WhisperSlot
from core.documents.inputs import (_pil_data_url, _util_chunks, _util_item,
                                   _util_input_from_request)
from core.documents.read import (_document_result, _layout_markdown,
                                 _native_document_markdown, _ocr_pages, _pdf_pages)
from core.tools.builtin import (BUILTIN_TOOLS, _builtin_generator,
                                _builtin_stream, _builtin_tools_supported,
                                _fit_tool_output, builtin_tools_for)
from core.coding_mode import _coding_mode_error
from core.slots.route import _route_request
from core.slots.select import _slot_serviceable, _util_slot
from core.tools.registry import PYTHON_TOOL, WEB_SEARCH_TOOL
from core.web.fetch_page import (_web_search_answer, _web_search_hits,
                                 _web_search_run)
from core.web.fetch_page_cfg import _login_wall, _web_sources_for_results
from core.voice.session import VoiceStreamSession
from core.voice.think_filter import _ThinkFilter
from core.web.search import (_WEB_ANSWER_RESERVE, _searx, _web_grounded_blocks,
                             web_search_status)
from core.errors import _TurnError, openai_error
from core.genai.results import extract_perf, extract_text, explain_genai_error
from core.hardware.devices import (_device_mem_bytes, _gpu_has_xmx,
                                   _gpu_shares_system_ram, _OS_RESERVE_BYTES,
                                   _usable_gpu_bytes)
from core.hardware.memory import (_mem_status, _memory_snapshot, _process_memory,
                                  _settle_memory, _system_ram_bytes,
                                  _win_current_process)
from core.media.images import load_image, pil_to_tensor
from core.models.describe import _model_dirs_under, describe_model, scan_models
from core.models.geometry import (_kv_bytes_per_token, _model_max_context,
                                  _moe_expert_fraction, _text_config)
from core.models.identity import (_GENERIC_DIR_NAMES, _is_generative_dir,
                                  _is_model_dir, _NAME_SUFFIXES,
                                  _strip_name_suffixes, is_vlm,
                                  model_display_name, resolve_display_name)
from core.models.integrity import _dir_size_bytes, _verify_weights_integrity
from core.models.irinfo import _flatten_rt_info, read_ir_rt_info, weight_precision
from core.sandbox.python_exec import execute_python
from core.tools.parse import parse_tool_calls
from core.tools.render import (_suppress_think_for_tools, _tool_calls_to_text,
                               prepare_messages_for_tools, render_tools_prompt)
from core.tools.text import _strip_tool_markup, strip_thinking
from core.web.reader import (_guard_public_url, _html_to_markdown, _http_fetcher)


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------

def parse_messages(messages, max_dim):
    """Parse OpenAI messages. Returns (text_prompt, images, raw_messages)."""
    text_parts = []
    images = []
    raw_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            text_parts.append(content)
            raw_messages.append({"role": role, "content": content})
            continue

        msg_text = []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                msg_text.append(block.get("text", ""))
            elif btype == "image_url":
                url = block.get("image_url", {}).get("url", "")
                if url:
                    img = load_image(url, max_dim)
                    images.append(pil_to_tensor(img, max_dim))
                    # Anchor the image to ITS turn in the flattened prompt —
                    # without the tag, genai clusters all images before the
                    # prompt and the model answers about image #1 regardless
                    # of which turn asked the question.
                    msg_text.append(f"<ov_genai_image_{len(images) - 1}>")

        joined = " ".join(msg_text)
        text_parts.append(joined)
        raw_messages.append({"role": role, "content": joined})

    return "\n".join(text_parts), images, raw_messages


# ---------------------------------------------------------------------------
# Memory preflight — warn (never block) when a model won't fit its device
# ---------------------------------------------------------------------------


# The NPU pipeline is built with this prompt cap (its own default is 1024).
# It is a hard ceiling on that device, so it is also the context limit there.
#
# 8192, not 4096: measured 2026-08-19 with scripts/npu-context-probe.py on
# Qwen3-8B-int4-cw / Intel AI Boost. 8192 compiles (120 s cold, 4.8 s cached)
# and answers an 8,160-token needle-in-haystack prompt correctly.
#
# 8192 is the CEILING, and it is a compiler limit rather than a memory one.
# 9216, 10240 and 12288 all fail the same way -- the vpux compiler refuses to
# legalise the graph:
#     failed to legalize operation 'VPU.NCE.Reduce'
#     Failed Pass EnsureNCEOpsSizeRequirements
# Only powers of two get through. Raising it further needs a compiler fix, not
# a flag. In particular KV-cache compression does NOT help: 12288 fails
# identically with KV_CACHE_PRECISION=u8, because nothing here is short of
# memory (Qwen3-8B is 144 KB/token, so even 8192 tokens is only 1.1 GB).
# See TODONT.md.
# Measured 2026-08-19, same 5-token prompt, warm, best of 3:
#     4096 -> TTFT 1.09s
#     8192 -> TTFT 1.43s
# and prompt LENGTH barely matters at a given shape (7 tokens 1.43s, 225 tokens
# 1.46s), because the graph pads to the compiled size. So this is a straight
# context-vs-latency trade and --npu-prompt-len exposes it.

# Server-side Python turns are deliberately separate from client tool calling.
# The latter renders a whole client-supplied catalog and asks the client to run
# the loop, which excludes the NPU. This one small, fixed schema is injected
# only when the user asks for it and the server owns the loop.
# Coding mode: hold the machine for the model driving the coding agent. A real mode
# rather than a one-shot "free memory" button, because the audio and utility
# slots reload themselves the moment anything touches them -- opening Voice
# pulls Whisper and Kokoro straight back in. A button would be undone by the
# next click and the user would never know why the memory came back.
config.CODING_MODE = False

# Rounds for the server-owned tool loop (python, web_search).
BUILTIN_TOOL_MAX_ROUNDS = 2


def _python_tool_prompt():
    """The only prompt addition for a server-side Python turn.

    Keep this intentionally small: on the NPU this is part of the hard
    8192-token prompt budget. The surrounding render_tools_prompt wrapper is
    measured at startup and reported in /health so a model-specific tokenizer
    can be checked.
    """
    return render_tools_prompt([PYTHON_TOOL])


# ---------------------------------------------------------------------------
# Model introspection (--scan)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tool calling (function calling) — Qwen3-Coder native format
#
# VS Code Copilot Chat sends tool definitions in the request `tools` array and
# expects structured `tool_calls` back. OpenVINO GenAI applies the model's chat
# template but gives us no hook to pass `tools` through it, so we render the
# specs into a system prompt ourselves (the way Qwen3-Coder was trained) and
# parse the model's emitted calls back into OpenAI shape.
#
# Qwen3-Coder emits calls as:
#   <tool_call>
#   <function=NAME>
#   <parameter=KEY>
#   VALUE
#   </parameter>
#   </function>
#   </tool_call>
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Server-side Python calculation tool
# ---------------------------------------------------------------------------


# `winreg` is already in sys.modules before the child runs -- CPython imports
# it during startup on Windows -- so the import block cannot catch it and does
# not need to: every winreg.* operation is denied by the prefix rule above.
# Verified: `import winreg` succeeds, `winreg.OpenKey` raises.


# ---------------------------------------------------------------------------
# Device slot — holds one pipeline + its metadata
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Proxy slot — a generative model this process does not own
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Whisper (speech-to-text) slot
# ---------------------------------------------------------------------------

def _load_audio(file_storage):
    """Read uploaded audio file to float32 numpy array at 16 kHz."""
    if sf is None:
        raise RuntimeError("soundfile not installed. pip install soundfile")
    audio, sr = sf.read(io.BytesIO(file_storage.read()), dtype="float32")
    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Resample to 16 kHz if needed
    if sr != 16000:
        target_len = int(len(audio) * 16000 / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------


# Default repetition penalty, overridable in locally.ini ([generation] section)
# next to this file. Ollama ships 1.1, which breaks thinking-loops faster but
# is known to hurt code generation; 1.05 is the compromise default. Clients
# that send their own penalties (OpenAI frequency/presence_penalty, Ollama
# repeat_penalty) override this per-request — see apply_penalties().
import configparser as _configparser
_ini = _configparser.ConfigParser()
# Both names are read, old first, so an existing nollama.ini keeps working
# after the rename and a locally.ini beside it wins.
_ini.read([Path(__file__).parent / "nollama.ini",
           Path(__file__).parent / "locally.ini"])
REPETITION_PENALTY = _ini.getfloat("generation", "repetition_penalty", fallback=1.05)


def apply_penalties(gen, repetition=None, frequency=None, presence=None):
    """Set gen's penalties: per-request client values win over the
    locally.ini/default repetition penalty. frequency/presence map through
    when this openvino-genai build supports them (CB pipelines do; the
    static NPU pipeline may ignore them)."""
    gen.repetition_penalty = repetition if repetition is not None else REPETITION_PENALTY
    for attr, val in (("frequency_penalty", frequency), ("presence_penalty", presence)):
        if val is not None:
            try:
                setattr(gen, attr, float(val))
            except Exception:
                pass
_prewarm_hash = None  # debounce: only re-capture when the system prompt changes

app = Flask("locally",
            template_folder=os.path.join(config.SCRIPT_DIR, "templates"),
            static_folder=os.path.join(config.SCRIPT_DIR, "static"))
app.config["MAX_CONTENT_LENGTH"] = config.MAX_REQUEST_BYTES
# Jinja caches compiled templates for the life of the process unless this is on,
# while /static is served fresh from disk every request (Cache-Control:
# no-cache). A long-running server therefore pairs OLD html with NEW javascript
# after any UI edit — markup and handlers disagree, and buttons silently stop
# responding in a way that reads as a broken feature rather than a stale
# process. Re-reading one 16 KB template when its mtime changes costs nothing.
app.config["TEMPLATES_AUTO_RELOAD"] = True

                      # every device on the box — not just ones that already
                      # have a slot. Lets one slot MOVE between NPU and GPU.
RUNTIME_SLOTS = []    # the live list shared with the idle watchdog; onboarding
                      # can add audio slots after the server has already bound
OLLAMA_COMPAT_PORT = 0  # non-zero when locally owns its Ollama-compatible port
max_dim = 768
debug = False
vscode_compat = False  # report a real Ollama version so VS Code accepts us

# Ollama version VS Code expects; fake but recent enough to pass its checks.
VSCODE_OLLAMA_VERSION = "0.18.3"
_request_counter = itertools.count(1)  # thread-safe id generator


def make_id():
    return f"arc-{next(_request_counter):04d}"


def overall_status():
    """Ready when all configured devices are ready."""
    slots = [s for s in (runtime.primary, runtime.secondary) if s and s.status != "not_configured"]
    if not slots:
        return "not_configured"
    # If ANY slot is ready or idle_unloaded (will reload on demand), we can
    # serve requests. A dead secondary shouldn't kill the primary.
    if any(s.status in ("ready", "idle_unloaded") for s in slots):
        return "ready"
    if all(s.status == "error" for s in slots):
        return "error"
    return "loading"


def _sse_replay(completion_id, created, model, message, finish_reason):
    """Emit a buffered chat result as an OpenAI streaming SSE sequence.

    Used for tool-enabled turns, where we must buffer the full generation
    before we can hand back a structured tool_calls delta.
    """
    def chunk(delta, finish=None):
        return "data: " + json.dumps({
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }) + "\n\n"

    yield chunk({"role": "assistant"})
    if message.get("content"):
        yield chunk({"content": message["content"]})
    for i, tc in enumerate(message.get("tool_calls") or []):
        yield chunk({"tool_calls": [{
            "index": i, "id": tc["id"], "type": "function",
            "function": tc["function"],
        }]})
    yield chunk({}, finish_reason)
    yield "data: [DONE]\n\n"


def _sse_tool_stream(slot, raw_messages, gen, tools, completion_id, created, t0):
    """Buffered tool turn, streamed with keep-alive frames.

    A tool turn must be fully generated before we can emit a structured
    tool_calls delta — but prefilling a big agent prompt (e.g. an OpenClaw
    system prompt) on a small device can take minutes, longer than a client's
    idle watchdog. So run generation in a background thread and emit SSE pings
    until it finishes, then replay the parsed result. Without this the client
    sees nothing during prefill and aborts (and OpenVINO can't cancel a blocked
    prefill, so the abandoned generation keeps churning).
    """
    result = {}

    def _run():
        try:
            result["text"] = slot.generate_llm(raw_messages, gen)
        except Exception as e:  # noqa: BLE001 — surfaced to the client below
            result["error"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()

    # Immediate role frame so the client sees activity at once, then a ping
    # every config.HEARTBEAT_SECS while generation runs (no tokens exist yet).
    yield ("data: " + json.dumps({
        "id": completion_id, "object": "chat.completion.chunk",
        "created": created, "model": slot.model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"},
                     "finish_reason": None}],
    }) + "\n\n")
    while th.is_alive():
        th.join(timeout=config.HEARTBEAT_SECS)
        if th.is_alive():
            # Empty-content keep-alive (resets content- and byte-based client
            # watchdogs alike; empty string is a no-op for message assembly).
            yield ("data: " + json.dumps({
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": slot.model_name,
                "choices": [{"index": 0, "delta": {"content": ""},
                             "finish_reason": None}],
            }) + "\n\n")

    elapsed = time.perf_counter() - t0
    if result.get("error") is not None:
        err = explain_genai_error(result["error"])
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
              f"LLM error: {err}", flush=True)
        yield ("data: " + json.dumps({
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": slot.model_name,
            "choices": [{"index": 0, "delta": {"content": f"\n[error: {err}]"},
                         "finish_reason": "error"}],
        }) + "\n\n")
        yield "data: [DONE]\n\n"
        return

    text = result.get("text", "")
    n_words = len(text.split())
    ttft = (f", TTFT {slot.last_ttft_ms:.0f}ms"
            if slot.last_ttft_ms is not None else "")
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
          f"~{n_words} tokens in {elapsed:.1f}s "
          f"({n_words / max(elapsed, 1e-6):.1f} tok/s{ttft})", flush=True)

    text, tool_calls = parse_tool_calls(text, tools)
    if tool_calls:
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
              f"{len(tool_calls)} tool call(s): "
              f"{', '.join(tc['function']['name'] for tc in tool_calls)}", flush=True)
        message = {"role": "assistant", "content": text or None, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish_reason = "stop"

    # Reuse the replay emitter (it re-sends a role frame — harmless, clients
    # just set role twice) for the content/tool_calls/finish/[DONE] tail.
    for frame in _sse_replay(completion_id, created, slot.model_name, message, finish_reason):
        yield frame


def _maybe_capture_prewarm(raw_messages):
    """Save a big (agent) prompt to config.PREWARM_FILE so the next startup can warm
    the prefix cache. Debounced on the system prompt — written once per distinct
    system prompt, not every turn. Stale-safe: if the agent's prompt later
    changes, a fresh request overwrites the file; a mismatch only costs a cold
    first turn, never a wrong answer.
    """
    if not config.PREWARM_FILE or not raw_messages:
        return
    sys_text = "".join(str(m.get("content", "")) for m in raw_messages
                       if m.get("role") == "system")
    if len(sys_text) < config.PREWARM_MIN_CHARS:
        return
    global _prewarm_hash
    # sha256, not builtin hash(): hash() is PYTHONHASHSEED-salted per process,
    # which forced one redundant rewrite after every restart.
    h = hashlib.sha256(sys_text.encode("utf-8")).hexdigest()
    if h == _prewarm_hash:
        return
    # Save system + the first user turn — enough to cache the shared prefix.
    to_save = []
    for m in raw_messages:
        to_save.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        if m.get("role") == "user":
            break
    try:
        with open(config.PREWARM_FILE, "w", encoding="utf-8") as f:
            json.dump(to_save, f)
        _prewarm_hash = h
    except OSError:
        pass


def _prewarm_slot(slot):
    """Prefill the saved prompt at startup so the first real (cold) turn is a
    cache hit. Only meaningful on a GPU/CPU LLM slot with prefix caching on.
    """
    if not (config.PREWARM_FILE and config.PROMPT_CACHE and slot.model_type == "llm"
            and slot.device_name in ("GPU", "CPU")):
        return
    if not os.path.isfile(config.PREWARM_FILE):
        return
    try:
        with open(config.PREWARM_FILE, encoding="utf-8") as f:
            raw_messages = json.load(f)
    except (OSError, ValueError):
        return
    if not raw_messages:
        return
    try:
        gen = ovg.GenerationConfig()
        gen.max_new_tokens = 1
        gen.do_sample = False
        t0 = time.perf_counter()
        slot.generate_llm(raw_messages, gen)  # prefills -> populates prefix cache
        slot.prewarmed = True
        print(f"  [{slot.device_name}] pre-warmed prompt cache from "
              f"{os.path.basename(config.PREWARM_FILE)} ({time.perf_counter() - t0:.1f}s)",
              flush=True)
    except Exception as e:
        slot.prewarmed = False
        print(f"  [{slot.device_name}] pre-warm failed: {explain_genai_error(e)}", flush=True)


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

def _log_request(api_label):
    if not debug:
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


@app.before_request
def _debug_openai():
    _log_request("OpenAI")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/")
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
              }}
    if runtime.whisper_slot and runtime.whisper_slot.status != "not_configured":
        result["whisper"] = runtime.whisper_slot.info
    if runtime.tts_slot and runtime.tts_slot.status != "not_configured":
        result["tts"] = runtime.tts_slot.info
    # The Voice tab enables auto turn-taking from this. Reported only when the
    # socket can actually carry it, so the UI never offers a control that is
    # wired to nothing.
    if (runtime._vad_slot and runtime._vad_slot.status != "not_configured" and AUDIO_STREAM_OK):
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


# ---------------------------------------------------------------------------
# Update checks
#
# Opt-in (--check-updates), and off by default on purpose. This is the only
# code in the server that contacts anything on the internet without the user
# asking for it in that request, and a tool whose pitch is "your data never
# leaves the machine" does not get to quietly poll four services on boot.
# Everything here also has to be unable to hurt a start: the work runs on a
# background thread, every failure is swallowed, and the cached answer is what
# the endpoint serves. On a plane it returns "unknown" and nothing waits.
# ---------------------------------------------------------------------------

CHECK_UPDATES = False
_UPDATE_TTL = 86400          # a day; nobody needs to know sooner
_UPDATE_TIMEOUT = 6          # sources run together, so four dead ones cost ~6 s
_updates_cache = {"checked_at": 0.0, "sources": {}, "refreshing": False}
_updates_lock = threading.Lock()

# name -> (kind, locator, human URL). Kept as data so adding a project is a
# line, not a branch.
_UPDATE_SOURCES = {
    # locally does not need a release to exist: compare this checkout's HEAD
    # with public main. That keeps update notices useful between releases.
    "locally":  ("github_commit", "KXZer0/locally", "https://github.com/KXZer0/locally"),
    "odysseus": ("github_release", "odysseus-dev/odysseus", "https://github.com/odysseus-dev/odysseus"),
    "searxng":  ("github_release", "searxng/searxng", "https://github.com/searxng/searxng"),
    "opencode": ("npm", "opencode-ai", "https://github.com/anomalyco/opencode"),
}


def _fetch_latest(kind, locator):
    """Newest published version for one project, or None. Never raises."""
    try:
        if kind == "npm":
            url = f"https://registry.npmjs.org/{locator}/latest"
        elif kind == "github_commit":
            url = f"https://api.github.com/repos/{locator}/commits/main"
        else:
            url = f"https://api.github.com/repos/{locator}/releases/latest"
        req = urllib.request.Request(url, headers={
            # GitHub rejects a missing UA, and naming ourselves is the polite
            # thing when we are the ones doing the polling.
            "User-Agent": "locally-update-check",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=_UPDATE_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        if kind == "npm":
            return data.get("version")
        if kind == "github_commit":
            return data.get("sha")
        return (data.get("tag_name") or data.get("name") or "").lstrip("v") or None
    except Exception:
        return None


def _local_version(name):
    """What is installed here, or None when we cannot tell.

    Deliberately returns None rather than guessing: reporting "0.0.0" would
    make every project look out of date forever, which trains people to ignore
    the notice — the exact failure this feature exists to avoid.
    """
    try:
        if name == "locally":
            out = subprocess.run(["git", "-C", config.SCRIPT_DIR, "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=5)
            return (out.stdout or "").strip() or None
        if name == "opencode":
            out = subprocess.run(["opencode", "--version"], capture_output=True,
                                 text=True, timeout=10, shell=(os.name == "nt"))
            return (out.stdout or "").strip().split()[0].lstrip("v") or None
        if name in ("odysseus", "searxng"):
            env_name = "ODYSSEUS_DIR" if name == "odysseus" else "SEARXNG_ROOT"
            candidates = [os.environ.get(env_name)]
            if name == "odysseus":
                candidates += [os.path.join(os.path.dirname(config.SCRIPT_DIR), "odysseus"),
                               os.path.join(os.path.expanduser("~"), "odysseus")]
            else:
                candidates += [SEARXNG_ROOT,
                               os.path.join(os.path.dirname(config.SCRIPT_DIR), "searxng-src")]
            for path in candidates:
                if not path or not os.path.isdir(os.path.join(path, ".git")):
                    continue
                out = subprocess.run(["git", "-C", path, "describe", "--tags",
                                      "--always"], capture_output=True, text=True,
                                     timeout=5)
                value = (out.stdout or "").strip().lstrip("v")
                if value:
                    return value
    except Exception:
        pass
    return None


def _version_differs(kind, installed, latest):
    """True only where both values are comparable, never merely different."""
    if not installed or not latest:
        return False
    if kind == "github_commit":
        return not (installed.startswith(latest) or latest.startswith(installed))
    # A dirty/tag-distance git describe is not comparable to a release tag.
    if "-g" in installed:
        return False
    return installed.lstrip("v") != latest.lstrip("v")


def _refresh_updates():
    """Re-check every source. Runs on a background thread; never raises."""
    def one(item):
        name, (kind, locator, page) = item
        latest = _fetch_latest(kind, locator)
        here = _local_version(name)
        return name, {
            "latest": latest, "installed": here, "url": page,
            # Only true when the forms are genuinely comparable. Unknown is
            # "cannot tell", never "out of date".
            "update_available": _version_differs(kind, here, latest),
        }

    try:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(_UPDATE_SOURCES)) as pool:
            sources = dict(pool.map(one, _UPDATE_SOURCES.items()))
    except Exception:
        sources = {}
    finally:
        with _updates_lock:
            _updates_cache["sources"] = sources
            _updates_cache["checked_at"] = time.time()
            _updates_cache["refreshing"] = False


def _schedule_update_refresh(force=False):
    """Start one refresh if needed. Returns whether a worker was started."""
    with _updates_lock:
        age = time.time() - _updates_cache["checked_at"]
        if _updates_cache["refreshing"] or (not force and age <= _UPDATE_TTL):
            return False
        _updates_cache["refreshing"] = True
    threading.Thread(target=_refresh_updates, daemon=True,
                     name="locally-update-check").start()
    return True


@app.route("/v1/updates", methods=["GET"])
def updates():
    """What is newer than what is installed. Cached for a day.

    Never blocks: a stale or empty cache is returned immediately and the
    refresh happens behind it. `?refresh=1` forces one, still asynchronously.
    """
    if not CHECK_UPDATES:
        return jsonify({"enabled": False, "sources": {},
                        "reason": "Update checks are off. Start with "
                                  "--check-updates to enable them."})
    _schedule_update_refresh(force=bool(request.args.get("refresh")))
    with _updates_lock:
        payload = dict(_updates_cache)
    return jsonify({"enabled": True, "checked_at": payload["checked_at"],
                    "refreshing": payload["refreshing"],
                    "sources": payload["sources"]})


# ---------------------------------------------------------------------------
# First-run setup
#
# The setup UI is allowed to download only this small, reviewed catalog. A
# caller cannot supply a repository, URL, filename, destination or command.
# Mutation endpoints are localhost-only because locally deliberately binds to
# the LAN for phone/tablet access, and a page on another device must not be
# able to fill this machine's disk or start an installer.
# ---------------------------------------------------------------------------

_SETUP_VAD_SHA256 = "7776b81ad1b0350c15d7f1555943b9232eb53e9ca5d989c6d0cea9ebc8664d87"
_SETUP_CATALOG = {
    "qwen3-8b-npu": {
        "component": "assistant", "kind": "hf_snapshot",
        "name": "Qwen3 8B", "detail": "Best verified everyday NPU model",
        "repo_id": "OpenVINO/Qwen3-8B-int4-cw-ov",
        "directory": "Qwen3-8B-int4-cw-ov", "size_gb": 5.0,
        "devices": ["NPU", "GPU", "CPU"], "capabilities": ["chat"],
        "license": "Apache-2.0", "verified": True,
    },
    "smollm3-3b": {
        "component": "assistant", "kind": "hf_snapshot",
        "name": "SmolLM3 3B", "detail": "Small, fast, and verified on Intel NPU",
        "repo_id": "aweussom/SmolLM3-3B-int4-cw-ov",
        "directory": "SmolLM3-3B-int4-cw-ov", "size_gb": 2.0,
        "devices": ["NPU", "GPU", "CPU"], "capabilities": ["chat"],
        "license": "See model card", "verified": True,
    },
    "qwen35-4b-vision": {
        "component": "assistant", "kind": "hf_snapshot",
        "name": "Qwen3.5 4B Vision", "detail": "Compact GPU model that can see images",
        "repo_id": "OpenVINO/Qwen3.5-4B-int4-ov",
        "directory": "Qwen3.5-4B-int4-ov", "size_gb": 3.0,
        "devices": ["GPU", "CPU"], "capabilities": ["chat", "vision"],
        "license": "See model card", "verified": True,
    },
    "qwen25-coder-7b": {
        "component": "assistant", "kind": "hf_snapshot",
        "name": "Qwen2.5 Coder 7B", "detail": "Compact agent model with tool calling",
        "repo_id": "OpenVINO/Qwen2.5-Coder-7B-Instruct-int4-ov",
        "directory": "Qwen2.5-Coder-7B-Instruct-int4-ov", "size_gb": 5.0,
        "devices": ["GPU", "CPU"], "capabilities": ["chat", "tools"],
        "license": "Apache-2.0", "verified": True,
    },
    "whisper-base": {
        "component": "stt", "kind": "hf_snapshot",
        "name": "Whisper Base", "detail": "Fastest speech recognition",
        "repo_id": "OpenVINO/whisper-base-int8-ov",
        "directory": "whisper-base-int8-ov", "size_gb": 0.2,
        "devices": ["GPU", "CPU"], "license": "Apache-2.0", "verified": True,
    },
    "whisper-small": {
        "component": "stt", "kind": "hf_snapshot",
        "name": "Whisper Small", "detail": "Recommended accuracy and speed",
        "repo_id": "OpenVINO/whisper-small-int8-ov",
        "directory": "whisper-small-int8-ov", "size_gb": 0.4,
        "devices": ["GPU", "CPU"], "license": "Apache-2.0", "verified": True,
    },
    "kokoro": {
        "component": "tts", "kind": "hf_snapshot",
        "name": "Kokoro", "detail": "Natural local voice, 54 speakers",
        "repo_id": "OpenVINO/Kokoro-82M-int8-ov",
        "directory": "Kokoro-82M-int8-ov", "size_gb": 0.16,
        "devices": ["CPU"], "license": "Apache-2.0", "verified": True,
    },
    "silero-vad": {
        "component": "vad", "kind": "url_file",
        "name": "Silero turn-taking", "detail": "Detects speech instead of room volume",
        "url": ("https://raw.githubusercontent.com/snakers4/silero-vad/master/"
                "src/silero_vad/data/silero_vad_openvino_16k.onnx"),
        "filename": "silero_vad_openvino_16k.onnx",
        "sha256": _SETUP_VAD_SHA256, "directory": "silero-vad",
        "size_gb": 0.002, "devices": ["CPU"],
        "license": "MIT", "verified": True,
    },
    "wespeaker": {
        "component": "speaker", "kind": "hf_file",
        "name": "WeSpeaker identity", "detail": "Stops other voices taking your turn",
        "repo_id": "Wespeaker/wespeaker-ecapa-tdnn512-LM",
        "filename": "voxceleb_ECAPA512_LM.onnx", "directory": "speaker",
        "size_gb": 0.03, "devices": ["CPU"],
        "license": "CC-BY-4.0", "verified": True,
    },
    "searxng": {
        "component": "web", "kind": "searxng",
        "name": "SearXNG search", "detail": "Private metasearch on this machine",
        "directory": "searxng-src", "size_gb": 0.35,
        "devices": [], "license": "AGPL-3.0", "verified": True,
    },
}

_setup_jobs = {}
_setup_jobs_lock = threading.Lock()
_setup_activation_lock = threading.Lock()


def _request_is_local():
    return request.remote_addr in ("127.0.0.1", "::1", "localhost")


def _setup_models_root():
    return os.path.abspath(os.path.expanduser(runtime.MODELS_DIR or "~/models"))


def _setup_searxng_root():
    if SEARXNG_ROOT:
        return os.path.abspath(SEARXNG_ROOT)
    return os.path.abspath(os.path.join(os.path.dirname(config.SCRIPT_DIR), "searxng-src"))


def _setup_destination(entry):
    if entry["kind"] == "searxng":
        return _setup_searxng_root()
    return os.path.join(_setup_models_root(), entry["directory"])


def _setup_entry_installed(entry):
    dest = _setup_destination(entry)
    if entry["kind"] == "searxng":
        return os.path.isfile(os.path.join(dest, "searx", "webapp.py"))
    if entry["kind"] in ("hf_file", "url_file"):
        return os.path.isfile(os.path.join(dest, entry["filename"]))
    if entry["component"] == "assistant":
        return os.path.isdir(dest) and _is_model_dir(dest)
    return os.path.isdir(dest) and any(Path(dest).glob("*.xml"))


def _setup_public_entry(entry_id, entry, recommended=False):
    # The source locator is intentionally visible: setup should be auditable,
    # and every one points to the official project/model owner. The destination
    # cannot be changed by the browser.
    source = entry.get("repo_id") or entry.get("url") or "searxng/searxng"
    return {
        "id": entry_id, "component": entry["component"],
        "name": entry["name"], "detail": entry["detail"],
        "size_gb": entry["size_gb"], "devices": entry.get("devices", []),
        "capabilities": entry.get("capabilities", []),
        "license": entry["license"], "verified": bool(entry.get("verified")),
        "installed": _setup_entry_installed(entry),
        "path": _setup_destination(entry), "source": source,
        "recommended": recommended,
    }


def _setup_recommendations(memory=None):
    memory = memory or _memory_data()
    free_mb = (memory.get("system") or {}).get("available_mb") or 0
    gpu_mb = (memory.get("gpu") or {}).get("usable_mb") or 0
    if "NPU" in runtime.DEVICES and free_mb >= 7 * 1024:
        assistant = "qwen3-8b-npu"
        reason = (f"NPU detected with {free_mb / 1024:.1f} GB of system memory "
                  "available; the verified 5 GB model fits with OS headroom.")
    elif "NPU" in runtime.DEVICES:
        assistant = "smollm3-3b"
        reason = (f"NPU detected, but only {free_mb / 1024:.1f} GB is available "
                  "right now; the 2 GB model leaves safer headroom.")
    elif "GPU" in runtime.DEVICES and gpu_mb >= 4 * 1024:
        assistant = "qwen35-4b-vision"
        reason = (f"The GPU can currently back {gpu_mb / 1024:.1f} GB; the "
                  "3 GB vision model fits without memory pressure.")
    else:
        assistant = "smollm3-3b"
        reason = "No suitable accelerator budget was reported; use the compact CPU fallback."
    return {
        "assistant": assistant,
        "assistant_reason": reason,
        "voice": ["whisper-small", "kokoro", "silero-vad", "wespeaker"],
    }


def _setup_nvidia():
    """NVIDIA is not an OpenVINO target, but it matters for Ollama routing."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        rows = []
        for line in (out.stdout or "").splitlines():
            name, _, memory = line.rpartition(",")
            if name.strip() and memory.strip().isdigit():
                rows.append({"name": name.strip(), "memory_mb": int(memory.strip())})
        return rows
    except Exception:
        return []


def _setup_ollama():
    # When locally owns :11434 this would only discover itself. An external
    # Ollama forces the compatibility port off during startup, so probe only
    # in that state.
    if OLLAMA_COMPAT_PORT:
        return {"available": False, "models": [], "reason": "locally owns port 11434"}
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags",
                                     headers={"User-Agent": "locally-setup"})
        with urllib.request.urlopen(req, timeout=1.0) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
        models = []
        for item in data.get("models", []):
            name = item.get("name") or item.get("model")
            if name:
                models.append({"name": name, "size": item.get("size")})
        return {"available": True, "models": models,
                "url": "http://127.0.0.1:11434"}
    except Exception:
        return {"available": False, "models": [], "reason": "not running"}


def _setup_config_path():
    override = os.environ.get("LOCALLY_SETUP_CONFIG")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return os.path.join(os.environ["LOCALAPPDATA"], "locally", "setup.json")
    return os.path.join(os.path.expanduser("~/.config"), "locally", "setup.json")


def _read_setup_config():
    try:
        with open(_setup_config_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_setup_config(data):
    path = _setup_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="setup-", suffix=".json",
                                     dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except OSError:
            pass
    return path


@app.route("/v1/setup", methods=["GET"])
def setup_state():
    memory = _memory_data()
    rec = _setup_recommendations(memory)
    catalog = [
        _setup_public_entry(item_id, entry,
                            item_id == rec["assistant"] or item_id in rec["voice"])
        for item_id, entry in _SETUP_CATALOG.items()
    ]
    devices = [
        {"kind": kind, "id": info.get("id", kind), "name": info.get("name", kind)}
        for kind, info in runtime.DEVICES.items()
    ]
    saved = _read_setup_config()
    return jsonify({
        "first_run": not bool(saved), "needs_assistant": overall_status() == "not_configured",
        "devices": devices, "memory": memory, "nvidia": _setup_nvidia(),
        "ollama": _setup_ollama(), "catalog": catalog,
        "recommended": rec, "configured": saved,
        "models_root": _setup_models_root(),
    })


def _setup_download_item(entry):
    dest = _setup_destination(entry)
    os.makedirs(dest, exist_ok=True)
    if entry["kind"] == "hf_snapshot":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is not installed") from exc
        snapshot_download(repo_id=entry["repo_id"], local_dir=dest)
    elif entry["kind"] == "hf_file":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is not installed") from exc
        hf_hub_download(repo_id=entry["repo_id"], filename=entry["filename"],
                        local_dir=dest)
    elif entry["kind"] == "url_file":
        target = os.path.join(dest, entry["filename"])
        fd, temp_path = tempfile.mkstemp(prefix="download-", suffix=".part", dir=dest)
        digest = hashlib.sha256()
        total = 0
        try:
            req = urllib.request.Request(entry["url"],
                                         headers={"User-Agent": "locally-setup"})
            with urllib.request.urlopen(req, timeout=30) as response, os.fdopen(fd, "wb") as out:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 8 * 1024 * 1024:
                        raise RuntimeError("the curated VAD download exceeded 8 MB")
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest().lower() != entry["sha256"].lower():
                raise RuntimeError("Silero VAD checksum did not match the tested file")
            os.replace(temp_path, target)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except OSError:
                pass
    elif entry["kind"] == "searxng":
        if os.name != "nt":
            raise RuntimeError("guided SearXNG installation is currently Windows-only")
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        script = os.path.join(config.SCRIPT_DIR, "scripts", "searxng.ps1")
        if not pwsh or not os.path.isfile(script):
            raise RuntimeError("PowerShell or scripts/searxng.ps1 is missing")
        result = subprocess.run(
            [pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script,
             "-Install", "-SearxRoot", dest],
            capture_output=True, text=True, timeout=1200)
        if result.returncode:
            tail = (result.stderr or result.stdout or "installation failed")[-1200:]
            raise RuntimeError(f"SearXNG installation failed: {tail.strip()}")
    if not _setup_entry_installed(entry):
        raise RuntimeError(f"{entry['name']} downloaded, but its required files are missing")


def _setup_install_worker(job_id, item_ids):
    try:
        for item_id in item_ids:
            entry = _SETUP_CATALOG[item_id]
            with _setup_jobs_lock:
                job = _setup_jobs[job_id]
                job["status"] = "downloading"
                job["current"] = item_id
                job["items"][item_id] = "downloading"
            if not _setup_entry_installed(entry):
                _setup_download_item(entry)
            with _setup_jobs_lock:
                _setup_jobs[job_id]["items"][item_id] = "installed"
        with _setup_jobs_lock:
            job = _setup_jobs[job_id]
            job["status"] = "complete"
            job["current"] = None
            job["finished_at"] = time.time()
    except Exception as exc:
        with _setup_jobs_lock:
            job = _setup_jobs[job_id]
            job["status"] = "error"
            job["error"] = str(exc)
            if job.get("current"):
                job["items"][job["current"]] = "error"
            job["finished_at"] = time.time()


@app.route("/v1/setup/install", methods=["POST"])
def setup_install():
    if not _request_is_local():
        return openai_error("Setup installation is local-only.",
                            "invalid_request_error", 403)
    body = request.get_json(silent=True) or {}
    item_ids = body.get("items") or []
    if not isinstance(item_ids, list) or not item_ids or len(item_ids) > 10:
        return openai_error("'items' must be a non-empty setup item list.")
    item_ids = list(dict.fromkeys(str(value) for value in item_ids))
    unknown = [value for value in item_ids if value not in _SETUP_CATALOG]
    if unknown:
        return openai_error(f"Unknown setup item(s): {', '.join(unknown)}")
    missing = [_SETUP_CATALOG[value] for value in item_ids
               if not _setup_entry_installed(_SETUP_CATALOG[value])]
    if missing:
        root = _setup_models_root()
        os.makedirs(root, exist_ok=True)
        free = shutil.disk_usage(root).free
        # Estimates are model-card rounded figures, so leave 15% download/
        # metadata slack plus 1 GB for the rest of the machine.
        needed = sum(float(entry.get("size_gb") or 0) for entry in missing)
        needed_bytes = int((needed * 1.15 + 1.0) * 2 ** 30)
        if free < needed_bytes:
            return openai_error(
                f"The selected setup needs about {needed:.1f} GB, but only "
                f"{free / 2 ** 30:.1f} GB is free on the models drive.",
                "server_error", 409)
    job_id = uuid.uuid4().hex
    job = {"id": job_id, "status": "queued", "current": None,
           "items": {item_id: "queued" for item_id in item_ids},
           "error": None, "started_at": time.time()}
    with _setup_jobs_lock:
        # Bound the in-memory history; active jobs are never discarded.
        old = sorted((j for j in _setup_jobs.values()
                      if j.get("status") in ("complete", "error")),
                     key=lambda j: j.get("finished_at", 0))
        for stale in old[:-7]:
            _setup_jobs.pop(stale["id"], None)
        _setup_jobs[job_id] = job
    threading.Thread(target=_setup_install_worker, args=(job_id, item_ids),
                     daemon=True, name=f"locally-setup-{job_id[:6]}").start()
    return jsonify(job), 202


@app.route("/v1/setup/install/<job_id>", methods=["GET"])
def setup_install_status(job_id):
    if not _request_is_local():
        return openai_error("Setup installation is local-only.",
                            "invalid_request_error", 403)
    with _setup_jobs_lock:
        job = dict(_setup_jobs.get(job_id) or {})
        if job.get("items"):
            job["items"] = dict(job["items"])
    if not job:
        return openai_error("Unknown setup job.", "invalid_request_error", 404)
    return jsonify(job)


def _runtime_device_id(kind):
    return runtime.DEVICES.get(kind, {}).get("id", kind)


def _setup_activate_audio(item_id):
    entry = _SETUP_CATALOG[item_id]
    component = entry["component"]
    path = _setup_destination(entry)
    if not _setup_entry_installed(entry):
        raise RuntimeError(f"{entry['name']} is not installed")
    current = {"stt": runtime.whisper_slot, "tts": runtime.tts_slot,
               "vad": runtime._vad_slot, "speaker": runtime._speaker_slot}.get(component)
    if (current and os.path.realpath(current.model_dir or "") == os.path.realpath(path)
            and current.status in ("ready", "idle_unloaded")):
        return current

    device = "GPU" if component == "stt" and "GPU" in runtime.DEVICES else "CPU"
    if component == "stt":
        slot = WhisperSlot(device, _runtime_device_id(device))
    elif component == "tts":
        slot = TtsSlot(device, _runtime_device_id(device))
    elif component == "vad":
        slot = VadSlot(device, _runtime_device_id(device))
    elif component == "speaker":
        slot = SpeakerSlot(device, _runtime_device_id(device))
    else:
        raise RuntimeError(f"{entry['name']} is not an audio component")
    slot.device_full = runtime.DEVICES.get(device, {}).get("name", device)

    if current:
        with current.lock:
            current.unload()
        try:
            RUNTIME_SLOTS.remove(current)
        except ValueError:
            pass
    slot.load(path)
    slot.warmup()
    RUNTIME_SLOTS.append(slot)
    if component == "stt":
        runtime.whisper_slot = slot
    elif component == "tts":
        runtime.tts_slot = slot
    elif component == "vad":
        runtime._vad_slot = slot
    else:
        runtime._speaker_slot = slot
    return slot


def _setup_activate_ollama(model):
    state = _setup_ollama()
    known = {entry["name"] for entry in state.get("models", [])}
    if not state.get("available") or model not in known:
        raise RuntimeError("That model is not advertised by the local Ollama server")
    proxy = ProxySlot("http://127.0.0.1:11434", model=model)
    proxy.load()
    proxy.warmup()
    old = runtime.primary
    if old:
        with old.lock:
            old.unload()
        try:
            index = RUNTIME_SLOTS.index(old)
            RUNTIME_SLOTS[index] = proxy
        except ValueError:
            RUNTIME_SLOTS.append(proxy)
    else:
        RUNTIME_SLOTS.append(proxy)
    runtime.primary = proxy
    return proxy


@app.route("/v1/setup/finish", methods=["POST"])
def setup_finish():
    global SEARXNG_ROOT, WEB_SEARCH_URL
    if not _request_is_local():
        return openai_error("Setup changes are local-only.",
                            "invalid_request_error", 403)
    body = request.get_json(silent=True) or {}
    assistant_id = body.get("assistant")
    backend = body.get("backend") or "openvino"
    voice_ids = body.get("voice") or []
    if not isinstance(voice_ids, list):
        return openai_error("'voice' must be a setup item list.")
    try:
        with _setup_activation_lock:
            if backend == "ollama":
                model = str(body.get("ollama_model") or "").strip()
                if not model:
                    raise RuntimeError("Choose an installed Ollama model")
                slot = _setup_activate_ollama(model)
                assistant_cfg = {"backend": "ollama", "url": "http://127.0.0.1:11434",
                                 "model": model}
                assistant_result = {"backend": "ollama", "model": slot.model_name,
                                    "loaded": True}
            else:
                if assistant_id not in _SETUP_CATALOG or \
                        _SETUP_CATALOG[assistant_id]["component"] != "assistant":
                    raise RuntimeError("Choose a curated assistant model")
                entry = _SETUP_CATALOG[assistant_id]
                if not _setup_entry_installed(entry):
                    raise RuntimeError(f"{entry['name']} is not installed")
                path = _setup_destination(entry)
                requested = str(body.get("device") or "").upper()
                compatible = [d for d in entry.get("devices", []) if d in runtime.DEVICES]
                device = requested if requested in compatible else (compatible[0] if compatible else "CPU")
                assistant_cfg = {"backend": "openvino", "model_dir": path,
                                 "device": device}
                loaded = bool(runtime.primary and runtime.primary.model_dir and
                              os.path.realpath(runtime.primary.model_dir) == os.path.realpath(path)
                              and _slot_serviceable(runtime.primary))
                assistant_result = {"backend": "openvino", "model": entry["name"],
                                    "path": path, "device": device, "loaded": loaded}

            audio_cfg, audio_result = {}, {}
            for item_id in dict.fromkeys(str(value) for value in voice_ids):
                entry = _SETUP_CATALOG.get(item_id)
                if not entry or entry["component"] not in ("stt", "tts", "vad", "speaker"):
                    raise RuntimeError(f"Unknown voice setup item: {item_id}")
                slot = _setup_activate_audio(item_id)
                key = f"{entry['component']}_dir"
                audio_cfg[key] = _setup_destination(entry)
                audio_result[entry["component"]] = slot.info

            web_cfg = {"searxng": bool(body.get("searxng")),
                       "check_updates": bool(body.get("check_updates"))}
            if web_cfg["searxng"]:
                entry = _SETUP_CATALOG["searxng"]
                if not _setup_entry_installed(entry):
                    raise RuntimeError("SearXNG is not installed")
                SEARXNG_ROOT = _setup_destination(entry)
                WEB_SEARCH_URL = "http://127.0.0.1:8080"
                web_cfg["root"] = SEARXNG_ROOT
                web_cfg["url"] = WEB_SEARCH_URL

            setup_cfg = {"version": 1, "assistant": assistant_cfg,
                         "voice": audio_cfg, "web": web_cfg,
                         "completed_at": datetime.now().isoformat(timespec="seconds")}
            config_path = _write_setup_config(setup_cfg)
    except Exception as exc:
        return openai_error(str(exc), "server_error", 500)
    return jsonify({"status": "ok", "assistant": assistant_result,
                    "audio": audio_result, "web": web_cfg,
                    "config_path": config_path})


@app.route("/health", methods=["GET"])
def health():
    return jsonify(_health_data())


@app.route("/v1/memory", methods=["GET"])
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
    if available is not None and available < _OS_RESERVE_BYTES:
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
            "reserve_mb": round(_OS_RESERVE_BYTES / mib),
        },
        "slots": slots,
    }


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify(_models_data())


def _models_data():
    data = []
    for slot in (runtime.primary, runtime.secondary):
        if slot and slot.status == "ready":
            data.append({
                "id": f"{slot.model_name}@{slot.device_name}",
                "object": "model",
                "created": 0,
                "owned_by": f"local-{slot.device_name.lower()}",
            })
    if runtime.whisper_slot and runtime.whisper_slot.status == "ready":
        data.append({
            "id": f"whisper@{runtime.whisper_slot.device_name}",
            "object": "model",
            "created": 0,
            "owned_by": f"local-{runtime.whisper_slot.device_name.lower()}",
        })
    return {"object": "list", "data": data}


def _available_models():
    """Model directories on disk that could be loaded into a slot.

    Searched: --models-dir if given, else the parents of whatever is already
    loaded (so `--model-dir ~/models/foo` makes all of ~/models visible)
    plus the script dir and ~/models. One model reached by two paths is one
    model — install.ps1 links model/ at a directory in ~/models.
    """
    roots = []
    if runtime.MODELS_DIR:
        roots.append(runtime.MODELS_DIR)
    else:
        for slot in (runtime.primary, runtime.secondary):
            if slot and slot.model_dir:
                roots.append(os.path.dirname(os.path.abspath(slot.model_dir)))
        roots += [config.SCRIPT_DIR, os.path.expanduser("~/models")]

    seen, out = set(), []
    for root in roots:
        for d in _model_dirs_under(os.path.normpath(os.path.expanduser(root)), 1):
            real = os.path.realpath(d)
            if real in seen or not _is_generative_dir(d):
                continue
            seen.add(real)
            out.append({"name": model_display_name(d), "path": d,
                        "type": "vlm" if is_vlm(d) else "llm"})
    return sorted(out, key=lambda m: m["name"].lower())


def _device_can_host(device_name, device_id, model_dir, vlm):
    """(ok, reason) — whether this device can serve this model at all.

    Capability first, memory second: a model that the NPU cannot execute is
    not "a tight fit", it's the wrong device, and saying so beats letting the
    driver fail ten minutes later.
    """
    if device_name == "NPU":
        if vlm:
            return False, "NPU has no working vision path"
        # Group-quantized int4 crashes the NPU driver compiler ("Found N
        # duplicated names") — channel-wise is required. Read what the
        # weights actually are, not what the folder is called.
        try:
            rt = read_ir_rt_info(model_dir)
            gs = rt.get("nncf/weight_compression/group_size")
            mode = (rt.get("nncf/weight_compression/mode") or "")
            if "int4" in mode and gs not in (None, "", "-1", -1):
                return False, f"NPU needs channel-wise int4 (this is group_size {gs})"
        except Exception:
            pass

    mem = _device_mem_bytes(device_name, device_id)
    weights = _dir_size_bytes(model_dir)
    if not mem or not weights:
        # The NPU exposes no memory-budget property — it allocates from system
        # RAM on demand — so "unknown" here is normal, not a warning sign.
        return True, ("no budget reported (allocates from system RAM)"
                      if device_name == "NPU" else "fit unknown")
    kv = (config.PROMPT_CACHE_GB * 2 ** 30
          if config.PROMPT_CACHE and not vlm and device_name in ("GPU", "CPU") else 0)
    need = (weights + kv) * 1.1
    if need > mem:
        # MoE disk offload keeps only part of the experts resident, so a model
        # over budget can still fit — don't rule the device out on size alone.
        # Only where offload actually does something, though: a dense model or
        # a non-XMX GPU would sail past this check and then fail to load.
        if (config.OFFLOAD_RATIO and device_name == "GPU"
                and _moe_expert_fraction(model_dir) and _gpu_has_xmx(device_id)):
            return True, ("over budget, fits by streaming expert weights"
                          if config.OFFLOAD_RATIO == "auto" else
                          f"over budget but --offload-ratio {config.OFFLOAD_RATIO} is on")
        return False, (f"needs ~{need / 2**30:.1f} GB, "
                       f"{device_name} budget is {mem / 2**30:.1f} GB")
    return True, f"fits ({need / 2**30:.1f}/{mem / 2**30:.1f} GB)"


def _choose_device(model_dir, preferred=None):
    """Pick which DEVICE should host a model. Returns (name, id, why).

    Considers every device on the machine, not only ones that already hold a
    slot — otherwise a single-slot ("one model at a time") setup could never
    use the NPU, and the one model the NPU can actually run would be stuck on
    the GPU. The slot is moved to the chosen device by the caller.
    """
    vlm = is_vlm(model_dir)
    # CPU is a legal target but a poor default (see TODONT.md), so it's only
    # considered when nothing else can host the model.
    order = [k for k in ("GPU", "NPU", "CPU") if k in runtime.DEVICES]
    notes = []

    def dev_id(kind):
        return runtime.DEVICES.get(kind, {}).get("id", kind)

    if preferred:
        if preferred not in runtime.DEVICES:
            return None, None, f"no device '{preferred}' on this machine"
        ok, why = _device_can_host(preferred, dev_id(preferred), model_dir, vlm)
        if ok:
            return preferred, dev_id(preferred), why
        notes.append(f"{preferred}: {why}")

    # NPU first when it can actually run the model. locally is NPU-first by
    # design, and the NPU is the low-power engine — a text model it can host
    # belongs there, leaving the GPU free. _device_can_host has already ruled
    # the NPU out for vision and group-quantized int4, so anything reaching
    # here genuinely runs. GPU next (it does vision and tool calling), CPU
    # last (see TODONT.md — Ollama is the better tool for CPU-only).
    rank = {"NPU": 0, "GPU": 1, "CPU": 2}

    for kind in sorted(order, key=lambda k: rank.get(k, 9)):
        if preferred and kind == preferred:
            continue
        ok, why = _device_can_host(kind, dev_id(kind), model_dir, vlm)
        if ok:
            if not notes:
                prefix = ""
            elif preferred:
                prefix = f"moved from {preferred} ({'; '.join(notes)}); "
            else:
                # No device was asked for — the notes explain what was skipped
                # on the way here, which is the useful part.
                prefix = f"ruled out {'; '.join(notes)}; "
            return kind, dev_id(kind), f"{prefix}{kind}: {why}"
        notes.append(f"{kind}: {why}")

    return None, None, "; ".join(notes) or "no device can host this model"


@app.route("/v1/models/available", methods=["GET"])
def list_available_models():
    return jsonify(_available_models_data())


def _available_models_data():
    """Models on disk, loaded or not — the menu for POST /v1/models/load."""
    loaded = {}
    for slot in (runtime.primary, runtime.secondary):
        if slot and slot.model_dir:
            loaded[os.path.realpath(slot.model_dir)] = slot.device_name
    data = []
    for m in _available_models():
        entry = dict(m)
        entry["loaded_on"] = loaded.get(os.path.realpath(m["path"]))
        data.append(entry)
    return {"object": "list", "data": data,
            "devices": [s.device_name for s in (runtime.primary, runtime.secondary) if s]}


@app.route("/v1/ui/bootstrap", methods=["GET"])
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


@app.route("/v1/models/load", methods=["POST"])
def load_model():
    """Swap the model in a slot without restarting the server.

    Ollama-style one-at-a-time: the slot's current model is unloaded before
    the new one is loaded, so peak memory is one model, not two. Synchronous
    — it returns when the model is ready to serve, which for a big IR can be
    a minute; that's honest about what's happening rather than reporting
    success on a model that can't answer yet.
    """
    body = request.get_json(silent=True) or {}
    name = (body.get("model") or body.get("name") or "").strip()
    if not name:
        return openai_error("'model' is required (name or directory path)")

    # Accept "name@DEVICE" like the rest of the API, an exact directory, or a
    # display name from /v1/models/available.
    device = (body.get("device") or "").upper()
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
        return openai_error(f"Unknown model '{name}'. Available: {known}")

    # Pick the slot automatically. An explicit device is honoured when that
    # device can actually host the model, and otherwise treated as a
    # preference rather than an order — a request that would OOM or hit the
    # NPU's vision/quantization limits gets placed where it can run instead
    # of failing.
    if device in ("AUTO", "ANY"):
        device = ""
    dev_name, dev_id, why = _choose_device(target, device or None)
    if dev_name is None:
        return openai_error(f"No device can host '{name}' ({why}).")
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
        return openai_error("That slot is already loading a model.",
                            "server_error", 409)

    err = _verify_weights_integrity(target)
    if err:
        return openai_error(err)

    # Hold the lock across unload+load so a concurrent request can't reach a
    # half-swapped slot. A generation in flight keeps the lock, so we wait
    # for it rather than yanking the pipeline out from under it.
    t0 = time.perf_counter()
    print(f"\n{datetime.now():%H:%M:%S} <> [{slot.device_name}] Swap "
          f"{slot.model_name or '(empty)'} -> {model_display_name(target)}"
          + (f" (moving {moved_from} -> {dev_name})" if moved_from else ""), flush=True)
    with slot.lock:
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
                time.sleep(1.5 * attempt)
        if last_err is not None:
            slot.status = "error"
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
                  f"Swap failed: {last_err}", flush=True)
            return openai_error(
                f"Failed to load '{name}': {explain_genai_error(last_err)}",
                "server_error", 500)
    # warmup() takes the lock itself, so it must run after the block above.
    slot.warmup()
    slot.prewarmed = False
    slot.last_ttft_ms = None
    elapsed = time.perf_counter() - t0
    print(f"{datetime.now():%H:%M:%S} <> [{slot.device_name}] Swap done "
          f"({elapsed:.1f}s)", flush=True)

    return jsonify({"status": "ok", "model": slot.model_name,
                    "device": slot.device_name, "type": slot.model_type,
                    "placement": placement,
                    "load_seconds": round(elapsed, 1)})


@app.route("/v1/models/unload", methods=["POST"])
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
    want = (body.get("device") or "").upper()

    targets = [s for s in (runtime.primary, runtime.secondary, runtime.whisper_slot, runtime.tts_slot,
                           runtime.util_npu, runtime.util_gpu, runtime.util_cpu)
               if s and s.status not in ("not_configured", "idle_unloaded")]
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


@app.route("/v1/cancel", methods=["POST"])
def cancel_generation():
    """Stop any in-progress generation. Returns immediately."""
    for slot in (runtime.primary, runtime.secondary):
        if slot:
            slot.cancel()
    return jsonify({"status": "ok"})


def _register_audio_stream():
    """Wire up the VAD socket, if flask-sock is installed.

    Optional on purpose: the socket only carries auto turn-taking, and voice
    already degrades to push-to-talk without it. A missing dependency should
    cost that feature, not the server.
    """
    try:
        from flask_sock import Sock
    except ImportError:
        return False

    sock = Sock(app)

    @sock.route("/v1/audio/stream")
    def audio_stream(ws):                       # noqa: ANN001  (flask-sock API)
        if not runtime._vad_slot or not _slot_serviceable(runtime._vad_slot):
            ws.send(json.dumps({"type": "error", "message": "No VAD model loaded."}))
            return
        try:
            runtime._vad_slot.ensure_loaded()
        except Exception as e:
            ws.send(json.dumps({"type": "error", "message": f"VAD load failed: {e}"}))
            return

        session = VoiceStreamSession()
        while True:
            msg = ws.receive()
            if msg is None:
                break
            if isinstance(msg, (bytes, bytearray)):
                pcm = np.frombuffer(msg, dtype="<i2").astype(np.float32) / 32768.0
                for i in range(0, len(pcm) - VadSlot.FRAME + 1, VadSlot.FRAME):
                    for ev in session.feed(pcm[i:i + VadSlot.FRAME]):
                        ws.send(json.dumps(ev))
                # Transcription finishes on a worker thread; it is handed back
                # here so every frame on this socket has exactly one writer.
                for ev in session.drain():
                    ws.send(json.dumps(ev))
                continue
            try:
                cfg = json.loads(msg)
            except (ValueError, TypeError):
                continue
            if cfg.get("type") == "config":
                session.threshold = float(cfg.get("threshold", session.threshold))
                session.patience_ms = float(cfg.get("patienceMs", session.patience_ms))
                session.language = cfg.get("language") or ""
                session.enabled = bool(cfg.get("enabled", True))
            elif cfg.get("type") == "state":
                session.voice_state = cfg.get("voice", "idle")
                # Leaving the answer behind clears any half-built interruption.
                if session.voice_state != "speaking":
                    session.speech_ms = 0.0
        return ""

    return True


AUDIO_STREAM_OK = _register_audio_stream()


@app.route("/v1/audio/warm", methods=["POST"])
def audio_warm():
    """Bring the audio models resident before the first turn needs them.

    Idle-unload is doing its job when it evicts Whisper and the TTS model, but
    the cost of the reload (measured ~5.9 s ASR, ~8.7 s TTS) then lands inside
    the user's first spoken turn, where it reads as "voice mode is slow". The
    web UI calls this when the Voice tab is opened, which is the earliest
    moment we know a turn is coming.

    Best-effort: a slot that fails to load is reported, not raised, because a
    warm-up failing must not stop someone from trying to talk.
    """
    blocked = _coding_mode_error("the audio models")
    if blocked is not None:
        return blocked
    out = {}
    for key, slot in (("whisper", runtime.whisper_slot), ("tts", runtime.tts_slot),
                      ("vad", runtime._vad_slot)):
        if not slot or not _slot_serviceable(slot):
            out[key] = "unavailable"
            continue
        t0 = time.perf_counter()
        try:
            slot.ensure_loaded()
            out[key] = {"ready": True, "ms": round((time.perf_counter() - t0) * 1000)}
        except Exception as e:
            out[key] = {"ready": False, "error": str(e)}
    return jsonify(out)


@app.route("/v1/audio/enroll", methods=["GET", "POST", "DELETE"])
def audio_enroll():
    """Manage the enrolled voice profile.

    GET returns the profile's state, POST folds one more utterance into it,
    DELETE forgets it. Enrolment is additive on purpose: a profile built from
    several short recordings, in whatever voice the room actually produces,
    generalises better than one careful reading. The client is told how many
    more seconds it needs rather than being given a pass/fail, so the UI can
    show progress instead of a mystery.
    """
    if not runtime._speaker_slot or runtime._speaker_slot.status == "not_configured":
        return openai_error(
            "No speaker-identification model loaded. Use --speaker-dir.",
            "server_error", 503)

    if request.method == "GET":
        return jsonify(runtime._speaker_slot.profile_state)

    if request.method == "DELETE":
        runtime._speaker_slot.clear_profile()
        print(f"{datetime.now():%H:%M:%S} -- [speaker] profile cleared", flush=True)
        return jsonify(runtime._speaker_slot.profile_state)

    try:
        runtime._speaker_slot.ensure_loaded()
    except Exception as e:
        return openai_error(f"Failed to load speaker model: {e}", "server_error", 500)

    if "file" not in request.files:
        return openai_error("'file' is required (multipart form upload)")
    try:
        audio = _load_audio(request.files["file"])
    except Exception as e:
        return openai_error(f"Could not read audio: {e}")

    try:
        state = runtime._speaker_slot.enroll(audio)
    except ValueError as e:
        return openai_error(str(e))
    except Exception as e:
        return openai_error(f"Enrolment failed: {e}", "server_error", 500)

    print(f"{datetime.now():%H:%M:%S} -- [speaker] enrolled "
          f"{len(audio) / SpeakerSlot.SR:.1f}s "
          f"({state['seconds']:.0f}/{state['needed_seconds']:.0f}s, "
          f"{'ready' if state['ready'] else 'more needed'})", flush=True)
    return jsonify(state)


@app.route("/v1/audio/verify", methods=["POST"])
def audio_verify():
    """Score one recording against the enrolled profile, without gating anything.

    Exists so the threshold can be set from evidence. Record yourself, record
    someone else, and the gap between the two cosines is what
    --speaker-threshold should sit in. Guessing a threshold and discovering it
    was wrong through a week of turns being dropped is the alternative.
    """
    if not runtime._speaker_slot or runtime._speaker_slot.status == "not_configured":
        return openai_error(
            "No speaker-identification model loaded. Use --speaker-dir.",
            "server_error", 503)
    if "file" not in request.files:
        return openai_error("'file' is required (multipart form upload)")
    try:
        runtime._speaker_slot.ensure_loaded()
        audio = _load_audio(request.files["file"])
    except Exception as e:
        return openai_error(f"Could not read audio: {e}")
    verdict, score = runtime._speaker_slot.verify(audio)
    return jsonify({
        "verdict": verdict,
        "score": round(score, 4) if score is not None else None,
        "threshold": runtime._speaker_slot.threshold,
        "seconds": round(len(audio) / SpeakerSlot.SR, 2),
        # Says WHY when there is no score, which is the difference between
        # "this is a stranger" and "I could not tell".
        "reason": (None if verdict != "abstain" else
                   ("no profile enrolled" if not runtime._speaker_slot.profile_ready
                    else f"needs at least {SpeakerSlot.MIN_VERIFY_S}s of audio")),
    })


@app.route("/v1/audio/transcriptions", methods=["POST"])
def audio_transcriptions():
    """OpenAI-compatible speech-to-text. Accepts multipart form with audio file."""
    blocked = _coding_mode_error("speech-to-text")
    if blocked is not None:
        return blocked
    if not runtime.whisper_slot or not _slot_serviceable(runtime.whisper_slot):
        return openai_error(
            "No speech-to-text model loaded. Use --whisper-dir.", "server_error", 503,
        )
    try:
        runtime.whisper_slot.ensure_loaded()   # reload if idle-unloaded or unloaded on request
    except Exception as e:
        return openai_error(f"Failed to load speech-to-text model: {e}",
                            "server_error", 500)

    if "file" not in request.files:
        return openai_error("'file' is required (multipart form upload)")

    audio_file = request.files["file"]
    language = request.form.get("language")
    response_format = request.form.get("response_format", "json")

    try:
        audio_samples = _load_audio(audio_file)
    except Exception as e:
        return openai_error(f"Failed to read audio: {e}")

    duration = len(audio_samples) / 16000
    lang_tag = f", lang={language}" if language else ""
    print(f"\n{datetime.now():%H:%M:%S} <- [{runtime.whisper_slot.device_name}] "
          f"Whisper {duration:.1f}s audio{lang_tag}", flush=True)

    t0 = time.perf_counter()
    try:
        text = runtime.whisper_slot.transcribe(audio_samples, language=language)
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} !! [{runtime.whisper_slot.device_name}] "
              f"Whisper error: {e}", flush=True)
        return openai_error(f"Transcription failed: {e}", "server_error", 500)

    elapsed = time.perf_counter() - t0
    print(f"{datetime.now():%H:%M:%S} -> [{runtime.whisper_slot.device_name}] "
          f"Whisper {len(text)} chars in {elapsed:.1f}s", flush=True)

    if response_format == "text":
        return Response(text, mimetype="text/plain")

    return jsonify({"text": text})


@app.route("/v1/audio/speech", methods=["POST"])
def audio_speech():
    """OpenAI-compatible text-to-speech. Returns a WAV body.

    No streaming: openvino-genai synthesizes the whole utterance before
    returning, so this responds once with the complete waveform.
    """
    if not runtime.tts_slot or not _slot_serviceable(runtime.tts_slot):
        return openai_error(
            "No text-to-speech model loaded. Use --tts-dir.", "server_error", 503,
        )

    blocked = _coding_mode_error("text-to-speech")
    if blocked is not None:
        return blocked
    body = request.get_json(silent=True) or {}
    text = (body.get("input") or "").strip()
    if not text:
        return openai_error("'input' is required")

    fmt = (body.get("response_format") or "wav").lower()
    if fmt not in ("wav", "pcm"):
        return openai_error(
            f"response_format '{fmt}' not supported (wav or pcm only — "
            "no encoder is bundled)."
        )

    print(f"\n{datetime.now():%H:%M:%S} <- [{runtime.tts_slot.device_name}] "
          f"TTS {len(text)} chars", flush=True)

    try:
        runtime.tts_slot.ensure_loaded()
        audio, rate = runtime.tts_slot.synthesize(text, voice=(body.get("voice") or None))
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} !! [{runtime.tts_slot.device_name}] "
              f"TTS error: {e}", flush=True)
        return openai_error(f"Speech synthesis failed: {e}", "server_error", 500)

    duration = len(audio) / rate if rate else 0
    print(f"{datetime.now():%H:%M:%S} -> [{runtime.tts_slot.device_name}] "
          f"TTS {duration:.1f}s audio in {runtime.tts_slot.last_synth_ms / 1000:.1f}s "
          f"({duration / max(runtime.tts_slot.last_synth_ms / 1000, 1e-6):.1f}x realtime)",
          flush=True)

    if fmt == "pcm":
        pcm = np.clip(audio, -1.0, 1.0)
        return Response((pcm * 32767).astype("<i2").tobytes(),
                        mimetype="audio/pcm",
                        headers={"X-Sample-Rate": str(rate),
                                 "X-Device": runtime.tts_slot.device_name})

    if sf is None:
        return openai_error("soundfile not installed. pip install soundfile",
                            "server_error", 500)
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), mimetype="audio/wav",
                    headers={"X-Sample-Rate": str(rate),
                             "X-Device": runtime.tts_slot.device_name})


# ---------------------------------------------------------------------------
# Utilities (OCR, native documents, images, and local semantic search)
# ---------------------------------------------------------------------------

# URL reading. A search result is nav bar, cookie banner and footer wrapped
# around a few paragraphs, and on the NPU that soup is spent out of a 8192-token
# window that the article itself needs. Fetching is I/O, so 20 s is a slow site
# rather than a hung one; 8 MB is far past any article and well short of a file
# somebody linked by accident.
_UTIL_URL_TIMEOUT_S = 20
_UTIL_URL_MAX_BYTES = 8 * 1024 * 1024
_UTIL_URL_MAX_REDIRECTS = 5
# ~50k tokens: deliberately larger than MAX_PROMPT_LEN, because this reader also
# feeds GPU slots and the Util tab's copy-out, and truncating every page to the
# NPU's window would throw away text the caller may have wanted. It is a bound
# on the JSON we will build, not a promise that the result fits a prompt.
_UTIL_URL_MAX_CHARS = 200_000
_markitdown_instance = None
_util_search_indexes = {}
_util_search_lock = threading.Lock()
_UTIL_SEARCH_MAX_INDEXES = 8
_UTIL_SEARCH_MAX_FILES = 20


def _url_result(url):
    """Fetch one public page and return it in the document response shape.

    No accelerator touches this — it is an HTTP request and an HTML-to-Markdown
    conversion — so `engine` is None and the header says LOCAL, exactly as the
    MarkItDown path does. Claiming a device that did not run is the one thing
    this endpoint's callers cannot check for themselves.
    """
    creq = _http_fetcher()
    _guard_public_url(url)

    t0 = time.perf_counter()
    try:
        # impersonate gives a real Chrome TLS/JA3 fingerprint, which is the
        # difference between a page answering and refusing python-requests.
        page = creq.get(url, timeout=_UTIL_URL_TIMEOUT_S,
                        impersonate="chrome", allow_redirects=True,
                        max_redirects=_UTIL_URL_MAX_REDIRECTS)
    except _TurnError:
        raise
    except Exception as e:
        raise _TurnError(openai_error(f"Failed to fetch {url}: {e}"))
    fetch_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Re-check where we actually landed. A public hostname that 302s to
    # 127.0.0.1 would otherwise smuggle a LAN service's response back out.
    for hop in list(getattr(page, "history", None) or []) + [page]:
        hop_url = getattr(hop, "url", hop)
        if hop_url and str(hop_url) != url:
            _guard_public_url(str(hop_url))

    status = int(getattr(page, "status_code", 0) or 0)
    if status >= 400:
        raise _TurnError(openai_error(
            f"{url} returned HTTP {status} {getattr(page, 'reason', '') or ''}".strip()))

    content_type = (page.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type and not (content_type.startswith("text/")
                             or content_type.endswith(("html", "xml"))):
        raise _TurnError(openai_error(
            f"{url} served '{content_type}', which is not a web page. Download "
            f"it and send it as a file — /v1/util/read reads PDFs and Office "
            f"documents from an upload."))

    body = getattr(page, "content", b"") or b""
    size = len(body if isinstance(body, (bytes, bytearray))
               else body.encode("utf-8", "ignore"))
    if size > _UTIL_URL_MAX_BYTES:
        raise _TurnError(openai_error(
            f"{url} is {size / 1e6:.1f} MB, over the {_UTIL_URL_MAX_BYTES / 1e6:.0f} MB "
            f"page limit."))

    try:
        markdown = _html_to_markdown(page.text)
    except _TurnError:
        raise
    except Exception as e:
        raise _TurnError(openai_error(f"Failed to convert {url} to Markdown: {e}",
                                      "server_error", 500))

    truncated = len(markdown) > _UTIL_URL_MAX_CHARS
    if truncated:
        markdown = markdown[:_UTIL_URL_MAX_CHARS].rstrip() + "\n\n*[truncated]*"

    return {
        "text": markdown, "markdown": markdown, "blocks": [],
        "regions": 0, "dropped_low_conf": 0,
        "source": "url", "engine": None,
        "timings_ms": {"fetch": fetch_ms},
        "pages_total": None, "pages_processed": None, "truncated": truncated,
        "url": str(getattr(page, "url", url)), "status": status,
        "bytes": size,
    }


@app.route("/v1/util/read", methods=["POST"])
def util_read():
    """Extract Markdown from an image, a local document, or a public web page.

    Images and scanned PDF pages use OCR on the requested engine (NPU default).
    Text-bearing Office/PDF/HTML/EPUB/tabular files use a native parser —
    inference would be slower and less accurate than reading their embedded
    XML/text. A `url` is fetched over plain HTTP and converted to Markdown, so
    a page can be handed to a model as prose instead of as tag soup. User
    uploads stay in memory; nothing here is written to disk.
    """
    body = request.get_json(silent=True) or {}
    engine = request.form.get("engine") or body.get("engine") or "npu"
    if engine.lower() not in ("npu", "gpu", "cpu"):
        return openai_error(f"Unknown engine '{engine}'. Use 'npu', 'gpu' or 'cpu'.")

    url = str(request.form.get("url") or body.get("url") or "").strip()
    if url:
        print(f"\n{datetime.now():%H:%M:%S} <- [LOCAL] util/read {url}", flush=True)
        try:
            result = _url_result(url)
        except _TurnError as e:
            return e.response
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [LOCAL] util/read error: {e}",
                  flush=True)
            return openai_error(f"Read failed: {e}", "server_error", 500)
        print(f"{datetime.now():%H:%M:%S} -> [LOCAL] util/read "
              f"{len(result['markdown'])} chars from {result['bytes']} bytes "
              f"({result['timings_ms']['fetch']}ms)", flush=True)
        return jsonify(result), 200, {"X-Device": "LOCAL",
                                      "X-Document-Source": result["source"]}

    try:
        item = _util_input_from_request()
    except _TurnError as e:
        return e.response

    if item["kind"] == "document":
        print(f"\n{datetime.now():%H:%M:%S} <- [LOCAL] util/read "
              f"{item['filename']}", flush=True)
        try:
            result = _document_result(item, engine)
        except _TurnError as e:
            return e.response
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [LOCAL] util/read error: {e}",
                  flush=True)
            return openai_error(f"Read failed: {e}", "server_error", 500)
        print(f"{datetime.now():%H:%M:%S} -> [LOCAL] util/read "
              f"{len(result['markdown'])} chars via {result['source']}", flush=True)
        headers = {"X-Device": (result["engine"] or "LOCAL").upper(),
                   "X-Document-Source": result["source"]}
        return jsonify(result), 200, headers

    try:
        slot = _util_slot(engine)
        img = item["image"]
    except _TurnError as e:
        return e.response
    try:
        slot.ensure_loaded()
    except Exception as e:
        return openai_error(f"Failed to load utility models: {e}",
                            "server_error", 500)

    print(f"\n{datetime.now():%H:%M:%S} <- [{slot.device_name}] "
          f"util/read {img.width}x{img.height}", flush=True)
    try:
        result = slot.read(img)
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
              f"util/read error: {e}", flush=True)
        return openai_error(f"Read failed: {e}", "server_error", 500)

    # Structure the flat OCR reading when a layout model is installed. This is
    # strictly additive: any failure leaves the caller with exactly the text
    # they would have had, because losing a correct OCR result to a layout
    # model's bad day is a far worse trade than returning unstructured text.
    layout_regions = []
    try:
        layout_regions, layout_timings = slot.run_utility("layout", img)
        structured = _layout_markdown(result["blocks"], layout_regions)
        if structured:
            result["markdown"] = structured
        result["timings_ms"]["layout"] = layout_timings["infer"]
    except UtilityUnavailable:
        pass                       # not installed on this engine; fine
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
              f"layout failed, returning unstructured text: {e}", flush=True)

    t = result["timings_ms"]
    layout_note = (f", layout {len(layout_regions)} regions "
                   f"({t['layout']}ms)" if layout_regions else "")
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
          f"util/read {len(result['text'])} chars, {result['regions']} regions "
          f"(detect {t['detect']}ms, read {t['recognise']}ms)"
          f"{layout_note}", flush=True)

    return jsonify({
        "text": result["text"],
        "markdown": result["markdown"],
        "blocks": result["blocks"],
        "regions": result["regions"],
        # Labelled page regions in reading order, when a layout model ran.
        "layout": layout_regions,
        "dropped_low_conf": result["dropped_low_conf"],
        "source": "ocr",
        "engine": slot.device_name.lower(),
        "timings_ms": t,
        "pages_total": 1,
        "pages_processed": 1,
        "truncated": False,
    }), 200, {"X-Device": slot.device_name}


def _util_image_turn(operation):
    """Resolve an image utility request to a ready slot and PIL image."""
    body = request.get_json(silent=True) or {}
    engine = request.form.get("engine") or body.get("engine") or "npu"
    item = _util_input_from_request()
    if item["kind"] != "image":
        raise _TurnError(openai_error(
            f"{operation} accepts PNG, JPEG, WebP, GIF, BMP, or TIFF images."))
    slot = _util_slot(engine)
    try:
        slot.ensure_loaded()
    except Exception as e:
        raise _TurnError(openai_error(
            f"Failed to load utility models: {e}", "server_error", 500))
    return slot, item["image"]


@app.route("/v1/util/background", methods=["POST"])
def util_background():
    """Return a transparent PNG with the foreground retained."""
    try:
        slot, image = _util_image_turn("Background removal")
        result, timings = slot.run_utility("remove_background", image)
    except _TurnError as e:
        return e.response
    except UtilityUnavailable as e:
        # Deliberately unavailable is 503, not 500 — see _require().
        return openai_error(str(e), "server_error", 503)
    except Exception as e:
        return openai_error(f"Background removal failed: {e}", "server_error", 500)
    return jsonify({
        "image": _pil_data_url(result), "engine": slot.device_name.lower(),
        "width": result.width, "height": result.height,
        "timings_ms": timings,
    }), 200, {"X-Device": slot.device_name}


@app.route("/v1/util/upscale", methods=["POST"])
def util_upscale():
    """Upscale an image through the static 3x Open Model Zoo network."""
    try:
        slot, image = _util_image_turn("Upscaling")
        result, details = slot.run_utility("upscale", image)
    except _TurnError as e:
        return e.response
    except UtilityUnavailable as e:
        # Deliberately unavailable is 503, not 500 — see _require().
        return openai_error(str(e), "server_error", 503)
    except Exception as e:
        return openai_error(f"Upscale failed: {e}", "server_error", 500)
    timings = {"infer": details.pop("infer")}
    return jsonify({
        "image": _pil_data_url(result), "engine": slot.device_name.lower(),
        "timings_ms": timings, **details,
    }), 200, {"X-Device": slot.device_name}


@app.route("/v1/util/detect", methods=["POST"])
def util_detect():
    """Detect common COCO objects and return an annotated image."""
    try:
        slot, image = _util_image_turn("Object detection")
        # RF-DETR's exported scores have a long high-confidence tail. A 0.5
        # default floods a normal street image with hundreds of tiny false
        # positives; 0.99 retained the bus + four people in the probe image.
        threshold = float(request.form.get("threshold") or 0.99)
        threshold = min(0.999, max(0.5, threshold))
        annotated, detections, timings = slot.run_utility(
            "detect", image, threshold=threshold)
    except _TurnError as e:
        return e.response
    except UtilityUnavailable as e:
        # Deliberately unavailable is 503, not 500 — see _require().
        return openai_error(str(e), "server_error", 503)
    except Exception as e:
        return openai_error(f"Detection failed: {e}", "server_error", 500)
    return jsonify({
        "image": _pil_data_url(annotated), "detections": detections,
        "engine": slot.device_name.lower(), "timings_ms": timings,
    }), 200, {"X-Device": slot.device_name}


@app.route("/v1/util/python", methods=["POST"])
def util_python():
    """Execute one bounded local calculation outside the Flask process.

    This endpoint is intentionally off by default. It is meant to give a
    small local model exact arithmetic, not to promise arbitrary untrusted
    code execution safety: the child is isolated, capped and killed on
    timeout, while Windows still has native escape paths we do not claim to
    close.
    """
    # Loopback only. The server binds 0.0.0.0 on purpose -- phones use the chat
    # UI -- so without this, enabling --python-tool would hand every device on
    # the network a code-execution endpoint. `request.remote_addr` is the socket
    # peer, not a header, so a client cannot spoof it.
    if request.remote_addr not in ("127.0.0.1", "::1", "localhost"):
        return openai_error("This endpoint is local-only (it runs code on the "
                            "server's machine).", "invalid_request_error", 403)
    if not PYTHON_TOOL_ENABLED:
        return openai_error(
            "Python calculations are off. Start locally with --python-tool to "
            "enable the local child-process tool.", "server_error", 503)
    body = request.get_json(silent=True) or {}
    code = body.get("code")
    if not isinstance(code, str) or not code.strip():
        return openai_error("'code' is required", "invalid_request_error", 400)
    return jsonify(execute_python(code))


@app.route("/v1/util/generate", methods=["POST"])
def util_generate():
    """Generate a 512px image when a complete local GenAI pipeline exists."""
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return openai_error("'prompt' is required")
    if len(prompt) > 2000:
        return openai_error("'prompt' is too long (maximum 2000 characters)")
    try:
        slot = _util_slot(body.get("engine") or "npu")
        slot.ensure_loaded()
        steps = min(30, max(1, int(body.get("steps", 12))))
        seed = int(body.get("seed", 0))
        image, timings = slot.run_utility(
            "generate", prompt, steps=steps, seed=seed)
    except _TurnError as e:
        return e.response
    except UtilityUnavailable as e:
        # Deliberately unavailable is 503, not 500 — see _require().
        return openai_error(str(e), "server_error", 503)
    except Exception as e:
        return openai_error(f"Image generation failed: {e}", "server_error", 500)
    return jsonify({
        "image": _pil_data_url(image), "engine": slot.device_name.lower(),
        "width": image.width, "height": image.height,
        "steps": steps, "seed": seed, "timings_ms": timings,
    }), 200, {"X-Device": slot.device_name}


@app.route("/v1/util/index", methods=["POST"])
def util_index():
    """Build a bounded in-memory semantic index from local uploads."""
    uploads = request.files.getlist("files") or request.files.getlist("file")
    if not uploads:
        return openai_error("Send one or more multipart files in 'files'.")
    if len(uploads) > _UTIL_SEARCH_MAX_FILES:
        return openai_error(
            f"Too many files (maximum {_UTIL_SEARCH_MAX_FILES} per index).")
    engine = request.form.get("engine") or "npu"
    try:
        slot = _util_slot(engine)
        slot.ensure_loaded()
        chunks = []
        for upload in uploads:
            item = _util_item(upload.read(), upload.filename, upload.mimetype)
            if item["kind"] == "image":
                read = slot.read(item["image"])
                text = read["markdown"]
            else:
                text = _document_result(item, engine)["markdown"]
            chunks.extend(_util_chunks(text, item["filename"]))
            if len(chunks) >= _UTIL_SEARCH_MAX_CHUNKS:
                chunks = chunks[:_UTIL_SEARCH_MAX_CHUNKS]
                break
        if not chunks:
            return openai_error("No readable text was found in those files.")
        vectors, timings = slot.run_utility(
            "embed", [chunk["text"] for chunk in chunks])
    except _TurnError as e:
        return e.response
    except UtilityUnavailable as e:
        # Deliberately unavailable is 503, not 500 — see _require().
        return openai_error(str(e), "server_error", 503)
    except Exception as e:
        return openai_error(f"Indexing failed: {e}", "server_error", 500)

    index_id = uuid.uuid4().hex
    with _util_search_lock:
        while len(_util_search_indexes) >= _UTIL_SEARCH_MAX_INDEXES:
            oldest = next(iter(_util_search_indexes))
            del _util_search_indexes[oldest]
        _util_search_indexes[index_id] = {
            "chunks": chunks, "vectors": vectors, "created": time.time(),
            "engine": slot.device_name.lower(),
        }
    return jsonify({
        "index_id": index_id, "files": len(uploads), "chunks": len(chunks),
        "engine": slot.device_name.lower(), "timings_ms": timings,
        "storage": "memory",
    }), 200, {"X-Device": slot.device_name}


@app.route("/v1/util/search", methods=["POST"])
def util_search():
    """Search an in-memory utility index with a MiniLM query embedding."""
    body = request.get_json(silent=True) or {}
    index_id = str(body.get("index_id") or "")
    query = str(body.get("query") or "").strip()
    if not index_id or not query:
        return openai_error("'index_id' and 'query' are required")
    with _util_search_lock:
        index = _util_search_indexes.get(index_id)
    if index is None:
        return openai_error("That in-memory index no longer exists.", status=404)
    try:
        slot = _util_slot(index["engine"])
        slot.ensure_loaded()
        query_vector, timings = slot.run_utility("embed", [query])
    except _TurnError as e:
        return e.response
    except UtilityUnavailable as e:
        # Deliberately unavailable is 503, not 500 — see _require().
        return openai_error(str(e), "server_error", 503)
    except Exception as e:
        return openai_error(f"Search failed: {e}", "server_error", 500)
    scores = np.asarray(index["vectors"]) @ query_vector[0]
    top_k = min(10, max(1, int(body.get("top_k", 5))))

    # Retrieve wide, then let the cross-encoder settle the order. Measured on
    # this corpus: the bi-encoder puts the right chunk in the top 5 every time
    # (recall@5 1.000) but ranks it first only 7 times in 9; reranking that
    # pool took top-1 to 8/9 and MRR from 0.889 to 0.944. Widening retrieval is
    # therefore free accuracy — the work is in the ordering, not the recall.
    pool_size = min(len(index["chunks"]), max(top_k, _UTIL_RERANK_POOL))
    pool = list(np.argsort(scores)[::-1][:pool_size])
    order, reranked = pool[:top_k], False

    cross_scores = {}
    if body.get("rerank", True):
        try:
            passages = [index["chunks"][int(i)]["text"] for i in pool]
            rerank_scores, rerank_timings = slot.run_utility(
                "rerank", query, passages)
            cross_scores = {int(i): s for s, i in zip(rerank_scores, pool)}
            ranked = sorted(zip(rerank_scores, pool),
                            key=lambda pair: pair[0], reverse=True)
            order = [i for _score, i in ranked[:top_k]]
            timings = {**timings, **{f"rerank_{k}": v
                                     for k, v in rerank_timings.items()}}
            reranked = True
        except UtilityUnavailable:
            # Reranking is an improvement, not a dependency: without the model
            # installed, search still answers in embedding order rather than
            # 503-ing on a feature the caller never asked for by name.
            pass
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
                  f"rerank failed, using embedding order: {e}", flush=True)

    # Carry both scores when reranking ran. The embedding cosine no longer
    # explains the order once a cross-encoder has reordered it — results come
    # back with a lower cosine above a higher one, which reads as a bug unless
    # the number the sort actually used is visible next to it.
    results = []
    for i in order:
        hit = {**index["chunks"][int(i)], "score": round(float(scores[i]), 4)}
        if int(i) in cross_scores:
            hit["rerank_score"] = round(cross_scores[int(i)], 3)
        results.append(hit)
    return jsonify({
        "results": results, "engine": slot.device_name.lower(),
        "timings_ms": timings,
        # Which model decided this order. `score` stays the embedding cosine so
        # it means the same thing in both modes; `rerank_score` is what the
        # sort used when this is true.
        "reranked": reranked,
    }), 200, {"X-Device": slot.device_name}


def _filter_tools(tools):
    """Keep only the tools in --agent-tools. Unset = keep everything.

    Tool schemas dominate an agent request — 99,044 of Claude Code's 141,959
    characters, 68%, across 30 tools (measured). Every one of them is
    prefilled on every turn, and on an iGPU prefill is the whole wait. Six
    tools is a working coding agent; the other 24 are prose the model reads
    and never uses.

    Claude Code's CLI can do this itself (`--tools`), but its VS Code
    extension exposes no such flag, and neither do most clients — so the
    server has to be able to say no. Dropped tools are logged, never silently
    swallowed: a model that can't see a tool will confidently work around it,
    and you want to know that's why.
    """
    if not config.AGENT_TOOLS or not tools:
        return tools

    def name_of(t):
        return (t.get("name")
                or (t.get("function") or {}).get("name", ""))

    kept = [t for t in tools if name_of(t).split("(")[0] in config.AGENT_TOOLS]
    dropped = len(tools) - len(kept)
    if dropped:
        print(f"{datetime.now():%H:%M:%S} -- tool filter: kept {len(kept)} of "
              f"{len(tools)} ({dropped} dropped: "
              f"{', '.join(sorted(name_of(t) for t in tools if t not in kept))[:120]})",
              flush=True)
    return kept


def _prepare_turn(body):
    """Everything a chat turn needs before generation, for either dialect.

    Parse, route to a device, reload it if it was idle-unloaded, and build the
    generation config. The OpenAI and Anthropic endpoints differ only in how
    the request and the answer are *shaped* — the work between those two points
    is identical, and duplicating it is how the two would drift apart.
    """
    if overall_status() != "ready":
        raise _TurnError(openai_error(
            f"Server not ready (status: {overall_status()}). "
            "Check GET /health.", "server_error", 503))
    if not body:
        raise _TurnError(openai_error("Request body must be JSON"))
    messages = body.get("messages")
    if not messages:
        raise _TurnError(openai_error("'messages' is required"))

    max_tokens = body.get("max_tokens", 4096)
    tools = _filter_tools(body.get("tools") or [])

    try:
        text_prompt, images, raw_messages = parse_messages(messages, max_dim)
    except (FileNotFoundError, ValueError) as e:
        raise _TurnError(openai_error(str(e)))
    except Exception as e:
        raise _TurnError(openai_error(f"Failed to parse request: {e}"))

    slot = _route_request(bool(images), body.get("model", ""))

    # Images with no vision path: read them with the NPU utilities and hand a
    # TEXT model the result, instead of refusing the turn. Done here rather
    # than in the web UI so every client gets it — Ollama, OpenAI SDKs and
    # Claude Code all become document-capable against an NPU chat model.
    vision_path = None
    if images and (slot is None or slot.model_type == "llm"):
        text_slot = slot if (slot and slot.model_type == "llm") else \
            _route_request(False, body.get("model", ""))
        ocr_block, trimmed = _ocr_for_text_model(messages, text_slot)
        if ocr_block:
            slot = text_slot
            images = []                       # answered as a plain text turn
            # The marker parse_messages left behind is exactly where the image
            # was, so the text lands in the right turn rather than at the end.
            marker = re.compile(r"<ov_genai_image_\d+>")
            text_prompt = marker.sub("", text_prompt).strip()
            text_prompt = f"{ocr_block}\n\n{text_prompt}" if text_prompt else ocr_block
            for msg in reversed(raw_messages):
                if msg.get("role") == "user":
                    stripped = marker.sub("", msg.get("content", "")).strip()
                    msg["content"] = (f"{ocr_block}\n\n{stripped}"
                                      if stripped else ocr_block)
                    break
            vision_path = "npu-ocr"
            note = f", {trimmed}" if trimmed else ""
            print(f"{datetime.now():%H:%M:%S} -- [{slot.device_name}] "
                  f"read {len(_image_urls_in(messages))} image(s) via OCR "
                  f"({len(ocr_block)} chars{note})", flush=True)

    if slot is None:
        if images:
            raise _TurnError(openai_error(
                "No vision model loaded. Send text only, or load a VLM."))
        raise _TurnError(openai_error(
            "No model ready to handle this request.", "server_error", 503))

    # Offered on every ordinary chat turn the slot can support, rather than
    # switched on by the caller. The old body flag came from a composer button;
    # deciding whether a question needs a calculation or a search is the
    # model's job, and the tool descriptions say when.
    builtin_specs = builtin_tools_for(slot, {
        "web_search": body.get("web_search", True) is not False,
    })
    python_tool_active = bool(builtin_specs) and not tools
    if python_tool_active:
        try:
            # Re-render from raw_messages when OCR has already rewritten them.
            # Parsing the ORIGINAL `messages` here threw the OCR work away: the
            # block above injects the read text into raw_messages/text_prompt
            # and clears `images`, and re-parsing the request body undid all
            # three -- handing a TEXT model its images back and dropping what
            # had just been read out of them.
            source = raw_messages if vision_path == "npu-ocr" else messages
            text_prompt, images, raw_messages = parse_messages(
                prepare_messages_for_tools(source, builtin_specs), max_dim)
        except Exception as e:
            raise _TurnError(openai_error(f"Failed to prepare Python turn: {e}"))

    # Tool calling is GPU/iGPU + CPU only. Only when such a slot serves the turn
    # do we render tool specs into the prompt and (later) parse calls back out;
    # on the NPU the request is answered as a plain chat turn.
    tools_active = bool(tools) and _tool_capable(slot, tools) and not python_tool_active
    if tools_active:
        try:
            text_prompt, images, raw_messages = parse_messages(
                prepare_messages_for_tools(messages, tools, slot), max_dim)
        except Exception as e:
            raise _TurnError(openai_error(f"Failed to parse request: {e}"))
    elif tools:
        print(f"{datetime.now():%H:%M:%S} -- [{slot.device_name}] "
              f"{_tools_refused_note(slot, tools)}", flush=True)

    if images and slot.model_type == "llm":
        # Only reachable when OCR fusion above could not run (no utility models
        # installed, or every image failed to read).
        raise _TurnError(openai_error(
            f"Model '{slot.model_name}' on {slot.device_name} does not support "
            "images, and the OCR utilities are not available to read them. "
            "Load a VLM, or fetch the utility models with "
            "`python scripts/npu-probe.py --all`."))

    try:
        slot.ensure_loaded()
    except Exception as e:
        raise _TurnError(openai_error(f"Failed to reload model: {e}",
                                      "server_error", 500))

    gen = ovg.GenerationConfig()
    gen.max_new_tokens = max_tokens
    temperature = body.get("temperature", 0.0)
    if temperature and temperature > 0.01:
        gen.do_sample = True
        gen.temperature = temperature
        gen.top_p = body.get("top_p", 1.0)
    else:
        gen.do_sample = False
        gen.top_k = 1
    apply_penalties(gen,
                    repetition=body.get("repetition_penalty"),
                    frequency=body.get("frequency_penalty"),
                    presence=body.get("presence_penalty"))

    stream = bool(body.get("stream", False))
    n_images = len(images)
    tag = f"{n_images} image{'s' if n_images != 1 else ''}, " if n_images else ""
    print(f"\n{datetime.now():%H:%M:%S} <- [{slot.device_name}] {tag}"
          f"{len(text_prompt)} chars, max_tokens={max_tokens}"
          f"{' (stream)' if stream else ''}", flush=True)

    return {"slot": slot, "gen": gen, "stream": stream, "tools": tools,
            "tools_active": tools_active, "text_prompt": text_prompt,
            "images": images, "raw_messages": raw_messages,
            "python_tool_active": python_tool_active,
            "builtin_specs": builtin_specs,
            "max_dim": max_dim,
            "max_tokens": max_tokens, "t0": time.perf_counter(),
            # "npu-ocr" when images were read as text rather than seen. Callers
            # surface it as X-Vision-Path so this is never mistaken for vision.
            "vision_path": vision_path}


# ---------------------------------------------------------------------------
# Built-in tools the SERVER owns and runs
#
# Distinct from the client's `tools` array, which locally only relays: these
# are offered to the model on every ordinary chat turn and executed here. That
# is what let the composer's mode buttons go. A "search the web" toggle asked
# the user to answer a question the model is better placed to answer -- it
# forced a search on "hello" and skipped one on a question about last week --
# and a "calculate" toggle asked the user to know in advance that a number was
# going to be hard. The tool's own description is the whole instruction; no
# system-prompt text is spent on either, which matters because that prompt is
# re-sent every turn.
#
# The gate is _tools_supported, so this is GPU/CPU only. On the NPU these are
# not offered at all -- its 8k prompt cap is the reason -- and a turn there
# answers as plain chat exactly as before.
# ---------------------------------------------------------------------------


def _builtin_tool_turn(generate, raw_messages, specs, slot_for_fit=None):
    """Run the bounded server-side tool loop and return answer + audit data.

    Bounded on purpose. An unbounded loop on a small model is how a whole
    context window gets spent re-deciding; two rounds is enough for "call the
    tool, read the result, answer" and stops a model that has decided every
    reply needs another search.

    A call to a tool that is not ours is not our business: it is left in the
    text and returned, so a client that sent its own `tools` array still gets
    its call back the way it always did.
    """
    messages = list(raw_messages)
    _builtin_runs().clear()
    for round_no in range(BUILTIN_TOOL_MAX_ROUNDS + 1):
        try:
            generated = generate(messages)
        except Exception as e:
            raise RuntimeError(explain_genai_error(e))
        text, calls = parse_tool_calls(generated, specs)
        accepted = []
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name")
            if name not in BUILTIN_TOOLS:
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (ValueError, TypeError):
                args = {}
            accepted.append((call, name, args if isinstance(args, dict) else {}))
        if not accepted:
            return text.strip(), list(_builtin_runs())
        if round_no >= BUILTIN_TOOL_MAX_ROUNDS:
            return (text.strip() or
                    "I stopped after the tool-round limit."), list(_builtin_runs())

        results = []
        for call, name, args in accepted:
            try:
                results.append((name, _fit_tool_output(
                    slot_for_fit, BUILTIN_TOOLS[name]["run"](args))))
            except Exception as e:
                # A tool that fails must not kill the turn -- the model can say
                # it could not check, which is far better than a 500.
                results.append((name, "The tool failed: " + str(e)))

        call_text = _tool_calls_to_text([c for c, _n, _a in accepted])
        messages.append({"role": "assistant",
                         "content": (text + "\n" + call_text).strip()})
        body = "\n\n".join('<tool_response name="' + n + '">\n' + r +
                           "\n</tool_response>" for n, r in results)
        messages.append({
            "role": "user",
            "content": (body + "\n\nUse the result above to answer the user. "
                        "Do not call another tool unless it is genuinely "
                        "necessary."),
        })
    return "I could not complete that.", list(_builtin_runs())


# ---------------------------------------------------------------------------
# Web search — the answer engine
#
# The point is not "the model can browse". It is that a small local model
# guesses when it does not know, and guessing is what sends you to Google. So
# the model is never asked to recall: it is handed passages and told to quote
# them or say it does not know.
#
# The accuracy comes from the reranker, not the model. Retrieval is deliberately
# wide (a bi-encoder over every chunk) and then reordered by the BGE cross-
# encoder on the NPU — measured on this project's own corpus, that took top-1
# from 7/9 to 8/9 and MRR from 0.889 to 0.944. Widening retrieval is free
# accuracy; the work is in the ordering.
#
# This is the server's FIRST outbound HTTP. It is opt-in behind --search-url and
# does nothing at all when unset, so "works on a plane" survives: with no flag,
# locally makes no external connection, exactly as before.
#
# Backend is a self-hosted SearXNG (JSON output, no API key, no per-query
# billing, no queries leaving the box beyond the engines it proxies). Brave
# removed its free tier in February 2026 and scraping DuckDuckGo breaks silently.


def python_tool_status():
    """Describe the opt-in local calculation tool for /health and the UI."""
    prompt = _python_tool_prompt()
    token_count = None
    if PYTHON_TOOL_ENABLED and runtime.primary and runtime.primary.status != "not_configured":
        try:
            token_count = _count_tokens(runtime.primary, prompt)
        except Exception:
            pass
    return {
        "enabled": bool(PYTHON_TOOL_ENABLED),
        "timeout_s": config.PYTHON_TOOL_TIMEOUT,
        "output_bytes": config.PYTHON_TOOL_OUTPUT_BYTES,
        "max_code_bytes": config.PYTHON_TOOL_MAX_CODE_BYTES,
        "max_rounds": BUILTIN_TOOL_MAX_ROUNDS,
        "prompt_bytes": len(prompt.encode("utf-8")),
        "prompt_tokens": token_count,
        "network": "blocked imports only; not a hostile-code jail on Windows",
        "reason": None if PYTHON_TOOL_ENABLED else
                  "start with --python-tool to enable local calculations",
    }


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


_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_READ_URL_MAX_CHARS = 24000     # what we hand the model from one page


@app.route("/v1/read-url", methods=["POST"])
def read_url():
    """Read one page and return its text, so a pasted link becomes context.

    This is the honest version of "read the site I am on". A separate window
    cannot see the user's tabs -- that needs a browser extension -- but a URL
    pasted into the chat is the same intent with none of the machinery, and it
    reuses the fetcher the web-search path already hardened (gzip, one retry,
    and a stated reason on failure).

    Deliberately not gated on --search-url: reading a link the user explicitly
    pasted is a different act from searching the web on their behalf. It is
    still outbound HTTP, so it does nothing unless asked.
    """
    body = request.get_json(silent=True) or {}
    url = str(body.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return openai_error("'url' must be an http:// or https:// address")

    t0 = time.perf_counter()
    text, reason = _fetch_page_text(url)
    if not text:
        return openai_error(f"Could not read that page - {reason}",
                            "server_error", 502)
    limit = min(int(body.get("max_chars", _READ_URL_MAX_CHARS)), 200000)
    return jsonify({
        "url": url,
        "text": text[:limit],
        "chars": len(text),
        "truncated": len(text) > limit,
        "took_ms": round((time.perf_counter() - t0) * 1000),
    })


@app.route("/v1/search", methods=["POST"])
def web_search():
    """Search the web, rank passages on the NPU, and cite what was used.

    JSON by default. With "stream": true the same run is delivered as SSE, one
    event per stage, so a 20-33 s turn can say what it is doing instead of
    showing a blinking dot.
    """
    body = request.get_json(silent=True) or {}
    reuse = (str(body.get("mode") or "").lower() == "analyze"
             and bool(body.get("passages")))
    if not WEB_SEARCH_URL and not reuse:
        return openai_error(
            "Web search is off. Start locally with --search-url pointing at a "
            "SearXNG instance (e.g. --search-url http://localhost:8080). "
            "Without it the server makes no outbound connections.",
            "server_error", 503)

    if not body.get("stream"):
        for kind, value in _web_search_run(body):
            if kind == "error":
                message, status = value
                return openai_error(
                    message,
                    "invalid_request_error" if status == 400 else "server_error",
                    status)
            if kind == "done":
                return jsonify(value)
        return openai_error("Search produced no result", "server_error", 500)

    def events():
        try:
            for kind, value in _web_search_run(body):
                if kind == "error":
                    message, _status = value
                    yield ("data: " + json.dumps({"event": "error",
                                                  "message": message}) + "\n\n")
                    return
                if kind == "stage":
                    yield "data: " + json.dumps({"event": "stage", **value}) + "\n\n"
                elif kind == "done":
                    yield "data: " + json.dumps({"event": "done", **value}) + "\n\n"
        except GeneratorExit:
            raise
        except Exception as e:
            yield ("data: " + json.dumps({"event": "error",
                                          "message": str(e)}) + "\n\n")
        yield "data: [DONE]\n\n"

    return Response(events(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    try:
        turn = _prepare_turn(request.get_json(silent=True))
    except _TurnError as e:
        return e.response

    slot, gen = turn["slot"], turn["gen"]
    stream, tools, tools_active = turn["stream"], turn["tools"], turn["tools_active"]
    python_tool_active = turn["python_tool_active"]
    text_prompt, images, raw_messages = (turn["text_prompt"], turn["images"],
                                         turn["raw_messages"])
    completion_id = make_id()
    created = int(time.time())
    t0 = turn["t0"]

    if python_tool_active and stream:
        # Streams live and still runs tools -- see _builtin_stream.
        return Response(
            _builtin_stream(slot, raw_messages, gen, turn["builtin_specs"],
                            turn, completion_id, created, t0),
            mimetype="text/event-stream", headers=_turn_headers(turn, slot))

    if python_tool_active:
        try:
            text, python_runs = _builtin_tool_turn(
                _builtin_generator(slot, gen, turn), raw_messages,
                turn["builtin_specs"], slot)
        except Exception as e:
            err = explain_genai_error(e)
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
                  f"Python turn error: {err}", flush=True)
            return openai_error(f"Python turn failed: {err}", "server_error", 500)
        elapsed = time.perf_counter() - t0
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] Python turn "
              f"({len(python_runs)} calculation(s)) in {elapsed:.1f}s",
              flush=True)
        resp = jsonify({
            "id": completion_id, "object": "chat.completion",
            "created": created, "model": slot.model_name,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "python_runs": python_runs,
            "usage": {"prompt_tokens": -1, "completion_tokens": -1,
                       "total_tokens": -1},
        })
        resp.headers.extend(_turn_headers(turn, slot))
        return resp

    if stream and not tools_active:
        return Response(
            slot.stream_llm(raw_messages, gen, completion_id, created, t0),
            mimetype="text/event-stream",
            headers=_turn_headers(turn, slot),
        )

    # --- VLM path ---
    if slot.model_type == "vlm":
        if stream:
            return Response(
                slot.stream_vlm(text_prompt, images, gen, completion_id, created, t0),
                mimetype="text/event-stream",
                headers=_turn_headers(turn, slot),
            )

        try:
            text = slot.generate_vlm(text_prompt, images, gen)
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] VLM error: {e}", flush=True)
            return openai_error(f"Inference failed: {e}", "server_error", 500)

        elapsed = time.perf_counter() - t0
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
              f"{len(text)} chars in {elapsed:.1f}s", flush=True)

        resp = jsonify({
            "id": completion_id, "object": "chat.completion",
            "created": created, "model": slot.model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
        })
        resp.headers.extend(_turn_headers(turn, slot))
        return resp

    # Capture a big agent prompt so the next startup can pre-warm it (no-op
    # unless --prewarm is set).
    _maybe_capture_prewarm(raw_messages)

    # --- LLM path ---
    # Tool turns must be buffered (we need the whole generation before emitting a
    # structured tool_calls delta). When streaming, buffer in a background thread
    # and emit SSE keep-alive frames so a long prefill on a big agent prompt
    # doesn't trip the client's idle watchdog (see _sse_tool_stream).
    if stream and tools_active:
        return Response(
            _sse_tool_stream(slot, raw_messages, gen, tools, completion_id, created, t0),
            mimetype="text/event-stream",
            headers=_turn_headers(turn, slot),
        )

    # Non-streaming (with or without tools): one blocking generate + JSON reply.
    try:
        text = slot.generate_llm(raw_messages, gen)
    except Exception as e:
        err = explain_genai_error(e)
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] LLM error: {err}", flush=True)
        return openai_error(f"Inference failed: {err}", "server_error", 500)

    elapsed = time.perf_counter() - t0
    n_words = len(text.split())
    ttft = (f", TTFT {slot.last_ttft_ms:.0f}ms"
            if slot.last_ttft_ms is not None else "")
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
          f"~{n_words} tokens in {elapsed:.1f}s "
          f"({n_words / max(elapsed, 1e-6):.1f} tok/s{ttft}"
          f"{slot.prefill_note(text_prompt)})", flush=True)

    tool_calls = []
    if tools_active:
        text, tool_calls = parse_tool_calls(text, tools)
        if tool_calls:
            print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
                  f"{len(tool_calls)} tool call(s): "
                  f"{', '.join(tc['function']['name'] for tc in tool_calls)}",
                  flush=True)

    if tool_calls:
        message = {"role": "assistant", "content": text or None, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish_reason = "stop"

    resp = jsonify({
        "id": completion_id, "object": "chat.completion",
        "created": created, "model": slot.model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    })
    resp.headers.extend(_turn_headers(turn, slot))
    return resp


# ---------------------------------------------------------------------------
# Anthropic Messages API — so Claude Code talks to locally directly
# ---------------------------------------------------------------------------
#
# Same idea as the Ollama shim below: speak someone else's dialect so their
# client works unchanged. Claude Code needs POST /v1/messages (+ count_tokens),
# and with it you point the real CLI at a local NPU/iGPU model with two env
# vars and no proxy process in between:
#
#   ANTHROPIC_BASE_URL=http://localhost:8000
#   ANTHROPIC_AUTH_TOKEN=local        # required by the client, unused here
#   ANTHROPIC_MODEL=<your model dir name>
#
# Translation happens only at the edges. The request is converted to the
# OpenAI shape that _prepare_turn already understands, and the answer is
# converted back — including SSE, which is adapted by *consuming* the existing
# OpenAI stream rather than by teaching DeviceSlot a second frame format. That
# keeps one tested streaming path (cancel, heartbeat, tool buffering) instead
# of two that can disagree.

def _anthropic_text(content):
    """Flatten an Anthropic content field (string or block list) to text."""
    if isinstance(content, str):
        return content
    parts = [b.get("text", "") for b in (content or [])
             if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p)


def _anthropic_to_openai(body):
    """Anthropic Messages request -> the OpenAI shape _prepare_turn takes."""
    messages = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _anthropic_text(system)})

    for msg in body.get("messages") or []:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        parts, tool_calls, tool_results = [], [], []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif btype == "image":
                src = block.get("source") or {}
                if src.get("type") == "base64":
                    url = (f"data:{src.get('media_type', 'image/png')};base64,"
                           f"{src.get('data', '')}")
                else:
                    url = src.get("url", "")
                if url:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", make_id()), "type": "function",
                    "function": {"name": block.get("name", ""),
                                 "arguments": json.dumps(block.get("input") or {})},
                })
            elif btype == "tool_result":
                # Anthropic carries results inside the *user* turn; OpenAI wants
                # them as their own messages, after the assistant's call.
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _anthropic_text(block.get("content")) or "",
                })

        if parts or tool_calls:
            out = {"role": role,
                   "content": parts if len(parts) != 1 or parts[0]["type"] != "text"
                   else parts[0]["text"]}
            if tool_calls:
                out["tool_calls"] = tool_calls
            messages.append(out)
        messages.extend(tool_results)

    out = {"messages": messages,
           "model": body.get("model", ""),
           "max_tokens": body.get("max_tokens", 4096),
           "stream": bool(body.get("stream", False))}
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if body.get("tools"):
        out["tools"] = [{"type": "function",
                         "function": {"name": t.get("name", ""),
                                      "description": t.get("description", ""),
                                      "parameters": t.get("input_schema") or {}}}
                        for t in body["tools"] if isinstance(t, dict)]
    return out


def _anthropic_blocks(text, tool_calls):
    """OpenAI text + tool_calls -> Anthropic content blocks."""
    blocks = []
    if text:
        blocks.append({"type": "text", "text": text})
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        blocks.append({"type": "tool_use", "id": tc.get("id") or make_id(),
                       "name": fn.get("name", ""), "input": args})
    return blocks


def _turn_headers(turn, slot):
    """Response headers for a chat turn: which engine, and how it saw images.

    X-Vision-Path is only present when images were READ as text rather than
    seen, so a client can tell the difference. Silence would let an OCR answer
    pass for real vision.
    """
    headers = {"X-Device": slot.device_name, "X-Model": slot.model_name}
    if turn.get("vision_path"):
        headers["X-Vision-Path"] = turn["vision_path"]
    return headers


def _image_urls_in(messages):
    """Every image URL in an OpenAI message list, in order."""
    urls = []
    for msg in messages or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "image_url":
                url = (block.get("image_url") or {}).get("url", "")
                if url:
                    urls.append(url)
    return urls


def _ocr_for_text_model(messages, slot):
    """Read attached images with the NPU utilities so a text model can 'see'.

    The NPU has no working vision path, so an image sent to an NPU chat model
    used to be a hard error. Reading it with the OCR utilities and handing the
    model the text is not a substitute for vision — it cannot describe a
    photograph — but for the things people actually paste at a local model
    (screenshots, receipts, scanned pages, error dialogs) it is *better* than a
    small VLM, which is the finding this whole feature rests on.

    Returns (text_block, truncated_note) or (None, None) when OCR isn't
    possible, so the caller can fall back to its original error.
    """
    util = next((u for u in (runtime.util_npu, runtime.util_gpu, runtime.util_cpu)
                 if u and _slot_serviceable(u)), None)
    if util is None:
        return None, None

    urls = _image_urls_in(messages)
    if not urls:
        return None, None

    try:
        util.ensure_loaded()
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} !! [{util.device_name}] "
              f"util load failed, cannot read images: {e}", flush=True)
        return None, None

    parts = []
    for i, url in enumerate(urls):
        try:
            # Full resolution on purpose: max_dim exists to fit a VLM's budget,
            # and downscaling a screenshot is how fine print stops being legible.
            img = load_image(url, 0)
            result = util.read(img)
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [{util.device_name}] "
                  f"OCR failed on image {i + 1}: {e}", flush=True)
            continue
        text = (result.get("text") or "").strip()
        label = f"Image {i + 1}" if len(urls) > 1 else "Image"
        parts.append(f"{label}:\n{text}" if text
                     else f"{label}: (no readable text found)")

    if not parts:
        return None, None

    block = ("Text read from the attached image(s) by local OCR:\n\n"
             + "\n\n".join(parts))

    # The NPU slot is built with MAX_PROMPT_LEN=4096 and that is a hard cap, so
    # a dense multi-page scan cannot simply be pasted in. Trim the OCR text —
    # never the user's own question — and say so, because an answer drawn from
    # half a document without a word of warning is the worst outcome here.
    note = None
    if slot is not None and slot.device_name == "NPU":
        budget = 4096 - 700            # leaves room for system prompt + reply
        n = _count_tokens(slot, block)
        if n and n > budget:
            keep = max(200, int(len(block) * budget / n))
            block = block[:keep].rstrip()
            note = f"{n} tokens of OCR text trimmed to ~{budget}"
            block += ("\n\n[Truncated: the page did not fit this model's "
                      "prompt limit. Switch the chat model to the GPU for the "
                      "whole document.]")
    return block, note


def _sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _anthropic_stream(openai_frames, model, msg_id, input_tokens):
    """Re-dress the OpenAI SSE stream as Anthropic events.

    Anthropic's protocol is block-structured where OpenAI's is a flat delta
    list: text arrives inside an explicit content block that has to be opened
    and closed, and a tool call is its own block whose arguments stream as
    partial JSON. We buffer tool calls anyway (see _sse_tool_stream), so each
    tool block is emitted whole.
    """
    yield _sse("message_start", {
        "type": "message_start",
        "message": {"id": msg_id, "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens or 0,
                              "output_tokens": 0}},
    })
    yield _sse("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    index, out_tokens, stop_reason = 0, 0, "end_turn"
    text_open = True
    think = _ThinkFilter()
    emitted = 0
    for frame in openai_frames:
        for line in frame.splitlines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}

            text = delta.get("content")
            if text:
                out_tokens += 1
                text = think.feed(text)
                if text:
                    emitted += len(text)
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta", "index": index,
                        "delta": {"type": "text_delta", "text": text},
                    })
            elif text == "":
                # locally's keep-alive during a long prefill. Anthropic has a
                # first-class event for exactly this, so it survives the trip.
                yield _sse("ping", {"type": "ping"})

            for tc in delta.get("tool_calls") or []:
                if text_open:
                    yield _sse("content_block_stop",
                               {"type": "content_block_stop", "index": index})
                    text_open = False
                index += 1
                fn = tc.get("function") or {}
                yield _sse("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use",
                                      "id": tc.get("id") or make_id(),
                                      "name": fn.get("name", ""), "input": {}},
                })
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta",
                              "partial_json": fn.get("arguments") or "{}"},
                })
                yield _sse("content_block_stop",
                           {"type": "content_block_stop", "index": index})
                stop_reason = "tool_use"

            if choice.get("finish_reason") in ("length",):
                stop_reason = "max_tokens"

    tail = think.flush()
    if not emitted and not tail and stop_reason != "tool_use" and think.discarded:
        tail = think.discarded          # reasoning was all we got — show it
    if tail and text_open:
        yield _sse("content_block_delta", {
            "type": "content_block_delta", "index": index,
            "delta": {"type": "text_delta", "text": tail},
        })
    if text_open:
        yield _sse("content_block_stop",
                   {"type": "content_block_stop", "index": index})
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": out_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})


@app.route("/v1/messages", methods=["POST"])
def anthropic_messages():
    raw = request.get_json(silent=True)
    if not raw:
        return openai_error("Request body must be JSON")

    if debug:
        # Where the bytes actually are decides what is worth trimming: the
        # client's preamble, the conversation, or the tool schemas we render
        # ourselves. Guessing that wrong costs a 3-minute test run.
        sysn = len(_anthropic_text(raw.get("system") or ""))
        msgn = len(json.dumps(raw.get("messages") or []))
        tooln = len(json.dumps(raw.get("tools") or []))
        print(f"{datetime.now():%H:%M:%S} -- anthropic request: system={sysn} "
              f"messages={msgn} tools={tooln} chars "
              f"({len(raw.get('tools') or [])} tools)", flush=True)

    try:
        turn = _prepare_turn(_anthropic_to_openai(raw))
    except _TurnError as e:
        return e.response

    slot, gen, t0 = turn["slot"], turn["gen"], turn["t0"]
    msg_id = f"msg_{make_id()}"
    created = int(time.time())
    input_tokens = _count_tokens(slot, turn["text_prompt"])

    if turn["stream"]:
        if slot.model_type == "vlm":
            frames = slot.stream_vlm(turn["text_prompt"], turn["images"], gen,
                                     msg_id, created, t0)
        elif turn["tools_active"]:
            frames = _sse_tool_stream(slot, turn["raw_messages"], gen,
                                      turn["tools"], msg_id, created, t0)
        else:
            frames = slot.stream_llm(turn["raw_messages"], gen, msg_id, created, t0)
        return Response(
            _anthropic_stream(frames, slot.model_name, msg_id, input_tokens),
            mimetype="text/event-stream",
            headers=_turn_headers(turn, slot),
        )

    try:
        if slot.model_type == "vlm":
            text = slot.generate_vlm(turn["text_prompt"], turn["images"], gen)
        else:
            _maybe_capture_prewarm(turn["raw_messages"])
            text = slot.generate_llm(turn["raw_messages"], gen)
    except Exception as e:
        err = explain_genai_error(e)
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
              f"error: {err}", flush=True)
        return openai_error(f"Inference failed: {err}", "server_error", 500)

    tool_calls = []
    if turn["tools_active"]:
        text, tool_calls = parse_tool_calls(text, turn["tools"])
    stripped = strip_thinking(text)
    if not stripped and not tool_calls:
        # Same rule as the streaming path: never hand back an empty message
        # just because the model spent its whole budget reasoning. Keep the
        # reasoning as the answer, but without the tags — they mean nothing
        # to an API client.
        stripped = _strip_tool_markup(
            text.replace("<think>", "").replace("</think>", ""))
    text = stripped

    elapsed = time.perf_counter() - t0
    out_tokens = _count_tokens(slot, text) or len(text.split())
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
          f"{out_tokens} tokens in {elapsed:.1f}s "
          f"({out_tokens / max(elapsed, 1e-6):.1f} tok/s)", flush=True)

    resp = jsonify({
        "id": msg_id, "type": "message", "role": "assistant",
        "model": slot.model_name,
        "content": _anthropic_blocks(text, tool_calls),
        "stop_reason": "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens or 0, "output_tokens": out_tokens},
    })
    resp.headers.extend(_turn_headers(turn, slot))
    return resp


BROWSER_EXES = ("zen.exe", "firefox.exe", "librewolf.exe", "chrome.exe",
                "msedge.exe", "brave.exe", "opera.exe", "vivaldi.exe")


def _allow_foreground_handoff():
    """Let the process we are about to launch take the foreground.

    Windows only grants SetForegroundWindow to a process that already owns the
    foreground or has just received input. This server has neither -- the click
    happened inside the browser -- so without this the browser opens the tab
    behind the locally window and only flashes its taskbar button.

    ASFW_ANY (-1) hands that right to whichever process acts next, which is the
    documented way to do a launch-and-focus handoff.
    """
    try:
        ctypes.windll.user32.AllowSetForegroundWindow(-1)
    except Exception:
        pass


def _raise_browser_window(timeout=3.0):
    """Bring the default browser's window to the front after opening a URL.

    ShellExecute returns as soon as the browser is told about the URL, not when
    it has drawn a window, so this polls. Matches on the process executable
    rather than the window title, because a title is whatever page happens to
    be loading and can be anything.

    Best-effort by design: if the window cannot be found or Windows refuses the
    activation, the tab is still open and the user can alt-tab. Failing loudly
    here would turn a cosmetic miss into a broken feature.
    """
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi

    def exe_of(hwnd):
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        # QUERY_LIMITED_INFORMATION: enough for the image name, and unlike
        # QUERY_INFORMATION it is granted for processes at other integrity
        # levels, which a browser often is.
        h = kernel32.OpenProcess(0x1000, False, pid.value)
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(260)
            psapi.GetModuleFileNameExW(h, None, buf, 260)
            return os.path.basename(buf.value).lower()
        finally:
            kernel32.CloseHandle(h)

    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        if user32.GetWindowTextLengthW(hwnd) == 0:
            return True          # toolwindows and hidden hosts
        if exe_of(hwnd) not in BROWSER_EXES:
            return True
        # locally's own PWA window IS a browser window -- it runs inside
        # msedge.exe -- so without this the server can raise itself and the
        # link still appears not to have opened. Measured: enumeration returned
        # msedge/"locally" ahead of zen/"<page title>".
        title = ctypes.create_unicode_buffer(160)
        user32.GetWindowTextW(hwnd, title, 160)
        if title.value.strip() == "locally":
            return True
        found.append(hwnd)
        return False             # first non-self browser window is enough

    deadline = time.time() + timeout
    while time.time() < deadline:
        found.clear()
        try:
            user32.EnumWindows(visit, 0)
        except Exception:
            return False
        if found:
            hwnd = found[0]
            try:
                if user32.IsIconic(hwnd):
                    user32.ShowWindow(hwnd, 9)       # SW_RESTORE
                return bool(user32.SetForegroundWindow(hwnd))
            except Exception:
                return False
        time.sleep(0.15)
    return False


def _open_default_browser(url):
    """Open a URL through the user's default browser and raise its window."""
    raised = False
    if sys.platform == "win32":
        # os.startfile uses the shell association, which IS the default
        # browser. webbrowser.open on Windows can resolve to whatever is
        # first on PATH instead.
        _allow_foreground_handoff()
        os.startfile(url)                                      # noqa: S606
        # Opening the tab is not the whole job: without this the browser
        # comes up BEHIND the locally window and only flashes its taskbar
        # button, so the link appears not to have worked at all.
        raised = _raise_browser_window()
    elif sys.platform == "darwin":
        subprocess.Popen(["open", url])
        raised = True                                          # `open` activates the app itself
    else:
        subprocess.Popen(["xdg-open", url])
    return raised


@app.route("/v1/open", methods=["POST"])
def open_external():
    """Open a URL in the user's DEFAULT browser.

    The chat UI runs inside an Edge PWA window, so a plain target="_blank" opens
    the link in Edge no matter what the user's default browser is. The page
    cannot escape its own host, so the server has to do it.

    Same two rules as /v1/code/launch, because this also acts on the machine:

    1. **Localhost only.** The server binds 0.0.0.0 on purpose (phones use the
       chat UI), so without this every device on the network could pop windows
       on this one. `request.remote_addr` is the socket peer, not a header.
    2. **http/https only, and never interpolated into a shell.** A URL is
       caller-supplied data, and `file://`, `smb://` or a shell metacharacter
       reaching a command line is the whole attack. The scheme is checked
       against an allowlist and the URL is passed as an argv element.
    """
    if request.remote_addr not in ("127.0.0.1", "::1", "localhost"):
        return openai_error("This endpoint is local-only (it opens a window on "
                            "the server's machine).", "invalid_request_error", 403)

    url = str((request.get_json(silent=True) or {}).get("url") or "").strip()
    try:
        parts = urllib.parse.urlsplit(url)
    except Exception:
        parts = None
    if not parts or parts.scheme not in ("http", "https") or not parts.netloc:
        return openai_error("Only http:// and https:// URLs can be opened.")

    try:
        raised = _open_default_browser(url)
    except Exception as e:
        return openai_error(f"Could not open the link: {e}", "server_error", 500)
    return jsonify({"opened": url, "raised": raised})


@app.route("/v1/code/launch", methods=["POST"])
def launch_claude_code():
    """Open Claude Code in a terminal, pointed at this server.

    The web UI can't start a process, so the button posts here. Two rules make
    that safe enough to ship:

    1. **Localhost only.** The server binds 0.0.0.0 (that's the point — phones
       and other machines use the chat UI), so without this check every device
       on the network could spawn processes on this one. `request.remote_addr`
       is the socket peer, not a header, so it can't be spoofed by a client.
    2. **No caller-supplied command.** The only input is a directory, and it
       is passed as an argv element to a fixed script — never interpolated
       into a shell string. The worst a caller can do is open a terminal in a
       directory they name.

    The terminal runs scripts/claude-local.ps1, the same script a user would
    run by hand, so there is one code path to debug. If no terminal can be
    opened we still return the command, so the UI can offer it for copy-paste
    rather than dead-ending.
    """
    if request.remote_addr not in ("127.0.0.1", "::1", "localhost"):
        return openai_error("This endpoint is local-only (it starts a process "
                            "on the server's machine).", "invalid_request_error",
                            403)

    body = request.get_json(silent=True) or {}
    cwd = os.path.expanduser(str(body.get("cwd") or config.SCRIPT_DIR))
    if not os.path.isdir(cwd):
        return openai_error(f"No such directory: {cwd}")

    script = os.path.join(config.SCRIPT_DIR, "scripts", "claude-local.ps1")
    if not os.path.isfile(script):
        return openai_error(f"Launcher script missing: {script}",
                            "server_error", 500)

    base_url = f"http://127.0.0.1:{config.SERVER_PORT}"
    model, warning = "", None
    for slot in (runtime.primary, runtime.secondary):
        if _slot_serviceable(slot):
            model = slot.model_name
            # Measured on this hardware: Claude Code sends a ~35k-token system
            # prompt and asks for up to 32k of output, and the KV pool has to
            # hold both at once. A 32k pool doesn't just truncate — it dies
            # with CL_OUT_OF_RESOURCES eight minutes into the first turn, which
            # is a miserable way to find out. Say it before launching.
            if slot.context_tokens and slot.context_tokens < config.CODE_MIN_CONTEXT:
                warning = (f"This model has only {slot.context_tokens // 1000}k "
                           f"of context. Claude Code needs ~67k (a ~35k-token "
                           f"system prompt plus room to answer) and will fail "
                           f"mid-turn below that — restart with "
                           f"--context-tokens 90000.")
            if not _tools_supported(slot):
                warning = (f"{slot.model_name} is on the {slot.device_name}, "
                           f"which has no agent path — Claude Code needs tool "
                           f"calling, so load the model on GPU or CPU.")
            break

    inner = ["pwsh", "-NoExit", "-File", script,
             "-BaseUrl", base_url, "-ProjectDir", cwd]
    if model:
        inner += ["-Model", model]
    # Shown to the user verbatim when we can't open a terminal for them.
    manual = (f'pwsh -File "{script}" -BaseUrl {base_url} '
              f'-ProjectDir "{cwd}"' + (f' -Model "{model}"' if model else ""))

    # Windows Terminal opens a real tab; without it, fall back to a bare
    # console. Non-Windows gets the command to run — guessing at someone's
    # terminal emulator is how you end up launching the wrong one.
    candidates = []
    if os.name == "nt":
        candidates.append(["wt.exe", "-d", cwd] + inner)
        candidates.append(["cmd.exe", "/c", "start", ""] + inner)

    import subprocess
    for argv in candidates:
        try:
            subprocess.Popen(argv, cwd=cwd, close_fds=True)
            print(f"{datetime.now():%H:%M:%S} <> Claude Code launched in {cwd} "
                  f"(model {model or 'unset'})", flush=True)
            return jsonify({"status": "ok", "cwd": cwd, "model": model,
                            "base_url": base_url, "command": manual,
                            "warning": warning})
        except Exception:
            continue

    return jsonify({"status": "manual", "cwd": cwd, "model": model,
                    "base_url": base_url, "command": manual, "warning": warning,
                    "reason": "Could not open a terminal window from here."})


def _port_is_listening(port, timeout=0.25):
    """Return whether localhost accepts a TCP connection on *port*."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


@app.route("/v1/coding-mode", methods=["POST", "GET"])
def coding_mode():
    """Hold the machine for the model that drives a coding agent.

    A mode, not a button. Whisper, Kokoro and the utility engines all reload
    themselves the moment anything touches them -- entering the Voice tab calls
    /v1/audio/warm and pulls two of them straight back in -- so a one-shot
    "free memory" click would be quietly undone and the user would watch the
    memory return with no explanation. While this is on, those loads are
    refused with a message that names the toggle.

    It reports `agent_ready` as well, because freeing memory is not what makes
    a local model able to code. Tool calling is GPU/CPU only and a coding agent
    needs ~70k of context; an 8k NPU slot cannot drive one no matter how much
    RAM is free, and a toggle that said "on" while the model still could not
    call a tool would be a lie.
    """

    if request.method == "GET":
        return jsonify(_coding_mode_state())

    body = request.get_json(silent=True) or {}
    want = bool(body.get("enabled", True))
    if want == config.CODING_MODE:
        return jsonify(dict(_coding_mode_state(), changed=False))

    payload = {"changed": True}
    if want:
        before = _memory_snapshot()
        freed = []
        for slot in (runtime.whisper_slot, runtime.tts_slot, runtime.util_npu, runtime.util_gpu, runtime.util_cpu):
            if not slot or slot.status in ("not_configured", "idle_unloaded"):
                continue
            if not slot.lock.acquire(blocking=False):
                continue          # never yank a pipeline out of a live turn
            try:
                name = slot.model_name
                slot.unload()
                freed.append({"device": slot.device_name, "model": name})
            finally:
                slot.lock.release()
        config.CODING_MODE = True
        gc.collect()
        after, settled = _settle_memory()
        a, b = before.get("system_available_mb"), after.get("system_available_mb")
        payload["freed"] = freed
        payload["returned_mb"] = None if a is None or b is None else b - a
        payload["settle_seconds"] = settled
    else:
        config.CODING_MODE = False
        payload["freed"] = []
    print(f"{datetime.now():%H:%M:%S} <> coding mode "
          f"{'on' if config.CODING_MODE else 'off'}", flush=True)
    return jsonify(dict(_coding_mode_state(), **payload))


def _coding_mode_state():
    """What coding mode is, and whether the loaded model can actually code."""
    slot = next((x for x in (runtime.primary, runtime.secondary) if _slot_serviceable(x)), None)
    ready, reason = False, "No model is loaded."
    if slot:
        ctx = slot.context_tokens or 0
        if not _tools_supported(slot):
            reason = (f"{slot.model_name} is on the {slot.device_name}, which "
                      f"has no tool-calling path — an agent cannot edit files "
                      f"or run commands there. Load it on GPU or CPU.")
        elif ctx and ctx < config.CODE_MIN_CONTEXT:
            reason = (f"{slot.model_name} has {ctx // 1000}k of context; a "
                      f"coding agent needs about "
                      f"{config.CODE_MIN_CONTEXT // 1000}k. Restart with "
                      f"--context-tokens 90000.")
        else:
            ready, reason = True, None
    return {
        "enabled": bool(config.CODING_MODE),
        "agent_ready": ready,
        "reason": reason,
        "model": slot.model_name if slot else None,
        "device": slot.device_name if slot else None,
    }


def _opencode_command():
    """Resolved OpenCode executable and version, without launching it."""
    command = shutil.which("opencode")
    if not command:
        return None, None
    try:
        if os.name == "nt" and command.lower().endswith((".cmd", ".bat")):
            comspec = os.environ.get("COMSPEC") or "cmd.exe"
            argv = [comspec, "/d", "/c", command, "--version"]
        else:
            argv = [command, "--version"]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=8)
        version = (result.stdout or result.stderr or "").strip().splitlines()[0]
        return command, version or None
    except Exception:
        return command, None


def _opencode_local_provider():
    """Runtime-only provider overlay for the model locally currently serves.

    OPENCODE_CONFIG_CONTENT has higher precedence than project config but is
    merged with it, so this adds locally without deleting the user's existing
    providers (including free/subscription providers they switch to when one
    account is out of credits). No API key is persisted or written to a repo.
    """
    slot = next((item for item in (runtime.primary, runtime.secondary)
                 if _slot_serviceable(item)), None)
    if not slot:
        return {}, None
    device_name = getattr(slot, "device_name", "API")
    context = int(getattr(slot, "context_tokens", None)
                  or (8192 if device_name == "NPU" else 32768))
    output = max(2048, min(8192, context // 4))
    model_id = str(getattr(slot, "model_name", None) or "locally")
    provider = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "locally",
        "options": {
            "baseURL": f"http://127.0.0.1:{config.SERVER_PORT}/v1",
            "apiKey": "local",
        },
        "models": {
            model_id: {
                "name": f"{model_id} · {device_name}",
                "limit": {"context": context, "output": output},
            }
        },
    }
    return provider, model_id


def _opencode_status_data():
    command, version = _opencode_command()
    coding = _coding_mode_state()
    _, model_id = _opencode_local_provider()
    return {
        "installed": bool(command), "version": version,
        "local_model": model_id, "local_agent_ready": coding["agent_ready"],
        "local_reason": coding["reason"],
        "workspace": config.SCRIPT_DIR,
    }


@app.route("/v1/opencode/status", methods=["GET"])
def opencode_status():
    return jsonify(_opencode_status_data())


@app.route("/v1/opencode/launch", methods=["POST"])
def launch_opencode():
    """Open the OpenCode TUI with locally added as one provider.

    Localhost-only because this creates an interactive process on the server's
    machine. The browser may choose only an existing directory; the command is
    fixed, and the directory travels through an environment variable rather
    than being interpolated into PowerShell.
    """
    if not _request_is_local():
        return openai_error("This endpoint is local-only (it opens a terminal "
                            "on the server's machine).", "invalid_request_error", 403)
    command, version = _opencode_command()
    if not command:
        return openai_error("OpenCode is not installed. Install the opencode-ai "
                            "package, then try again.", "server_error", 503)

    body = request.get_json(silent=True) or {}
    workspace = os.path.abspath(os.path.expanduser(
        str(body.get("workspace") or config.SCRIPT_DIR).strip()))
    if not os.path.isdir(workspace):
        return openai_error(f"Workspace directory not found: {workspace}")

    env = dict(os.environ)
    overlay = {}
    raw_existing = env.get("OPENCODE_CONFIG_CONTENT")
    if raw_existing:
        try:
            parsed = json.loads(raw_existing)
            if isinstance(parsed, dict):
                overlay = parsed
        except ValueError:
            pass
    provider, model_id = _opencode_local_provider()
    if provider:
        providers = dict(overlay.get("provider") or {})
        providers["locally"] = provider
        overlay["provider"] = providers
        if _coding_mode_state()["agent_ready"] and not overlay.get("model"):
            overlay["model"] = f"locally/{model_id}"
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(overlay, separators=(",", ":"))
    env["LOCALLY_OPENCODE_WORKSPACE"] = workspace

    try:
        if os.name == "nt":
            powershell = shutil.which("pwsh") or shutil.which("powershell")
            if not powershell:
                raise RuntimeError("PowerShell was not found")
            # The command string is constant. The user-selected path is read
            # from an environment variable, so &, |, quotes or spaces in a
            # legitimate folder name never become shell syntax.
            argv = [powershell, "-NoExit", "-NoProfile", "-Command",
                    "$host.UI.RawUI.WindowTitle='OpenCode · locally'; "
                    "& opencode --dir $env:LOCALLY_OPENCODE_WORKSPACE"]
            process = subprocess.Popen(
                argv, cwd=workspace, env=env,
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        else:
            terminal = (shutil.which("x-terminal-emulator") or
                        shutil.which("gnome-terminal") or shutil.which("konsole"))
            if not terminal:
                return openai_error("No supported terminal launcher was found. "
                                    f"Run `opencode --dir {workspace}` manually.",
                                    "server_error", 501)
            name = os.path.basename(terminal).lower()
            if "gnome-terminal" in name:
                argv = [terminal, "--", command, "--dir", workspace]
            else:
                argv = [terminal, "-e", command, "--dir", workspace]
            process = subprocess.Popen(argv, cwd=workspace, env=env)
    except Exception as exc:
        return openai_error(f"Could not open OpenCode: {exc}", "server_error", 500)

    state = _opencode_status_data()
    return jsonify({"status": "started", "pid": process.pid,
                    "workspace": workspace, "version": version,
                    "local_model": model_id,
                    "local_agent_ready": state["local_agent_ready"],
                    "warning": state["local_reason"]})


@app.route("/v1/messages/count_tokens", methods=["POST"])
def anthropic_count_tokens():
    raw = request.get_json(silent=True) or {}
    slot = _route_request(False, raw.get("model", ""))
    if slot is None:
        return openai_error("No model ready.", "server_error", 503)
    body = _anthropic_to_openai(raw)
    try:
        text_prompt, _images, _raw = parse_messages(body.get("messages") or [], max_dim)
    except Exception as e:
        return openai_error(f"Failed to parse request: {e}")
    # chars/4 is the standard fallback when the slot is unloaded and we would
    # rather answer than wake a model just to count.
    n = _count_tokens(slot, text_prompt)
    return jsonify({"input_tokens": n if n is not None else len(text_prompt) // 4})


# ---------------------------------------------------------------------------
# Ollama-compatible API (port 11434)
# ---------------------------------------------------------------------------

ollama_app = Flask("locally-Ollama")
ollama_app.config["MAX_CONTENT_LENGTH"] = config.MAX_REQUEST_BYTES

OLLAMA_PORT = 11434


@ollama_app.before_request
def _debug_ollama():
    _log_request("Ollama")


@ollama_app.route("/")
def ollama_health():
    return "Ollama is running"


@ollama_app.route("/api/version", methods=["GET"])
def ollama_version():
    # VS Code's Ollama client rejects non-numeric versions, so when
    # --vscode-compat is set we report a real Ollama version to please it.
    version = VSCODE_OLLAMA_VERSION if vscode_compat else "locally-0.1.0"
    return jsonify({"version": version})


@ollama_app.route("/api/tags", methods=["GET"])
def ollama_tags():
    models = []
    for slot in (runtime.primary, runtime.secondary):
        if slot and slot.status == "ready":
            models.append({
                "name": slot.model_name,
                "model": slot.model_name,
                "size": slot.info.get("size", 0),
                "details": {
                    "family": slot.model_name.split("-")[0],
                    "parameter_size": "",
                    "quantization_level": "int4",
                },
            })
    return jsonify({"models": models})


@ollama_app.route("/api/show", methods=["POST"])
def ollama_show():
    body = request.get_json(silent=True) or {}
    model_name = body.get("model", "")

    model_info = {}
    # Copilot Chat uses `general.basename` to label models in the picker, but
    # it prefers the name returned by /api/tags. Echo back what we returned there
    # so the picker shows the same name the user sees in /api/tags.
    if request.headers.get("User-Agent", "").startswith("GitHubCopilotChat/"):
        model_info["general.basename"] = model_name

    for slot in (runtime.primary, runtime.secondary):
        if slot and slot.model_name == model_name:
            # Deliberately _tools_supported, not _tool_capable: an advert is
            # answered before any tool set exists, and the NPU's budget is a
            # property of the REQUEST. Claiming `tools` here would invite
            # Copilot to pick an NPU model for agent mode and then send it 30
            # schemas — 4,524 tokens, over half the window — which is the case
            # the budget exists to refuse. An NPU slot that will happily serve
            # a small tool set still shows as completion-only, and clients
            # that simply send `tools` (Odysseus does) get them honored.
            caps = ["completion"] + (["tools"] if _tools_supported(slot) else [])
            return jsonify({
                "model": model_name,
                "details": {
                    "family": model_name.split("-")[0],
                    "parameter_size": "",
                    "quantization_level": "int4",
                },
                "model_info": model_info,
                "capabilities": caps,
            })
    # Unknown model — we can't confirm it's on a GPU, so don't claim tools.
    return jsonify({"model": model_name, "details": {}, "model_info": model_info,
                    "capabilities": ["completion"]})


@ollama_app.route("/api/chat", methods=["POST"])
def ollama_chat():
    if overall_status() != "ready":
        return jsonify({"error": "model not ready"}), 503

    body = request.get_json(silent=True) or {}
    ollama_messages = body.get("messages", [])
    stream = body.get("stream", True)  # Ollama defaults to streaming
    requested_model = body.get("model", "")

    max_tokens = body.get("options", {}).get("num_predict", 2048)
    temperature = body.get("options", {}).get("temperature", 0.0)
    tools = body.get("tools") or []

    # Translate Ollama messages to internal format
    has_images = False
    internal_messages = []
    for msg in ollama_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        msg_images = msg.get("images", [])

        if msg_images:
            has_images = True
            blocks = [{"type": "text", "text": content}]
            for img_b64 in msg_images:
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })
            internal_messages.append({"role": role, "content": blocks})
        else:
            internal_messages.append({"role": role, "content": content})

    # Route to device
    slot = _route_request(has_images, requested_model)
    if slot is None:
        return jsonify({"error": "no model ready"}), 503

    # GPU/CPU/REMOTE always; the NPU when this particular tool set fits its
    # budget (see _npu_tools_affordable).
    tools_active = bool(tools) and _tool_capable(slot, tools)
    if tools_active:
        # Tool turns are text-only: render tool specs + prior calls into the
        # prompt. (Images + tools simultaneously is not a supported path.)
        internal_messages = prepare_messages_for_tools(ollama_messages, tools, slot)
    elif tools:
        print(f"{datetime.now():%H:%M:%S} -- [{slot.device_name}] [Ollama] "
              f"{_tools_refused_note(slot, tools)}", flush=True)

    # Parse through same pipeline as OpenAI
    try:
        text_prompt, images, raw_messages = parse_messages(internal_messages, max_dim)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if has_images and slot.model_type == "llm":
        return jsonify({"error": f"model '{slot.model_name}' does not support images"}), 400

    try:
        slot.ensure_loaded()
    except Exception as e:
        return jsonify({"error": f"Failed to reload model: {e}"}), 500

    # Capture a big agent prompt so the next startup can pre-warm it (no-op
    # unless --prewarm is set). Mirrors the OpenAI path at chat_completions —
    # clients on the plain Ollama API (pre-0.53 Copilot, Open WebUI, ...)
    # reach us through this handler, not /v1.
    if slot.model_type == "llm":
        _maybe_capture_prewarm(raw_messages)

    # Build generation config
    gen = ovg.GenerationConfig()
    gen.max_new_tokens = max_tokens
    if temperature and temperature > 0.01:
        gen.do_sample = True
        gen.temperature = temperature
    else:
        gen.do_sample = False
        gen.top_k = 1
    _opts = body.get("options", {})
    apply_penalties(gen,
                    repetition=_opts.get("repeat_penalty"),
                    frequency=_opts.get("frequency_penalty"),
                    presence=_opts.get("presence_penalty"))

    print(f"\n{datetime.now():%H:%M:%S} <- [{slot.device_name}] [Ollama] "
          f"{'image, ' if has_images else ''}{len(text_prompt)} chars"
          f"{' (stream)' if stream else ''}", flush=True)

    t0 = time.perf_counter()

    # VLM path (no streaming)
    if slot.model_type == "vlm":
        try:
            text = slot.generate_vlm(text_prompt, images, gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        elapsed = time.perf_counter() - t0
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] [Ollama] "
              f"{len(text)} chars in {elapsed:.1f}s", flush=True)

        return jsonify({
            "model": slot.model_name,
            "message": {"role": "assistant", "content": text},
            "done": True,
            "total_duration": int(elapsed * 1e9),
        })

    # LLM path. Tool turns are buffered (see chat_completions for why).
    if stream and not tools_active:
        return Response(
            _ollama_stream_chat(slot, raw_messages, gen, t0),
            mimetype="application/x-ndjson",
        )

    try:
        text = slot.generate_llm(raw_messages, gen)
    except Exception as e:
        return jsonify({"error": explain_genai_error(e)}), 500

    elapsed = time.perf_counter() - t0
    ttft = (f", TTFT {slot.last_ttft_ms:.0f}ms"
            if slot.last_ttft_ms is not None else "")
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] [Ollama] "
          f"~{len(text.split())} tokens in {elapsed:.1f}s{ttft}", flush=True)

    message = {"role": "assistant", "content": text}
    if tools_active:
        text, tool_calls = parse_tool_calls(text, tools)
        message["content"] = text
        if tool_calls:
            # Ollama shape: arguments are an object, not a JSON string.
            message["tool_calls"] = [{
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"]),
                }
            } for tc in tool_calls]
            print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] [Ollama] "
                  f"{len(tool_calls)} tool call(s): "
                  f"{', '.join(tc['function']['name'] for tc in tool_calls)}",
                  flush=True)

    final = {
        "model": slot.model_name,
        "message": message,
        "done": True,
        "total_duration": int(elapsed * 1e9),
    }
    if stream:
        # tools + stream: emit the buffered result as a single ndjson line.
        return Response(json.dumps(final) + "\n", mimetype="application/x-ndjson")
    return jsonify(final)


def _ollama_stream_chat(slot, raw_messages, gen, t0):
    """Ollama streaming: newline-delimited JSON (not SSE)."""
    history = ovg.ChatHistory()
    for msg in raw_messages:
        history.append({"role": msg["role"], "content": msg["content"]})

    token_queue = Queue()
    token_count = 0

    def streamer_callback(token):
        if slot._cancel.is_set():
            return True
        token_queue.put(token)
        return False

    def _generate():
        try:
            with slot.lock:
                slot._cancel.clear()
                slot.pipe.generate(history, gen, streamer_callback)
                slot.last_used = time.time()
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] [Ollama] "
                  f"generate error: {explain_genai_error(e)}", flush=True)
        finally:
            token_queue.put(None)

    t = threading.Thread(target=_generate, daemon=True)
    t.start()

    try:
        while True:
            try:
                token = token_queue.get(timeout=120)
            except Empty:
                break
            if token is None:
                break
            if token_count == 0:
                # Wall-clock TTFT: prefill is over when the first token lands.
                slot.last_ttft_ms = (time.perf_counter() - t0) * 1000
            token_count += 1
            yield json.dumps({
                "model": slot.model_name,
                "message": {"role": "assistant", "content": token},
                "done": False,
            }) + "\n"

        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0

        yield json.dumps({
            "model": slot.model_name,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "total_duration": int(elapsed * 1e9),
            "eval_count": token_count,
        }) + "\n"
    finally:
        slot._cancel.set()

    ttft = (f", TTFT {slot.last_ttft_ms:.0f}ms" if token_count and
            slot.last_ttft_ms is not None else "")
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] [Ollama] "
          f"{token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s{ttft})", flush=True)


@ollama_app.route("/api/generate", methods=["POST"])
def ollama_generate():
    """Single-turn completion (no chat history)."""
    if overall_status() != "ready":
        return jsonify({"error": "model not ready"}), 503

    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "")
    stream = body.get("stream", True)
    requested_model = body.get("model", "")
    max_tokens = body.get("options", {}).get("num_predict", 2048)
    temperature = body.get("options", {}).get("temperature", 0.0)

    # Images in generate endpoint
    images_b64 = body.get("images", [])
    has_images = bool(images_b64)

    slot = _route_request(has_images, requested_model)
    if slot is None:
        return jsonify({"error": "no model ready"}), 503

    try:
        slot.ensure_loaded()
    except Exception as e:
        return jsonify({"error": f"Failed to reload model: {e}"}), 500

    gen = ovg.GenerationConfig()
    gen.max_new_tokens = max_tokens
    if temperature and temperature > 0.01:
        gen.do_sample = True
        gen.temperature = temperature
    else:
        gen.do_sample = False
        gen.top_k = 1
    _opts = body.get("options", {})
    apply_penalties(gen,
                    repetition=_opts.get("repeat_penalty"),
                    frequency=_opts.get("frequency_penalty"),
                    presence=_opts.get("presence_penalty"))

    t0 = time.perf_counter()

    # VLM with images
    if has_images and slot.model_type == "vlm":
        img_tensors = []
        for b64 in images_b64:
            img = load_image(f"data:image/jpeg;base64,{b64}", max_dim)
            img_tensors.append(pil_to_tensor(img, max_dim))
        try:
            text = slot.generate_vlm(prompt, img_tensors, gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        elapsed = time.perf_counter() - t0
        return jsonify({
            "model": slot.model_name,
            "response": text,
            "done": True,
            "total_duration": int(elapsed * 1e9),
        })

    if has_images and slot.model_type == "llm":
        return jsonify({"error": "model does not support images"}), 400

    # Text-only generate → wrap as single-turn chat
    raw_messages = [{"role": "user", "content": prompt}]

    if stream and slot.model_type == "llm":
        return Response(
            _ollama_stream_generate(slot, raw_messages, gen, t0),
            mimetype="application/x-ndjson",
        )

    # Non-streaming
    if slot.model_type == "vlm":
        try:
            text = slot.generate_vlm(prompt, [], gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        try:
            text = slot.generate_llm(raw_messages, gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elapsed = time.perf_counter() - t0
    return jsonify({
        "model": slot.model_name,
        "response": text,
        "done": True,
        "total_duration": int(elapsed * 1e9),
    })


def _ollama_stream_generate(slot, raw_messages, gen, t0):
    """Ollama /api/generate streaming."""
    history = ovg.ChatHistory()
    for msg in raw_messages:
        history.append({"role": msg["role"], "content": msg["content"]})

    token_queue = Queue()
    token_count = 0

    def streamer_callback(token):
        if slot._cancel.is_set():
            return True
        token_queue.put(token)
        return False

    def _generate():
        try:
            with slot.lock:
                slot._cancel.clear()
                slot.pipe.generate(history, gen, streamer_callback)
                slot.last_used = time.time()
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] [Ollama] "
                  f"generate error: {explain_genai_error(e)}", flush=True)
        finally:
            token_queue.put(None)

    t = threading.Thread(target=_generate, daemon=True)
    t.start()

    try:
        while True:
            try:
                token = token_queue.get(timeout=120)
            except Empty:
                break
            if token is None:
                break
            token_count += 1
            yield json.dumps({
                "model": slot.model_name,
                "response": token,
                "done": False,
            }) + "\n"

        elapsed = time.perf_counter() - t0
        yield json.dumps({
            "model": slot.model_name,
            "response": "",
            "done": True,
            "total_duration": int(elapsed * 1e9),
            "eval_count": token_count,
        }) + "\n"
    finally:
        slot._cancel.set()


# Stubs — clients expect these to exist
@ollama_app.route("/api/pull", methods=["POST"])
def ollama_pull():
    return jsonify({"status": "success"})


@ollama_app.route("/api/delete", methods=["DELETE"])
def ollama_delete():
    return "", 200


@ollama_app.route("/api/copy", methods=["POST"])
def ollama_copy():
    return "", 200


# Copilot Chat 0.53+ sends actual chat via /v1/chat/completions on the Ollama
# port rather than /api/chat — delegate to the same handler.
@ollama_app.route("/v1/chat/completions", methods=["POST"])
def ollama_v1_chat_completions():
    return chat_completions()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def check_port(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False


def detect_devices():
    """Return {kind: {"id": ov_id, "name": full_name}} of usable devices.

    kind is the canonical category ("NPU", "GPU", "CPU"). For "GPU", "id"
    may be "GPU.0", "GPU.1", etc. when OpenVINO enumerates multiple GPUs;
    callers must pass "id" to OpenVINO, not "kind".

    Non-Intel GPUs are filtered out: OpenVINO's intel_gpu plugin enumerates
    any OpenCL-capable GPU (NVIDIA, AMD), but its kernels only run on Intel
    hardware. Selecting a non-Intel GPU produces hundreds of compile errors
    and crashes at warmup with CL_INVALID_VALUE — better not to offer it.
    """
    devices = {}
    core = ov.Core()
    for d in core.get_available_devices():
        try:
            full_name = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            full_name = d
        if d.startswith("GPU"):
            if "intel" not in full_name.lower():
                continue
            if "GPU" not in devices:  # first Intel GPU wins
                devices["GPU"] = {"id": d, "name": full_name}
        elif d in ("NPU", "CPU"):
            devices[d] = {"id": d, "name": full_name}
    return devices


def _idle_watchdog(slots, idle_timeout, check_interval=30):
    """Background thread: unload slots that have been idle too long."""
    while True:
        time.sleep(check_interval)
        now = time.time()
        for slot in slots:
            if not slot or slot.status != "ready":
                continue
            if now - slot.last_used < idle_timeout:
                continue
            # Try non-blocking lock acquire — skip if a request is in progress
            if not slot.lock.acquire(blocking=False):
                continue
            try:
                slot.unload()
            finally:
                slot.lock.release()
        # SearXNG follows the same rule as every slot: resident only while it
        # is being used. Only ever stops a process this server started.
        _searx.maybe_stop_idle()


_banner_lock = threading.Lock()
_banner_printed = False


def _load_in_background(slot, model_dir, devices, port, ollama_port, banner_slots):
    """Background thread: load model + warmup on one device."""
    global _banner_printed
    try:
        if isinstance(slot, ProxySlot):
            # device_full already names the upstream URL, which is the only
            # honest answer to "where does this run"; the device table has no
            # entry for a machine we are merely talking to. Prewarm is skipped
            # for the same reason: it exists to fill a local prefix cache this
            # slot does not have, so it would spend a real upstream request to
            # warm nothing.
            slot.load()
            slot.warmup()
        else:
            slot.device_full = devices.get(slot.device_name, {}).get("name", slot.device_name)
            slot.load(model_dir)
            slot.warmup()
            _prewarm_slot(slot)
    except Exception as e:
        slot.status = "error"
        print(f"\n  [{slot.device_name}] ERROR: Failed to load model: {explain_genai_error(e)}")
        if (not isinstance(slot, ProxySlot)
                and not any(s in str(e) for s in ("Could not find a model",
                                                  "is truncated",
                                                  "Compilation failed"))):
            # Device-contention hint only where it's plausible — for a
            # missing/truncated model or a compiler failure it sends people
            # chasing ghosts (#17, #20 — the latter is literally titled
            # after this hint).
            print(f"  Is another process using the {slot.device_name}?", flush=True)

    # Print banner when all slots are done — only one thread wins
    with _banner_lock:
        if _banner_printed:
            return
        all_done = all(
            s.status in ("ready", "error", "not_configured")
            for s in banner_slots
        )
        if not all_done:
            return
        _banner_printed = True

    if any(s.status == "ready" for s in banner_slots):
        lines = []
        for s in banner_slots:
            if s.status == "ready":
                lines.append(f"    {s.device_name:5s}: {s.model_name} ({s.model_type.upper()}) "
                             f"-- {s.device_full}")
        url = f"http://localhost:{port}"
        api_lines = [f"    API  : {url}  (OpenAI)"]
        if ollama_port:
            api_lines.append(f"    API  : http://localhost:{ollama_port}  (Ollama)")
        print(f"""
================================================
  locally ready
{chr(10).join(lines)}
{chr(10).join(api_lines)}
================================================
""", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_model = str(Path(__file__).parent / "model")
    p.add_argument("--model-dir", default=default_model,
                   help="Primary model directory (default: model/)")
    p.add_argument("--device", default="auto",
                   help="Device for primary model: NPU, GPU, CPU, or auto (default: auto)")
    # A machine with no Intel silicon has no device to place a generative model
    # on, which used to make the whole app unreachable there — utilities, voice
    # and UI included — for want of a chat model. Pointing the primary slot at
    # an OpenAI-compatible server (Ollama, vLLM, llama.cpp) runs the same
    # program on that machine. --model-dir is not required in this mode.
    p.add_argument("--proxy-url", default=None,
                   help="Serve the primary slot from an OpenAI-compatible server "
                        "instead of a local model, e.g. http://localhost:11434 "
                        "(Ollama). Replaces --model-dir/--device for chat.")
    p.add_argument("--proxy-model", default=None,
                   help="Model id to request upstream (default: the first the "
                        "upstream advertises)")
    p.add_argument("--proxy-key", default=None,
                   help="Bearer token for the upstream, if it needs one")
    p.add_argument("--gpu-model-dir", default=None,
                   help="Secondary GPU model (enables dual mode: NPU chat + GPU vision/LLM)")
    p.add_argument("--port", type=int, default=8000,
                   help="OpenAI API port (default: 8000)")
    # Note: the CLI flag is --ollama-port (dash), read back as args.ollama_port
    # (underscore) — argparse converts dashes to underscores in attribute names
    # by design, per Python convention. Same for every --two-word flag here.
    p.add_argument("--ollama-port", type=int, default=11434,
                   help="Ollama API port (default: 11434, 0 to disable)")
    p.add_argument("--max-dim", type=int, default=768,
                   help="Max image dimension before resize (default: 768)")
    p.add_argument("--whisper-dir", default=None,
                   help="Whisper model directory for speech-to-text (enables /v1/audio/transcriptions)")
    p.add_argument("--no-model-cache", action="store_true",
                   help="Disable the OpenVINO compiled-model cache. The cache "
                        "trades disk for startup: the first load of a model "
                        "compiles it, later loads read the compiled blob back "
                        "(the NPU otherwise rebuilds its graph on every start).")
    p.add_argument("--model-cache-dir", default=None, metavar="DIR",
                   help="Where to keep compiled-model blobs "
                        "(default: <script dir>/.ov-cache).")
    p.add_argument("--models-dir", default=None, metavar="DIR",
                   help="Directory holding model folders, offered by "
                        "GET /v1/models/available and loadable at runtime via "
                        "POST /v1/models/load (no restart). Defaults to the "
                        "parent of the loaded models plus ~/models.")
    p.add_argument("--tts-dir", default=None,
                   help="Text-to-speech model directory (SpeechT5 or Kokoro IR). "
                        "Enables POST /v1/audio/speech.")
    p.add_argument("--tts-device", default="CPU",
                   help="Device for TTS (default: CPU — keeps the GPU free for "
                        "ASR and the NPU free for chat).")
    p.add_argument("--whisper-device", default="CPU",
                   help="Device for Whisper: CPU or GPU (default: CPU)")
    p.add_argument("--vad-dir", default=None,
                   help="Silero voice-activity model (silero_vad.onnx, or the "
                        "folder holding it). Enables auto turn-taking in the "
                        "Voice tab; without it voice is push-to-talk.")
    p.add_argument("--vad-device", default="CPU",
                   help="Device for the VAD (default: CPU — it is ~1M "
                        "parameters and belongs nowhere near the NPU).")
    p.add_argument("--speaker-dir", default=None,
                   help="Speaker-embedding model (ECAPA-TDNN / WeSpeaker .onnx "
                        "or .xml, or the folder holding it). Gates turn-taking "
                        "on WHOSE voice it is; without it any speech in the "
                        "room can open a turn or interrupt the assistant.")
    p.add_argument("--speaker-device", default="CPU",
                   help="Device for speaker ID (default: CPU — it runs once "
                        "per turn, not per frame).")
    p.add_argument("--speaker-threshold", type=float, default=0.35,
                   help="Cosine similarity above which a voice is the enrolled "
                        "user (default: 0.35). Every verification logs its "
                        "score, so tune from your own numbers, not this one.")
    p.add_argument("--util-models-dir", default=None,
                   help="Directory of utility models holding ocr-det/, "
                        "ocr-rec/, and optional background/, upscale/, "
                        "detect-rfdetr/, and embed/ folders. Enables the "
                        "Utilities sidebar and /v1/util/* endpoints. Auto-detected "
                        "from ~/models/util when omitted. Fetch OCR with "
                        "`python scripts/npu-probe.py --all`.")
    p.add_argument("--no-util", action="store_true",
                   help="Don't load utility models even if they are found.")
    p.add_argument("--check-updates", action="store_true",
                   help="Check GitHub/npm once a day for newer locally, "
                        "Odysseus, SearXNG and OpenCode versions. Off by "
                        "default; never delays startup or auto-updates.")
    p.add_argument("--util-engines", default="auto",
                   help="Which engines load the utility models: 'npu', 'gpu', "
                        "'cpu', a comma-separated list, or 'auto' (default). "
                        "Auto takes every Intel accelerator detected, and falls "
                        "back to CPU only when there is none — which is the "
                        "machine --proxy-url exists for, where the utilities are "
                        "the whole reason to run locally at all. The NPU stays "
                        "the default *engine* for a request because the GPU is "
                        "usually busy holding a chat model — but some tasks are "
                        "GPU-only (Swin2SR upscaling will not compile on the "
                        "NPU), so a slot has to exist there or that tier is "
                        "unreachable.")
    p.add_argument("--idle-timeout", type=int, default=1800,
                   help="Change idle-unload timeout in seconds "
                        "(default: 1800 = 30 min). Use 0 to disable unloading "
                        "(recommended for agent use; also auto-enables --prewarm).")
    p.add_argument("--debug", action="store_true",
                   help="Log every inbound API request (method, path, User-Agent, body)")
    p.add_argument("--vscode-compat", action="store_true",
                   help=f"Report a real Ollama version ({VSCODE_OLLAMA_VERSION}) on "
                        f"/api/version so VS Code's Ollama client accepts the server")
    p.add_argument("--no-prompt-cache", action="store_true",
                   help="Disable prefix (KV) caching on GPU/CPU LLM slots. Caching is "
                        "ON by default — it prefills a repeated prompt prefix (e.g. an "
                        "agent's fixed system prompt) once instead of every turn.")
    p.add_argument("--cache-size-gb", type=int, default=config.PROMPT_CACHE_GB,
                   help=f"KV-cache pool size in GB when prefix caching is on "
                        f"(default: {config.PROMPT_CACHE_GB})")
    p.add_argument("--prewarm", default=None, metavar="FILE",
                   help="Prefill a saved agent prompt at startup so the first turn is a "
                        "cache hit (no cold-prefill stall). The file auto-populates from the "
                        "first big prompt served, so: run once, then restart with --prewarm. "
                        "Auto-enabled (as prewarm-<port>.json) when --idle-timeout is 0.")
    p.add_argument("--no-prewarm", action="store_true",
                   help="Disable the automatic prewarm that --idle-timeout 0 turns on.")
    p.add_argument("--offload-ratio", default="auto", metavar="PCT",
                   help="Stream PCT%% of MoE expert weights from disk instead of "
                        "keeping them GPU-resident (OpenVINO 2026.3+ disk offload). "
                        "Lets 30B-class MoE models run on 16 GB-class GPUs at the "
                        "cost of decode speed. Default 'auto': compute the smallest "
                        "ratio that makes the model fit alongside its KV pool, and "
                        "stay off entirely when it already fits. '0' disables it; "
                        "1-99 pins a value. GPU + XMX + MoE only — it does nothing "
                        "on dense models or non-XMX GPUs, by design of the plugin.")
    p.add_argument("--context-tokens", default=None, metavar="N",
                   help="Size the KV pool by the context you need, instead of "
                        "guessing gigabytes with --cache-size-gb: N tokens is "
                        "converted using the model's own KV geometry, and "
                        "'auto' takes whatever the device has left after the "
                        "weights. Capped at the model's real context length. "
                        "This is the ceiling an agent session actually hits — "
                        "overrunning it fails the request outright.")
    p.add_argument("--npu-prompt-len", type=int, default=None, metavar="N",
                   help="NPU context window (default 8192; 4096 cuts ~0.34s off "
                        "time-to-first-token). The NPU compiles a static graph, "
                        "so prefill costs the same whatever the prompt length - "
                        "this trades context for latency. 8192 is the compiler "
                        "ceiling; see TODONT.md.")
    p.add_argument("--searxng-root", default=None, metavar="DIR",
                   help="SearXNG checkout to start on demand and stop when "
                        "idle. Without it, SearXNG must already be running.")
    p.add_argument("--searxng-idle", type=int, default=600, metavar="SECS",
                   help="Stop an auto-started SearXNG after this many idle "
                        "seconds (0 keeps it up).")
    p.add_argument("--search-url", default=None, metavar="URL",
                   help="SearXNG base URL for web search, e.g. "
                        "http://localhost:8080. Unset (the default) means the "
                        "server makes no outbound connections at all.")
    p.add_argument("--python-tool", action="store_true",
                   help="Enable the server-side local Python calculation tool "
                        "(off by default; use only for local model turns).")
    p.add_argument("--python-timeout", type=float, default=config.PYTHON_TOOL_TIMEOUT,
                   metavar="SECS",
                   help="Hard timeout for one Python calculation child "
                        f"(default: {config.PYTHON_TOOL_TIMEOUT:g}s).")
    p.add_argument("--python-output-bytes", type=int,
                   default=config.PYTHON_TOOL_OUTPUT_BYTES, metavar="N",
                   help="Maximum stdout/stderr bytes from one calculation "
                        f"(default: {config.PYTHON_TOOL_OUTPUT_BYTES}).")
    p.add_argument("--kv-precision", default=None, choices=["u8", "f16"],
                   metavar="P",
                   help="KV cache element type. 'u8' halves cache bytes per "
                        "token (96 KB -> ~48 KB on a 30B), which is what lets a "
                        "100k-token session fit a 24 GB GPU budget. Quality cost "
                        "is far below that of int4 weights. GPU/CPU only.")
    p.add_argument("--prompt-cache", dest="prompt_cache", action="store_true",
                   default=None,
                   help="Force the prefix-caching (continuous-batching) backend "
                        "on. Default is per-device: on for CPU, OFF for GPU, "
                        "where it measured ~10x slower and hangs on repeated "
                        "prefixes (TODONT.md).")
    p.add_argument("--agent-tools", default=None, metavar="NAMES",
                   help="Comma-separated tool names to keep from a client's "
                        "tool list; the rest are dropped before the prompt is "
                        "built. Tool schemas are ~68%% of an agent request "
                        "(30 tools = 99k of Claude Code's 142k chars), and "
                        "prefilling them is most of the wait on an iGPU. Try "
                        "\"Bash,Edit,Read,Write,Glob,Grep\".")
    p.add_argument("--scan", nargs="*", default=None, metavar="DIR",
                   help="Report what each model directory actually contains "
                        "(name, precision, architecture, integrity) and exit. "
                        "Searches the locally directory and ~/models by "
                        "default; pass directories to search those instead.")
    return p.parse_args()


def main():
    global max_dim, debug, vscode_compat

    global WEB_SEARCH_URL
    global CHECK_UPDATES, OLLAMA_COMPAT_PORT, RUNTIME_SLOTS
    global SEARXNG_ROOT, SEARXNG_IDLE
    global PYTHON_TOOL_ENABLED

    args = parse_args()

    # First-run choices live outside the repository and only fill values the
    # caller did not explicitly provide. Command-line flags remain the final
    # authority, while the launcher can become one-click after onboarding.
    saved_setup = _read_setup_config()
    saved_assistant = saved_setup.get("assistant") or {}
    default_model = os.path.abspath(os.path.join(config.SCRIPT_DIR, "model"))
    using_default_model = (os.path.abspath(os.path.expanduser(args.model_dir)) ==
                           default_model)
    if not args.proxy_url and using_default_model:
        if saved_assistant.get("backend") == "ollama":
            args.proxy_url = saved_assistant.get("url") or "http://127.0.0.1:11434"
            args.proxy_model = saved_assistant.get("model") or args.proxy_model
        elif (saved_assistant.get("backend") == "openvino" and
              os.path.isdir(saved_assistant.get("model_dir") or "")):
            args.model_dir = saved_assistant["model_dir"]
            args.device = saved_assistant.get("device") or args.device
    saved_voice = saved_setup.get("voice") or {}
    if not args.whisper_dir and os.path.isdir(saved_voice.get("stt_dir") or ""):
        args.whisper_dir = saved_voice["stt_dir"]
    if not args.tts_dir and os.path.isdir(saved_voice.get("tts_dir") or ""):
        args.tts_dir = saved_voice["tts_dir"]
    if not args.vad_dir and os.path.isdir(saved_voice.get("vad_dir") or ""):
        args.vad_dir = saved_voice["vad_dir"]
    if not args.speaker_dir and os.path.isdir(saved_voice.get("speaker_dir") or ""):
        args.speaker_dir = saved_voice["speaker_dir"]
    saved_web = saved_setup.get("web") or {}
    if saved_web.get("check_updates"):
        args.check_updates = True
    if not args.search_url and saved_web.get("searxng"):
        args.search_url = saved_web.get("url") or "http://127.0.0.1:8080"
        if not args.searxng_root:
            args.searxng_root = saved_web.get("root")

    # --scan is a report, not a server: no ports, no devices, no model load.
    if args.scan is not None:
        print(flush=True)
        scan_models(args.scan)
        return

    model_dir = os.path.expanduser(args.model_dir)
    config.SERVER_PORT = args.port
    CHECK_UPDATES = bool(args.check_updates)
    config.AGENT_TOOLS = ({t.strip() for t in args.agent_tools.split(",") if t.strip()}
                   if args.agent_tools else None)
    max_dim = args.max_dim
    debug = args.debug
    vscode_compat = args.vscode_compat
    # Tri-state: explicit flags win, otherwise per-device (see config.PROMPT_CACHE).
    config.PROMPT_CACHE = (False if args.no_prompt_cache
                    else (True if args.prompt_cache else None))
    config.KV_PRECISION = args.kv_precision
    if args.npu_prompt_len:
        if args.npu_prompt_len > 8192:
            print(f"WARNING: --npu-prompt-len {args.npu_prompt_len} is above the "
                  f"measured vpux ceiling of 8192 and will fail to compile; "
                  f"using 8192", flush=True)
        config.NPU_MAX_PROMPT_LEN = min(8192, max(1024, args.npu_prompt_len))
    SEARXNG_IDLE = max(0, int(args.searxng_idle))
    root = (args.searxng_root or "").strip()
    if not root:
        # Convention over configuration: a sibling checkout is the layout
        # scripts/searxng.ps1 -Install produces.
        for guess in (os.path.join(os.path.dirname(config.SCRIPT_DIR), "searxng-src"),
                      os.path.join(config.SCRIPT_DIR, "searxng-src")):
            if os.path.isdir(guess):
                root = guess
                break
    SEARXNG_ROOT = root if root and os.path.isdir(root) else None
    WEB_SEARCH_URL = (args.search_url or "").strip() or None
    if WEB_SEARCH_URL:
        print(f"  Web search via {WEB_SEARCH_URL} "
              f"(the only outbound network path this server has)", flush=True)
    PYTHON_TOOL_ENABLED = bool(args.python_tool)
    config.PYTHON_TOOL_TIMEOUT = max(0.1, min(60.0, float(args.python_timeout)))
    config.PYTHON_TOOL_OUTPUT_BYTES = max(1024, min(16 * 1024 * 1024,
                                              int(args.python_output_bytes)))
    if PYTHON_TOOL_ENABLED:
        print(f"  Python calculations enabled (child timeout "
              f"{config.PYTHON_TOOL_TIMEOUT:g}s, output cap "
              f"{config.PYTHON_TOOL_OUTPUT_BYTES} bytes)", flush=True)
    config.PROMPT_CACHE_GB = args.cache_size_gb
    if args.context_tokens is not None:
        raw = str(args.context_tokens).strip().lower()
        if raw == "auto":
            config.CONTEXT_TOKENS = "auto"
        else:
            try:
                config.CONTEXT_TOKENS = int(raw.replace("k", "000").replace(",", ""))
            except ValueError:
                print(f"WARNING: --context-tokens '{args.context_tokens}' is not "
                      f"a number or 'auto' — ignoring.", flush=True)
    raw_offload = str(args.offload_ratio).strip().lower()
    if raw_offload == "auto":
        config.OFFLOAD_RATIO = "auto"
    else:
        try:
            config.OFFLOAD_RATIO = max(0, min(99, int(raw_offload)))
        except ValueError:
            print(f"WARNING: --offload-ratio '{args.offload_ratio}' is not a "
                  f"number or 'auto' — using auto.", flush=True)
            config.OFFLOAD_RATIO = "auto"
    # A pinned ratio on a GPU without XMX is silently ignored by the plugin —
    # say so up front instead of letting the user believe their model got
    # smaller (see TODONT.md). 'auto' needs no warning: it checks for XMX
    # itself and simply stays off.
    if config.OFFLOAD_RATIO not in ("auto", 0) and not _gpu_has_xmx():
        print("WARNING: --offload-ratio set, but this GPU has no XMX "
              "(OPTIMIZATION_CAPABILITIES lacks GPU_HW_MATMUL). MoE disk "
              "offload will silently do nothing — the model must fit in "
              "GPU memory.", flush=True)
    if args.prewarm:
        config.PREWARM_FILE = os.path.expanduser(args.prewarm)
    elif args.idle_timeout == 0 and not args.no_prewarm and config.PROMPT_CACHE:
        # Agent/server mode (--idle-timeout 0 keeps models loaded forever) is
        # exactly where prewarm pays off — and the idle unload that would
        # throw the warmed cache away can't happen, so turn it on. Port-scoped
        # filename so two instances on one install don't overwrite each other.
        config.PREWARM_FILE = os.path.join(config.SCRIPT_DIR, f"prewarm-{args.port}.json")
    else:
        config.PREWARM_FILE = None

    # Quiet Flask/Werkzeug startup noise: kills the "Serving Flask app" /
    # "Debug mode: off" / "Running on http://..." / "Press CTRL+C to quit"
    # block (printed twice — once per Flask app), the dev-server warning,
    # and per-request access logs that would otherwise flood the console
    # in normal use. locally.py has its own app-level request logging.
    import logging
    import flask.cli
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    flask.cli.show_server_banner = lambda *a, **k: None

    print(flush=True)

    # 1. Check ports
    if not check_port(args.port):
        print(f"ERROR: Port {args.port} is already in use.")
        print(f"Use --port <number> to pick another port.")
        sys.exit(1)
    if args.ollama_port and not check_port(args.ollama_port):
        print(f"WARNING: Ollama port {args.ollama_port} is in use. "
              f"Ollama API disabled. (Is Ollama already running?)")
        args.ollama_port = 0
    OLLAMA_COMPAT_PORT = args.ollama_port

    # 2. Detect devices
    devices = detect_devices()
    runtime.DEVICES.update(devices)   # runtime placement can then move a slot to any device
    print("  Devices:", flush=True)
    for kind, info in devices.items():
        suffix = f" [{info['id']}]" if info['id'] != kind else ""
        print(f"    {kind}{suffix}: {info['name']}")
    print()

    def _id_of(kind):
        return devices.get(kind, {}).get("id", kind)

    # 3. Resolve primary device
    device = args.device.upper()
    if device == "AUTO":
        if args.gpu_model_dir:
            # Dual mode: primary goes on NPU (or CPU if no NPU)
            device = "NPU" if "NPU" in devices else "CPU"
        elif "NPU" in devices:
            device = "NPU"
        elif "GPU" in devices:
            device = "GPU"
        else:
            device = "CPU"

    if device not in devices and device != "CPU":
        print(f"ERROR: Device {device} not available. Found: {list(devices.keys())}")
        sys.exit(1)

    # 4. Verify model directories
    setup_only = not args.proxy_url and not os.path.isdir(model_dir)
    if setup_only:
        print(f"  No assistant model found at {model_dir}.")
        print("  Starting in setup mode — the web UI can install one.", flush=True)
    if args.gpu_model_dir and not os.path.isdir(args.gpu_model_dir):
        print(f"ERROR: GPU model directory not found: {args.gpu_model_dir}")
        sys.exit(1)
    if args.whisper_dir and not os.path.isdir(args.whisper_dir):
        print(f"ERROR: Whisper model directory not found: {args.whisper_dir}")
        sys.exit(1)
    if args.tts_dir and not os.path.isdir(args.tts_dir):
        print(f"ERROR: TTS model directory not found: {args.tts_dir}")
        sys.exit(1)
    if args.util_models_dir:
        if not os.path.isdir(args.util_models_dir):
            print(f"ERROR: --util-models-dir not found: {args.util_models_dir}")
            sys.exit(1)
        runtime.UTIL_DIR = os.path.abspath(args.util_models_dir)
    elif not args.no_util:
        # Find the utility models without being told where they are. An
        # explicit flag meant every existing launch script silently omitted it,
        # so the models sat on disk while the UI reported "not installed" —
        # a setup error dressed up as a missing download. Detection is cheap
        # and unambiguous: the OCR pair is what the Util tab needs at minimum.
        for cand in (os.path.join(os.path.expanduser("~"), "models", "util"),
                     os.path.join(config.SCRIPT_DIR, "models", "util"),
                     os.path.join(config.SCRIPT_DIR, "util-models")):
            if all(os.path.isfile(os.path.join(cand, sub, "inference.onnx"))
                   for sub in ("ocr-det", "ocr-rec")):
                runtime.UTIL_DIR = os.path.abspath(cand)
                print(f"  Utilities: found models in {runtime.UTIL_DIR}", flush=True)
                break
    if args.models_dir:
        if not os.path.isdir(args.models_dir):
            print(f"ERROR: --models-dir not found: {args.models_dir}")
            sys.exit(1)
        runtime.MODELS_DIR = os.path.abspath(args.models_dir)
    elif setup_only:
        # The setup downloader needs one stable destination even before a
        # first model exists. Do not create it until the user clicks Install.
        runtime.MODELS_DIR = os.path.abspath(os.path.expanduser("~/models"))

    if not args.no_model_cache:
        config.MODEL_CACHE_DIR = os.path.abspath(
            args.model_cache_dir or os.path.join(config.SCRIPT_DIR, ".ov-cache"))
        try:
            os.makedirs(config.MODEL_CACHE_DIR, exist_ok=True)
        except OSError as e:
            print(f"  WARNING: can't use model cache ({e}); compiling every start")
            config.MODEL_CACHE_DIR = None

    # 5. Create device slots
    if args.proxy_url:
        runtime.primary = ProxySlot(args.proxy_url, model=args.proxy_model,
                            api_key=args.proxy_key)
        print(f"  Primary: proxied to {args.proxy_url}"
              + (f" ({args.proxy_model})" if args.proxy_model else ""), flush=True)
    else:
        runtime.primary = DeviceSlot(device, _id_of(device))
    all_slots = [runtime.primary]
    RUNTIME_SLOTS = all_slots

    if args.gpu_model_dir:
        if "GPU" not in devices:
            print("WARNING: --gpu-model-dir given but no GPU detected. Ignoring.")
        else:
            runtime.secondary = DeviceSlot("GPU", _id_of("GPU"))
            all_slots.append(runtime.secondary)

    if args.whisper_dir:
        whisper_device = args.whisper_device.upper()
        if whisper_device not in devices and whisper_device != "CPU":
            print(f"WARNING: Whisper device {whisper_device} not available, falling back to CPU.")
            whisper_device = "CPU"
        runtime.whisper_slot = WhisperSlot(whisper_device, _id_of(whisper_device))
        all_slots.append(runtime.whisper_slot)

    if args.tts_dir:
        tts_device = args.tts_device.upper()
        if tts_device not in devices and tts_device != "CPU":
            print(f"WARNING: TTS device {tts_device} not available, falling back to CPU.")
            tts_device = "CPU"
        runtime.tts_slot = TtsSlot(tts_device, _id_of(tts_device))
        all_slots.append(runtime.tts_slot)

    if args.vad_dir:
        vad_device = args.vad_device.upper()
        if vad_device not in devices and vad_device != "CPU":
            print(f"WARNING: VAD device {vad_device} not available, falling back to CPU.")
            vad_device = "CPU"
        runtime._vad_slot = VadSlot(vad_device, _id_of(vad_device))
        all_slots.append(runtime._vad_slot)
        if not AUDIO_STREAM_OK:
            print("WARNING: --vad-dir given but flask-sock is not installed, so "
                  "auto turn-taking has no socket to run over. "
                  "`pip install flask-sock`. Voice falls back to push-to-talk.")

    if args.speaker_dir:
        spk_device = args.speaker_device.upper()
        if spk_device not in devices and spk_device != "CPU":
            print(f"WARNING: speaker device {spk_device} not available, "
                  f"falling back to CPU.")
            spk_device = "CPU"
        runtime._speaker_slot = SpeakerSlot(spk_device, _id_of(spk_device),
                                    threshold=args.speaker_threshold)
        all_slots.append(runtime._speaker_slot)
        if not args.vad_dir:
            # Verification runs on the utterance the VAD delimits. Without the
            # VAD there are no server-side turns to gate -- push-to-talk is
            # already the user deciding when to speak.
            print("WARNING: --speaker-dir without --vad-dir does nothing: "
                  "speaker ID gates the VAD's turns, and push-to-talk has none.")

    if runtime.UTIL_DIR:
        # "auto" means every engine this box actually has. It is the default
        # because the tier map now contains GPU-only models (the vpux compiler
        # rejects Swin2SR), and with an NPU-only default that tier had no slot
        # to run on — the capability existed and nothing could reach it. Slots
        # are cheap: only OCR loads eagerly (~160 MB), everything else compiles
        # on first use and idle-unloads afterwards.
        auto = args.util_engines.strip().lower() == "auto"
        # Under "auto", CPU is the fallback rather than an addition: it is only
        # taken when the box has no Intel accelerator at all. That is the
        # machine the proxy slot exists for -- an NVIDIA desktop where Ollama
        # generates and locally supplies the UI and the utilities. Adding a CPU
        # slot alongside an NPU would cost real memory to duplicate models that
        # already have a faster home.
        accel = [e for e in ("NPU", "GPU") if e in devices]
        wanted = ((accel or ["CPU"]) if auto else
                  [e.strip().upper() for e in args.util_engines.split(",") if e.strip()])
        for engine in wanted:
            if engine not in ("NPU", "GPU", "CPU"):
                print(f"WARNING: unknown --util-engines value {engine!r}, ignoring.")
                continue
            if engine not in devices:
                # Silent under "auto" — asking for what the machine has cannot
                # be a mistake worth warning about on every start.
                if not auto:
                    print(f"WARNING: --util-engines asked for {engine} but none "
                          f"was detected. Utilities will not be available there.")
                continue
            slot = UtilSlot(engine, _id_of(engine))
            if engine == "NPU":
                runtime.util_npu = slot
            elif engine == "GPU":
                runtime.util_gpu = slot
            else:
                runtime.util_cpu = slot
            all_slots.append(slot)

    # 6. Start Flask, load models in background
    ports_msg = f"port {args.port}"
    if args.ollama_port:
        ports_msg += f" + Ollama on {args.ollama_port}"
    print(f"  Starting server on {ports_msg}...", flush=True)

    threads = []
    if args.proxy_url or not setup_only:
        t = threading.Thread(
            target=_load_in_background,
            args=(runtime.primary, model_dir, devices, args.port, args.ollama_port, all_slots),
            daemon=True,
        )
        threads.append(t)
    else:
        print(f"  Setup UI: http://localhost:{args.port}", flush=True)

    if runtime.secondary:
        t2 = threading.Thread(
            target=_load_in_background,
            args=(runtime.secondary, args.gpu_model_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        )
        threads.append(t2)

    if runtime.whisper_slot:
        tw = threading.Thread(
            target=_load_in_background,
            args=(runtime.whisper_slot, args.whisper_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        )
        threads.append(tw)

    if runtime.tts_slot:
        tt = threading.Thread(
            target=_load_in_background,
            args=(runtime.tts_slot, args.tts_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        )
        threads.append(tt)

    if runtime._vad_slot:
        threads.append(threading.Thread(
            target=_load_in_background,
            args=(runtime._vad_slot, args.vad_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        ))

    if runtime._speaker_slot:
        threads.append(threading.Thread(
            target=_load_in_background,
            args=(runtime._speaker_slot, args.speaker_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        ))

    for util in (runtime.util_npu, runtime.util_gpu, runtime.util_cpu):
        if util:
            threads.append(threading.Thread(
                target=_load_in_background,
                args=(util, runtime.UTIL_DIR, devices, args.port,
                      args.ollama_port, all_slots),
                daemon=True,
            ))

    for t in threads:
        t.start()

    # Idle watchdog — unload models after inactivity
    if args.idle_timeout > 0:
        print(f"  Idle unload after {args.idle_timeout}s of inactivity", flush=True)
        if config.PREWARM_FILE:
            # The prefix cache lives in the pipeline; unload drops both and the
            # reload path doesn't re-warm (a synchronous re-warm would stall
            # the triggering request for the whole prefill).
            print(f"  WARNING: --prewarm + idle unload: the warmed cache is lost "
                  f"when the model idle-unloads and not rebuilt until restart. "
                  f"Use --idle-timeout 0 to keep it.", flush=True)
        watchdog = threading.Thread(
            target=_idle_watchdog,
            args=(all_slots, args.idle_timeout),
            daemon=True,
        )
        watchdog.start()
    elif config.PREWARM_FILE and not args.prewarm:
        print(f"  Prewarm auto-enabled (--idle-timeout 0): "
              f"{os.path.basename(config.PREWARM_FILE)} (--no-prewarm to disable)", flush=True)

    # Suppress Flask's default "Serving Flask app" banner — we have our own
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    # Start Ollama API on separate port in background thread
    if args.ollama_port:
        print(f"  Ollama API on port {args.ollama_port}", flush=True)
        def _run_ollama():
            try:
                ollama_app.run(
                    host="0.0.0.0", port=args.ollama_port, threaded=True,
                )
            except Exception as e:
                print(f"  WARNING: Ollama API failed to start: {e}", flush=True)
        ollama_thread = threading.Thread(target=_run_ollama, daemon=True)
        ollama_thread.start()

    # OpenAI API on main thread
    print(f"  OpenAI API on port {args.port}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
