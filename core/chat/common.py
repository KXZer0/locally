"""Request parsing, OpenAI SSE framing, and prompt prewarming."""

import configparser
import hashlib
import itertools
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import openvino_genai as ovg

from core import config, runtime
from core.genai.results import explain_genai_error
from core.media.images import load_image, pil_to_tensor
from core.tools.parse import parse_tool_calls

_ini = configparser.ConfigParser()
_ini.read([Path(config.SCRIPT_DIR) / "nollama.ini",
           Path(config.SCRIPT_DIR) / "locally.ini"])
REPETITION_PENALTY = _ini.getfloat("generation", "repetition_penalty", fallback=1.05)
_prewarm_hash = None
_request_counter = itertools.count(1)

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
    generation_finish = getattr(text, "finish_reason", "stop")
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
          f"Buffered tool response completed in {elapsed:.1f}s", flush=True)

    text, tool_calls = parse_tool_calls(text, tools)
    if tool_calls:
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
              f"{len(tool_calls)} tool call(s): "
              f"{', '.join(tc['function']['name'] for tc in tool_calls)}", flush=True)
        message = {"role": "assistant", "content": text or None, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish_reason = generation_finish

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


