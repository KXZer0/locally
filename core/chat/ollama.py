"""Ollama-compatible request and response adapters."""

import base64
import json
import time
import threading
from queue import Empty, Queue
from datetime import datetime

import openvino_genai as ovg
from flask import Response, jsonify, request

from core import config, runtime
from core.chat.common import (apply_penalties, _maybe_capture_prewarm,
                              overall_status, parse_messages)
from core.chat.openai import chat_completions
from core.errors import _TurnError, openai_error
from core.genai.results import extract_perf, explain_genai_error, extract_finish_reason
from core.media.images import load_image, pil_to_tensor
from core.slots.capability import (_tool_capable, _tools_refused_note,
                                   _tools_supported)
from core.slots.route import _route_request
from core.tools.parse import parse_tool_calls
from core.tools.render import prepare_messages_for_tools
from core.system.status import _log_request

# Ollama-compatible API (port 11434)
# ---------------------------------------------------------------------------

VSCODE_OLLAMA_VERSION = "0.18.3"


def _debug_ollama():
    _log_request("Ollama")


def ollama_health():
    return "Ollama is running"


def ollama_version():
    # VS Code's Ollama client rejects non-numeric versions, so when
    # --vscode-compat is set we report a real Ollama version to please it.
    version = VSCODE_OLLAMA_VERSION if config.VSCODE_COMPAT else "locally-0.1.0"
    return jsonify({"version": version})


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
    try:
        slot = _route_request(has_images, requested_model)
    except _TurnError as error:
        response, status = error.response
        return jsonify({"error": response.get_json()["error"]["message"]}), status
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
        text_prompt, images, raw_messages = parse_messages(
            internal_messages, config.MAX_IMAGE_DIM)
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
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] [Ollama] "
          f"Response completed in {elapsed:.1f}s", flush=True)

    generation_finish = getattr(text, "finish_reason", "stop")
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
        "done_reason": generation_finish,
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
    pieces = []
    generated = [None]
    failure = [None]
    ttft_ms = None

    def streamer_callback(token):
        if slot._cancel.is_set():
            return True
        token_queue.put(token)
        return False

    def _generate():
        try:
            with slot.lock:
                slot._cancel.clear()
                generated[0] = slot.pipe.generate(history, generation_config=gen,
                                                  streamer=streamer_callback)
                slot.last_used = time.time()
        except Exception as e:
            failure[0] = e
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
                ttft_ms = slot.last_ttft_ms
            token_count += 1
            pieces.append(token)
            yield json.dumps({
                "model": slot.model_name,
                "message": {"role": "assistant", "content": token},
                "done": False,
            }) + "\n"

        elapsed = time.perf_counter() - t0
        reason = ('error' if failure[0] else 'cancelled' if slot._cancel.is_set()
                  else extract_finish_reason(generated[0]))
        metric = slot._record_turn(slot._message_text(raw_messages), ''.join(pieces),
                                   None, t0, reason, ttft_ms=ttft_ms, result=generated[0])

        final = {
            "model": slot.model_name,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "total_duration": int(elapsed * 1e9),
            "done_reason": reason,
        }
        if metric['completion_tokens'] is not None:
            final['eval_count'] = metric['completion_tokens']
        if failure[0]:
            final['error'] = explain_genai_error(failure[0])
        yield json.dumps(final) + "\n"
    finally:
        slot._cancel.set()

from core.chat.ollama_generate import (ollama_copy, ollama_delete,
                                         ollama_generate, ollama_pull,
                                         ollama_v1_chat_completions)
