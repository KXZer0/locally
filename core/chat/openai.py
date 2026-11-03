"""OpenAI chat-completions response path."""

import time
import threading
from datetime import datetime

from flask import Response, jsonify, request

from core.chat import structured
from core.chat.common import (_debug_raw, _maybe_capture_prewarm, _sse_tool_stream,
                              _tool_message,
                              _turn_headers, make_id)
from core.chat.turn import _builtin_tool_turn, _prepare_turn
from core.errors import _TurnError, openai_error
from core.genai.results import GeneratedText, explain_genai_error, usage_block
from core.tools.builtin import _builtin_generator, _builtin_stream
from core.tools.parse import parse_tool_calls

def chat_completions():
    try:
        turn = _prepare_turn(request.get_json(silent=True), native=True)
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
                turn["builtin_specs"], slot, native=turn.get("native_builtin", False))
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
                         "finish_reason": getattr(text, "finish_reason", "stop")}],
            "python_runs": python_runs,
            "usage": usage_block(text),
        })
        resp.headers.extend(_turn_headers(turn, slot))
        return resp

    if turn.get("native_prompt"):
        return _native_response(turn, completion_id, created, t0)

    if stream and not tools_active and not images:
        return Response(
            structured.watch_stream(
                slot.stream_llm(raw_messages, gen, completion_id, created, t0), turn),
            mimetype="text/event-stream",
            headers=_turn_headers(turn, slot),
        )

    # --- VLM path ---
    if slot.model_type == "vlm":
        if stream and tools_active:
            # A tool turn has to be whole before tool_calls can be emitted.
            return Response(
                _sse_tool_stream(slot, raw_messages, gen, tools, completion_id,
                                 created, t0,
                                 generate=lambda: slot.generate_vlm(text_prompt, images, gen)),
                mimetype="text/event-stream",
                headers=_turn_headers(turn, slot),
            )
        if stream:
            return Response(
                structured.watch_stream(
                    slot.stream_vlm(text_prompt, images, gen, completion_id, created, t0),
                    turn),
                mimetype="text/event-stream",
                headers=_turn_headers(turn, slot),
            )

        try:
            text = slot.generate_vlm(text_prompt, images, gen)
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] VLM error: {e}", flush=True)
            return (structured.error_response(turn, e) or
                    openai_error(f"Inference failed: {e}", "server_error", 500))

        elapsed = time.perf_counter() - t0
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
              f"{len(text)} chars in {elapsed:.1f}s", flush=True)

        finish = getattr(text, "finish_reason", "stop")
        usage = usage_block(text)
        message = {"role": "assistant", "content": text}
        if tools_active:
            _debug_raw("vlm", text)
            text, vlm_calls = parse_tool_calls(text, tools)
            if vlm_calls:
                message = {"role": "assistant", "content": text or None,
                           "tool_calls": vlm_calls}
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": text}
        resp = jsonify({
            "id": completion_id, "object": "chat.completion",
            "created": created, "model": slot.model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
        })
        resp.headers.extend(_turn_headers(turn, slot))
        _judge(resp, turn, text)
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
        return (structured.error_response(turn, e) or
                openai_error(f"Inference failed: {err}", "server_error", 500))

    elapsed = time.perf_counter() - t0
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
          f"Response completed in {elapsed:.1f}s", flush=True)

    generation_finish = getattr(text, "finish_reason", "stop")
    # Read before parse_tool_calls: it returns a plain str, and the counts
    # ride on the GeneratedText it is handed.
    usage = usage_block(text)
    tool_calls = []
    if tools_active:
        _debug_raw("llm", text)
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
        finish_reason = generation_finish

    resp = jsonify({
        "id": completion_id, "object": "chat.completion",
        "created": created, "model": slot.model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    })
    resp.headers.extend(_turn_headers(turn, slot))
    _judge(resp, turn, GeneratedText(text, generation_finish))
    return resp


def _native_response(turn, completion_id, created, t0):
    """A Gemma 4 turn answered from its own chat template (core/tools/gemma.py):
    tool calls in the model's dialect, the thought channel as reasoning_content.
    Buffered, because a structured tool_calls delta needs the whole generation."""
    slot, gen, tools = turn["slot"], turn["gen"], turn["tools"]
    prompt = turn["native_prompt"]

    cancelled = threading.Event()

    def generate():
        return slot.generate_native(prompt, gen, cancel_event=cancelled)

    if turn["stream"]:
        def frames():
            try:
                yield from _sse_tool_stream(slot, None, gen, tools, completion_id, created, t0,
                                           generate=generate)
            finally:
                # Closing the HTTP stream cancels THIS request, not another
                # request currently using the shared model.
                cancelled.set()
        return Response(
            frames(),
            mimetype="text/event-stream", headers=_turn_headers(turn, slot))
    try:
        text = generate()
    except Exception as e:
        err = explain_genai_error(e)
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] LLM error: {err}", flush=True)
        return (structured.error_response(turn, e) or
                openai_error(f"Inference failed: {err}", "server_error", 500))
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
          f"Response completed in {time.perf_counter() - t0:.1f}s", flush=True)
    finish, usage = getattr(text, "finish_reason", "stop"), usage_block(text)
    message, tool_calls = _tool_message(text, tools)
    if tool_calls:
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
              f"{len(tool_calls)} tool call(s): "
              f"{', '.join(tc['function']['name'] for tc in tool_calls)}", flush=True)
        finish = "tool_calls"
    resp = jsonify({
        "id": completion_id, "object": "chat.completion",
        "created": created, "model": slot.model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage,
    })
    resp.headers.extend(_turn_headers(turn, slot))
    _judge(resp, turn, GeneratedText(message.get("content") or "", finish))
    return resp


def _judge(resp, turn, text):
    """Replace the 'constrained' header with what the answer turned out to be."""
    if turn.get("structured") is not None:
        resp.headers["X-Structured-Output"] = structured.verdict(
            turn["slot"], turn["structured"], text)


# ---------------------------------------------------------------------------
