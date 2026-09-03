"""OpenAI chat-completions response path."""

import time
from datetime import datetime

from flask import Response, jsonify, request

from core.chat.common import (_maybe_capture_prewarm, _sse_tool_stream,
                              _turn_headers, make_id)
from core.chat.turn import _builtin_tool_turn, _prepare_turn
from core.errors import _TurnError, openai_error
from core.genai.results import explain_genai_error
from core.tools.builtin import _builtin_generator, _builtin_stream
from core.tools.parse import parse_tool_calls

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
