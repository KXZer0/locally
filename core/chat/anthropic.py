"""Anthropic Messages protocol adapter."""

import json
import time
from datetime import datetime

from flask import Response, jsonify, request

from core import config
from core.chat.common import (_maybe_capture_prewarm, _sse_tool_stream,
                              _turn_headers, make_id, parse_messages)
from core.chat.turn import _prepare_turn
from core.errors import _TurnError, openai_error
from core.genai.results import explain_genai_error
from core.genai.tokens import _count_tokens
from core.slots.route import _route_request
from core.tools.parse import parse_tool_calls
from core.tools.text import _strip_tool_markup, strip_thinking
from core.voice.think_filter import _ThinkFilter

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


def anthropic_messages():
    raw = request.get_json(silent=True)
    if not raw:
        return openai_error("Request body must be JSON")

    if config.DEBUG_REQUESTS:
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




def anthropic_count_tokens():
    raw = request.get_json(silent=True) or {}
    slot = _route_request(False, raw.get("model", ""))
    if slot is None:
        return openai_error("No model ready.", "server_error", 503)
    body = _anthropic_to_openai(raw)
    try:
        text_prompt, _images, _raw = parse_messages(body.get("messages") or [], config.MAX_IMAGE_DIM)
    except Exception as e:
        return openai_error(f"Failed to parse request: {e}")
    # chars/4 is the standard fallback when the slot is unloaded and we would
    # rather answer than wake a model just to count.
    n = _count_tokens(slot, text_prompt)
    return jsonify({"input_tokens": n if n is not None else len(text_prompt) // 4})


# ---------------------------------------------------------------------------
