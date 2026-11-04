"""Streaming a tool turn live: OpenAI SSE in, gated OpenAI SSE out.

The pipelines' own streams (DeviceSlot.stream_llm / stream_vlm, and
stream_native below for Gemma's dialect) already own the hard parts -- the
prefill heartbeat, cancel on disconnect, the finish reason the pipeline
reported, the per-turn metrics. So a tool turn is not a second generation
path: it is those frames with ToolStreamGate (core/tools/stream_gate.py)
deciding which text may go out, and the parsed calls appended at the end in
the same frames the buffered path sent. The Anthropic and Ollama adapters
consume these frames, so neither needed a tool-specific path of its own.

Two things the gate's hold costs, each paid back here:

- **Silence while a call is written.** A call can be long (a Write tool with
  a whole file in it), and every token of it is held, so an empty-content
  keep-alive goes out every HEARTBEAT_SECS -- the same frame the prefill
  heartbeat uses -- or a client watchdog would abort a turn that is working.
- **A constraint the device rejects.** When `retry` is given (the tool-call
  constraint, core/tools/constrain.py) and the stream fails before any text
  went out, it is asked for a replacement stream; the replacement's role
  frame is dropped so the client sees one stream.
"""

import json
import threading
import time
from datetime import datetime
from queue import Empty, Queue

from core import config
from core.genai.results import explain_genai_error
from core.tools.stream_gate import ToolStreamGate


def _chunk(completion_id, created, model, delta, finish=None):
    return "data: " + json.dumps({
        "id": completion_id, "object": "chat.completion.chunk",
        "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }) + "\n\n"


def _payload(frame):
    if not frame.startswith("data: {"):
        return None
    try:
        return json.loads(frame[6:])
    except ValueError:
        return None


def gated_stream(frames, tools, completion_id, created, model, native=False,
                 retry=None, judge=None, log=None):
    """Re-emit `frames` (OpenAI SSE) with tool-call text held back and parsed."""
    gate = ToolStreamGate(tools, native=native)
    finish, role_sent, last = None, False, time.monotonic()

    def out(pieces):
        for kind, text in pieces:
            key = "reasoning_content" if kind == "reasoning" else "content"
            yield _chunk(completion_id, created, model, {key: text})

    try:
        while frames is not None:
            source, frames = frames, None
            try:
                for frame in source:
                    p = _payload(frame)
                    if p is None:
                        continue                   # [DONE]: ours goes out last
                    choice = (p.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    if "role" in delta:
                        if not role_sent:
                            role_sent = True
                            yield frame
                        continue
                    text = delta.get("content")
                    if choice.get("finish_reason") == "error":
                        if retry is not None and not gate.buf:
                            frames = retry(text or "")
                            if frames is not None:
                                break
                        tail, _calls, _r = gate.finish()
                        yield from out(tail)
                        yield frame
                        yield "data: [DONE]\n\n"
                        return
                    if text == "":
                        last = time.monotonic()
                        yield frame                # the prefill heartbeat
                    elif text:
                        pieces = gate.feed(text)
                        if pieces:
                            last = time.monotonic()
                            yield from out(pieces)
                        elif time.monotonic() - last >= config.HEARTBEAT_SECS:
                            last = time.monotonic()
                            yield _chunk(completion_id, created, model, {"content": ""})
                    finish = choice.get("finish_reason") or finish
            finally:
                _close(source)             # a disconnect cancels generation
    finally:
        _close(frames)

    tail, calls, _reasoning = gate.finish()
    yield from out(tail)
    for i, tc in enumerate(calls):
        yield _chunk(completion_id, created, model, {"tool_calls": [{
            "index": i, "id": tc["id"], "type": "function", "function": tc["function"]}]})
    if calls and log:
        log(calls)
    if judge is not None:
        judge(gate.buf, calls, finish)
    yield _chunk(completion_id, created, model, {},
                 "tool_calls" if calls else (finish or "stop"))
    yield "data: [DONE]\n\n"


def _close(gen):
    close = getattr(gen, "close", None)
    if close:
        close()


def log_calls(slot):
    def log(calls):
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
              f"{len(calls)} tool call(s): "
              f"{', '.join(tc['function']['name'] for tc in calls)}", flush=True)
    return log


def stream_native(slot, prompt, gen, completion_id, created, t0, cancel_event):
    """Gemma's native dialect as OpenAI SSE frames, raw special tokens kept.

    generate_native decodes with skip_special_tokens=False, which is why the
    dialect is readable at all; its `on_text` hook streams that same decoding.
    """
    q = Queue()
    result = {}

    def run():
        try:
            result["text"] = slot.generate_native(prompt, gen, cancel_event=cancel_event,
                                                  on_text=q.put)
        except Exception as e:  # noqa: BLE001 -- surfaced as the error frame
            result["error"] = e
        finally:
            q.put(None)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    model = slot.model_name
    yield _chunk(completion_id, created, model, {"role": "assistant"})
    try:
        while True:
            try:
                piece = q.get(timeout=config.HEARTBEAT_SECS)
            except Empty:
                yield _chunk(completion_id, created, model, {"content": ""})
                continue
            if piece is None:
                break
            if piece:
                yield _chunk(completion_id, created, model, {"content": piece})
    finally:
        cancel_event.set()                 # a closed stream cancels THIS request
    if "error" in result:
        err = explain_genai_error(result["error"])
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] LLM error: {err}", flush=True)
        yield _chunk(completion_id, created, model, {"content": f"\n[error: {err}]"}, "error")
    else:
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
              f"Native response completed in {time.perf_counter() - t0:.1f}s", flush=True)
        yield _chunk(completion_id, created, model, {},
                     getattr(result.get("text"), "finish_reason", "stop"))
    yield "data: [DONE]\n\n"


def ollama_frames(frames, model, t0):
    """OpenAI SSE frames -> Ollama /api/chat ndjson lines (tool turns)."""
    calls, finish, error = [], "stop", None
    for frame in frames:
        p = _payload(frame)
        if p is None:
            continue
        choice = (p.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if choice.get("finish_reason") == "error":
            error = (delta.get("content") or "").strip()
            continue
        for tc in delta.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            calls.append({"function": {"name": fn.get("name", ""), "arguments": args}})
        text = delta.get("content")
        if text is not None and "role" not in delta:
            yield json.dumps({"model": model, "message": {"role": "assistant", "content": text},
                              "done": False}) + "\n"
        finish = choice.get("finish_reason") or finish
    if calls:
        yield json.dumps({"model": model, "message": {
            "role": "assistant", "content": "", "tool_calls": calls}, "done": False}) + "\n"
    final = {"model": model, "message": {"role": "assistant", "content": ""}, "done": True,
             "done_reason": "stop" if finish == "tool_calls" else finish,
             "total_duration": int((time.perf_counter() - t0) * 1e9)}
    if error:
        final["error"] = error
    yield json.dumps(final) + "\n"
