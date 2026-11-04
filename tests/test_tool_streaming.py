"""Tool turns stream live: text goes out as it arrives, call markup never does."""

import json
import random

import pytest

from core.chat.anthropic import _anthropic_stream
from core.chat.common import _tool_message
from core.chat.tool_stream import gated_stream, ollama_frames
from core.tools.stream_gate import ToolStreamGate

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}}}}}]

CALLS = {
    "qwen-xml": "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
                "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
    "hermes": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}\n</tool_call>',
    "bare-function": "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
                     "<parameter=days>\n3\n</parameter>\n</function>",
    "mistral": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}]',
    "llama": '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris", "days": 3}}',
    "deepseek": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
                '```json\n{"city": "Paris", "days": 3}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
    "lfm": '<|tool_call_start|>[get_weather(city="Paris", days=3)]<|tool_call_end|>',
    "gemma": '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3}<tool_call|>',
    "bare-json": '{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}',
}


def frames_of(tokens, finish="stop", heartbeat_at=None):
    yield 'data: {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}\n\n'
    for i, tok in enumerate(tokens):
        if heartbeat_at == i:
            yield 'data: {"choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": null}]}\n\n'
        yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": tok},
                                                   "finish_reason": None}]}) + "\n\n"
    yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}) + "\n\n"
    yield "data: [DONE]\n\n"


def run(tokens, native=False, finish="stop", **kw):
    """(content, reasoning, tool_calls, finish_reason, frames) of a gated stream."""
    out = list(gated_stream(frames_of(tokens, finish, **kw), TOOLS, "id", 0, "m", native=native))
    content, reasoning, calls, fin = [], [], [], None
    for f in out:
        if not f.startswith("data: {"):
            continue
        ch = json.loads(f[6:])["choices"][0]
        d = ch["delta"]
        content.append(d.get("content") or "")
        reasoning.append(d.get("reasoning_content") or "")
        calls += d.get("tool_calls") or []
        fin = ch["finish_reason"] or fin
    return "".join(content), "".join(reasoning), calls, fin, out


def chars(s):
    return list(s)


def args_of(call):
    return json.loads(call["function"]["arguments"])


@pytest.mark.parametrize("fmt", sorted(CALLS))
def test_call_only_char_by_char_never_leaks(fmt):
    content, _r, calls, fin, _ = run(chars(CALLS[fmt]), native=(fmt == "gemma"))
    assert content == ""
    assert fin == "tool_calls"
    assert [c["function"]["name"] for c in calls] == ["get_weather"]
    assert args_of(calls[0]) == {"city": "Paris", "days": 3}


@pytest.mark.parametrize("fmt", sorted(set(CALLS) - {"bare-json"}))
def test_text_then_call_streams_the_text_first(fmt):
    text = "Let me check the weather for you. "
    gate = ToolStreamGate(TOOLS, native=(fmt == "gemma"))
    shown = []
    for ch in text:
        shown += [t for _k, t in gate.feed(ch)]
    # Everything but a few held characters is out before the call starts.
    assert "".join(shown).startswith("Let me check the weather for you.")
    for ch in CALLS[fmt]:
        assert not gate.feed(ch), "call markup must never be released"
    tail, calls, _ = gate.finish()
    assert [c["function"]["name"] for c in calls] == ["get_weather"]
    assert "".join(t for _k, t in tail).strip() == ""


def test_text_only_streams_before_generation_ends():
    tokens = ["The", " weather", " is", " nice", " today", "."]
    gate = ToolStreamGate(TOOLS)
    first = None
    for i, t in enumerate(tokens):
        if gate.feed(t) and first is None:
            first = i
    assert first == 0
    content, _r, calls, fin, _ = run(tokens)
    assert content == "The weather is nice today." and not calls and fin == "stop"


@pytest.mark.parametrize("prose", [
    "Use the <tool> tag in HTML, or <tools> for a list.",
    "Compare a<b and [TOOL] notes; <|python is not a tag.",
    "A fenced block:\n```json\n{\"name\": \"get_weather\"}\n```\n",
])
def test_false_alarms_are_released(prose):
    content, _r, calls, fin, _ = run(chars(prose))
    assert content == prose and not calls and fin == "stop"


def test_split_opener_is_held_then_released():
    gate = ToolStreamGate(TOOLS)
    assert gate.feed("Use <tool") == [("content", "Use ")]
    assert gate.feed("box> now") == [("content", "<toolbox> now")]


def test_json_answer_that_is_not_a_call_is_sent_whole_at_the_end():
    answer = '{"temperature": 21, "unit": "C"}'
    gate = ToolStreamGate(TOOLS)
    assert all(not gate.feed(c) for c in answer)
    tail, calls, _ = gate.finish()
    assert not calls and "".join(t for _k, t in tail) == answer


def test_no_tools_declared_means_json_streams():
    gate = ToolStreamGate([])
    assert gate.feed('{"a"') == [("content", '{"a"')]


def test_think_then_bare_json_call():
    toks = ["<think>", "I should", " look it up", "</think>", "\n\n"] + chars(CALLS["bare-json"])
    content, _r, calls, fin, _ = run(toks)
    assert content.startswith("<think>I should look it up</think>")
    assert fin == "tool_calls" and args_of(calls[0])["city"] == "Paris"


def test_gemma_thought_channel_becomes_reasoning():
    toks = ["<|channel>", "thought", "\nThe user", " wants weather", "<channel|>", "Sure", ". "] + \
        chars(CALLS["gemma"]) + ["<|tool_response>"]
    content, reasoning, calls, fin, _ = run(toks, native=True)
    assert reasoning == "The user wants weather"
    assert content == "Sure. " and fin == "tool_calls"
    assert args_of(calls[0]) == {"city": "Paris", "days": 3}


def test_gemma_text_turn_drops_end_of_turn_tokens():
    content, reasoning, calls, fin, _ = run(["Hello", " world", ".", "<turn|>"], native=True)
    assert content == "Hello world." and not calls and not reasoning


def test_text_after_the_call_is_sent_at_the_end():
    toks = ["Checking", ". "] + chars(CALLS["qwen-xml"]) + ["\nOne moment."]
    content, _r, calls, fin, _ = run(toks)
    assert fin == "tool_calls" and len(calls) == 1
    assert content.startswith("Checking.") and content.rstrip().endswith("One moment.")


def test_heartbeat_and_length_survive():
    content, _r, calls, fin, out = run(["Long", " answer"], finish="length", heartbeat_at=0)
    assert any('"content": ""' in f for f in out)
    assert content == "Long answer" and fin == "length" and not calls
    assert out[-1] == "data: [DONE]\n\n"


def test_one_role_frame_and_retry_on_early_error():
    def broken():
        yield 'data: {"choices": [{"delta": {"role": "assistant"}, "finish_reason": null}]}\n\n'
        yield 'data: {"choices": [{"delta": {"content": "\\n[error: xgrammar grammar_parser]"}, "finish_reason": "error"}]}\n\n'
        yield "data: [DONE]\n\n"
    seen = []
    def retry(err):
        seen.append(err)
        return frames_of(["fine"])
    out = list(gated_stream(broken(), TOOLS, "id", 0, "m", retry=retry))
    assert len(seen) == 1 and "grammar_parser" in seen[0]
    assert sum('"role"' in f for f in out) == 1
    assert any('"fine"' in f for f in out)


def test_error_without_retry_is_passed_through():
    def broken():
        yield 'data: {"choices": [{"delta": {"content": "Hi"}, "finish_reason": null}]}\n\n'
        yield 'data: {"choices": [{"delta": {"content": "\\n[error: boom]"}, "finish_reason": "error"}]}\n\n'
    out = list(gated_stream(broken(), TOOLS, "id", 0, "m"))
    assert any('"finish_reason": "error"' in f for f in out) and out[-1] == "data: [DONE]\n\n"


def test_closing_the_stream_closes_the_pipeline_stream():
    closed = []
    def src():
        try:
            yield from frames_of(["a", "b", "c"])
        finally:
            closed.append(True)
    g = gated_stream(src(), TOOLS, "id", 0, "m")
    next(g); next(g)
    g.close()
    assert closed == [True]


@pytest.mark.parametrize("seed", range(40))
def test_random_splits_match_the_buffered_parse(seed):
    rng = random.Random(seed)
    fmt = rng.choice(sorted(CALLS))
    native = fmt == "gemma"
    pre = rng.choice(["", "Sure. ", "Let me look <that> up.\n", "<think>hmm</think>\n"])
    if native and pre.startswith("<think>"):
        pre = "<|channel>thought\nhmm<channel|>"
    post = rng.choice(["", "\nDone."])
    text = (pre + CALLS[fmt] + post) if rng.random() < 0.8 else (pre + "No call here.")
    toks, i = [], 0
    while i < len(text):
        n = rng.randint(1, 6)
        toks.append(text[i:i + n]); i += n
    content, _r, calls, fin, _ = run(toks, native=native)
    if native:
        from core.tools.gemma import split_reasoning
        _reasoning, text_vis = split_reasoning(text)
        message, want = _tool_message(text, TOOLS)
    else:
        message, want = _tool_message(text, TOOLS)
    assert [c["function"] for c in calls] == [c["function"] for c in want]
    assert content.strip().replace("\n", "") == (message.get("content") or "").strip().replace("\n", "")
    assert fin == ("tool_calls" if want else "stop")


def _anthropic_events(frames):
    events = []
    for sse in _anthropic_stream(frames, "m", "msg", 5):
        lines = sse.strip().splitlines()
        events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


def test_anthropic_text_block_then_tool_use_block():
    toks = ["I'll", " check", ". "] + chars(CALLS["qwen-xml"])
    gated = gated_stream(frames_of(toks), TOOLS, "id", 0, "m")
    ev = _anthropic_events(gated)
    kinds = [(e, (p.get("content_block") or {}).get("type")) for e, p in ev
             if e in ("content_block_start", "content_block_stop")]
    assert kinds == [("content_block_start", "text"), ("content_block_stop", None),
                     ("content_block_start", "tool_use"), ("content_block_stop", None)]
    text = "".join(p["delta"].get("text", "") for e, p in ev if e == "content_block_delta")
    assert text == "I'll check. "
    args = [p["delta"]["partial_json"] for e, p in ev
            if e == "content_block_delta" and p["delta"]["type"] == "input_json_delta"]
    assert json.loads(args[0]) == {"city": "Paris", "days": 3}
    stop = [p for e, p in ev if e == "message_delta"][0]["delta"]["stop_reason"]
    assert stop == "tool_use"


def test_ollama_lines_stream_text_then_carry_the_call():
    toks = ["Checking", "."] + chars(CALLS["hermes"])
    lines = [json.loads(x) for x in
             ollama_frames(gated_stream(frames_of(toks), TOOLS, "o", 0, "m"), "m", 0.0)]
    text = "".join(x["message"]["content"] for x in lines if not x["done"])
    assert text == "Checking."
    call = [x for x in lines if x["message"].get("tool_calls")]
    assert call and call[0]["message"]["tool_calls"][0]["function"]["arguments"]["days"] == 3
    assert lines[-1]["done"] and lines[-1]["done_reason"] == "stop"
