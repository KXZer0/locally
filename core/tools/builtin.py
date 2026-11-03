"""Tools the SERVER owns and runs, as opposed to the client's `tools` array.

Distinct from core/tools/render.py, which only relays what a client sent.
These are offered on every ordinary chat turn and executed here, and that
is what let the composer's mode buttons go: a 'search the web' toggle asked
the user to answer a question the model is better placed to answer -- it
forced a search on 'hello' and skipped one on a question about last week.

The gate is _builtin_tools_supported, deliberately NOT _tools_supported:
that one is about driving a client's agent loop, and the NPU exclusion
there is about 30-schema catalogues and multi-step planning, neither of
which describes two small tools costing ~212 tokens."""

import json
import re
from core.genai.tokens import _count_tokens
from core.sandbox.python_exec import execute_python
from core.tools.parse import parse_tool_calls

import threading

from core import config
from core.tools.registry import PYTHON_TOOL, WEB_SEARCH_TOOL
from core.tools.activity import record, request_id
from core.tools.calculator import calculate, CALCULATOR_TOOL
from core.web import search as web_search_mod
from core.tools.render import _tool_calls_to_text
from core.web.fetch_page import _web_search_run
from core.voice.think_filter import _HOLD_CHARS

# Audit trail for the turn in flight: what code actually ran, so the UI can
# show the receipts instead of asking the user to trust a number.
_BUILTIN_RUNS = threading.local()


# Openers every format parse_tool_calls knows. A turn is held back only while
# its visible text could still be growing into one of these.
_TOOL_OPENERS = ("<tool_call", "<function=", "[TOOL_CALLS]", "<|python_tag|>",
                 "｜tool▁calls▁begin｜", "<|tool_call_start|>", "<|tool_call>", '{"name"')



def _builtin_runs():
    if not hasattr(_BUILTIN_RUNS, "items"):
        _BUILTIN_RUNS.items = []
    return _BUILTIN_RUNS.items


def _run_python_tool(args):
    code = args.get("code") if isinstance(args, dict) else None
    if not isinstance(code, str):
        return "No code was given."
    record("calculating", "Calculating with local Python…", code=code[:6000])
    run = execute_python(code)
    record("calculated", "Calculation finished", result=str(run.get("stdout") or run.get("stderr") or "")[:2000])
    _builtin_runs().append(dict(run, tool="python"))
    return ("stdout:\n" + (run["stdout"] or "(empty)") +
            "\nstderr:\n" + (run["stderr"] or "(empty)") +
            "\nexit_status: " + str(run["exit_status"]) +
            "\ntimed_out: " + str(run["timed_out"]))


def _run_web_search_tool(args):
    """Retrieval only. The model still does the answering, as it always has."""
    query = (args.get("query") or "").strip() if isinstance(args, dict) else ""
    # A model often wraps the query in quotes; as a search operator that means
    # "exact phrase", and the engine returns nothing for a whole sentence.
    query = query.strip("\"' ")
    if not query:
        return "No query was given."
    if not web_search_mod.WEB_SEARCH_URL:
        return ("Web search is not configured on this server. Answer from what "
                "you know, and say you could not check.")
    record("searching", "Searching the web…", query=query)
    payload = None
    for kind, value in _web_search_run({"query": query, "answer": False,
                                        "top_k": 5, "per_url": 1}):
        if kind == "stage":
            record("reading" if value.get("stage") == "reading" else "searching", value.get("detail") or "Searching the web…")
        if kind == "error":
            record("tool_error", "Search failed: " + str(value[0])[:200])
            return "The search failed: " + str(value[0])
        if kind == "done":
            payload = value
    payload = payload or {}
    passages = payload.get("results") or payload.get("passages") or []
    if not passages:
        return "The search returned nothing usable."
    _builtin_runs().append({"tool": "web_search", "query": query,
                            "sources": payload.get("sources") or []})
    # One block per page, best passage first. Ranked passages alone let one
    # long page (a docs index full of nav links) fill every slot, and the
    # engine's own snippet -- short, dated, usually where the answer sits --
    # never reached the model because the ranker penalises snippets.
    snippets = payload.get("snippets") or {}
    blocks, seen = [], set()
    for psg in passages:
        url = str(psg.get("url") or "")
        if url in seen:
            continue
        seen.add(url)
        body = str(snippets.get(url) or "").strip()
        text = str(psg.get("text") or "").strip()
        if text and text != body:
            body = (body + "\n" + text[:500]).strip()
        blocks.append("[" + str(len(blocks) + 1) + "] "
                      + str(psg.get("title") or "") + "\n" + url + "\n" + body)
        if len(blocks) >= 5:
            break
    receipts = [{"url":str(psg.get("url") or ""),"title":str(psg.get("title") or "")} for psg in passages if psg.get("url")]
    record("sources", "Writing an answer from the sources…", sources=receipts[:5])
    print(f"  [search] model query: {query!r} -> {len(blocks)} sources",
          flush=True)
    if request_id():
        return "\n\n".join(blocks) + "\n\nAnswer from the evidence. Sources appear below the answer in the app; do not include [1] or any numbered citation markers or a source list in your text."
    return ("\n\n".join(blocks) + "\n\n"
            "Answer from these sources and cite them inline as [1], [2]. If "
            "they do not contain the answer, say so plainly.")


def _builtin_tools_supported(slot):
    """Can this slot handle OUR one or two small tools?

    Deliberately NOT _tools_supported, which stays exactly as it is. That gate
    answers a different question -- can this slot drive a CLIENT's agent loop --
    and its NPU exclusion is about 30-schema catalogs and multi-step planning,
    neither of which describes this. Measured: the whole rendered block for the
    built-in tools is ~212 tokens, 2.6% of the NPU's 8192, about what the
    default system prompt already costs.

    Splitting rather than loosening matters: /api/show keeps advertising
    tools: false for an NPU slot, so Copilot still will not offer an NPU model
    for agent mode and a 30-tool request is still ignored there. Only ours
    cross the line.
    """
    return bool(slot) and slot.status in ("ready", "idle_unloaded")


def _fit_tool_output(slot, text):
    """Trim tool output to what this slot's prompt can actually hold.

    This is the real NPU risk, and it is not the tool spec -- it is the result.
    The NPU pipeline REFUSES an over-long prompt rather than truncating it, so
    feeding back an unbounded print() or a page of search passages is a hard
    turn failure, not a degraded answer. Same method the OCR path uses: count,
    trim, and say so, because a partial number that reads as a whole one is
    worse than an obvious cut.

    A no-op wherever the window is roomy, which is every GPU/CPU slot.
    """
    budget = getattr(slot, "context_tokens", None) or 0
    if not budget or budget > 32000:
        return text
    # Leave room for the conversation already in the prompt and for an answer.
    allow = max(256, int(budget * 0.4))
    try:
        if _count_tokens(slot, text) <= allow:
            return text
    except Exception:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            fits = _count_tokens(slot, text[:mid]) <= allow
        except Exception:
            break
        if fits:
            lo = mid
        else:
            hi = mid - 1
    return (text[:lo].rstrip() +
            "\n\n[output truncated to fit this model's prompt limit]")


def builtin_tools_for(slot, allow=None):
    """The built-in specs this slot may be offered, or [] when it may not."""
    if not _builtin_tools_supported(slot):
        return []
    return [t["spec"] for name, t in BUILTIN_TOOLS.items()
            if t["enabled"]() and (allow is None or allow.get(name, True))]


def _visible_so_far(text):
    """The answer text, with reasoning removed -- closed blocks and an open one."""
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    out = re.sub(r"<think>.*$", "", out, flags=re.S)
    return out.strip()


def _could_be_tool_call(tail):
    """True while `tail` is still a prefix of some opener, or already is one."""
    if not tail:
        return True
    for op in _TOOL_OPENERS:
        if tail.startswith(op) or op.startswith(tail):
            return True
    return False


def _sse_text(frame):
    """The delta text in one SSE frame, or None if it carries none."""
    if not frame.startswith("data: ") or frame.strip().endswith("[DONE]"):
        return None
    try:
        payload = json.loads(frame[6:].strip())
        return (payload["choices"][0].get("delta") or {}).get("content")
    except Exception:
        return None


def _builtin_streamer(slot, gen, turn, completion_id, created, t0):
    """Frames for one round, from whichever pipeline this slot actually has."""
    if slot.model_type == "vlm":
        # Reuse the (prompt, images) pair _prepare_turn already built. Do NOT
        # re-parse: parse_messages EXTRACTS images out of the messages it
        # returns, so parsing raw_messages a second time yields a prompt that
        # still carries the image marker and an image list that is empty --
        # "Missing image/video with index 0" from inputs_embedder.cpp, on the
        # next text-only turn of any conversation that ever held a picture.
        base = turn.get("text_prompt") or ""
        imgs = turn.get("images") or []
        seen = len(turn.get("raw_messages") or [])

        def run(messages):
            extra = "".join("\n" + str(m.get("content") or "")
                            for m in messages[seen:])
            return slot.stream_vlm(base + extra, imgs, gen,
                                   completion_id, created, t0)
        return run
    return lambda messages: slot.stream_llm(messages, gen, completion_id,
                                            created, t0)


def _builtin_stream(slot, raw_messages, gen, specs, turn,
                    completion_id, created, t0):
    """Stream a turn that MAY call a tool, without giving up live tokens.

    The old behaviour buffered every tool-capable turn, because the loop has to
    see a whole tool-call block before it can run it. That cost live streaming
    on exactly the slots people use most. The fix is the hold-back _ThinkFilter
    already uses for split tags, applied to a different question: forward
    frames as they arrive, but withhold the first few characters of VISIBLE
    text until they can no longer be the start of a tool call. Reasoning is not
    visible text, so a thinking model still streams its <think> block live and
    the decision is made on the answer that follows it.

    Cost when the model does not call a tool: the first ~24 characters arrive
    together instead of one by one. Cost when it does: nothing is shown, which
    is the point -- the user never sees raw XML.
    """
    held, decided, forwarding = [], False, False
    accumulated = ""
    stream_round = _builtin_streamer(slot, gen, turn, completion_id, created, t0)

    def replay():
        """Emit everything held back as one frame."""
        text = "".join(held)
        held.clear()
        if not text:
            return None
        return "data: " + json.dumps({
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": slot.model_name,
            "choices": [{"index": 0, "delta": {"content": text},
                         "finish_reason": None}],
        }) + "\n\n"

    for frame in stream_round(raw_messages):
        if forwarding:
            yield frame
            continue
        delta = _sse_text(frame)
        if delta is None:
            if decided:            # role frame, [DONE], heartbeats
                yield frame
            continue
        accumulated += delta
        held.append(delta)
        if decided:
            continue
        tail = _visible_so_far(accumulated)
        if _could_be_tool_call(tail):
            if len(tail) >= _HOLD_CHARS:
                decided = True     # it IS a call -- stay silent, keep reading
            continue
        # Ordinary prose. Release everything and go live for the rest.
        decided, forwarding = True, True
        out = replay()
        if out:
            yield out

    text, calls = parse_tool_calls(accumulated, specs)
    ours = [c for c in calls
            if (c.get("function") or {}).get("name") in BUILTIN_TOOLS]
    if forwarding:
        return                     # its own finish + [DONE] already went out
    if not ours:
        # Nothing to run. Anything still held is the whole answer.
        out = replay()
        if out:
            yield out
        yield "data: " + json.dumps({
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": slot.model_name,
            "choices": [{"index": 0, "delta": {},
                         "finish_reason": "stop"}],
        }) + "\n\n"
        yield "data: [DONE]\n\n"
        return

    # A tool call, and nothing of it reached the user. Run it, then stream the
    # answer it produces -- that second round is what the user actually reads,
    # and it streams token by token like any other turn.
    _builtin_runs().clear()
    results = []
    for call in ours:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        try:
            out = BUILTIN_TOOLS[fn["name"]]["run"](args if isinstance(args, dict) else {})
        except Exception as e:
            out = "The tool failed: " + str(e)
        results.append((fn["name"], _fit_tool_output(slot, out)))

    messages = list(raw_messages)
    messages.append({"role": "assistant",
                     "content": (text + "\n" + _tool_calls_to_text(ours)).strip()})
    body = "\n\n".join('<tool_response name="' + n + '">\n' + r +
                       "\n</tool_response>" for n, r in results)
    messages.append({"role": "user",
                     "content": body + "\n\nUse the result above to answer the "
                                       "user. Do not call another tool unless "
                                       "it is genuinely necessary."})
    for frame in stream_round(messages):
        yield frame


def _builtin_generator(slot, gen, turn):
    """How this slot produces text for one round of the tool loop.

    A VLM slot is a legitimate agent host -- gemma-4-26b on GPU does a full
    tool round trip -- but it is driven by (prompt, images) rather than a
    message list, so each round has to be re-rendered for it.
    """
    if turn.get("native_builtin"):
        return lambda messages: slot.generate_native(
            slot.native_chat().render(messages, turn["builtin_specs"], turn.get("thinking", False)), gen)
    if slot.model_type == "vlm":
        # Same rule as _builtin_streamer: reuse the pair, never re-parse.
        base = turn.get("text_prompt") or ""
        imgs = turn.get("images") or []
        seen = len(turn.get("raw_messages") or [])

        def run(messages):
            extra = "".join("\n" + str(m.get("content") or "")
                            for m in messages[seen:])
            return slot.generate_vlm(base + extra, imgs, gen)
        return run
    return lambda messages: slot.generate_llm(messages, gen)


def _run_calculator(args):
    expression = str(args.get("expression") or "")
    record("calculating", "Calculating…", expression=expression)
    result = calculate(args)
    record("calculated", "Calculation finished", result=result)
    _builtin_runs().append({"tool":"calculator", "expression":expression, "result":result})
    return result


BUILTIN_TOOLS = {
    "calculator": {"spec":CALCULATOR_TOOL,"run":_run_calculator,"enabled":lambda:True},
    # Read through the module every time. Both of these are set AFTER import,
    # by core/launch/configure.py from the command line, so a bound copy is
    # frozen at the default and the tool can never turn on -- which is exactly
    # what happened: /health reported `builtin_tools: []` no matter what was
    # passed, and neither --python-tool nor --search-url did anything at all.
    "python": {"spec": PYTHON_TOOL, "run": _run_python_tool,
               "enabled": lambda: config.PYTHON_TOOL_ENABLED},
    "web_search": {"spec": WEB_SEARCH_TOOL, "run": _run_web_search_tool,
                   "enabled": lambda: bool(web_search_mod.WEB_SEARCH_URL)},
}
