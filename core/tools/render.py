"""Turning a client's tools array into prompt text the model will answer in.

Carries today's date (~15 tokens, every device): with no clock a model calls
the calendar at its training cutoff and then reports that the results look
'way in the future'."""

import json
from datetime import datetime


def render_tools_prompt(tools):
    """Build a system-prompt block describing tools in Qwen3-Coder format."""
    specs = []
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if fn.get("name"):
            specs.append(json.dumps(fn, ensure_ascii=False))
    if not specs:
        return ""
    return (
        "You have access to the following tools. When you need to call one, "
        "emit it exactly in this format (one <tool_call> block per call, and "
        "nothing else in that turn):\n"
        "<tool_call>\n<function=TOOL_NAME>\n<parameter=ARG_NAME>\nARG_VALUE\n"
        "</parameter>\n</function>\n</tool_call>\n\n"
        # An assistant's tools are mostly about time — calendars, reminders,
        # "tomorrow", "this week" — and a model has no clock. Measured on the
        # NPU 2026-08-29: asked what was on the calendar tomorrow, Qwen3-8B
        # called get_calendar with start and end of 2023-10-11, its training
        # cutoff, then noticed the results "were way in the future" and said
        # so to the user. Fifteen tokens of date fixes a whole class of that.
        f"Today's date is {datetime.now():%Y-%m-%d (%A)}. Resolve relative "
        "dates like \"tomorrow\" against it.\n\n"
        "Available tools (JSON schema):\n<tools>\n"
        + "\n".join(specs)
        + "\n</tools>"
    )


def _tool_calls_to_text(tool_calls):
    """Render assistant tool_calls (from history) back into Qwen3-Coder XML."""
    blocks = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        params = "".join(
            f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
            for k, v in args.items()
        )
        blocks.append(f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>")
    return "\n".join(blocks)


def prepare_messages_for_tools(messages, tools, slot=None):
    """Normalize OpenAI messages for a tool-enabled turn.

    - Injects/extends a system message describing the tools.
    - Renders prior assistant tool_calls back into the model's XML so the
      conversation stays coherent across turns.
    - Folds `tool` result messages into tagged user content (works regardless
      of whether the chat template knows the `tool` role).
    - On the NPU, suppresses the reasoning block (see _suppress_think_for_tools).
    """
    tool_prompt = render_tools_prompt(tools)
    out = []
    injected = False
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "system":
            sys_text = content if isinstance(content, str) else (content or "")
            if tool_prompt and not injected:
                sys_text = (sys_text + "\n\n" + tool_prompt).strip()
                injected = True
            out.append({"role": "system", "content": sys_text})
            continue

        if role == "assistant" and msg.get("tool_calls"):
            rendered = _tool_calls_to_text(msg["tool_calls"])
            base = content if isinstance(content, str) and content else ""
            out.append({"role": "assistant", "content": (base + "\n" + rendered).strip()})
            continue

        if role == "tool":
            name = msg.get("name", "")
            result = content if isinstance(content, str) else json.dumps(content)
            tag = f' name="{name}"' if name else ""
            out.append({"role": "user",
                        "content": f"<tool_response{tag}>\n{result}\n</tool_response>"})
            continue

        # plain user/assistant (content may be None on a tool-call-only turn)
        out.append({"role": role, "content": content if content is not None else ""})

    if tool_prompt and not injected:
        out.insert(0, {"role": "system", "content": tool_prompt})
    return _suppress_think_for_tools(out, slot)


def _suppress_think_for_tools(messages, slot):
    """Append /no_think to the last user turn on an NPU tool request.

    Measured on the NPU 2026-08-29, Qwen3-8B, 8-tool assistant set, "Remind me
    to call the dentist tomorrow.": the model opened a <think> block, spent the
    whole 400-token budget reasoning about which tool to use, and emitted **no
    tool call at all** — 272 tokens and 23.2 s to produce nothing callable. The
    identical request with reasoning suppressed emits create_task directly.

    Scoped to the NPU because that is where both costs bite: an 8192-token
    window the reasoning competes for, and ~12 tok/s decode that turns a think
    block into twenty seconds of silence. On the GPU, reasoning before a tool
    call is affordable and sometimes better, so it is left alone.

    Choosing a tool from eight well-described options is a classification, not
    a problem that rewards deliberation — which is why suppressing it costs
    nothing here. /no_think is a real Qwen3 control token; an English sentence
    asking for the same thing does not work (see the voice path, which learned
    this the same way).
    """
    if not slot or slot.device_name != "NPU" or not messages:
        return messages
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            text = messages[i].get("content")
            if isinstance(text, str) and "/no_think" not in text:
                out = list(messages)
                out[i] = dict(messages[i])
                out[i]["content"] = text.rstrip() + " /no_think"
                return out
            break
    return messages
