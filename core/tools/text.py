"""Stripping the machinery back out of a model's text: orphaned tool tags and
<think> blocks.

This is the whole-string form. The streaming twin is _ThinkFilter, which has
to decide from stream state rather than a regex over a half-written document."""

import re


_ORPHAN_TOOL_TAG_RE = re.compile(r"</?tool_call>|\[TOOL_CALLS\]|<\|python_tag\|>")


def _strip_tool_markup(content):
    """Drop wrapper tags left behind when a model doesn't close them.

    A model that emits `<tool_call>` then a bare `<function=...>` block and
    stops (no `</tool_call>`) defeats the paired-tag regex: the call is still
    recovered by the fallback paths, but the opening tag stays in the visible
    answer. Observed on Qwen3-8B; the user sees a stray `<tool_call>` at the
    end of every agent turn.
    """
    return _ORPHAN_TOOL_TAG_RE.sub("", content).strip()


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text):
    """Remove reasoning blocks, including one left unclosed at the cutoff.

    The web UI renders `<think>` as a collapsible block, so the OpenAI path
    keeps it. An API client has nowhere to put it — it is neither an answer
    nor a tool call — so the Anthropic endpoint strips it instead.
    """
    text = _THINK_RE.sub("", text)
    if "<think>" in text:                    # opened and never closed
        text = text.split("<think>", 1)[0]
    return text.replace("</think>", "").strip()
