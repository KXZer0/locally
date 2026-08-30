"""Preparing a conversation for a web-grounded answer, and spotting a page
that is really a login wall rather than the article it claimed to be.
"""
import re


# A login wall answers 200 with a real page, so nothing upstream catches it.
# Measured: x.com/intel returned 5,921 characters beginning "Log in", and
# linkedin.com/feed 1,882 beginning "Sign in / New to LinkedIn? Join now".
# Both would have been handed to the model as though they were the article,
# and a summary of a signup page is worse than an honest failure -- it is
# wrong without looking wrong.
_LOGIN_MARKERS = re.compile(
    r"\b(log ?in|sign ?in|sign ?up|create an account|join now|subscribe to (read|continue)"
    r"|to continue reading|members only|register to (read|continue))\b", re.I)


_LOGIN_WALL_MAX_CHARS = 6000


def _web_history_text(history):
    return "\n\n".join(str(m.get("content") or "") for m in history)


def _web_message_copy(messages):
    """Flatten OpenAI text blocks into the string form ChatHistory accepts."""
    copied = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(block.get("text") or "")
                                  for block in content
                                  if isinstance(block, dict) and block.get("type") == "text")
        copied.append({"role": str(msg.get("role") or "user"),
                       "content": str(content)})
    return copied
