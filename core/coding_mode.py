"""Coding mode's one rule: while it is on, the utility and audio paths are
refused rather than quietly competing for the device holding the coder."""

from core import config, runtime
from core.errors import openai_error


def _coding_mode_error(what):
    """Refuse a load that coding mode is deliberately holding closed.

    Returns a Flask response to return, or None when the path is allowed. The
    message names the toggle, because a feature that silently does nothing is
    the failure this repo keeps writing comments about.
    """
    if not config.CODING_MODE:
        return None
    return openai_error(
        f"Coding mode is on, so {what} stays unloaded and the memory stays "
        f"with the model driving the coding agent. Turn coding mode off in "
        f"Settings to use it again.", "server_error", 503)
