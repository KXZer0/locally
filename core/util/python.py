"""Bounded local Python utility endpoint."""

from flask import jsonify, request

from core import config
from core.errors import openai_error
from core.sandbox.python_exec import execute_python

def util_python():
    """Execute one bounded local calculation outside the Flask process.

    This endpoint is intentionally off by default. It is meant to give a
    small local model exact arithmetic, not to promise arbitrary untrusted
    code execution safety: the child is isolated, capped and killed on
    timeout, while Windows still has native escape paths we do not claim to
    close.
    """
    # Loopback only. The server binds 0.0.0.0 on purpose -- phones use the chat
    # UI -- so without this, enabling --python-tool would hand every device on
    # the network a code-execution endpoint. `request.remote_addr` is the socket
    # peer, not a header, so a client cannot spoof it.
    if request.remote_addr not in ("127.0.0.1", "::1", "localhost"):
        return openai_error("This endpoint is local-only (it runs code on the "
                            "server's machine).", "invalid_request_error", 403)
    if not config.PYTHON_TOOL_ENABLED:
        return openai_error(
            "Python calculations are off. Start locally with --python-tool to "
            "enable the local child-process tool.", "server_error", 503)
    body = request.get_json(silent=True) or {}
    code = body.get("code")
    if not isinstance(code, str) or not code.strip():
        return openai_error("'code' is required", "invalid_request_error", 400)
    return jsonify(execute_python(code))


