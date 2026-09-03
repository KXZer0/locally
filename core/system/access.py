"""Who is allowed to call an endpoint.

A leaf on purpose: this imports flask and nothing else. The predicate used
to live in setup_info, and pulling it into coding.py closed the loop
status -> coding -> setup_info -> status. A two-line request check should
never decide the import graph.
"""

from flask import request


def _request_is_local():
    """True when the caller is on this machine.

    The gate on every endpoint that runs a command, launches a window or
    reads the filesystem: locally binds 0.0.0.0 so a phone on the LAN can
    chat, and those endpoints must not follow it there.
    """
    return request.remote_addr in ("127.0.0.1", "::1", "localhost")
