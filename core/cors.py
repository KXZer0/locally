"""Cross-origin headers, so browser-based clients can reach the API.

A local server is a *cross-origin* server to every app that is not a page it
served itself. Obsidian's renderer runs at `app://obsidian.md`, so a plugin
calling locally does a cross-origin `fetch`, and a `fetch` whose response
carries no `Access-Control-Allow-Origin` is rejected by the browser before any
plugin code sees it. What the user is shown is the bare string "Failed to
fetch" -- no status, no body -- and there is nothing in locally's log to match
it against, because the request *did* arrive and *was* answered 200. That
mismatch is the whole reason this file has a docstring: the symptom points at
the network and the cause is a missing response header.

The allowlist is load-bearing, not decoration. locally binds 0.0.0.0 and
answers unauthenticated, so `Access-Control-Allow-Origin: *` would let any
page in any open tab drive the model, enumerate `/v1/models` and read whatever
a util endpoint will read. Echoing the caller's own `Origin` unconditionally
is the same thing wearing a hat. Ollama shipped `OLLAMA_ORIGINS` for exactly
this; `--cors-origin` is that flag.
"""

import fnmatch

from flask import request

from core import config

# Sent to the browser as readable on a cross-origin response. Without this a
# client can read the body and not `X-Device`, which is the header saying
# which engine actually answered.
EXPOSED_HEADERS = "X-Device, X-Model, X-Vision-Path"

# Preflight fallback. Normally the browser's own
# `Access-Control-Request-Headers` is echoed instead -- an OpenAI client sends
# `authorization`, an Anthropic one sends `x-api-key` and `anthropic-version`,
# and echoing means a client we have never heard of works too.
DEFAULT_ALLOW_HEADERS = ("Authorization, Content-Type, X-Api-Key, "
                         "Anthropic-Version, Anthropic-Beta, X-Requested-With")

ALLOW_METHODS = "GET, POST, DELETE, OPTIONS"


def origin_allowed(origin, patterns):
    """The origin to echo back, or None. `fnmatch`, so '*' and ':*' work."""
    if not origin or not patterns:
        return None
    for pattern in patterns:
        if pattern == "*":
            # Deliberate: --cors-origin '*'. Echo rather than literal '*' so
            # the answer stays valid if credentials are ever turned on.
            return origin
        if fnmatch.fnmatchcase(origin, pattern):
            return origin
    return None


def _vary_on_origin(response):
    # The same URL answers with different headers per Origin, so a cache that
    # does not vary on it will serve one client's ACAO to another.
    existing = response.headers.get("Vary")
    if not existing:
        response.headers["Vary"] = "Origin"
    elif "origin" not in existing.lower():
        response.headers["Vary"] = existing + ", Origin"


def attach_cors(app):
    """Add the headers to every response, including Flask's automatic OPTIONS.

    An `after_request` hook rather than a route: preflight is answered by
    Flask's own automatic OPTIONS handler for every route that declares POST,
    and error responses (413 on an oversized image, 500 on a failed load) need
    the header too or the client reports "Failed to fetch" for what is
    actually a perfectly legible error.
    """
    @app.after_request
    def _cors(response):
        _vary_on_origin(response)
        allowed = origin_allowed(request.headers.get("Origin"),
                                 config.CORS_ORIGINS)
        if not allowed:
            return response
        response.headers["Access-Control-Allow-Origin"] = allowed
        response.headers["Access-Control-Expose-Headers"] = EXPOSED_HEADERS
        if request.method == "OPTIONS":
            asked = request.headers.get("Access-Control-Request-Headers")
            response.headers["Access-Control-Allow-Headers"] = (
                asked or DEFAULT_ALLOW_HEADERS)
            response.headers["Access-Control-Allow-Methods"] = ALLOW_METHODS
            response.headers["Access-Control-Max-Age"] = "600"
        return response

    return app
