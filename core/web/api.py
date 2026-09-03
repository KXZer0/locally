"""Explicit URL reading and local SearXNG answer routes."""

import json
import time

from flask import Response, jsonify, request

from core.errors import openai_error
from core.web.fetch_page import _fetch_page_text, _web_search_run
from core.web import search as web_search_mod

_READ_URL_MAX_CHARS = 24000

def read_url():
    """Read one page and return its text, so a pasted link becomes context.

    This is the honest version of "read the site I am on". A separate window
    cannot see the user's tabs -- that needs a browser extension -- but a URL
    pasted into the chat is the same intent with none of the machinery, and it
    reuses the fetcher the web-search path already hardened (gzip, one retry,
    and a stated reason on failure).

    Deliberately not gated on --search-url: reading a link the user explicitly
    pasted is a different act from searching the web on their behalf. It is
    still outbound HTTP, so it does nothing unless asked.
    """
    body = request.get_json(silent=True) or {}
    url = str(body.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return openai_error("'url' must be an http:// or https:// address")

    t0 = time.perf_counter()
    text, reason = _fetch_page_text(url)
    if not text:
        return openai_error(f"Could not read that page - {reason}",
                            "server_error", 502)
    limit = min(int(body.get("max_chars", _READ_URL_MAX_CHARS)), 200000)
    return jsonify({
        "url": url,
        "text": text[:limit],
        "chars": len(text),
        "truncated": len(text) > limit,
        "took_ms": round((time.perf_counter() - t0) * 1000),
    })


def web_search():
    """Search the web, rank passages on the NPU, and cite what was used.

    JSON by default. With "stream": true the same run is delivered as SSE, one
    event per stage, so a 20-33 s turn can say what it is doing instead of
    showing a blinking dot.
    """
    body = request.get_json(silent=True) or {}
    reuse = (str(body.get("mode") or "").lower() == "analyze"
             and bool(body.get("passages")))
    if not web_search_mod.WEB_SEARCH_URL and not reuse:
        return openai_error(
            "Web search is off. Start locally with --search-url pointing at a "
            "SearXNG instance (e.g. --search-url http://localhost:8080). "
            "Without it the server makes no outbound connections.",
            "server_error", 503)

    if not body.get("stream"):
        for kind, value in _web_search_run(body):
            if kind == "error":
                message, status = value
                return openai_error(
                    message,
                    "invalid_request_error" if status == 400 else "server_error",
                    status)
            if kind == "done":
                return jsonify(value)
        return openai_error("Search produced no result", "server_error", 500)

    def events():
        try:
            for kind, value in _web_search_run(body):
                if kind == "error":
                    message, _status = value
                    yield ("data: " + json.dumps({"event": "error",
                                                  "message": message}) + "\n\n")
                    return
                if kind == "stage":
                    yield "data: " + json.dumps({"event": "stage", **value}) + "\n\n"
                elif kind == "done":
                    yield "data: " + json.dumps({"event": "done", **value}) + "\n\n"
        except GeneratorExit:
            raise
        except Exception as e:
            yield ("data: " + json.dumps({"event": "error",
                                          "message": str(e)}) + "\n\n")
        yield "data: [DONE]\n\n"

    return Response(events(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})

