#!/usr/bin/env python
"""Serve the web UI with no model, no OpenVINO, no device.

The frontend work in REBUILD-PLAN §2.1-2.3 is measured in a real browser
(see scripts/css-oracle.js), and the real server needs 10-40s to compile a
model onto a device before it will serve a single byte of HTML. That is a
cost the CSS and DOM work has no reason to pay, and on a box that is busy
exporting a model it may not be payable at all.

This serves templates/index.html and static/ verbatim, plus the smallest
possible stubs for the endpoints the page calls on boot, so the shell renders
in the state it renders in for real. It is a measuring rig; it never answers
a chat request. Run it, then point the oracle at it:

    python scripts/uiserve.py --port 8777
"""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parent.parent

# Enough of /health for the shell to paint its normal boot state: one GPU slot
# holding one model. Fabricating a device here is honest -- the page is being
# measured, not used -- but keep it boring so nothing in the UI lights up a
# path (offloading, critical memory) it would not normally be in.
HEALTH = {
    "status": "ok",
    "version": "ui-serve",
    "prompt_cache": False,
    "slots": [{
        "device": "GPU", "device_name": "GPU", "model": "Qwen3-8B-int4-ov",
        "loaded": True, "kind": "llm", "context_tokens": 32768,
        "kv_pool_gb": 4, "last_ttft_ms": 0, "prewarmed": False,
    }],
    "util": {}, "opencode_web": {"running": False},
}
MODELS = {"object": "list", "data": [
    {"id": "Qwen3-8B-int4-ov", "object": "model", "owned_by": "locally"}]}
AVAILABLE = {"models": []}
MEMORY = {
    "devices": [{
        "device": "GPU", "total_bytes": 27_200_000_000,
        "usable_bytes": 18_500_000_000, "weights_bytes": 4_700_000_000,
        "state": "resident",
    }],
    "system": {"total_bytes": 33_800_000_000, "available_bytes": 15_000_000_000},
}
# The onboarding overlay opens whenever /v1/setup fails or reports first run,
# and a modal over the shell is not the state the shell is being measured in.
# Answering "configured, nothing needed" is what a real machine that has been
# through setup returns.
SETUP = {
    "first_run": False, "needs_assistant": False,
    "devices": [{"kind": "GPU", "id": "GPU", "name": "Intel Arc B390"}],
    "memory": MEMORY, "nvidia": None, "ollama": None, "catalog": [],
    "recommended": {"assistant": None, "voice": []},
    "configured": {"assistant": "Qwen3-8B-int4-ov"},
    "models_root": str(ROOT / "models"),
}
# The web UI's real boot is ONE request. /v1/ui/bootstrap answers health,
# models, available models and memory together, and system/bootstrap.js falls
# back to the four public endpoints only when it fails. Serving the four and
# not this one meant every capture ever taken on this rig booted through the
# `catch` -- readable as `document.documentElement.dataset.bootstrapSource ==
# "fallback"` -- so the rig was measuring a path the app does not take, with a
# failed request inside every measurement. Composed from the same four
# payloads: one source of truth per endpoint, so the fallback and the fast
# path can never disagree here in a way they would not disagree for real.
BOOTSTRAP = {
    "health": HEALTH,
    "models": MODELS,
    "available_models": AVAILABLE,
    "memory": MEMORY,
}
STUBS = {
    "/health": HEALTH,
    "/v1/setup": SETUP,
    "/v1/models": MODELS,
    "/v1/models/available": AVAILABLE,
    "/v1/memory": MEMORY,
    "/v1/ui/bootstrap": BOOTSTRAP,
}

MIME = {
    ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8", ".json": "application/json",
    ".svg": "image/svg+xml", ".woff2": "font/woff2", ".png": "image/png",
    ".ico": "image/x-icon", ".webmanifest": "application/manifest+json",
}


def _strip_element(html, element_id):
    """Remove the element carrying id="<element_id>", tag and all.

    Deliberately dumb: it finds the tag that opens with that id and drops
    through its matching close, and it only handles the flat, non-nested
    elements the strip flag is used on. Anything cleverer would be an HTML
    parser, and this is a measuring rig.
    """
    marker = f'id="{element_id}"'
    at = html.find(marker)
    if at < 0:
        return html
    start = html.rfind("<", 0, at)
    tag = re.match(r"<([a-zA-Z0-9-]+)", html[start:]).group(1)
    end = html.find(f"</{tag}>", at)
    if end < 0:
        return html
    return html[:start] + html[end + len(tag) + 3:]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # one line per asset drowns the console
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path in STUBS:
            return self._json(STUBS[path])
        if path in ("/", "/index.html"):
            html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
            # ?strip=id removes one element from the served markup. A new
            # element shifts every later sibling's nth-child index, so an
            # oracle capture taken with it cannot be compared path-for-path
            # with one taken without it. Serving the page both ways is the
            # only way to ask "did adding this move anything else" and get an
            # answer rather than a renumbering.
            for name in parse_qs(query).get("strip", []):
                html = _strip_element(html, name)
            return self._send(200, html.encode("utf-8"), MIME[".html"])
        if path.startswith("/static/"):
            target = (ROOT / path.lstrip("/")).resolve()
            if ROOT in target.parents:
                return self._file(target)
        self._json({"error": "not found", "path": path}, 404)

    def do_POST(self):
        # Drain the body first. On HTTP/1.1 the connection is kept alive, so an
        # unread body is parsed as the NEXT request line -- which showed up as a
        # 400 on a perfectly good /health that followed a POST.
        n = int(self.headers.get("Content-Length") or 0)
        while n > 0:
            n -= len(self.rfile.read(min(n, 65536)))
        # Nothing here generates. A POST that reaches this rig is a request the
        # page should not be making at boot, so say so rather than pretending.
        self._json({"error": "uiserve is a static measuring rig"}, 501)

    def _file(self, target):
        if not target.is_file():
            return self._json({"error": "not found", "path": self.path}, 404)
        self._send(200, target.read_bytes(),
                   MIME.get(target.suffix, "application/octet-stream"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"ui-serve on http://127.0.0.1:{args.port}  (root {ROOT})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
