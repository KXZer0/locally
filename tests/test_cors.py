"""A browser rejects a cross-origin response with no Access-Control-Allow-Origin
before the calling code sees it, so the client reports a bare "Failed to fetch"
against a server that answered 200 and logged nothing wrong. That is what these
guard: the header being present for the clients we support, and absent for
everyone else -- this server binds 0.0.0.0 and takes no credentials, so a
reflected origin is a page in any open tab driving the model.
"""

import unittest

from flask import Flask, jsonify

from core import config
from core.cors import attach_cors, origin_allowed

OBSIDIAN = "app://obsidian.md"


def _app():
    app = Flask("cors-test")
    attach_cors(app)

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat():
        return jsonify({"ok": True}), 200, {"X-Device": "NPU"}

    return app.test_client()


class OriginMatchTests(unittest.TestCase):
    def test_obsidian_desktop_and_mobile_are_allowed_by_default(self):
        for origin in (OBSIDIAN, "capacitor://localhost", "http://localhost"):
            self.assertEqual(origin_allowed(origin, config.CORS_ORIGINS), origin)

    def test_localhost_matches_on_any_port(self):
        # The Odysseus and OpenCode pages are localhost with a port.
        for origin in ("http://localhost:7000", "http://127.0.0.1:4747"):
            self.assertEqual(origin_allowed(origin, config.CORS_ORIGINS), origin)

    def test_unknown_origin_is_refused(self):
        self.assertIsNone(origin_allowed("https://evil.example", config.CORS_ORIGINS))

    def test_suffix_lookalike_is_refused(self):
        # Substring matching would let these through; fnmatch does not.
        for origin in ("https://localhost.evil.example", "app://obsidian.md.evil"):
            self.assertIsNone(origin_allowed(origin, config.CORS_ORIGINS))

    def test_empty_allowlist_refuses_everything(self):
        self.assertIsNone(origin_allowed(OBSIDIAN, []))

    def test_star_echoes_the_caller(self):
        self.assertEqual(origin_allowed("https://evil.example", ["*"]),
                         "https://evil.example")


class ResponseHeaderTests(unittest.TestCase):
    def setUp(self):
        self.origins = list(config.CORS_ORIGINS)
        self.client = _app()

    def tearDown(self):
        config.CORS_ORIGINS = self.origins

    def test_preflight_is_answered_with_the_headers_the_browser_asked_for(self):
        r = self.client.options("/v1/chat/completions", headers={
            "Origin": OBSIDIAN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,authorization"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("Access-Control-Allow-Origin"), OBSIDIAN)
        # Echoed, so a client sending x-api-key or anthropic-version works too.
        self.assertEqual(r.headers.get("Access-Control-Allow-Headers"),
                         "content-type,authorization")
        self.assertIn("POST", r.headers.get("Access-Control-Allow-Methods", ""))

    def test_post_carries_the_origin_and_exposes_x_device(self):
        r = self.client.post("/v1/chat/completions", json={},
                             headers={"Origin": OBSIDIAN})
        self.assertEqual(r.headers.get("Access-Control-Allow-Origin"), OBSIDIAN)
        self.assertIn("X-Device", r.headers.get("Access-Control-Expose-Headers", ""))

    def test_vary_on_origin_is_always_set(self):
        # Set even when the origin is refused: the same URL answers with
        # different headers per Origin, and a cache that misses this serves
        # one client's answer to another.
        r = self.client.post("/v1/chat/completions", json={},
                             headers={"Origin": "https://evil.example"})
        self.assertIn("Origin", r.headers.get("Vary", ""))

    def test_refused_origin_gets_no_header(self):
        r = self.client.post("/v1/chat/completions", json={},
                             headers={"Origin": "https://evil.example"})
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.headers.get("Access-Control-Allow-Origin"))

    def test_no_cors_empties_the_allowlist(self):
        config.CORS_ORIGINS = []
        r = self.client.post("/v1/chat/completions", json={},
                             headers={"Origin": OBSIDIAN})
        self.assertIsNone(r.headers.get("Access-Control-Allow-Origin"))

    def test_same_origin_request_is_untouched(self):
        r = self.client.post("/v1/chat/completions", json={})
        self.assertIsNone(r.headers.get("Access-Control-Allow-Origin"))


class FlagTests(unittest.TestCase):
    """core.cli pulls in the native stack, so skip rather than fail where
    OpenVINO is not installed -- the same rule the other suites follow."""

    def test_parser_accepts_repeated_cors_origin_and_no_cors(self):
        try:
            from core.cli import parse_args
        except (ImportError, OSError) as exc:
            self.skipTest(f"core.cli unavailable: {exc}")
        args = parse_args(["--cors-origin", "vscode-webview://*",
                           "--cors-origin", "http://a.b"])
        self.assertEqual(args.cors_origin, ["vscode-webview://*", "http://a.b"])
        self.assertFalse(args.no_cors)
        self.assertTrue(parse_args(["--no-cors"]).no_cors)


if __name__ == "__main__":
    unittest.main()
