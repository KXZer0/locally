"""Exercise a real server process with a synthetic upstream; loads no models."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.terminal import ApiError, Client, OwnedServer

ANSWER = "# Homework\n\n**Result:** π ≈ 3.14\n\n```python\nprint(2 + 2)\n```\n"


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": "fixture"}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if body.get("stream") else "application/json")
        self.end_headers()
        if body.get("stream"):
            for text in (ANSWER[:10], ANSWER[10:]):
                frame = {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
                self.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode())
            self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        else:
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "Hi"}}]}).encode())


def main():
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    owned = OwnedServer(port, ["--proxy-url", f"http://127.0.0.1:{upstream.server_port}",
                               "--no-util", "--no-model-cache", "--device", "CPU"])
    try:
        owned.start()
        client = Client(f"http://127.0.0.1:{port}")
        deadline = time.monotonic() + 30
        while True:
            if owned.process.poll() is not None:
                raise AssertionError(Path(owned.log_path).read_text(errors="replace"))
            try:
                if client.json("/health", timeout=1)["status"] == "ready":
                    break
            except ApiError:
                pass
            if time.monotonic() > deadline:
                raise AssertionError("Readiness timed out: " + str(owned.log_path))
            time.sleep(0.2)
        assert client.json("/")["mode"] == "headless"
        assert client.json("/v1/models")["data"][0]["id"] == "fixture@REMOTE"
        assert "".join(client.stream([{"role": "user", "content": "Explain 2 + 2"}], "fixture@REMOTE", 64)) == ANSWER
        attached = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] / "locally.py"),
             "chat", "--port", str(port), "--plain"],
            input="Explain 2 + 2\n/exit\n", text=True, encoding="utf-8",
            capture_output=True, timeout=15)
        assert attached.returncode == 0, attached.stdout + attached.stderr
        assert "Result:" in attached.stdout, attached.stdout + attached.stderr
        assert client.json("/")["mode"] == "headless", "Attached chat stopped an external server"
        try:
            client.json("/v1/ui/bootstrap")
            raise AssertionError("Retired browser route is still reachable")
        except ApiError:
            pass
        print("PASS: API ready, model listing, real HTTP streaming, exact Markdown, browser routes removed")
    finally:
        owned.close()
        upstream.shutdown()
        upstream.server_close()
    assert owned.process.poll() is not None
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0, "Owned server left a listener behind"
    print("PASS: owned process tree stopped and API port released")


if __name__ == "__main__":
    main()
