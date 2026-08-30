"""Web search: a SearXNG instance this server may start, and the grounding
blocks built from its results.

SearXNG follows the same rule as every slot -- resident only while it is
being used -- and only ever stops a process this server started."""

import os
import socket
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from core.errors import _TurnError
import concurrent.futures
import numpy as np

try:
    from utility_pipeline import UtilityUnavailable
except ImportError:
    class UtilityUnavailable(RuntimeError):
        """Stand-in so the except clauses always resolve."""


_SEARX_START_TIMEOUT = 45      # cold start compiles nothing but does load engines


_WEB_REACH_TTL = 60          # seconds


_WEB_UA = "locally/1.0 (local assistant; +https://github.com/aweussom/NoLlama)"


_web_reach_cache = {"at": 0.0, "ok": False}


SEARXNG_ROOT = None            # --searxng-root: enables on-demand start


SEARXNG_IDLE = 600             # seconds; 0 disables the idle stop


WEB_SEARCH_URL = None          # --search-url, e.g. http://localhost:8080


class _SearxManager:
    """Starts SearXNG on demand and stops it when it goes idle."""

    def __init__(self):
        self.proc = None           # set only when WE started it
        self.last_used = 0.0
        self.lock = threading.Lock()
        self.last_error = None

    # -- state ---------------------------------------------------------------

    def port(self):
        if not WEB_SEARCH_URL:
            return None
        try:
            return int(urllib.parse.urlsplit(WEB_SEARCH_URL).port or 8080)
        except Exception:
            return 8080

    def listening(self):
        p = self.port()
        if not p:
            return False
        s = socket.socket()
        s.settimeout(0.3)
        try:
            return s.connect_ex(("127.0.0.1", p)) == 0
        finally:
            s.close()

    def managed(self):
        return self.proc is not None and self.proc.poll() is None

    def can_start(self):
        return bool(SEARXNG_ROOT and os.path.isdir(SEARXNG_ROOT))

    # -- lifecycle -----------------------------------------------------------

    def ensure_running(self):
        """Return (ok, message). Starts SearXNG if it is configured and down."""
        self.last_used = time.time()
        if self.listening():
            return True, None
        if not self.can_start():
            return False, ("SearXNG is not running and no --searxng-root is set, "
                           "so it cannot be started automatically. Start it with "
                           "scripts/searxng.ps1.")
        with self.lock:
            if self.listening():          # another request won the race
                return True, None
            py = os.path.join(SEARXNG_ROOT, ".venv", "Scripts", "python.exe")
            if not os.path.exists(py):
                py = os.path.join(SEARXNG_ROOT, ".venv", "bin", "python")
            settings = next(
                (s for s in (os.path.join(SEARXNG_ROOT, "settings-locally.yml"),
                             os.path.join(SEARXNG_ROOT, "settings-nollama.yml"))
                 if os.path.exists(s)),
                os.path.join(SEARXNG_ROOT, "settings-locally.yml"))
            if not os.path.exists(py):
                return False, f"No SearXNG venv under {SEARXNG_ROOT}"
            env = dict(os.environ)
            if os.path.exists(settings):
                env["SEARXNG_SETTINGS_PATH"] = settings
            print("  [search] starting SearXNG on demand...", flush=True)
            t0 = time.perf_counter()
            try:
                self.proc = subprocess.Popen(
                    [py, "-m", "searx.webapp"], cwd=SEARXNG_ROOT, env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                self.proc = None
                self.last_error = str(e)
                return False, f"Could not start SearXNG: {e}"
            deadline = time.time() + _SEARX_START_TIMEOUT
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    self.proc = None
                    return False, ("SearXNG exited during startup — run "
                                   "scripts/searxng.ps1 to see why.")
                if self.listening():
                    print(f"  [search] SearXNG ready in "
                          f"{time.perf_counter() - t0:.1f}s", flush=True)
                    self.last_used = time.time()
                    return True, None
                time.sleep(0.4)
            self.stop()
            return False, (f"SearXNG did not come up within "
                           f"{_SEARX_START_TIMEOUT}s")

    def stop(self):
        """Stop only a process we started. A user-run instance is left alone."""
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return False
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        except Exception:
            pass
        print("  [search] SearXNG stopped (idle)", flush=True)
        return True

    def maybe_stop_idle(self):
        if not SEARXNG_IDLE or not self.managed():
            return
        if time.time() - self.last_used < SEARXNG_IDLE:
            return
        if not self.lock.acquire(blocking=False):
            return
        try:
            self.stop()
        finally:
            self.lock.release()

    def status(self):
        return {"managed": self.managed(), "can_start": self.can_start(),
                "root": SEARXNG_ROOT, "idle_timeout": SEARXNG_IDLE}


_searx = _SearxManager()


_WEB_ANSWER_RESERVE = 700


def web_search_status():
    """Is web search configured, and is the backend actually answering?

    The UI needs both: a toggle that is merely *configured* still looks broken
    when the backend is down, and a toggle greyed out for no stated reason is
    worse than one that works. Cached, because /health is polled every 15s and
    a dead backend costs a full TCP timeout each time.
    """
    if not WEB_SEARCH_URL:
        return {"enabled": False, "reachable": False, "url": None,
                "on_demand": False,
                "reason": "start with --search-url to enable web search"}
    # A backend that is DOWN but startable is not unavailable -- it is asleep,
    # and the UI must not grey the control out for that. This is the same
    # distinction as a model slot that is idle_unloaded rather than missing.
    if _searx.can_start():
        return {"enabled": True, "reachable": True, "url": WEB_SEARCH_URL,
                "on_demand": True, "running": _searx.listening(),
                "reason": None}
    now = time.time()
    if now - _web_reach_cache["at"] > _WEB_REACH_TTL:
        _web_reach_cache["at"] = now
        try:
            import urllib.request
            req = urllib.request.Request(
                WEB_SEARCH_URL.rstrip("/") + "/search?q=ping&format=json",
                headers={"User-Agent": _WEB_UA})
            with urllib.request.urlopen(req, timeout=3) as r:
                _web_reach_cache["ok"] = r.status == 200
        except Exception:
            _web_reach_cache["ok"] = False
    ok = _web_reach_cache["ok"]
    return {"enabled": True, "reachable": ok, "url": WEB_SEARCH_URL,
            "on_demand": False, "running": ok,
            "reason": None if ok else f"search backend at {WEB_SEARCH_URL} is not responding"}


def _web_grounded_blocks(results, sources):
    src_index = {src.get("url"): n for n, src in enumerate(sources, 1)
                 if src.get("url")}
    by_source = {}
    for result in results:
        n = src_index.get(result.get("url"))
        if n:
            by_source.setdefault(n, []).append(result["text"])
    blocks = []
    for n in sorted(by_source):
        blocks.append("[" + str(n) + "] " + sources[n - 1].get("title", "Source")
                     + "\n" + "\n".join(by_source[n]))
    return "\n\n".join(blocks)


