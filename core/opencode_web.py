"""The OpenCode WEB server: pin its port, adopt or start it, stop it cleanly.

In here: the lifecycle of the headless `opencode serve` process whose web UI
the Code tab shows, and the reachability facts the UI needs to decide between
showing it and explaining why it cannot.
Not in here: the TUI launcher (that is still locally.py's /v1/opencode/launch),
and any provider-overlay building -- the caller passes the environment in, so
there is one place that knows how locally is added as a provider.

Three measured facts decide this module's shape, all taken from OpenCode
1.18.25 on 2026-08-30:

1. **`serve` already serves the web UI.** `opencode web` is `serve` plus
   opening the user's default browser at it. We want the UI inside locally's
   Code tab, so opening a second browser window would be the app fighting
   itself for where the thing appears. We run `serve` and own the URL.
2. **`--port` defaults to 0**, i.e. a random free port. A caller that does not
   pin the port cannot link to, embed, or health-check what it just started.
   Pinning the port is therefore not process-management enthusiasm; it is the
   minimum required to be able to name the result.
3. **It refuses nothing about framing.** The response carries a CSP with no
   `frame-ancestors`, no `X-Frame-Options` and no HSTS, and an iframe of it
   loads and renders. That is the opposite of Odysseus (see docs/ODYSSEUS.md),
   which is why this one is embedded and that one is a link.

The bind address is 127.0.0.1 and is not configurable here. `opencode serve`
prints "OPENCODE_SERVER_PASSWORD is not set; server is unsecured" on startup,
and the thing it serves runs arbitrary commands in a workspace. locally binds
0.0.0.0 on purpose so phones can use the chat UI; handing the same reach to an
unauthenticated agent that can execute code would be giving every device on the
network a shell. Loopback is the whole security model, so it is a constant.
"""

import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from core.portprobe import CachedProbe

# The pinned port. Chosen high and out of the way of locally (8000), Odysseus
# (7000), SearXNG (8080/8081), ChromaDB (8100) and Ollama (11434).
DEFAULT_PORT = 4747
HOST = "127.0.0.1"

_lock = threading.Lock()
_process = None          # the child we started, or None when we adopted one
_started_port = None


def command():
    """Resolved `opencode` executable, or None."""
    return shutil.which("opencode")


def _port_is_open(port, timeout=0.35):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((HOST, int(port))) == 0


def _serves_opencode(port, timeout=0.4):
    """True when whatever holds the port answers as the OpenCode web UI.

    An open port is not proof: the user may have something else on it, and
    embedding an unrelated page in the Code tab would be worse than saying
    nothing. The check is the page title, which is what the app actually
    serves at `/`.
    """
    try:
        request = urllib.request.Request(f"http://{HOST}:{int(port)}/",
                                         headers={"Accept": "text/html"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            head = response.read(4096).decode("utf-8", "replace").lower()
        return "<title>opencode</title>" in head
    except (urllib.error.URLError, OSError, ValueError):
        return False


# /health calls status() on every poll, so the probe there must never be the
# thing the request waits on. See core/portprobe.py for the measurement.
_probe = CachedProbe(lambda: _serves_opencode(DEFAULT_PORT))


def url(port=None):
    return f"http://{HOST}:{int(port or _started_port or DEFAULT_PORT)}"


def status(port=None):
    """What the Code tab needs to decide what to show, without starting anything."""
    port = int(port or _started_port or DEFAULT_PORT)
    executable = command()
    # Cached: this is the /health hot path. start()/stop() call set()/
    # refresh_now(), so a state change we caused is reflected immediately
    # rather than up to a TTL later.
    running = _probe.get() if port == DEFAULT_PORT else _serves_opencode(port)
    with _lock:
        ours = bool(_process and _process.poll() is None)
    return {
        "installed": bool(executable),
        "running": running,
        # Distinguishing the two matters for the stop button: we may stop a
        # child we started, but killing a server the user ran by hand would be
        # taking away something we did not provide.
        "managed": ours and running,
        "port": port,
        "url": url(port) if running else None,
        # The iframe can only work when the browser is on this machine, since
        # the server is loopback-bound by design. The page checks its own
        # hostname; this is here so /health tells the same story.
        "host": HOST,
    }


def start(port=None, env=None, cwd=None, wait_secs=25):
    """Adopt an already-listening OpenCode server, or start one.

    Adoption comes first on purpose. A user who already has `opencode serve`
    on this port gets their session shown rather than a second server fighting
    for the port and losing with a confusing bind error.

    Returns (ok, detail).
    """
    global _process, _started_port
    port = int(port or DEFAULT_PORT)

    if _serves_opencode(port):
        _started_port = port
        _probe.set(True)
        return True, "adopted"

    executable = command()
    if not executable:
        return False, "OpenCode is not installed."

    if _port_is_open(port):
        return False, (f"Port {port} is in use by something that is not "
                       f"OpenCode. Free it, then try again.")

    argv = [executable, "serve", "--port", str(port), "--hostname", HOST]
    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        with _lock:
            _process = subprocess.Popen(
                argv, cwd=cwd or None, env=env or None,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=creation)
    except Exception as exc:
        return False, f"Could not start the OpenCode server: {exc}"

    # Poll rather than sleep a fixed amount: it is ready in about two seconds
    # on a warm cache and much longer on the first run of a new version, and
    # returning before it can serve would hand the UI a URL that 404s.
    deadline = time.time() + wait_secs
    while time.time() < deadline:
        with _lock:
            died = _process.poll() is not None
        if died:
            return False, ("The OpenCode server exited immediately. Run "
                           f"`opencode serve --port {port}` to see why.")
        if _serves_opencode(port):
            _started_port = port
            _probe.set(True)
            return True, "started"
        time.sleep(0.4)
    return False, (f"The OpenCode server did not answer on port {port} within "
                   f"{wait_secs}s.")


def _kill_tree(process):
    """Kill the child AND its descendants, then say whether it worked.

    On Windows `shutil.which("opencode")` resolves to `opencode.CMD`, so our
    direct child is a cmd.exe wrapper and the real `opencode.exe` is its child.
    `terminate()` reaps only the wrapper: measured 2026-08-30, the UI reported
    "Stopped the OpenCode server" while the server kept answering 200 on the
    port. Worse than the lie is what follows -- the orphan outlives locally,
    keeps the pinned port, and is then ADOPTED by the next start, so the user
    gets a server running against a stale config with no way to see why.

    So the tree is killed as one, and it must be killed BEFORE the wrapper is
    reaped: taskkill walks live parent links, and terminating the wrapper first
    is what breaks the chain to the process actually holding the port.
    """
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True, timeout=15)
        except Exception:
            pass
    try:
        process.terminate()
        try:
            process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=4)
    except Exception:
        pass


def stop(port=None):
    """Stop a server this process started. Never one the user ran themselves.

    Returns True only when the port has actually stopped serving OpenCode --
    the caller shows this to a user, and "stopped" has to mean stopped.
    """
    global _process, _started_port
    with _lock:
        process, _process = _process, None
    port = int(port or _started_port or DEFAULT_PORT)
    _started_port = None
    if not process or process.poll() is not None:
        return False

    _kill_tree(process)

    # The port is the only honest test. The kill can succeed against a wrapper
    # that was never the listener, so ask the thing we actually care about.
    deadline = time.time() + 6
    while time.time() < deadline:
        if not _serves_opencode(port, timeout=0.6):
            _probe.set(False)
            return True
        time.sleep(0.3)
    return False
