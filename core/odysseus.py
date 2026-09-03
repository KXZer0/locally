"""Odysseus's lifecycle: find the checkout, adopt or start its stack, stop it.

In here: everything locally needs to answer "is the assistant up, and can I
bring it up?" without the user leaving the browser. Odysseus is a Docker
Compose app (services `odysseus`, `chromadb`, `searxng`, `ntfy`) serving on
127.0.0.1:7000 — see docs/ODYSSEUS.md for the split: Odysseus owns the
assistant, locally owns the silicon.

Not in here: any opinion about how the UI shows it. Odysseus is a LINK in the
sidebar and never an iframe (HSTS + its own cookie host + a service worker each
break an embed independently, docs/ODYSSEUS.md §1). This module only reports
facts and runs compose.

Three shape decisions, each from something that can actually go wrong here:

1. **`<engine> info`, not just `shutil.which(...)`.** Docker Desktop installs an
   executable that runs fine while the daemon is stopped (and Podman's runs
   fine before `podman machine start`) — it prints an error
   and exits non-zero. That is by far the most likely failure on a laptop, and
   "docker is installed" would be a true statement that sends the user looking
   in the wrong place. The two states get separate fields and separate
   messages.
2. **A directory only counts when it holds a compose file.** `~/odysseus` can
   exist as an empty clone target or a leftover; calling that "installed" makes
   the start button appear and then fail with a compose error nobody asked for.
3. **`stop`, never `down`.** See stop().
"""

import json
import os
import signal
import shutil
import sys
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

# The module, never the names: main() reassigns ODYSSEUS_PORT/ODYSSEUS_DIR
# after the command line is parsed.
from core import config
from core.portprobe import CachedProbe

# Odysseus's own default (APP_PORT, compose publishes
# ${APP_BIND:-127.0.0.1}:${APP_PORT:-7000}:7000). Confirmed against the repo.
DEFAULT_PORT = 7000
HOST = "127.0.0.1"

# All four spellings Compose itself accepts, in its own precedence order.
COMPOSE_FILES = ("docker-compose.yml", "docker-compose.yaml",
                 "compose.yml", "compose.yaml")

_lock = threading.Lock()
_managed = False        # True only when THIS process ran `compose up`
_managed_machine = False  # True only when THIS process started Podman's VM
_started_dir = None

# `docker info` costs ~0.3 s warm and several seconds when the daemon is down,
# and status() is on the /health path the UI polls. Cache the verdict briefly;
# start()/stop() clear it so a user who just started Docker is not told for
# another half minute that it is off.
_DOCKER_TTL = 30.0
_docker_cache = (0.0, False)


def _run(argv, cwd=None, timeout=30, env=None):
    """Run a command with its output captured. Returns the CompletedProcess or None.

    Never inherits stdio: compose writes progress bars and image-pull output,
    and letting that into locally's console would bury the startup summary.
    """
    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = None
    try:
        # encoding/errors are NOT optional. `text=True` alone decodes with the
        # locale codec -- cp1252 on this machine -- and compose output contains
        # bytes it cannot represent. Measured 2026-08-30 on a real
        # `podman compose up`: the reader thread died with
        # UnicodeDecodeError: 'charmap' codec can't decode byte 0x81, which
        # loses the output we then try to show the user in the failure path.
        proc = subprocess.Popen(
            argv, cwd=cwd or None, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, creationflags=creation,
            start_new_session=(os.name != "nt"), env=env,
            encoding="utf-8", errors="replace")
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(
            argv, proc.returncode, stdout=stdout, stderr=stderr)
    except subprocess.TimeoutExpired:
        # `podman compose` delegates to podman-compose, which then launches
        # more podman processes. Killing only the wrapper leaves those children
        # holding stdout/stderr open, so communicate() never finishes and the
        # HTTP request can outlive its own timeout. Measured on Windows with an
        # orphaned `podman wait --condition=running` four processes deep.
        try:
            if os.name == "nt" and proc is not None:
                subprocess.run(
                    ["taskkill.exe", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=15, creationflags=creation)
            elif proc is not None:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                if proc is not None:
                    proc.kill()
            except Exception:
                pass
        try:
            if proc is not None:
                proc.communicate(timeout=5)
        except Exception:
            pass
        return None
    except Exception:
        try:
            if proc is not None:
                proc.kill()
        except Exception:
            pass
        return None


# Docker first, then Podman. Both are asked for by name only -- nothing here
# assumes Docker semantics beyond `<engine> compose up -d` and `stop`, which
# Podman 4.1+ implements (it delegates to docker-compose or podman-compose).
#
# Podman works for Odysseus specifically because the ONE Docker-shaped thing
# this architecture depends on is not Docker magic: Odysseus's compose file
# declares `extra_hosts: ["host.docker.internal:host-gateway"]` explicitly, and
# `host-gateway` is honoured by Podman too. That hostname is how the container
# reaches locally on the host (docs/ODYSSEUS.md §3), so if it resolved only
# under Docker, none of this would be portable. It is declared, so it is.
ENGINES = ("docker", "podman")


# Where the installers actually put these, for when PATH does not have them.
#
# PATH is not reliable here and the failure is silent: a locally process
# inherits the environment of whatever started it, so one launched from a shell
# or session that PREDATES the container-engine install sees no engine at all
# and reports "Docker is not installed" on a machine where Podman is installed,
# running, and serving. Measured 2026-08-30: shutil.which returned None for
# both engines while `podman info` succeeded from another shell.
_WELL_KNOWN = {
    "podman": [
        os.path.join(os.environ.get("LOCALAPPDATA", ""),
                     "Programs", "Podman", "podman.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""),
                     "RedHat", "Podman", "podman.exe"),
        "/usr/bin/podman", "/usr/local/bin/podman", "/opt/homebrew/bin/podman",
    ],
    "docker": [
        os.path.join(os.environ.get("ProgramFiles", ""),
                     "Docker", "Docker", "resources", "bin", "docker.exe"),
        "/usr/bin/docker", "/usr/local/bin/docker",
    ],
}


def _well_known(name):
    """Engine path from a known install location, or None."""
    for candidate in _WELL_KNOWN.get(name, ()):
        try:
            if candidate and os.path.isfile(candidate):
                return candidate
        except OSError:
            continue
    return None


def command():
    """Resolved container engine executable, or None.

    Honours --container-engine when set; otherwise takes the first of
    ENGINES that is on PATH.
    """
    pinned = getattr(config, "CONTAINER_ENGINE", None)
    if pinned:
        return shutil.which(pinned) or _well_known(pinned)
    for name in ENGINES:
        found = shutil.which(name) or _well_known(name)
        if found:
            return found
    return None


def _compose_env():
    """Environment for a compose call, with our own venv on PATH.

    `podman compose` is a THIN WRAPPER, not an implementation: it shells out to
    an external `docker-compose` or `podman-compose` binary and fails outright
    when neither is on PATH. Found by running it, 2026-08-30:

        Error: looking up compose provider failed
          * exec: "docker-compose": executable file not found in %PATH%
          * exec: "podman-compose": executable file not found in %PATH%

    locally ships a venv, and `pip install podman-compose` puts the provider in
    the same directory as the interpreter running this process. Handing that
    directory to the subprocess makes the dependency self-contained rather than
    requiring a global install -- and it is derived from sys.executable, not a
    guessed "venv/Scripts", so it is right on POSIX and in any venv layout.
    """
    env = dict(os.environ)
    own_bin = os.path.dirname(sys.executable)
    if own_bin and own_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = own_bin + os.pathsep + env.get("PATH", "")
    return env


def engine_name():
    """Bare name of the engine in use ('docker' / 'podman'), or None."""
    path = command()
    if not path:
        return None
    base = os.path.basename(path).lower()
    for ext in (".exe", ".cmd", ".bat"):
        if base.endswith(ext):
            base = base[:-len(ext)]
    return base


def docker_available(refresh=False):
    """True when the engine CLI exists AND its backend answers.

    Kept under this name because it is what /health and the UI already read.
    `info` is the cheapest call that requires a working backend; `--version`
    succeeds against a stopped Docker daemon and against a Podman machine that
    has never been started, which is exactly the state we need to detect.
    """
    global _docker_cache
    age, value = _docker_cache
    if not refresh and (time.time() - age) < _DOCKER_TTL:
        return value
    ok = False
    exe = command()
    if exe:
        # Podman is daemonless and has no .ServerVersion in the Docker sense,
        # so ask for nothing and judge on the exit code alone -- `info` fails
        # when the backend (or the podman machine) is not usable, which is the
        # only thing this needs to know.
        out = _run([exe, "info"], timeout=20)
        ok = bool(out and out.returncode == 0)
    _docker_cache = (time.time(), ok)
    return ok


def _start_podman_machine():
    """Start Podman's VM without opening Podman Desktop or a console window.

    Podman Desktop is an optional GUI, not a daemon dependency.  On Windows and
    macOS the CLI talks to a small VM, so an on-demand start is the quiet path:
    no login app, no terminal the user has to keep open, and no idle VM until
    Odysseus is actually requested.
    """
    global _managed_machine
    exe = command()
    if engine_name() != "podman" or not exe:
        return False, "Podman is not installed."
    result = _run([exe, "machine", "start"], timeout=120)
    _docker_cache_clear()
    if result and result.returncode == 0 and docker_available(refresh=True):
        _managed_machine = True
        return True, "started"
    detail = _tail(result.stderr if result else "") or _tail(
        result.stdout if result else "") or "no output"
    return False, f"Podman's machine did not start: {detail}"


def _stop_managed_podman_machine_if_idle():
    """Release Podman's VM only when we started it and no container is running."""
    global _managed_machine
    with _lock:
        managed, _managed_machine = _managed_machine, False
    exe = command()
    if not managed or engine_name() != "podman" or not exe:
        return False

    # Never interrupt an unrelated container.  `podman ps -q` is deliberately
    # used after the Odysseus stack has stopped, so an empty answer means the VM
    # is now pure idle overhead.
    running = _run([exe, "ps", "-q"], timeout=20)
    if not running or running.returncode != 0 or running.stdout.strip():
        return False
    stopped = _run([exe, "machine", "stop"], timeout=90)
    _docker_cache_clear()
    return bool(stopped and stopped.returncode == 0)


def compose_file(directory):
    """The compose file in `directory`, or None. A checkout without one is not an install."""
    if not directory or not os.path.isdir(directory):
        return None
    for name in COMPOSE_FILES:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def find_dir(explicit=None):
    """Locate an Odysseus checkout, or None.

    The search order is ODYSSEUS_DIR, then the sibling of this checkout, then
    ~/odysseus — deliberately identical to locally.py's `_local_version()`,
    which already looks Odysseus up for the update check. Two places that
    disagree about where Odysseus lives is a bug report nobody can reproduce.
    """
    candidates = [explicit, os.environ.get("ODYSSEUS_DIR"),
                  os.path.join(os.path.dirname(config.SCRIPT_DIR), "odysseus"),
                  os.path.join(os.path.expanduser("~"), "odysseus")]
    for path in candidates:
        if not path:
            continue
        path = os.path.abspath(os.path.expanduser(str(path)))
        if compose_file(path):
            return path
    return None


def _port_is_open(port, timeout=0.35):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(timeout)
            return probe.connect_ex((HOST, int(port))) == 0
    except OSError:
        return False


def _serves(port, timeout=0.4):
    """True when something answers HTTP on the port.

    Any response counts, including a 4xx/5xx — unlike OpenCode we do not match
    a page title. Odysseus's markup is not ours to depend on, it redirects to a
    login, and a version bump that changed the title would silently turn "up"
    into "down". An open socket alone is too weak (a half-open Docker port
    forward accepts and hangs), so the bar is: it spoke HTTP.
    """
    if not _port_is_open(port):
        return False
    try:
        request = urllib.request.Request(f"http://{HOST}:{int(port)}/",
                                         headers={"Accept": "text/html"})
        urllib.request.urlopen(request, timeout=timeout).close()
        return True
    except urllib.error.HTTPError:
        return True            # 401/403/500 is still Odysseus answering
    except (urllib.error.URLError, OSError, ValueError):
        return False


# Same reasoning as opencode_web: status() is on /health, which the UI polls,
# so the probe must not be what the request waits on. See core/portprobe.py.
_probe = CachedProbe(lambda: _serves(int(getattr(config, "ODYSSEUS_PORT", DEFAULT_PORT) or DEFAULT_PORT)))


def url(port=None):
    return f"http://{HOST}:{int(port or getattr(config, 'ODYSSEUS_PORT', DEFAULT_PORT))}"


def _autostart_path():
    return os.path.join(config.SCRIPT_DIR, "odysseus-autostart.json")


def load_autostart():
    """The persisted 'start it with locally' preference.

    A toggle that only lived in memory would be useless for this setting: the
    thing it controls happens at startup, so a value that does not survive a
    restart can never take effect. It is a small JSON file rather than a new
    section in locally.ini because writing that back through configparser
    would silently strip the user's comments out of a file they hand-edited.
    """
    try:
        with open(_autostart_path(), "r", encoding="utf-8") as fh:
            return bool(json.load(fh).get("autostart"))
    except (OSError, ValueError):
        return False


def save_autostart(enabled):
    try:
        with open(_autostart_path(), "w", encoding="utf-8") as fh:
            json.dump({"autostart": bool(enabled)}, fh)
        return True
    except OSError:
        return False


def status(port=None, directory=None):
    """Everything the UI needs to decide what to offer. Never raises."""
    try:
        port = int(port or getattr(config, "ODYSSEUS_PORT", DEFAULT_PORT))
    except (TypeError, ValueError):
        port = DEFAULT_PORT
    try:
        found = find_dir(directory or getattr(config, "ODYSSEUS_DIR", None))
        # Cached -- see the comment on _probe. start()/stop() record what they
        # learn, so a change we caused shows up at once.
        running = _probe.get()
        return {
            "installed": bool(found),
            "dir": found,
            "running": running,
            # Separated for the stop button, same discipline as opencode_web:
            # we may stop a stack we brought up, but taking down one the user
            # started themselves would be removing something we never provided.
            "managed": bool(_managed and running),
            "port": port,
            "url": url(port) if running else None,
            "docker_available": docker_available(),
            # Split out because `docker_available: false` has two causes with
            # two different fixes -- Docker is not installed, or it is
            # installed and the daemon is not running -- and a UI that cannot
            # tell them apart can only offer one wrong instruction.
            "docker_installed": bool(command()),
            "engine": engine_name(),
            "autostart": load_autostart(),
        }
    except Exception as exc:
        return {"installed": False, "dir": None, "running": False,
                "managed": False, "port": port, "url": None,
                "docker_available": False, "docker_installed": False,
                "autostart": False, "error": str(exc)}


def _tail(text, lines=8, limit=800):
    text = (text or "").strip()
    if not text:
        return ""
    return "\n".join(text.splitlines()[-lines:])[-limit:]


def start(port=None, directory=None, timeout=180):
    """Adopt a running Odysseus, or bring the compose stack up. Returns (ok, detail).

    Adoption comes first and touches Docker not at all: a user who already has
    Odysseus up gets their session, not a redundant `compose up` that races
    their own containers for the port.

    Every failure returns a sentence a person can act on. "Failed" plus a
    Docker exit code is what makes people uninstall things.
    """
    global _managed, _started_dir
    port = int(port or getattr(config, "ODYSSEUS_PORT", DEFAULT_PORT))

    if _serves(port):
        _probe.set(True)
        _started_dir = _started_dir or find_dir(
            directory or getattr(config, "ODYSSEUS_DIR", None))
        return True, "adopted"

    found = find_dir(directory or getattr(config, "ODYSSEUS_DIR", None))
    if not found:
        return False, ("No Odysseus checkout found. Clone it next to this one "
                       "(a sibling `odysseus` directory), into ~/odysseus, or "
                       "set ODYSSEUS_DIR / --odysseus-dir to where it lives.")
    if not command():
        return False, ("Odysseus runs as a Compose stack and neither `docker` "
                       "nor `podman` is on PATH. Install one of them (Docker "
                       "Desktop or Podman Desktop), then try again.")
    if not docker_available(refresh=True):
        # The two engines fail in different places and need different fixes:
        # Docker has a daemon that can be stopped, Podman on Windows/macOS runs
        # a VM that has to be started with `podman machine start`. "The backend
        # is not answering" would be true for both and useful for neither.
        if engine_name() == "podman":
            ready, detail = _start_podman_machine()
            if not ready:
                return False, detail
        else:
            return False, ("Docker is installed but its daemon is not answering. "
                           "Start Docker Desktop and wait for it to say it is "
                           "running, then try again.")

    with _lock:
        # `up -d` returns once the containers are created; on a first run it
        # also builds/pulls images, which is minutes. It gets the whole budget,
        # and the port poll below gets whatever is left.
        result = _run([command(), "compose", "up", "-d"], cwd=found,
                      timeout=max(30, int(timeout)), env=_compose_env())
    if result is None:
        _stop_managed_podman_machine_if_idle()
        return False, (f"`{engine_name()} compose up -d` did not finish within "
                       f"{int(timeout)}s in {found}. First runs build images "
                       f"and can take longer — run it in a terminal to watch.")
    if result.returncode != 0:
        detail = _tail(result.stderr) or _tail(result.stdout) or "no output"
        _stop_managed_podman_machine_if_idle()
        if "compose provider" in detail or "docker-compose" in detail:
            # Name the actual fix: this failure reads as "Odysseus is broken"
            # when it means "the engine has no compose implementation".
            return False, (
                f"{engine_name()} has no compose provider installed. Install "
                f'one with:  \"{sys.executable}\" -m pip install podman-compose  '
                f"or put Docker Compose v2 on PATH.\n\nOriginal error:\n{detail}")
        return False, f"`{engine_name()} compose up -d` failed in {found}:\n{detail}"

    # Compose reporting success means the containers exist, not that the app
    # inside them has finished booting. Poll the port; a URL handed over before
    # it answers is a link that dead-ends on the user's first click.
    deadline = time.time() + max(10, int(timeout))
    while time.time() < deadline:
        if _serves(port):
            _probe.set(True)
            _managed = True
            _started_dir = found
            return True, "started"
        time.sleep(1.0)
    _managed = True          # the containers ARE ours even though it never answered
    _started_dir = found
    return False, (f"Odysseus's containers started but nothing answered on port "
                   f"{port} within {int(timeout)}s. Check `{engine_name()} compose logs` "
                   f"in {found}, and that APP_PORT is {port}.")


def stop(port=None):
    """Stop a stack this process started. Returns True only when the port went quiet.

    **`stop`, never `down`.** `docker compose down` removes the containers and,
    with volumes, the data in them — and this container holds the user's own
    assistant: their notes, calendar, email and chat history. A convenience
    button in an inference server has no business being able to delete that.
    `stop` halts the containers and leaves everything on disk, so the next
    `up -d` is the same Odysseus the user had.

    Only ever our own stack, same discipline as opencode_web.stop(): a stack
    the user brought up by hand is theirs to take down.
    """
    global _managed, _started_dir
    with _lock:
        managed, _managed = _managed, False
        directory, _started_dir = _started_dir, None
    port = int(port or getattr(config, "ODYSSEUS_PORT", DEFAULT_PORT))
    if not managed or not directory:
        return False

    _run([command() or "docker", "compose", "stop"], cwd=directory, timeout=120)
    _docker_cache_clear()

    # The port is the only honest test: compose can report success for a
    # project name that was never the thing holding the port.
    deadline = time.time() + 30
    while time.time() < deadline:
        if not _serves(port, timeout=0.6):
            _probe.set(False)
            _stop_managed_podman_machine_if_idle()
            return True
        time.sleep(0.5)
    return False


def _docker_cache_clear():
    global _docker_cache
    _docker_cache = (0.0, False)
