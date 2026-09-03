"""Podman lifecycle and command construction for the Python tool sandbox.

Health readers never run Podman synchronously.  A calculation that actually
needs the container pays the machine probe/start cost, and only that path may
stop a machine this process started.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import uuid

from core import config
from core.portprobe import CachedProbe


IMAGE = "docker.io/library/python:3.12-alpine"
MACHINE = "podman-machine-default"
CONTAINER_WORKSPACE = "/workspace"

_MACHINE_TTL = 5.0
_lifecycle_lock = threading.Lock()


def _well_known_paths():
    """Install locations used by Podman Desktop and the standalone MSI."""
    return (
        os.path.join(os.environ.get("ProgramFiles", ""),
                     "RedHat", "Podman", "podman.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""),
                     "Podman", "podman.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""),
                     "Programs", "Podman", "podman.exe"),
        "/usr/bin/podman",
        "/usr/local/bin/podman",
        "/opt/homebrew/bin/podman",
    )


def podman_command():
    """Return a runnable-looking Podman CLI path without starting its VM."""
    found = shutil.which("podman")
    if found:
        return found
    for candidate in _well_known_paths():
        try:
            if candidate and os.path.isfile(candidate):
                return candidate
        except OSError:
            continue
    return None


def podman_available():
    """Whether the CLI is installed; deliberately independent of VM state."""
    return bool(podman_command())


def _creation_flags():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _run_control(argv, timeout):
    """Run a short Podman control command with bounded, decoded output."""
    try:
        return subprocess.run(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout, check=False,
            text=True, encoding="utf-8", errors="replace",
            creationflags=_creation_flags())
    except (OSError, subprocess.TimeoutExpired):
        return None


def _machine_is_required():
    return os.name == "nt" or sys.platform == "darwin"


def _probe_machine_running():
    if not podman_available():
        return False
    if not _machine_is_required():
        return True
    result = _run_control(
        [podman_command(), "machine", "inspect", MACHINE], timeout=10)
    if not result or result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout)
        item = payload[0] if isinstance(payload, list) else payload
        state = str((item or {}).get("State", "")).lower()
        return state == "running"
    except (TypeError, ValueError, AttributeError):
        return False


_machine_probe = CachedProbe(_probe_machine_running, ttl=_MACHINE_TTL)


def machine_running():
    """Return cached VM state immediately and refresh it in the background."""
    return _machine_probe.get()


def refresh_machine_running():
    """Pay for a current answer only on the execution/lifecycle path."""
    return _machine_probe.refresh_now()


def sandbox_status():
    """Facts used by /health.  No call here waits for Podman or its VM."""
    requested = str(getattr(config, "PYTHON_SANDBOX", "auto") or "auto")
    available = podman_available()
    effective = requested
    if requested == "auto":
        effective = "podman" if available else "subprocess"
    return {
        "sandbox_requested": requested,
        "sandbox": effective,
        "podman_available": available,
        "machine_running": machine_running() if available else False,
        "image": IMAGE if effective == "podman" else None,
    }


def _start_machine(exe):
    if not _machine_is_required():
        _machine_probe.set(True)
        return True, ""
    result = _run_control([exe, "machine", "start", MACHINE], timeout=120)
    if result and result.returncode == 0:
        _machine_probe.set(True)
        return True, ""
    detail = ((result.stderr or result.stdout).strip() if result else
              "the start command failed or timed out")
    return False, detail[-1200:]


def _stop_machine_if_idle(exe):
    """Stop only a VM we started, and never over an unrelated container."""
    if not _machine_is_required():
        return None
    running = _run_control([exe, "ps", "-q"], timeout=20)
    if not running or running.returncode != 0 or running.stdout.strip():
        return False
    stopped = _run_control(
        [exe, "machine", "stop", MACHINE], timeout=90)
    ok = bool(stopped and stopped.returncode == 0)
    if ok:
        _machine_probe.set(False)
    return ok


def _kill_process(proc):
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill.exe", "/PID", str(proc.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=15,
                creationflags=_creation_flags(), check=False)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _container_argv(exe, workspace, relative_dir, container_name):
    """The auditable hardening contract for one calculation container."""
    mount = f"{workspace}:{CONTAINER_WORKSPACE}:rw"
    # Normalize explicitly as well as through os.sep so a Windows-style path
    # stays valid when command construction is unit-tested on another host.
    relative_dir = relative_dir.replace("\\", "/").replace(os.sep, "/")
    workdir = f"{CONTAINER_WORKSPACE}/{relative_dir}"
    return [
        exe, "run", "--rm", "--network=none", "--read-only",
        "--memory=512m", "--pids-limit=64", "--cap-drop=ALL",
        "--security-opt", "no-new-privileges", "--name", container_name,
        "--volume", mount, "--workdir", workdir,
        IMAGE, "python", "-I", "-B", "run.py",
    ]


def run_python(root, stdout_path, stderr_path, timeout):
    """Run run.py under *root* and return execution/lifecycle facts."""
    exe = podman_command()
    if not exe:
        return {"error": "Podman is not installed or is not on PATH",
                "returncode": None, "timed_out": False,
                "machine_started": False, "machine_stopped": None}

    workspace = os.path.realpath(config.SCRIPT_DIR)
    sandbox_root = os.path.realpath(root)
    try:
        inside = os.path.commonpath((workspace, sandbox_root)) == workspace
    except ValueError:
        inside = False
    if not inside or sandbox_root == workspace:
        return {"error": "sandbox directory is outside the workspace",
                "returncode": None, "timed_out": False,
                "machine_started": False, "machine_stopped": None}

    relative_dir = os.path.relpath(sandbox_root, workspace)
    container_name = "locally-python-" + uuid.uuid4().hex[:12]
    started_here = False
    stopped = None
    timed_out = False
    proc = None
    outcome = None

    # Serialising this short path prevents one request from stopping the VM
    # while a second calculation is still using it.
    with _lifecycle_lock:
        if not refresh_machine_running():
            ok, detail = _start_machine(exe)
            if not ok:
                return {"error": "Podman's machine did not start: " + detail,
                        "returncode": None, "timed_out": False,
                        "machine_started": False, "machine_stopped": None}
            started_here = True
        try:
            with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
                proc = subprocess.Popen(
                    _container_argv(exe, workspace, relative_dir, container_name),
                    cwd=workspace, stdin=subprocess.DEVNULL,
                    stdout=out, stderr=err,
                    creationflags=_creation_flags(),
                    start_new_session=(os.name != "nt"))
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _kill_process(proc)
                    _run_control([exe, "rm", "-f", container_name], timeout=20)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            outcome = {"error": None,
                       "returncode": None if timed_out else proc.returncode,
                       "timed_out": timed_out,
                       "machine_started": started_here,
                       "machine_stopped": None}
        except OSError as exc:
            outcome = {"error": str(exc), "returncode": None,
                       "timed_out": timed_out,
                       "machine_started": started_here,
                       "machine_stopped": None}
        finally:
            if started_here:
                stopped = _stop_machine_if_idle(exe)
                if stopped is False:
                    _machine_probe.set(True)
        outcome["machine_stopped"] = stopped
        return outcome
