"""Live verification for the Podman Python boundary.

Run from the project root with::

    python -m core.sandbox.verify_podman

This starts and stops ``podman-machine-default`` but never loads a generative
model.  It prints JSON so the exact results can be pasted into the handoff.
"""

import json
import time

from core import config
from core.sandbox import podman
from core.sandbox.python_exec import execute_python


CHECKS = {
    "benign": "print(sum(range(10)))",
    "network_none": (
        "import urllib.request\n"
        "urllib.request.urlopen('http://example.com', timeout=2)"
    ),
    "read_only_root": "open('/outside-workspace.txt', 'w').write('no')",
    "memory_512m": "x = bytearray(2 * 1024**3); print(len(x))",
    "pids_64": (
        "import os, time\n"
        "children = []\n"
        "for _ in range(256):\n"
        "    try:\n"
        "        pid = os.fork()\n"
        "    except OSError as exc:\n"
        "        print('fork refused after', len(children), 'children:', exc)\n"
        "        break\n"
        "    if pid == 0:\n"
        "        time.sleep(1)\n"
        "        os._exit(0)\n"
        "    children.append(pid)\n"
        "for pid in children:\n"
        "    os.waitpid(pid, 0)\n"
        "print('reaped', len(children))"
    ),
    "ctypes_host": (
        "import ctypes, os\n"
        "print('container_os', os.name)\n"
        "print('windows_host_visible', os.path.exists('/mnt/c/Windows'))\n"
        "ctypes.CDLL('kernel32.dll')"
    ),
}


def _health_timings(samples=20):
    """Measure the real Flask route in-process, without starting a server."""
    from locally import app

    client = app.test_client()
    elapsed = []
    statuses = []
    for _ in range(samples):
        started = time.perf_counter()
        response = client.get("/health")
        elapsed.append((time.perf_counter() - started) * 1000)
        statuses.append(response.status_code)
    return {
        "samples": samples,
        "min_ms": round(min(elapsed), 3),
        "median_ms": round(sorted(elapsed)[len(elapsed) // 2], 3),
        "max_ms": round(max(elapsed), 3),
        "status_codes": sorted(set(statuses)),
    }


def _stop_if_idle(exe):
    stopped = podman._stop_machine_if_idle(exe)
    if stopped is False:
        raise RuntimeError(
            "Podman has a running container or could not be queried; refusing "
            "to stop a machine the verifier cannot prove is idle")
    return stopped


def main():
    config.PYTHON_SANDBOX = "podman"
    # The controlled PID test waits one second for children to exit.  Keep the
    # product cap small while leaving enough margin for a busy CI host.
    config.PYTHON_TOOL_TIMEOUT = 8.0
    report = {
        "podman_cli": podman.podman_command(),
        "podman_available": podman.podman_available(),
        "image": podman.IMAGE,
        "machine": podman.MACHINE,
    }
    exe = podman.podman_command()
    if not exe:
        raise SystemExit(json.dumps(report, indent=2))

    was_running = podman.refresh_machine_running()
    report["machine_running_initially"] = was_running
    if was_running:
        _stop_if_idle(exe)
    podman._machine_probe.set(False)
    report["health_machine_stopped"] = _health_timings()

    # execute_python owns this cold start and must stop it again itself.
    cold = execute_python(CHECKS["benign"])
    report["cold_start"] = cold
    if cold.get("stdout") != "45\n" or cold.get("exit_status") != 0:
        raise RuntimeError("the cold benign calculation failed")
    if not cold.get("machine_started") or cold.get("machine_stopped") is not True:
        raise RuntimeError("the cold path did not start and stop its machine")

    ok, detail = podman._start_machine(exe)
    if not ok:
        raise RuntimeError("could not start Podman for warm checks: " + detail)
    try:
        report["health_machine_running"] = _health_timings()
        report["warm_start"] = execute_python(CHECKS["benign"])
        report["checks"] = {
            name: execute_python(code)
            for name, code in CHECKS.items() if name != "benign"
        }
    finally:
        _stop_if_idle(exe)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
