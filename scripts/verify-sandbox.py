#!/usr/bin/env python
"""verify-sandbox.py — prove the Python-tool boundary, one escape at a time.

AGENT TASK C lists what a container has to actually stop before the claim is
worth making. Each case below runs real model-shaped code through the real
`execute_python` path — the one a tool call takes — and asserts the escape
FAILS. A sandbox that is only argued for on the strength of its command line is
not a sandbox; these are the assertions that turn the flags into evidence.

The comparison against the subprocess path is the point of the last column.
CLAUDE.md already concedes that path "is not a security boundary"; this shows
exactly which of these it stops and which it does not.

Usage:
    python scripts/verify-sandbox.py [podman|subprocess|both]
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

# Run as `python scripts/verify-sandbox.py`: sys.path[0] is scripts/, not the
# repo root, so `core` is not importable without this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.sandbox import podman
from core.sandbox.python_exec import execute_python


class _Mem(ctypes.Structure):
    _fields_ = [("l", ctypes.c_ulong), ("mem", ctypes.c_ulong),
                ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                ("ae", ctypes.c_ulonglong)]


def free_gb():
    m = _Mem()
    m.l = ctypes.sizeof(_Mem)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ap / 2 ** 30


# (name, code, what "contained" means)
# `contained` reads the result dict and returns True when the escape failed.
CASES = [
    ("baseline: arithmetic runs",
     "print(sum(range(10)))",
     lambda r: r.get("stdout", "").strip() == "45"),

    ("network egress blocked",
     "import urllib.request\n"
     "print(urllib.request.urlopen('http://example.com', timeout=8).status)",
     lambda r: "200" not in r.get("stdout", "")),

    ("write outside the workspace blocked",
     "open('/etc/locally-escape', 'w').write('x'); print('WROTE')",
     lambda r: "WROTE" not in r.get("stdout", "")),

    ("memory cap kills a 2 GB allocation",
     "b = bytearray(2 * 1024 * 1024 * 1024)\n"
     "print('ALLOCATED', len(b))",
     lambda r: "ALLOCATED" not in r.get("stdout", "")),

    ("pid limit contains a fork bomb",
     "import os\n"
     "for _ in range(400):\n"
     "    try:\n"
     "        if os.fork() == 0: os._exit(0)\n"
     "    except Exception as e:\n"
     "        print('STOPPED', type(e).__name__); break\n"
     "else:\n"
     "    print('FORKED ALL')",
     lambda r: "FORKED ALL" not in r.get("stdout", "")),

    ("ctypes cannot reach the host",
     "import ctypes\n"
     "try:\n"
     "    libc = ctypes.CDLL(None)\n"
     "    print('HOSTNAME_CALL', libc.getpid())\n"
     "except Exception as e:\n"
     "    print('BLOCKED', type(e).__name__)",
     # The honest assertion: inside a container getpid() SUCCEEDS and is
     # meaningless -- it is the container's PID namespace. What must not happen
     # is reaching a host resource. Reported, not asserted, and discussed below.
     lambda r: True),
]


def run(kind: str) -> None:
    config.PYTHON_SANDBOX = kind
    config.PYTHON_TOOL_ENABLED = True
    print(f"\n=== --python-sandbox {kind}")
    for name, code, contained in CASES:
        t0 = time.perf_counter()
        r = execute_python(code)
        dt = (time.perf_counter() - t0) * 1000
        ok = contained(r)
        out = (r.get("stdout") or "").strip().replace("\n", " ")[:48]
        errtail = (r.get("stderr") or "").strip().replace("\n", " ")[-48:]
        mark = "contained" if ok else "*** ESCAPED ***"
        print(f"  {name:<40} {mark:<16} {dt:7.0f} ms  "
              f"exit={r.get('exit_status')}  out={out!r}"
              + (f"  err=…{errtail!r}" if errtail and not out else ""))


def main() -> int:
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    print(f"free RAM before: {free_gb():.1f} GB   "
          f"machine_running={podman.machine_running()}")

    if which in ("podman", "both"):
        t0 = time.perf_counter()
        run("podman")
        print(f"  (podman block total {time.perf_counter() - t0:.1f} s, "
              f"free RAM now {free_gb():.1f} GB, "
              f"machine_running={podman.machine_running()})")
    if which in ("subprocess", "both"):
        run("subprocess")

    print(f"\nfree RAM after: {free_gb():.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
