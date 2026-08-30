"""Running one short calculation in a killed child process.

This is defence-in-depth against a small model's arithmetic mistakes and
against prompt-injected pages -- it is NOT a hostile-code sandbox on Windows.
ctypes and native escape paths are outside what a local convenience feature
can enforce, and the import block says so rather than implying otherwise."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from core import config


# Everything the calculation tool may import. This is an ALLOWLIST, and it is
# enforced by pre-importing every name below and then denying the `import`
# audit event outright -- a blocklist was tried first and measured to be
# worthless: `import socket` was refused while `os.system("curl ...")` returned
# rc=0, so the feature looked network-isolated and was not. See TODONT.md.
_PYTHON_ALLOWED_IMPORTS = (
    "math", "cmath", "decimal", "fractions", "statistics", "itertools",
    "functools", "operator", "collections", "heapq", "bisect", "random",
    "re", "json", "string", "textwrap", "datetime", "hashlib", "unicodedata",
)


# sympy is the factoring workhorse this tool exists for, but it costs ~500 ms to
# import against a 10 s budget, so it is loaded only when the code mentions it.
_PYTHON_HEAVY_IMPORTS = ("sympy",)


_PYTHON_TRUNCATION_MARKER = "\n… output truncated by locally …"


# Audit events the sandbox refuses. `import` and `open` are handled separately
# (both are denied outright once the allowlist is resident); these are the
# prefixes that must never fire regardless.
_PYTHON_DENIED_EVENTS = (
    "os.system", "os.exec", "os.spawn", "os.fork", "os.posix_spawn",
    "os.startfile", "os.remove", "os.rename", "os.rmdir", "os.mkdir",
    "os.chmod", "os.link", "os.symlink", "os.truncate", "os.putenv",
    "os.unsetenv", "subprocess.", "socket.", "ctypes.", "winreg.",
    "shutil.", "urllib.", "ftplib.", "http.", "webbrowser.", "pty.",
    "msvcrt.", "glob.", "fcntl.", "mmap.", "resource.",
    # Reads, not writes -- but the tool tells the model it has no filesystem
    # access, and os.listdir("C:/") made that a lie.
    "os.listdir", "os.scandir", "os.stat", "os.walk", "os.chdir",
)


def _python_child_source(code, stdout_cap, stderr_cap, status_path):
    """Wrap user code with capped streams and a deny-by-default audit hook.

    The order here is the design. Every allowlisted module is imported *first*,
    while importing is still possible; only then does the audit hook go up, and
    from that point `import` is refused unconditionally. A module already in
    `sys.modules` is returned from cache without raising the `import` audit
    event, so allowed code keeps working and nothing new can be loaded -- no
    name matching, no way to spell a module so the check misses it.

    Honest scope: this is a guardrail against a small model emitting something
    destructive, or a prompt-injected page steering it. It is not a jail.
    Windows offers no seccomp/bubblewrap equivalent here, and real isolation
    needs a Job Object or AppContainer (TODONT.md). What it does close are the
    six holes that were measured open under the previous blocklist: filesystem
    read, filesystem write, ctypes, winreg, os.system, and network via a
    shelled-out curl.
    """
    wanted = list(_PYTHON_ALLOWED_IMPORTS)
    # Only pay for the heavy ones when the code actually reaches for them.
    wanted += [m for m in _PYTHON_HEAVY_IMPORTS if m in code]
    # JSON quoting keeps paths and caps data, rather than executable source, and
    # makes the wrapper safe for paths containing spaces or apostrophes.
    return f'''\
import sys


# Opened BEFORE the hook goes up: afterwards `open` is refused, and this handle
# is the only way the child can still report whether it hit an output cap.
_status = open({json.dumps(status_path)}, "w", encoding="utf-8")
_limits = {{"stdout": False, "stderr": False}}

class _Limited:
    def __init__(self, raw, cap, name):
        self.raw, self.cap, self.name, self.used = raw, cap, name, 0
    def write(self, value):
        data = str(value).encode("utf-8", "replace")
        room = max(0, self.cap - self.used)
        if len(data) > room:
            data = data[:room]
            _limits[self.name] = True
        if data:
            self.raw.write(data)
            self.raw.flush()
            self.used += len(data)
        return len(value)
    def flush(self):
        self.raw.flush()
    def isatty(self):
        return False

sys.stdout = _Limited(sys.__stdout__.buffer, {int(stdout_cap)}, "stdout")
sys.stderr = _Limited(sys.__stderr__.buffer, {int(stderr_cap)}, "stderr")

# Resident before the door closes. Anything not named here cannot be imported
# at all once the hook is installed, which is what makes this an allowlist.
for _m in {wanted!r}:
    try:
        __import__(_m)
    except Exception:
        pass

_DENIED = {_PYTHON_DENIED_EVENTS!r}

def _guard(event, args):
    if event == "import" or event == "open" or event.startswith(_DENIED):
        raise PermissionError(
            event + " is not available in locally's calculation sandbox")

sys.addaudithook(_guard)      # audit hooks cannot be removed once added

try:
    exec(compile({code!r}, "<locally-python>", "exec"), {{"__name__": "__main__"}})
finally:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    import json as _json
    _status.write(_json.dumps(_limits))
    _status.flush()
'''


def execute_python(code):
    """Run a calculation in a disposable, isolated child process.

    The child uses the same interpreter as the server with ``-I``; its current
    directory is a fresh temporary directory and it is killed, not merely
    abandoned, at the wall-clock deadline. This is a local convenience tool,
    not a hostile-code sandbox: the import block is useful defence-in-depth,
    but Windows has no portable in-process network jail here and ctypes/native
    escape paths remain outside what we can promise.
    """
    if not isinstance(code, str) or not code.strip():
        return {"code": code if isinstance(code, str) else "", "stdout": "",
                "stderr": "code is required", "exit_status": None,
                "elapsed_ms": 0, "truncated": False, "timed_out": False}
    code_bytes = code.encode("utf-8", "replace")
    if len(code_bytes) > config.PYTHON_TOOL_MAX_CODE_BYTES:
        return {"code": code, "stdout": "", "stderr": (
                    f"code exceeds the {config.PYTHON_TOOL_MAX_CODE_BYTES} byte limit"),
                "exit_status": None, "elapsed_ms": 0, "truncated": False,
                "timed_out": False}

    root = tempfile.mkdtemp(prefix="locally-python-")
    script_path = os.path.join(root, "run.py")
    status_path = os.path.join(root, "status.json")
    stdout_path = os.path.join(root, "stdout.bin")
    stderr_path = os.path.join(root, "stderr.bin")
    started = time.perf_counter()
    timed_out = False
    proc = None
    try:
        with open(script_path, "w", encoding="utf-8", newline="") as f:
            f.write(_python_child_source(code, config.PYTHON_TOOL_OUTPUT_BYTES,
                                         config.PYTHON_TOOL_OUTPUT_BYTES, status_path))
        with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
            # A scrubbed environment, because os.environ is filled in before
            # any audit hook can exist and so cannot be blocked from inside the
            # child. Keep only what Windows needs to start an interpreter.
            env = {k: v for k, v in os.environ.items()
                   if k in ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP",
                            "PATHEXT", "COMSPEC", "NUMBER_OF_PROCESSORS",
                            "PROCESSOR_ARCHITECTURE", "LANG", "LC_ALL")}
            proc = subprocess.Popen(
                [sys.executable, "-I", script_path], cwd=root, env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            )
            try:
                proc.wait(timeout=config.PYTHON_TOOL_TIMEOUT)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                proc.wait()

        def read_capped(path):
            with open(path, "rb") as f:
                raw = f.read(config.PYTHON_TOOL_OUTPUT_BYTES + 1)
            capped = len(raw) > config.PYTHON_TOOL_OUTPUT_BYTES
            raw = raw[:config.PYTHON_TOOL_OUTPUT_BYTES]
            return raw.decode("utf-8", "replace"), capped

        stdout, out_capped = read_capped(stdout_path)
        stderr, err_capped = read_capped(stderr_path)
        flags = {}
        try:
            with open(status_path, encoding="utf-8") as f:
                flags = json.load(f)
        except Exception:
            pass
        truncated = bool(out_capped or err_capped or flags.get("stdout")
                        or flags.get("stderr"))
        if out_capped or flags.get("stdout"):
            stdout += _PYTHON_TRUNCATION_MARKER
        if err_capped or flags.get("stderr"):
            stderr += _PYTHON_TRUNCATION_MARKER
        if timed_out:
            stderr = (stderr + "\nexecution timed out after "
                      f"{config.PYTHON_TOOL_TIMEOUT:g}s").strip()
        return {
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
            "exit_status": None if timed_out else proc.returncode,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "truncated": truncated,
            "timed_out": timed_out,
        }
    except Exception as e:
        return {"code": code, "stdout": "", "stderr": str(e),
                "exit_status": None, "elapsed_ms": round(
                    (time.perf_counter() - started) * 1000),
                "truncated": False, "timed_out": timed_out}
    finally:
        # The directory is private to this request and never the project root.
        shutil.rmtree(root, ignore_errors=True)
