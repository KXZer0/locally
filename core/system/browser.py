"""Local browser handoff and coding-agent launcher."""

import ctypes
import socket
import sys
import os
import shutil
import subprocess
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from datetime import datetime

from flask import jsonify, request

from core import config
from core import runtime
from core.slots.capability import _tools_supported
from core.slots.select import _slot_serviceable
from core.errors import openai_error

BROWSER_EXES = ("zen.exe", "firefox.exe", "librewolf.exe", "chrome.exe",
                "msedge.exe", "brave.exe", "opera.exe", "vivaldi.exe")


def _allow_foreground_handoff():
    """Let the process we are about to launch take the foreground.

    Windows only grants SetForegroundWindow to a process that already owns the
    foreground or has just received input. This server has neither -- the click
    happened inside the browser -- so without this the browser opens the tab
    behind the locally window and only flashes its taskbar button.

    ASFW_ANY (-1) hands that right to whichever process acts next, which is the
    documented way to do a launch-and-focus handoff.
    """
    try:
        ctypes.windll.user32.AllowSetForegroundWindow(-1)
    except Exception:
        pass


def _raise_browser_window(timeout=3.0):
    """Bring the default browser's window to the front after opening a URL.

    ShellExecute returns as soon as the browser is told about the URL, not when
    it has drawn a window, so this polls. Matches on the process executable
    rather than the window title, because a title is whatever page happens to
    be loading and can be anything.

    Best-effort by design: if the window cannot be found or Windows refuses the
    activation, the tab is still open and the user can alt-tab. Failing loudly
    here would turn a cosmetic miss into a broken feature.
    """
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi

    def exe_of(hwnd):
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        # QUERY_LIMITED_INFORMATION: enough for the image name, and unlike
        # QUERY_INFORMATION it is granted for processes at other integrity
        # levels, which a browser often is.
        h = kernel32.OpenProcess(0x1000, False, pid.value)
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(260)
            psapi.GetModuleFileNameExW(h, None, buf, 260)
            return os.path.basename(buf.value).lower()
        finally:
            kernel32.CloseHandle(h)

    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        if user32.GetWindowTextLengthW(hwnd) == 0:
            return True          # toolwindows and hidden hosts
        if exe_of(hwnd) not in BROWSER_EXES:
            return True
        # locally's own PWA window IS a browser window -- it runs inside
        # msedge.exe -- so without this the server can raise itself and the
        # link still appears not to have opened. Measured: enumeration returned
        # msedge/"locally" ahead of zen/"<page title>".
        title = ctypes.create_unicode_buffer(160)
        user32.GetWindowTextW(hwnd, title, 160)
        if title.value.strip() == "locally":
            return True
        found.append(hwnd)
        return False             # first non-self browser window is enough

    deadline = time.time() + timeout
    while time.time() < deadline:
        found.clear()
        try:
            user32.EnumWindows(visit, 0)
        except Exception:
            return False
        if found:
            hwnd = found[0]
            try:
                if user32.IsIconic(hwnd):
                    user32.ShowWindow(hwnd, 9)       # SW_RESTORE
                return bool(user32.SetForegroundWindow(hwnd))
            except Exception:
                return False
        time.sleep(0.15)
    return False


def _open_default_browser(url):
    """Open a URL through the user's default browser and raise its window."""
    raised = False
    if sys.platform == "win32":
        # os.startfile uses the shell association, which IS the default
        # browser. webbrowser.open on Windows can resolve to whatever is
        # first on PATH instead.
        _allow_foreground_handoff()
        os.startfile(url)                                      # noqa: S606
        # Opening the tab is not the whole job: without this the browser
        # comes up BEHIND the locally window and only flashes its taskbar
        # button, so the link appears not to have worked at all.
        raised = _raise_browser_window()
    elif sys.platform == "darwin":
        subprocess.Popen(["open", url])
        raised = True                                          # `open` activates the app itself
    else:
        subprocess.Popen(["xdg-open", url])
    return raised


def open_external():
    """Open a URL in the user's DEFAULT browser.

    The chat UI runs inside an Edge PWA window, so a plain target="_blank" opens
    the link in Edge no matter what the user's default browser is. The page
    cannot escape its own host, so the server has to do it.

    Same two rules as /v1/code/launch, because this also acts on the machine:

    1. **Localhost only.** The server binds 0.0.0.0 on purpose (phones use the
       chat UI), so without this every device on the network could pop windows
       on this one. `request.remote_addr` is the socket peer, not a header.
    2. **http/https only, and never interpolated into a shell.** A URL is
       caller-supplied data, and `file://`, `smb://` or a shell metacharacter
       reaching a command line is the whole attack. The scheme is checked
       against an allowlist and the URL is passed as an argv element.
    """
    if request.remote_addr not in ("127.0.0.1", "::1", "localhost"):
        return openai_error("This endpoint is local-only (it opens a window on "
                            "the server's machine).", "invalid_request_error", 403)

    url = str((request.get_json(silent=True) or {}).get("url") or "").strip()
    try:
        parts = urllib.parse.urlsplit(url)
    except Exception:
        parts = None
    if not parts or parts.scheme not in ("http", "https") or not parts.netloc:
        return openai_error("Only http:// and https:// URLs can be opened.")

    try:
        raised = _open_default_browser(url)
    except Exception as e:
        return openai_error(f"Could not open the link: {e}", "server_error", 500)
    return jsonify({"opened": url, "raised": raised})


def launch_claude_code():
    """Open Claude Code in a terminal, pointed at this server.

    The web UI can't start a process, so the button posts here. Two rules make
    that safe enough to ship:

    1. **Localhost only.** The server binds 0.0.0.0 (that's the point — phones
       and other machines use the chat UI), so without this check every device
       on the network could spawn processes on this one. `request.remote_addr`
       is the socket peer, not a header, so it can't be spoofed by a client.
    2. **No caller-supplied command.** The only input is a directory, and it
       is passed as an argv element to a fixed script — never interpolated
       into a shell string. The worst a caller can do is open a terminal in a
       directory they name.

    The terminal runs scripts/claude-local.ps1, the same script a user would
    run by hand, so there is one code path to debug. If no terminal can be
    opened we still return the command, so the UI can offer it for copy-paste
    rather than dead-ending.
    """
    if request.remote_addr not in ("127.0.0.1", "::1", "localhost"):
        return openai_error("This endpoint is local-only (it starts a process "
                            "on the server's machine).", "invalid_request_error",
                            403)

    body = request.get_json(silent=True) or {}
    cwd = os.path.expanduser(str(body.get("cwd") or config.SCRIPT_DIR))
    if not os.path.isdir(cwd):
        return openai_error(f"No such directory: {cwd}")

    script = os.path.join(config.SCRIPT_DIR, "scripts", "claude-local.ps1")
    if not os.path.isfile(script):
        return openai_error(f"Launcher script missing: {script}",
                            "server_error", 500)

    base_url = f"http://127.0.0.1:{config.SERVER_PORT}"
    model, warning = "", None
    for slot in (runtime.primary, runtime.secondary):
        if _slot_serviceable(slot):
            model = slot.model_name
            # Measured on this hardware: Claude Code sends a ~35k-token system
            # prompt and asks for up to 32k of output, and the KV pool has to
            # hold both at once. A 32k pool doesn't just truncate — it dies
            # with CL_OUT_OF_RESOURCES eight minutes into the first turn, which
            # is a miserable way to find out. Say it before launching.
            if slot.context_tokens and slot.context_tokens < config.CODE_MIN_CONTEXT:
                warning = (f"This model has only {slot.context_tokens // 1000}k "
                           f"of context. Claude Code needs ~67k (a ~35k-token "
                           f"system prompt plus room to answer) and will fail "
                           f"mid-turn below that — restart with "
                           f"--context-tokens 90000.")
            if not _tools_supported(slot):
                warning = (f"{slot.model_name} is on the {slot.device_name}, "
                           f"which has no agent path — Claude Code needs tool "
                           f"calling, so load the model on GPU or CPU.")
            break

    inner = ["pwsh", "-NoExit", "-File", script,
             "-BaseUrl", base_url, "-ProjectDir", cwd]
    if model:
        inner += ["-Model", model]
    # Shown to the user verbatim when we can't open a terminal for them.
    manual = (f'pwsh -File "{script}" -BaseUrl {base_url} '
              f'-ProjectDir "{cwd}"' + (f' -Model "{model}"' if model else ""))

    # Windows Terminal opens a real tab; without it, fall back to a bare
    # console. Non-Windows gets the command to run — guessing at someone's
    # terminal emulator is how you end up launching the wrong one.
    candidates = []
    if os.name == "nt":
        candidates.append(["wt.exe", "-d", cwd] + inner)
        candidates.append(["cmd.exe", "/c", "start", ""] + inner)

    import subprocess
    for argv in candidates:
        try:
            subprocess.Popen(argv, cwd=cwd, close_fds=True)
            print(f"{datetime.now():%H:%M:%S} <> Claude Code launched in {cwd} "
                  f"(model {model or 'unset'})", flush=True)
            return jsonify({"status": "ok", "cwd": cwd, "model": model,
                            "base_url": base_url, "command": manual,
                            "warning": warning})
        except Exception:
            continue

    return jsonify({"status": "manual", "cwd": cwd, "model": model,
                    "base_url": base_url, "command": manual, "warning": warning,
                    "reason": "Could not open a terminal window from here."})


def _port_is_listening(port, timeout=0.25):
    """Return whether localhost accepts a TCP connection on *port*."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False
