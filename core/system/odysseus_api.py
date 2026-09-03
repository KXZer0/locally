"""Odysseus and OpenCode launch controls."""

import os
import json
import shutil
import subprocess

from flask import jsonify, request

from core import config, odysseus, runtime
from core.errors import openai_error
from core.system.coding import (_opencode_command, _opencode_local_provider,
                                _opencode_overlay, _opencode_status_data)
from core.system.access import _request_is_local

def odysseus_status():
    """Is the assistant installed, up, and can Docker bring it up?

    Readable from anywhere: it starts nothing and names no path a caller did
    not already have. The two routes below that DO start a process are
    localhost-only.
    """
    return jsonify(odysseus.status())


def odysseus_start():
    """Start (or adopt) the Odysseus compose stack.

    Localhost-only, exactly as /v1/opencode/web: this starts a process on the
    server's machine, the argv is fixed, and nothing from the request body is
    interpolated into it.

    Synchronous on purpose here even though the startup hook is threaded — a
    user who pressed a button is waiting for an answer, and the answer has to
    be "it is serving" rather than "compose returned 0".
    """
    if not _request_is_local():
        return openai_error("This endpoint is local-only (it starts a process "
                            "on the server's machine).", "invalid_request_error", 403)
    ok, detail = odysseus.start(timeout=config.ODYSSEUS_START_TIMEOUT)
    state = odysseus.status()
    if not ok:
        return jsonify({**state, "detail": detail,
                        "error": {"message": detail, "type": "server_error"}}), 503
    return jsonify({**state, "detail": detail})


def odysseus_set_autostart():
    """Persist 'start Odysseus with locally'.

    Local-only because it writes a file on the server's machine and changes
    what that machine does on its next start.

    This deliberately does NOT start or stop anything. The row it belongs to
    says "when locally starts", and a toggle that also acted immediately would
    be two controls wearing one switch -- the Start button next to it is the
    one that acts now.
    """
    if not _request_is_local():
        return openai_error("This endpoint is local-only.",
                            "invalid_request_error", 403)
    body = request.get_json(silent=True) or {}
    enabled = bool(body.get("enabled"))
    if not odysseus.save_autostart(enabled):
        return openai_error("Could not save that setting (the folder may be "
                            "read-only).", "server_error", 500)
    config.ODYSSEUS_AUTOSTART = enabled
    return jsonify(odysseus.status())


def odysseus_stop():
    """Stop a stack locally started. Never one the user brought up themselves.

    `docker compose stop`, never `down` — see core/odysseus.stop().
    """
    if not _request_is_local():
        return openai_error("This endpoint is local-only.",
                            "invalid_request_error", 403)
    stopped = odysseus.stop()
    return jsonify({**odysseus.status(), "stopped": stopped})


def launch_opencode():
    """Open the OpenCode TUI with locally added as one provider.

    Localhost-only because this creates an interactive process on the server's
    machine. The browser may choose only an existing directory; the command is
    fixed, and the directory travels through an environment variable rather
    than being interpolated into PowerShell.
    """
    if not _request_is_local():
        return openai_error("This endpoint is local-only (it opens a terminal "
                            "on the server's machine).", "invalid_request_error", 403)
    command, version = _opencode_command()
    if not command:
        return openai_error("OpenCode is not installed. Install the opencode-ai "
                            "package, then try again.", "server_error", 503)

    body = request.get_json(silent=True) or {}
    workspace = os.path.abspath(os.path.expanduser(
        str(body.get("workspace") or config.SCRIPT_DIR).strip()))
    if not os.path.isdir(workspace):
        return openai_error(f"Workspace directory not found: {workspace}")

    env = dict(os.environ)
    provider, model_id = _opencode_local_provider()
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(
        _opencode_overlay(), separators=(",", ":"))
    env["LOCALLY_OPENCODE_WORKSPACE"] = workspace

    try:
        if os.name == "nt":
            powershell = shutil.which("pwsh") or shutil.which("powershell")
            if not powershell:
                raise RuntimeError("PowerShell was not found")
            # The command string is constant. The user-selected path is read
            # from an environment variable, so &, |, quotes or spaces in a
            # legitimate folder name never become shell syntax.
            argv = [powershell, "-NoExit", "-NoProfile", "-Command",
                    "$host.UI.RawUI.WindowTitle='OpenCode · locally'; "
                    "& opencode --dir $env:LOCALLY_OPENCODE_WORKSPACE"]
            process = subprocess.Popen(
                argv, cwd=workspace, env=env,
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        else:
            terminal = (shutil.which("x-terminal-emulator") or
                        shutil.which("gnome-terminal") or shutil.which("konsole"))
            if not terminal:
                return openai_error("No supported terminal launcher was found. "
                                    f"Run `opencode --dir {workspace}` manually.",
                                    "server_error", 501)
            name = os.path.basename(terminal).lower()
            if "gnome-terminal" in name:
                argv = [terminal, "--", command, "--dir", workspace]
            else:
                argv = [terminal, "-e", command, "--dir", workspace]
            process = subprocess.Popen(argv, cwd=workspace, env=env)
    except Exception as exc:
        return openai_error(f"Could not open OpenCode: {exc}", "server_error", 500)

    state = _opencode_status_data()
    return jsonify({"status": "started", "pid": process.pid,
                    "workspace": workspace, "version": version,
                    "local_model": model_id,
                    "local_agent_ready": state["local_agent_ready"],
                    "warning": state["local_reason"]})
