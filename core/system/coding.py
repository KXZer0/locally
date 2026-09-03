"""Coding mode and embedded OpenCode process controls."""

import json
import gc
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from flask import jsonify, request

from core import config, opencode_web, runtime
from core.hardware.memory import _mem_status, _settle_memory
from core.errors import openai_error
from core.hardware.memory import _memory_snapshot
from core.slots.capability import _tools_supported
from core.slots.select import _slot_serviceable
from core.system.access import _request_is_local

def coding_mode():
    """Hold the machine for the model that drives a coding agent.

    A mode, not a button. Whisper, Kokoro and the utility engines all reload
    themselves the moment anything touches them -- entering the Voice tab calls
    /v1/audio/warm and pulls two of them straight back in -- so a one-shot
    "free memory" click would be quietly undone and the user would watch the
    memory return with no explanation. While this is on, those loads are
    refused with a message that names the toggle.

    It reports `agent_ready` as well, because freeing memory is not what makes
    a local model able to code. Tool calling is GPU/CPU only and a coding agent
    needs ~70k of context; an 8k NPU slot cannot drive one no matter how much
    RAM is free, and a toggle that said "on" while the model still could not
    call a tool would be a lie.
    """

    if request.method == "GET":
        return jsonify(_coding_mode_state())

    body = request.get_json(silent=True) or {}
    want = bool(body.get("enabled", True))
    if want == config.CODING_MODE:
        return jsonify(dict(_coding_mode_state(), changed=False))

    payload = {"changed": True}
    if want:
        before = _memory_snapshot()
        freed = []
        for slot in (runtime.whisper_slot, runtime.tts_slot, runtime.util_npu, runtime.util_gpu, runtime.util_cpu):
            if not slot or slot.status in ("not_configured", "idle_unloaded"):
                continue
            if not slot.lock.acquire(blocking=False):
                continue          # never yank a pipeline out of a live turn
            try:
                name = slot.model_name
                slot.unload()
                freed.append({"device": slot.device_name, "model": name})
            finally:
                slot.lock.release()
        config.CODING_MODE = True
        gc.collect()
        after, settled = _settle_memory()
        a, b = before.get("system_available_mb"), after.get("system_available_mb")
        payload["freed"] = freed
        payload["returned_mb"] = None if a is None or b is None else b - a
        payload["settle_seconds"] = settled
    else:
        config.CODING_MODE = False
        payload["freed"] = []
    print(f"{datetime.now():%H:%M:%S} <> coding mode "
          f"{'on' if config.CODING_MODE else 'off'}", flush=True)
    return jsonify(dict(_coding_mode_state(), **payload))


def _coding_mode_state():
    """What coding mode is, and whether the loaded model can actually code."""
    slot = next((x for x in (runtime.primary, runtime.secondary) if _slot_serviceable(x)), None)
    ready, reason = False, "No model is loaded."
    if slot:
        ctx = slot.context_tokens or 0
        if not _tools_supported(slot):
            reason = (f"{slot.model_name} is on the {slot.device_name}, which "
                      f"has no tool-calling path — an agent cannot edit files "
                      f"or run commands there. Load it on GPU or CPU.")
        elif ctx and ctx < config.CODE_MIN_CONTEXT:
            reason = (f"{slot.model_name} has {ctx // 1000}k of context; a "
                      f"coding agent needs about "
                      f"{config.CODE_MIN_CONTEXT // 1000}k. Restart with "
                      f"--context-tokens 90000.")
        else:
            ready, reason = True, None
    return {
        "enabled": bool(config.CODING_MODE),
        "agent_ready": ready,
        "reason": reason,
        "model": slot.model_name if slot else None,
        "device": slot.device_name if slot else None,
    }


def _opencode_command():
    """Resolved OpenCode executable and version, without launching it."""
    command = shutil.which("opencode")
    if not command:
        return None, None
    try:
        if os.name == "nt" and command.lower().endswith((".cmd", ".bat")):
            comspec = os.environ.get("COMSPEC") or "cmd.exe"
            argv = [comspec, "/d", "/c", command, "--version"]
        else:
            argv = [command, "--version"]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=8)
        version = (result.stdout or result.stderr or "").strip().splitlines()[0]
        return command, version or None
    except Exception:
        return command, None


def _opencode_local_provider():
    """Runtime-only provider overlay for the model locally currently serves.

    OPENCODE_CONFIG_CONTENT has higher precedence than project config but is
    merged with it, so this adds locally without deleting the user's existing
    providers (including free/subscription providers they switch to when one
    account is out of credits). No API key is persisted or written to a repo.
    """
    slot = next((item for item in (runtime.primary, runtime.secondary)
                 if _slot_serviceable(item)), None)
    if not slot:
        return {}, None
    device_name = getattr(slot, "device_name", "API")
    context = int(getattr(slot, "context_tokens", None)
                  or (8192 if device_name == "NPU" else 32768))
    output = max(2048, min(8192, context // 4))
    model_id = str(getattr(slot, "model_name", None) or "locally")
    provider = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "locally",
        "options": {
            "baseURL": f"http://127.0.0.1:{config.SERVER_PORT}/v1",
            "apiKey": "local",
        },
        "models": {
            model_id: {
                "name": f"{model_id} · {device_name}",
                "limit": {"context": context, "output": output},
            }
        },
    }
    return provider, model_id


def _opencode_overlay():
    """The OPENCODE_CONFIG_CONTENT overlay adding locally as one provider.

    Shared by the TUI and web routes so there is a single place that decides
    how locally is offered to OpenCode. It MERGES with the user's own config
    rather than replacing it, so their existing providers survive.
    """
    overlay = {}
    raw_existing = os.environ.get("OPENCODE_CONFIG_CONTENT")
    if raw_existing:
        try:
            parsed = json.loads(raw_existing)
            if isinstance(parsed, dict):
                overlay = parsed
        except ValueError:
            pass
    provider, model_id = _opencode_local_provider()
    if provider:
        providers = dict(overlay.get("provider") or {})
        providers["locally"] = provider
        overlay["provider"] = providers
        if _coding_mode_state()["agent_ready"] and not overlay.get("model"):
            overlay["model"] = f"locally/{model_id}"
    return overlay


def _opencode_status_data():
    command, version = _opencode_command()
    coding = _coding_mode_state()
    _, model_id = _opencode_local_provider()
    return {
        "installed": bool(command), "version": version,
        "local_model": model_id, "local_agent_ready": coding["agent_ready"],
        "local_reason": coding["reason"],
        "workspace": config.SCRIPT_DIR,
        "web": opencode_web.status(),
    }


def opencode_status():
    return jsonify(_opencode_status_data())


def opencode_web_start():
    """Start (or adopt) the OpenCode WEB server and return its URL.

    Localhost-only and no caller-supplied command, exactly as
    /v1/opencode/launch: the only input is a directory, the argv is fixed, and
    the directory travels in an environment variable rather than being
    interpolated into a shell string.

    Unlike the TUI route this does not open a terminal. `opencode serve` hosts
    the same web interface `opencode web` opens a browser at, so locally starts
    it headless and shows it in the Code tab instead of spawning a window that
    competes with its own UI.
    """
    if not _request_is_local():
        return openai_error("This endpoint is local-only (it starts a process "
                            "on the server's machine).", "invalid_request_error", 403)

    body = request.get_json(silent=True) or {}
    workspace = os.path.abspath(os.path.expanduser(
        str(body.get("workspace") or config.SCRIPT_DIR).strip()))
    if not os.path.isdir(workspace):
        return openai_error(f"Workspace directory not found: {workspace}")

    env = dict(os.environ)
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(
        _opencode_overlay(), separators=(",", ":"))

    ok, detail = opencode_web.start(env=env, cwd=workspace)
    state = opencode_web.status()
    if not ok:
        return jsonify({**state, "error": {"message": detail,
                                           "type": "server_error"}}), 503
    provider, model_id = _opencode_local_provider()
    coding = _coding_mode_state()
    return jsonify({**state, "detail": detail, "workspace": workspace,
                    "local_model": model_id,
                    "local_agent_ready": coding["agent_ready"],
                    "warning": coding["reason"]})


def opencode_web_stop():
    """Stop an OpenCode web server locally started. Never one the user ran."""
    if not _request_is_local():
        return openai_error("This endpoint is local-only.",
                            "invalid_request_error", 403)
    stopped = opencode_web.stop()
    return jsonify({**opencode_web.status(), "stopped": stopped})
