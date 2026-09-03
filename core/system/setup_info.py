"""First-run setup catalog, discovery, and saved configuration."""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from pathlib import Path

from flask import jsonify, request

from core import config, runtime
from core.chat.common import overall_status
from core.models.identity import _is_model_dir
from core.system.status import _memory_data

# ---------------------------------------------------------------------------
# First-run setup
#
# The setup UI is allowed to download only this small, reviewed catalog. A
# caller cannot supply a repository, URL, filename, destination or command.
# Mutation endpoints are localhost-only because locally deliberately binds to
# the LAN for phone/tablet access, and a page on another device must not be
# able to fill this machine's disk or start an installer.
# ---------------------------------------------------------------------------

_SETUP_VAD_SHA256 = "7776b81ad1b0350c15d7f1555943b9232eb53e9ca5d989c6d0cea9ebc8664d87"
_SETUP_CATALOG = {
    "qwen3-8b-npu": {
        "component": "assistant", "kind": "hf_snapshot",
        "name": "Qwen3 8B", "detail": "Best verified everyday NPU model",
        "repo_id": "OpenVINO/Qwen3-8B-int4-cw-ov",
        "directory": "Qwen3-8B-int4-cw-ov", "size_gb": 5.0,
        "devices": ["NPU", "GPU", "CPU"], "capabilities": ["chat"],
        "license": "Apache-2.0", "verified": True,
    },
    "smollm3-3b": {
        "component": "assistant", "kind": "hf_snapshot",
        "name": "SmolLM3 3B", "detail": "Small, fast, and verified on Intel NPU",
        "repo_id": "aweussom/SmolLM3-3B-int4-cw-ov",
        "directory": "SmolLM3-3B-int4-cw-ov", "size_gb": 2.0,
        "devices": ["NPU", "GPU", "CPU"], "capabilities": ["chat"],
        "license": "See model card", "verified": True,
    },
    "qwen35-4b-vision": {
        "component": "assistant", "kind": "hf_snapshot",
        "name": "Qwen3.5 4B Vision", "detail": "Compact GPU model that can see images",
        "repo_id": "OpenVINO/Qwen3.5-4B-int4-ov",
        "directory": "Qwen3.5-4B-int4-ov", "size_gb": 3.0,
        "devices": ["GPU", "CPU"], "capabilities": ["chat", "vision"],
        "license": "See model card", "verified": True,
    },
    "qwen25-coder-7b": {
        "component": "assistant", "kind": "hf_snapshot",
        "name": "Qwen2.5 Coder 7B", "detail": "Compact agent model with tool calling",
        "repo_id": "OpenVINO/Qwen2.5-Coder-7B-Instruct-int4-ov",
        "directory": "Qwen2.5-Coder-7B-Instruct-int4-ov", "size_gb": 5.0,
        "devices": ["GPU", "CPU"], "capabilities": ["chat", "tools"],
        "license": "Apache-2.0", "verified": True,
    },
    "whisper-base": {
        "component": "stt", "kind": "hf_snapshot",
        "name": "Whisper Base", "detail": "Fastest speech recognition",
        "repo_id": "OpenVINO/whisper-base-int8-ov",
        "directory": "whisper-base-int8-ov", "size_gb": 0.2,
        "devices": ["GPU", "CPU"], "license": "Apache-2.0", "verified": True,
    },
    "whisper-small": {
        "component": "stt", "kind": "hf_snapshot",
        "name": "Whisper Small", "detail": "Recommended accuracy and speed",
        "repo_id": "OpenVINO/whisper-small-int8-ov",
        "directory": "whisper-small-int8-ov", "size_gb": 0.4,
        "devices": ["GPU", "CPU"], "license": "Apache-2.0", "verified": True,
    },
    "kokoro": {
        "component": "tts", "kind": "hf_snapshot",
        "name": "Kokoro", "detail": "Natural local voice, 54 speakers",
        "repo_id": "OpenVINO/Kokoro-82M-int8-ov",
        "directory": "Kokoro-82M-int8-ov", "size_gb": 0.16,
        "devices": ["CPU"], "license": "Apache-2.0", "verified": True,
    },
    "silero-vad": {
        "component": "vad", "kind": "url_file",
        "name": "Silero turn-taking", "detail": "Detects speech instead of room volume",
        "url": ("https://raw.githubusercontent.com/snakers4/silero-vad/master/"
                "src/silero_vad/data/silero_vad_openvino_16k.onnx"),
        "filename": "silero_vad_openvino_16k.onnx",
        "sha256": _SETUP_VAD_SHA256, "directory": "silero-vad",
        "size_gb": 0.002, "devices": ["CPU"],
        "license": "MIT", "verified": True,
    },
    "wespeaker": {
        "component": "speaker", "kind": "hf_file",
        "name": "WeSpeaker identity", "detail": "Stops other voices taking your turn",
        "repo_id": "Wespeaker/wespeaker-ecapa-tdnn512-LM",
        "filename": "voxceleb_ECAPA512_LM.onnx", "directory": "speaker",
        "size_gb": 0.03, "devices": ["CPU"],
        "license": "CC-BY-4.0", "verified": True,
    },
    "searxng": {
        "component": "web", "kind": "searxng",
        "name": "SearXNG search", "detail": "Private metasearch on this machine",
        "directory": "searxng-src", "size_gb": 0.35,
        "devices": [], "license": "AGPL-3.0", "verified": True,
    },
}

_setup_jobs = {}
_setup_jobs_lock = threading.Lock()



def _setup_models_root():
    return os.path.abspath(os.path.expanduser(runtime.MODELS_DIR or "~/models"))


def _setup_searxng_root():
    if config.SEARXNG_ROOT:
        return os.path.abspath(config.SEARXNG_ROOT)
    return os.path.abspath(os.path.join(os.path.dirname(config.SCRIPT_DIR), "searxng-src"))


def _setup_destination(entry):
    if entry["kind"] == "searxng":
        return _setup_searxng_root()
    return os.path.join(_setup_models_root(), entry["directory"])


def _setup_entry_installed(entry):
    dest = _setup_destination(entry)
    if entry["kind"] == "searxng":
        return os.path.isfile(os.path.join(dest, "searx", "webapp.py"))
    if entry["kind"] in ("hf_file", "url_file"):
        return os.path.isfile(os.path.join(dest, entry["filename"]))
    if entry["component"] == "assistant":
        return os.path.isdir(dest) and _is_model_dir(dest)
    return os.path.isdir(dest) and any(Path(dest).glob("*.xml"))


def _setup_public_entry(entry_id, entry, recommended=False):
    # The source locator is intentionally visible: setup should be auditable,
    # and every one points to the official project/model owner. The destination
    # cannot be changed by the browser.
    source = entry.get("repo_id") or entry.get("url") or "searxng/searxng"
    return {
        "id": entry_id, "component": entry["component"],
        "name": entry["name"], "detail": entry["detail"],
        "size_gb": entry["size_gb"], "devices": entry.get("devices", []),
        "capabilities": entry.get("capabilities", []),
        "license": entry["license"], "verified": bool(entry.get("verified")),
        "installed": _setup_entry_installed(entry),
        "path": _setup_destination(entry), "source": source,
        "recommended": recommended,
    }


def _setup_recommendations(memory=None):
    memory = memory or _memory_data()
    free_mb = (memory.get("system") or {}).get("available_mb") or 0
    gpu_mb = (memory.get("gpu") or {}).get("usable_mb") or 0
    if "NPU" in runtime.DEVICES and free_mb >= 7 * 1024:
        assistant = "qwen3-8b-npu"
        reason = (f"NPU detected with {free_mb / 1024:.1f} GB of system memory "
                  "available; the verified 5 GB model fits with OS headroom.")
    elif "NPU" in runtime.DEVICES:
        assistant = "smollm3-3b"
        reason = (f"NPU detected, but only {free_mb / 1024:.1f} GB is available "
                  "right now; the 2 GB model leaves safer headroom.")
    elif "GPU" in runtime.DEVICES and gpu_mb >= 4 * 1024:
        assistant = "qwen35-4b-vision"
        reason = (f"The GPU can currently back {gpu_mb / 1024:.1f} GB; the "
                  "3 GB vision model fits without memory pressure.")
    else:
        assistant = "smollm3-3b"
        reason = "No suitable accelerator budget was reported; use the compact CPU fallback."
    return {
        "assistant": assistant,
        "assistant_reason": reason,
        "voice": ["whisper-small", "kokoro", "silero-vad", "wespeaker"],
    }


def _setup_nvidia():
    """NVIDIA is not an OpenVINO target, but it matters for Ollama routing."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        rows = []
        for line in (out.stdout or "").splitlines():
            name, _, memory = line.rpartition(",")
            if name.strip() and memory.strip().isdigit():
                rows.append({"name": name.strip(), "memory_mb": int(memory.strip())})
        return rows
    except Exception:
        return []


def _setup_ollama():
    # When locally owns :11434 this would only discover itself. An external
    # Ollama forces the compatibility port off during startup, so probe only
    # in that state.
    if config.OLLAMA_COMPAT_PORT:
        return {"available": False, "models": [], "reason": "locally owns port 11434"}
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags",
                                     headers={"User-Agent": "locally-setup"})
        with urllib.request.urlopen(req, timeout=1.0) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
        models = []
        for item in data.get("models", []):
            name = item.get("name") or item.get("model")
            if name:
                models.append({"name": name, "size": item.get("size")})
        return {"available": True, "models": models,
                "url": "http://127.0.0.1:11434"}
    except Exception:
        return {"available": False, "models": [], "reason": "not running"}


def _setup_config_path():
    override = os.environ.get("LOCALLY_SETUP_CONFIG")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return os.path.join(os.environ["LOCALAPPDATA"], "locally", "setup.json")
    return os.path.join(os.path.expanduser("~/.config"), "locally", "setup.json")


def _read_setup_config():
    try:
        with open(_setup_config_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_setup_config(data):
    path = _setup_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="setup-", suffix=".json",
                                     dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except OSError:
            pass
    return path


def setup_state():
    memory = _memory_data()
    rec = _setup_recommendations(memory)
    catalog = [
        _setup_public_entry(item_id, entry,
                            item_id == rec["assistant"] or item_id in rec["voice"])
        for item_id, entry in _SETUP_CATALOG.items()
    ]
    devices = [
        {"kind": kind, "id": info.get("id", kind), "name": info.get("name", kind)}
        for kind, info in runtime.DEVICES.items()
    ]
    saved = _read_setup_config()
    return jsonify({
        "first_run": not bool(saved), "needs_assistant": overall_status() == "not_configured",
        "devices": devices, "memory": memory, "nvidia": _setup_nvidia(),
        "ollama": _setup_ollama(), "catalog": catalog,
        "recommended": rec, "configured": saved,
        "models_root": _setup_models_root(),
    })


