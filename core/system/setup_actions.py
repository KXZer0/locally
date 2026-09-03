"""Curated setup downloads and live slot activation."""

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from datetime import datetime

from flask import jsonify, request

from core import config, runtime
from core.errors import openai_error
from core.slots.device import DeviceSlot
from core.slots.proxy import ProxySlot
from core.slots.select import _slot_serviceable
from core.slots.speaker import SpeakerSlot
from core.slots.tts import TtsSlot
from core.slots.vad import VadSlot
from core.slots.whisper import WhisperSlot
from core.system.access import _request_is_local
from core.system.setup_info import (_SETUP_CATALOG, _setup_destination, _setup_entry_installed, _setup_jobs, _setup_jobs_lock, _setup_models_root, _setup_ollama, _write_setup_config)
from core.web import search as web_search_mod

_setup_activation_lock = threading.Lock()

def _setup_download_item(entry):
    dest = _setup_destination(entry)
    os.makedirs(dest, exist_ok=True)
    if entry["kind"] == "hf_snapshot":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is not installed") from exc
        snapshot_download(repo_id=entry["repo_id"], local_dir=dest)
    elif entry["kind"] == "hf_file":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is not installed") from exc
        hf_hub_download(repo_id=entry["repo_id"], filename=entry["filename"],
                        local_dir=dest)
    elif entry["kind"] == "url_file":
        target = os.path.join(dest, entry["filename"])
        fd, temp_path = tempfile.mkstemp(prefix="download-", suffix=".part", dir=dest)
        digest = hashlib.sha256()
        total = 0
        try:
            req = urllib.request.Request(entry["url"],
                                         headers={"User-Agent": "locally-setup"})
            with urllib.request.urlopen(req, timeout=30) as response, os.fdopen(fd, "wb") as out:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 8 * 1024 * 1024:
                        raise RuntimeError("the curated VAD download exceeded 8 MB")
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest().lower() != entry["sha256"].lower():
                raise RuntimeError("Silero VAD checksum did not match the tested file")
            os.replace(temp_path, target)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except OSError:
                pass
    elif entry["kind"] == "searxng":
        if os.name != "nt":
            raise RuntimeError("guided SearXNG installation is currently Windows-only")
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        script = os.path.join(config.SCRIPT_DIR, "scripts", "searxng.ps1")
        if not pwsh or not os.path.isfile(script):
            raise RuntimeError("PowerShell or scripts/searxng.ps1 is missing")
        result = subprocess.run(
            [pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script,
             "-Install", "-SearxRoot", dest],
            capture_output=True, text=True, timeout=1200)
        if result.returncode:
            tail = (result.stderr or result.stdout or "installation failed")[-1200:]
            raise RuntimeError(f"SearXNG installation failed: {tail.strip()}")
    if not _setup_entry_installed(entry):
        raise RuntimeError(f"{entry['name']} downloaded, but its required files are missing")


def _setup_install_worker(job_id, item_ids):
    try:
        for item_id in item_ids:
            entry = _SETUP_CATALOG[item_id]
            with _setup_jobs_lock:
                job = _setup_jobs[job_id]
                job["status"] = "downloading"
                job["current"] = item_id
                job["items"][item_id] = "downloading"
            if not _setup_entry_installed(entry):
                _setup_download_item(entry)
            with _setup_jobs_lock:
                _setup_jobs[job_id]["items"][item_id] = "installed"
        with _setup_jobs_lock:
            job = _setup_jobs[job_id]
            job["status"] = "complete"
            job["current"] = None
            job["finished_at"] = time.time()
    except Exception as exc:
        with _setup_jobs_lock:
            job = _setup_jobs[job_id]
            job["status"] = "error"
            job["error"] = str(exc)
            if job.get("current"):
                job["items"][job["current"]] = "error"
            job["finished_at"] = time.time()


def setup_install():
    if not _request_is_local():
        return openai_error("Setup installation is local-only.",
                            "invalid_request_error", 403)
    body = request.get_json(silent=True) or {}
    item_ids = body.get("items") or []
    if not isinstance(item_ids, list) or not item_ids or len(item_ids) > 10:
        return openai_error("'items' must be a non-empty setup item list.")
    item_ids = list(dict.fromkeys(str(value) for value in item_ids))
    unknown = [value for value in item_ids if value not in _SETUP_CATALOG]
    if unknown:
        return openai_error(f"Unknown setup item(s): {', '.join(unknown)}")
    missing = [_SETUP_CATALOG[value] for value in item_ids
               if not _setup_entry_installed(_SETUP_CATALOG[value])]
    if missing:
        root = _setup_models_root()
        os.makedirs(root, exist_ok=True)
        free = shutil.disk_usage(root).free
        # Estimates are model-card rounded figures, so leave 15% download/
        # metadata slack plus 1 GB for the rest of the machine.
        needed = sum(float(entry.get("size_gb") or 0) for entry in missing)
        needed_bytes = int((needed * 1.15 + 1.0) * 2 ** 30)
        if free < needed_bytes:
            return openai_error(
                f"The selected setup needs about {needed:.1f} GB, but only "
                f"{free / 2 ** 30:.1f} GB is free on the models drive.",
                "server_error", 409)
    job_id = uuid.uuid4().hex
    job = {"id": job_id, "status": "queued", "current": None,
           "items": {item_id: "queued" for item_id in item_ids},
           "error": None, "started_at": time.time()}
    with _setup_jobs_lock:
        # Bound the in-memory history; active jobs are never discarded.
        old = sorted((j for j in _setup_jobs.values()
                      if j.get("status") in ("complete", "error")),
                     key=lambda j: j.get("finished_at", 0))
        for stale in old[:-7]:
            _setup_jobs.pop(stale["id"], None)
        _setup_jobs[job_id] = job
    threading.Thread(target=_setup_install_worker, args=(job_id, item_ids),
                     daemon=True, name=f"locally-setup-{job_id[:6]}").start()
    return jsonify(job), 202


def setup_install_status(job_id):
    if not _request_is_local():
        return openai_error("Setup installation is local-only.",
                            "invalid_request_error", 403)
    with _setup_jobs_lock:
        job = dict(_setup_jobs.get(job_id) or {})
        if job.get("items"):
            job["items"] = dict(job["items"])
    if not job:
        return openai_error("Unknown setup job.", "invalid_request_error", 404)
    return jsonify(job)


def _runtime_device_id(kind):
    return runtime.DEVICES.get(kind, {}).get("id", kind)


def _setup_activate_audio(item_id):
    entry = _SETUP_CATALOG[item_id]
    component = entry["component"]
    path = _setup_destination(entry)
    if not _setup_entry_installed(entry):
        raise RuntimeError(f"{entry['name']} is not installed")
    current = {"stt": runtime.whisper_slot, "tts": runtime.tts_slot,
               "vad": runtime._vad_slot, "speaker": runtime._speaker_slot}.get(component)
    if (current and os.path.realpath(current.model_dir or "") == os.path.realpath(path)
            and current.status in ("ready", "idle_unloaded")):
        return current

    device = "GPU" if component == "stt" and "GPU" in runtime.DEVICES else "CPU"
    if component == "stt":
        slot = WhisperSlot(device, _runtime_device_id(device))
    elif component == "tts":
        slot = TtsSlot(device, _runtime_device_id(device))
    elif component == "vad":
        slot = VadSlot(device, _runtime_device_id(device))
    elif component == "speaker":
        slot = SpeakerSlot(device, _runtime_device_id(device))
    else:
        raise RuntimeError(f"{entry['name']} is not an audio component")
    slot.device_full = runtime.DEVICES.get(device, {}).get("name", device)

    if current:
        with current.lock:
            current.unload()
        try:
            runtime.slots.remove(current)
        except ValueError:
            pass
    slot.load(path)
    slot.warmup()
    runtime.slots.append(slot)
    if component == "stt":
        runtime.whisper_slot = slot
    elif component == "tts":
        runtime.tts_slot = slot
    elif component == "vad":
        runtime._vad_slot = slot
    else:
        runtime._speaker_slot = slot
    return slot


def _setup_activate_ollama(model):
    state = _setup_ollama()
    known = {entry["name"] for entry in state.get("models", [])}
    if not state.get("available") or model not in known:
        raise RuntimeError("That model is not advertised by the local Ollama server")
    proxy = ProxySlot("http://127.0.0.1:11434", model=model)
    proxy.load()
    proxy.warmup()
    old = runtime.primary
    if old:
        with old.lock:
            old.unload()
        try:
            index = runtime.slots.index(old)
            runtime.slots[index] = proxy
        except ValueError:
            runtime.slots.append(proxy)
    else:
        runtime.slots.append(proxy)
    runtime.primary = proxy
    return proxy


def setup_finish():
    if not _request_is_local():
        return openai_error("Setup changes are local-only.",
                            "invalid_request_error", 403)
    body = request.get_json(silent=True) or {}
    assistant_id = body.get("assistant")
    backend = body.get("backend") or "openvino"
    voice_ids = body.get("voice") or []
    if not isinstance(voice_ids, list):
        return openai_error("'voice' must be a setup item list.")
    try:
        with _setup_activation_lock:
            if backend == "ollama":
                model = str(body.get("ollama_model") or "").strip()
                if not model:
                    raise RuntimeError("Choose an installed Ollama model")
                slot = _setup_activate_ollama(model)
                assistant_cfg = {"backend": "ollama", "url": "http://127.0.0.1:11434",
                                 "model": model}
                assistant_result = {"backend": "ollama", "model": slot.model_name,
                                    "loaded": True}
            else:
                if assistant_id not in _SETUP_CATALOG or \
                        _SETUP_CATALOG[assistant_id]["component"] != "assistant":
                    raise RuntimeError("Choose a curated assistant model")
                entry = _SETUP_CATALOG[assistant_id]
                if not _setup_entry_installed(entry):
                    raise RuntimeError(f"{entry['name']} is not installed")
                path = _setup_destination(entry)
                requested = str(body.get("device") or "").upper()
                compatible = [d for d in entry.get("devices", []) if d in runtime.DEVICES]
                device = requested if requested in compatible else (compatible[0] if compatible else "CPU")
                assistant_cfg = {"backend": "openvino", "model_dir": path,
                                 "device": device}
                loaded = bool(runtime.primary and runtime.primary.model_dir and
                              os.path.realpath(runtime.primary.model_dir) == os.path.realpath(path)
                              and _slot_serviceable(runtime.primary))
                assistant_result = {"backend": "openvino", "model": entry["name"],
                                    "path": path, "device": device, "loaded": loaded}

            audio_cfg, audio_result = {}, {}
            for item_id in dict.fromkeys(str(value) for value in voice_ids):
                entry = _SETUP_CATALOG.get(item_id)
                if not entry or entry["component"] not in ("stt", "tts", "vad", "speaker"):
                    raise RuntimeError(f"Unknown voice setup item: {item_id}")
                slot = _setup_activate_audio(item_id)
                key = f"{entry['component']}_dir"
                audio_cfg[key] = _setup_destination(entry)
                audio_result[entry["component"]] = slot.info

            web_cfg = {"searxng": bool(body.get("searxng")),
                       "check_updates": bool(body.get("check_updates"))}
            if web_cfg["searxng"]:
                entry = _SETUP_CATALOG["searxng"]
                if not _setup_entry_installed(entry):
                    raise RuntimeError("SearXNG is not installed")
                config.SEARXNG_ROOT = _setup_destination(entry)
                web_search_mod.WEB_SEARCH_URL = "http://127.0.0.1:8080"
                web_cfg["root"] = config.SEARXNG_ROOT
                web_cfg["url"] = web_search_mod.WEB_SEARCH_URL

            setup_cfg = {"version": 1, "assistant": assistant_cfg,
                         "voice": audio_cfg, "web": web_cfg,
                         "completed_at": datetime.now().isoformat(timespec="seconds")}
            config_path = _write_setup_config(setup_cfg)
    except Exception as exc:
        return openai_error(str(exc), "server_error", 500)
    return jsonify({"status": "ok", "assistant": assistant_result,
                    "audio": audio_result, "web": web_cfg,
                    "config_path": config_path})
