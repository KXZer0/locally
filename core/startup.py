"""Device discovery, idle unloading, and model-loading workers."""

import socket
import threading
import time
from datetime import datetime

import openvino as ov

from core import runtime
from core.chat.common import _prewarm_slot
from core.genai.results import explain_genai_error
from core.slots.proxy import ProxySlot
from core.web.search import _searx

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def check_port(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False


def detect_devices():
    """Return {kind: {"id": ov_id, "name": full_name}} of usable devices.

    kind is the canonical category ("NPU", "GPU", "CPU"). For "GPU", "id"
    may be "GPU.0", "GPU.1", etc. when OpenVINO enumerates multiple GPUs;
    callers must pass "id" to OpenVINO, not "kind".

    Non-Intel GPUs are filtered out: OpenVINO's intel_gpu plugin enumerates
    any OpenCL-capable GPU (NVIDIA, AMD), but its kernels only run on Intel
    hardware. Selecting a non-Intel GPU produces hundreds of compile errors
    and crashes at warmup with CL_INVALID_VALUE — better not to offer it.
    """
    devices = {}
    core = ov.Core()
    for d in core.get_available_devices():
        try:
            full_name = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            full_name = d
        if d.startswith("GPU"):
            if "intel" not in full_name.lower():
                continue
            if "GPU" not in devices:  # first Intel GPU wins
                devices["GPU"] = {"id": d, "name": full_name}
        elif d in ("NPU", "CPU"):
            devices[d] = {"id": d, "name": full_name}
    return devices


def _idle_watchdog(slots, idle_timeout, check_interval=30):
    """Background thread: unload slots that have been idle too long."""
    while True:
        time.sleep(check_interval)
        now = time.time()
        for slot in slots:
            if not slot or slot.status != "ready":
                continue
            if now - slot.last_used < idle_timeout:
                continue
            # Try non-blocking lock acquire — skip if a request is in progress
            if not slot.lock.acquire(blocking=False):
                continue
            try:
                slot.unload()
            finally:
                slot.lock.release()
        # SearXNG follows the same rule as every slot: resident only while it
        # is being used. Only ever stops a process this server started.
        _searx.maybe_stop_idle()


_banner_lock = threading.Lock()
_banner_printed = False


def _load_in_background(slot, model_dir, devices, port, ollama_port, banner_slots):
    """Background thread: load model + warmup on one device."""
    global _banner_printed
    try:
        if isinstance(slot, ProxySlot):
            # device_full already names the upstream URL, which is the only
            # honest answer to "where does this run"; the device table has no
            # entry for a machine we are merely talking to. Prewarm is skipped
            # for the same reason: it exists to fill a local prefix cache this
            # slot does not have, so it would spend a real upstream request to
            # warm nothing.
            slot.load()
            slot.warmup()
        else:
            with slot.lock:
                slot.device_full = devices.get(slot.device_name, {}).get("name", slot.device_name)
                slot.load(model_dir)
                slot.warmup()
                _prewarm_slot(slot)
    except Exception as e:
        slot.status = "error"
        print(f"\n  [{slot.device_name}] ERROR: Failed to load model: {explain_genai_error(e)}")
        if (not isinstance(slot, ProxySlot)
                and not any(s in str(e) for s in ("Could not find a model",
                                                  "is truncated",
                                                  "Compilation failed"))):
            # Device-contention hint only where it's plausible — for a
            # missing/truncated model or a compiler failure it sends people
            # chasing ghosts (#17, #20 — the latter is literally titled
            # after this hint).
            print(f"  Is another process using the {slot.device_name}?", flush=True)

    # Print banner when all slots are done — only one thread wins
    with _banner_lock:
        if _banner_printed:
            return
        all_done = all(
            s.status in ("ready", "error", "not_configured")
            for s in banner_slots
        )
        if not all_done:
            return
        _banner_printed = True

    if any(s.status == "ready" for s in banner_slots):
        lines = []
        for s in banner_slots:
            if s.status == "ready":
                lines.append(f"    {s.device_name:5s}: {s.model_name} ({s.model_type.upper()}) "
                             f"-- {s.device_full}")
        url = f"http://localhost:{port}"
        api_lines = [f"    API  : {url}  (OpenAI)"]
        if ollama_port:
            api_lines.append(f"    API  : http://localhost:{ollama_port}  (Ollama)")
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.console import Group
            from rich.text import Text
            table = Table("Device", "Model", "Type", expand=True, box=None)
            for s in banner_slots:
                if s.status == "ready":
                    table.add_row(Text(s.device_name), Text(s.model_name), Text(s.model_type.upper()))
            endpoints = [f"OpenAI   {url}/v1"]
            if ollama_port:
                endpoints.append(f"Ollama   http://localhost:{ollama_port}")
            Console().print(Panel(Group(table, Text("\n" + "\n".join(endpoints)),
                                        Text("\nCtrl+C stops the API and releases its models · "
                                             "Shift+Enter opens terminal chat", style="dim")),
                                  title="locally · ready", border_style="dim"))
        except ImportError:
            print(f"""
================================================
  locally ready
{chr(10).join(lines)}
{chr(10).join(api_lines)}
================================================
  Ctrl+C stops the API and releases its models · Shift+Enter opens terminal chat
""", flush=True)
