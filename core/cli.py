"""Command-line parsing and process startup."""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

from core import cachetrim, config, odysseus, runtime
from core.chat.ollama import VSCODE_OLLAMA_VERSION
from core.hardware.devices import _gpu_has_xmx
from core.models.describe import scan_models
from core.slots.device import DeviceSlot
from core.slots.proxy import ProxySlot
from core.slots.speaker import SpeakerSlot
from core.slots.tts import TtsSlot
from core.slots.util import UtilSlot
from core.slots.vad import VadSlot
from core.slots.whisper import WhisperSlot
from core.startup import check_port, detect_devices, _idle_watchdog, _load_in_background
from core.routes import audio as audio_routes
from core.system.setup import _read_setup_config
from core.system.updates import _schedule_update_refresh
from core.launch.configure import configure
from core.launch.slots import build_slots
from core.launch.serve import serve


DESCRIPTION = """locally — OpenAI-compatible API server for Intel NPU / ARC GPU.

Auto-detects available devices (NPU, GPU, CPU) and model type (VLM/LLM).
NPU-first: works on any Intel Core Ultra laptop. ARC GPU optional.
Dual mode: NPU for chat + GPU for vision, simultaneously.

Usage:
    python locally.py                                        # auto-detect device
    python locally.py --device NPU                           # force NPU
    python locally.py --device GPU                           # force GPU
    python locally.py --gpu-model-dir gpu-model              # dual: NPU chat + GPU vision
    python locally.py --model-dir ~/models/qwen3-14b-int4-ov --device GPU  # big LLM on GPU
    python locally.py --whisper-dir whisper-model             # add speech-to-text
    python locally.py --scan                                 # what models do I have?
"""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_model = str(Path(config.SCRIPT_DIR) / "model")
    p.add_argument("--model-dir", default=default_model,
                   help="Primary model directory (default: model/)")
    p.add_argument("--device", default="auto",
                   help="Device for primary model: NPU, GPU, CPU, or auto (default: auto)")
    # A machine with no Intel silicon has no device to place a generative model
    # on, which used to make the whole app unreachable there — utilities, voice
    # and UI included — for want of a chat model. Pointing the primary slot at
    # an OpenAI-compatible server (Ollama, vLLM, llama.cpp) runs the same
    # program on that machine. --model-dir is not required in this mode.
    p.add_argument("--proxy-url", default=None,
                   help="Serve the primary slot from an OpenAI-compatible server "
                        "instead of a local model, e.g. http://localhost:11434 "
                        "(Ollama). Replaces --model-dir/--device for chat.")
    p.add_argument("--proxy-model", default=None,
                   help="Model id to request upstream (default: the first the "
                        "upstream advertises)")
    p.add_argument("--proxy-key", default=None,
                   help="Bearer token for the upstream, if it needs one")
    p.add_argument("--gpu-model-dir", default=None,
                   help="Secondary GPU model (enables dual mode: NPU chat + GPU vision/LLM)")
    p.add_argument("--port", type=int, default=8000,
                   help="OpenAI API port (default: 8000)")
    # Note: the CLI flag is --ollama-port (dash), read back as args.ollama_port
    # (underscore) — argparse converts dashes to underscores in attribute names
    # by design, per Python convention. Same for every --two-word flag here.
    p.add_argument("--ollama-port", type=int, default=11434,
                   help="Ollama API port (default: 11434, 0 to disable)")
    p.add_argument("--max-dim", type=int, default=768,
                   help="Max image dimension before resize (default: 768)")
    p.add_argument("--whisper-dir", default=None,
                   help="Whisper model directory for speech-to-text (enables /v1/audio/transcriptions)")
    p.add_argument("--model-cache-gb", type=float, default=12.0,
                   help="Upper bound on the compile cache (default 12 GB). "
                        "Least-recently-used entries are evicted at startup; "
                        "evicting one costs a single cold compile. 0 disables "
                        "trimming, which is how it reached 41.9 GB before.")
    p.add_argument("--no-model-cache", action="store_true",
                   help="Disable the OpenVINO compiled-model cache. The cache "
                        "trades disk for startup: the first load of a model "
                        "compiles it, later loads read the compiled blob back "
                        "(the NPU otherwise rebuilds its graph on every start).")
    p.add_argument("--model-cache-dir", default=None, metavar="DIR",
                   help="Where to keep compiled-model blobs "
                        "(default: <script dir>/.ov-cache).")
    p.add_argument("--models-dir", default=None, metavar="DIR",
                   help="Directory holding model folders, offered by "
                        "GET /v1/models/available and loadable at runtime via "
                        "POST /v1/models/load (no restart). Defaults to the "
                        "parent of the loaded models plus ~/models.")
    p.add_argument("--tts-dir", default=None,
                   help="Text-to-speech model directory (SpeechT5 or Kokoro IR). "
                        "Enables POST /v1/audio/speech.")
    p.add_argument("--tts-device", default="CPU",
                   help="Device for TTS (default: CPU — keeps the GPU free for "
                        "ASR and the NPU free for chat).")
    p.add_argument("--whisper-device", default="CPU",
                   help="Device for Whisper: CPU or GPU (default: CPU)")
    p.add_argument("--vad-dir", default=None,
                   help="Silero voice-activity model (silero_vad.onnx, or the "
                        "folder holding it). Enables auto turn-taking in the "
                        "Voice tab; without it voice is push-to-talk.")
    p.add_argument("--vad-device", default="CPU",
                   help="Device for the VAD (default: CPU — it is ~1M "
                        "parameters and belongs nowhere near the NPU).")
    p.add_argument("--speaker-dir", default=None,
                   help="Speaker-embedding model (ECAPA-TDNN / WeSpeaker .onnx "
                        "or .xml, or the folder holding it). Gates turn-taking "
                        "on WHOSE voice it is; without it any speech in the "
                        "room can open a turn or interrupt the assistant.")
    p.add_argument("--speaker-device", default="CPU",
                   help="Device for speaker ID (default: CPU — it runs once "
                        "per turn, not per frame).")
    p.add_argument("--speaker-threshold", type=float, default=0.35,
                   help="Cosine similarity above which a voice is the enrolled "
                        "user (default: 0.35). Every verification logs its "
                        "score, so tune from your own numbers, not this one.")
    p.add_argument("--util-models-dir", default=None,
                   help="Directory of utility models holding ocr-det/, "
                        "ocr-rec/, and optional background/, upscale/, "
                        "detect-rfdetr/, and embed/ folders. Enables the "
                        "Utilities sidebar and /v1/util/* endpoints. Auto-detected "
                        "from ~/models/util when omitted. Fetch OCR with "
                        "`python scripts/npu-probe.py --all`.")
    p.add_argument("--no-util", action="store_true",
                   help="Don't load utility models even if they are found.")
    p.add_argument("--check-updates", action="store_true",
                   help="Check GitHub/npm once a day for newer locally, "
                        "Odysseus, SearXNG and OpenCode versions. Off by "
                        "default; never delays startup or auto-updates.")
    p.add_argument("--util-engines", default="auto",
                   help="Which engines load the utility models: 'npu', 'gpu', "
                        "'cpu', a comma-separated list, or 'auto' (default). "
                        "Auto takes every Intel accelerator detected, and falls "
                        "back to CPU only when there is none — which is the "
                        "machine --proxy-url exists for, where the utilities are "
                        "the whole reason to run locally at all. The NPU stays "
                        "the default *engine* for a request because the GPU is "
                        "usually busy holding a chat model — but some tasks are "
                        "GPU-only (Swin2SR upscaling will not compile on the "
                        "NPU), so a slot has to exist there or that tier is "
                        "unreachable.")
    p.add_argument("--idle-timeout", type=int, default=1800,
                   help="Change idle-unload timeout in seconds "
                        "(default: 1800 = 30 min). Use 0 to disable unloading "
                        "(recommended for agent use; also auto-enables --prewarm).")
    p.add_argument("--debug", action="store_true",
                   help="Log every inbound API request (method, path, User-Agent, body)")
    p.add_argument("--vscode-compat", action="store_true",
                   help=f"Report a real Ollama version ({VSCODE_OLLAMA_VERSION}) on "
                        f"/api/version so VS Code's Ollama client accepts the server")
    p.add_argument("--no-prompt-cache", action="store_true",
                   help="Disable prefix (KV) caching on GPU/CPU LLM slots. Caching is "
                        "ON by default — it prefills a repeated prompt prefix (e.g. an "
                        "agent's fixed system prompt) once instead of every turn.")
    p.add_argument("--cache-size-gb", type=int, default=config.PROMPT_CACHE_GB,
                   help=f"KV-cache pool size in GB when prefix caching is on "
                        f"(default: {config.PROMPT_CACHE_GB})")
    p.add_argument("--prewarm", default=None, metavar="FILE",
                   help="Prefill a saved agent prompt at startup so the first turn is a "
                        "cache hit (no cold-prefill stall). The file auto-populates from the "
                        "first big prompt served, so: run once, then restart with --prewarm. "
                        "Auto-enabled (as prewarm-<port>.json) when --idle-timeout is 0.")
    p.add_argument("--no-prewarm", action="store_true",
                   help="Disable the automatic prewarm that --idle-timeout 0 turns on.")
    p.add_argument("--offload-ratio", default="auto", metavar="PCT",
                   help="Stream PCT%% of MoE expert weights from disk instead of "
                        "keeping them GPU-resident (OpenVINO 2026.3+ disk offload). "
                        "Lets 30B-class MoE models run on 16 GB-class GPUs at the "
                        "cost of decode speed. Default 'auto': compute the smallest "
                        "ratio that makes the model fit alongside its KV pool, and "
                        "stay off entirely when it already fits. '0' disables it; "
                        "1-99 pins a value. GPU + XMX + MoE only — it does nothing "
                        "on dense models or non-XMX GPUs, by design of the plugin.")
    p.add_argument("--context-tokens", default=None, metavar="N",
                   help="Size the KV pool by the context you need, instead of "
                        "guessing gigabytes with --cache-size-gb: N tokens is "
                        "converted using the model's own KV geometry, and "
                        "'auto' takes whatever the device has left after the "
                        "weights. Capped at the model's real context length. "
                        "This is the ceiling an agent session actually hits — "
                        "overrunning it fails the request outright.")
    p.add_argument("--npu-prompt-len", type=int, default=None, metavar="N",
                   help="NPU context window (default 8192; 4096 cuts ~0.34s off "
                        "time-to-first-token). The NPU compiles a static graph, "
                        "so prefill costs the same whatever the prompt length - "
                        "this trades context for latency. 8192 is the compiler "
                        "ceiling; see TODONT.md.")
    p.add_argument("--searxng-root", default=None, metavar="DIR",
                   help="SearXNG checkout to start on demand and stop when "
                        "idle. Without it, SearXNG must already be running.")
    p.add_argument("--searxng-idle", type=int, default=600, metavar="SECS",
                   help="Stop an auto-started SearXNG after this many idle "
                        "seconds (0 keeps it up).")
    p.add_argument("--odysseus-autostart", dest="odysseus_autostart",
                   action="store_true", default=None,
                   help="Bring an Odysseus checkout up at startup with "
                        "`docker compose up -d` (see docs/ODYSSEUS.md). Runs on "
                        "a background thread — a first run builds images and "
                        "takes minutes, and chat must not wait for it. Does "
                        "nothing when no checkout is found.")
    p.add_argument("--no-odysseus-autostart", dest="odysseus_autostart",
                   action="store_false",
                   help="Never start Odysseus, even if a checkout is found "
                        "(the default).")
    p.add_argument("--container-engine", choices=("docker", "podman"),
                   default=None,
                   help="Which container engine runs the Odysseus stack. "
                        "Default: whichever is on PATH, Docker first. Podman "
                        "works because Odysseus's compose file declares "
                        "host.docker.internal via host-gateway explicitly, "
                        "which Podman honours.")
    p.add_argument("--odysseus-port", type=int, default=config.ODYSSEUS_PORT,
                   metavar="N",
                   help=f"Port Odysseus serves on (default "
                        f"{config.ODYSSEUS_PORT}; its own APP_PORT default). "
                        f"This is what locally probes to tell running from not.")
    p.add_argument("--odysseus-dir", default=None, metavar="DIR",
                   help="Odysseus checkout to manage. Without it: $ODYSSEUS_DIR, "
                        "then a sibling `odysseus` directory, then ~/odysseus — "
                        "the same order the update check already uses.")
    p.add_argument("--search-url", default=None, metavar="URL",
                   help="SearXNG base URL for web search, e.g. "
                        "http://localhost:8080. Unset (the default) means the "
                        "server makes no outbound connections at all.")
    p.add_argument("--python-sandbox", choices=("auto", "podman", "subprocess"),
                   default="auto",
                   help="Boundary for Python calculations: a Podman container, "
                        "subprocess guardrails, or auto (default: Podman when it "
                        "is available, with an explicit subprocess fallback).")
    p.add_argument("--python-tool", action="store_true",
                   help="Enable the server-side local Python calculation tool "
                        "(off by default; use only for local model turns).")
    p.add_argument("--python-timeout", type=float, default=config.PYTHON_TOOL_TIMEOUT,
                   metavar="SECS",
                   help="Hard timeout for one Python calculation child "
                        f"(default: {config.PYTHON_TOOL_TIMEOUT:g}s).")
    p.add_argument("--python-output-bytes", type=int,
                   default=config.PYTHON_TOOL_OUTPUT_BYTES, metavar="N",
                   help="Maximum stdout/stderr bytes from one calculation "
                        f"(default: {config.PYTHON_TOOL_OUTPUT_BYTES}).")
    p.add_argument("--gpu-reserve-gb", type=float, default=3.0, metavar="N",
                   help="System RAM to keep outside the shared-GPU budget, in GB "
                        "(default: 3). The reserve is not arbitrary: an 18.3 GB "
                        "model once 'fit' a 23.6 GB advertised ceiling on a "
                        "machine with 0.9 GB actually free, Windows paged it, and "
                        "an A3B MoE touching a fresh expert set every token "
                        "decoded at 0.5 tok/s out of the pagefile. 0 restores "
                        "exactly that failure mode.")
    p.add_argument("--kv-precision", default=None, choices=["u8", "f16"],
                   metavar="P",
                   help="KV cache element type. 'u8' halves cache bytes per "
                        "token (96 KB -> ~48 KB on a 30B), which is what lets a "
                        "100k-token session fit a 24 GB GPU budget. Quality cost "
                        "is far below that of int4 weights. GPU/CPU only.")
    p.add_argument("--prompt-cache", dest="prompt_cache", action="store_true",
                   default=None,
                   help="Force the prefix-caching (continuous-batching) backend "
                        "on. Default is per-device: on for CPU, OFF for GPU, "
                        "where it measured ~10x slower and hangs on repeated "
                        "prefixes (TODONT.md).")
    p.add_argument("--agent-tools", default=None, metavar="NAMES",
                   help="Comma-separated tool names to keep from a client's "
                        "tool list; the rest are dropped before the prompt is "
                        "built. Tool schemas are ~68%% of an agent request "
                        "(30 tools = 99k of Claude Code's 142k chars), and "
                        "prefilling them is most of the wait on an iGPU. Try "
                        "\"Bash,Edit,Read,Write,Glob,Grep\".")
    p.add_argument("--scan", nargs="*", default=None, metavar="DIR",
                   help="Report what each model directory actually contains "
                        "(name, precision, architecture, integrity) and exit. "
                        "Searches the locally directory and ~/models by "
                        "default; pass directories to search those instead.")
    return p.parse_args()


def main(app=None, ollama_app=None):
    if app is None or ollama_app is None:
        from core.app import create_app, create_ollama_app
        app = app or create_app()
        ollama_app = ollama_app or create_ollama_app()
    args = parse_args()
    model_dir = configure(args)
    if model_dir is None:
        return
    devices, setup_only, all_slots = build_slots(args, model_dir)
    serve(app, ollama_app, args, model_dir, devices, setup_only, all_slots)
