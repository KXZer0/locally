"""Runtime settings, in one place, owned by nobody else.

**Import the module, never the names.** These are reassigned by `main()` once
the command line is parsed, so `from core.config import NPU_MAX_PROMPT_LEN`
binds the default forever and silently ignores the flag the user passed:

    from core import config
    if n > config.NPU_MAX_PROMPT_LEN:      # right: reads the live value
        ...

That one rule is why this file is a module rather than a set of constants
scattered through locally.py, where the same value was both a default and a
global three thousand lines apart.
"""
import os

# The project root — the directory holding locally.py, not this package.
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- server -----------------------------------------------------------------

SERVER_PORT = 8000                     # set from --port; used to tell clients our URL
MAX_REQUEST_BYTES = 50 * 1024 * 1024   # enough for large base64 images
HEARTBEAT_SECS = 15                    # SSE keep-alive during a long prefill
MAX_IMAGE_DIM = 768                    # --max-dim
DEBUG_REQUESTS = False                 # --debug
VSCODE_COMPAT = False                  # --vscode-compat
OLLAMA_COMPAT_PORT = 0                 # --ollama-port after availability checks
CHECK_UPDATES = False                  # --check-updates
SEARXNG_ROOT = None                    # --searxng-root, or detected checkout
SEARXNG_IDLE = 600                     # --searxng-idle
PYTHON_TOOL_ENABLED = False            # --python-tool


# --- the NPU's ceiling ------------------------------------------------------

# 8192 is the vpux compiler's limit, not a memory limit: 9216 / 10240 / 12288
# all fail graph legalisation, and KV compression does not move it.
# See TODONT.md and scripts/npu-context-probe.py.
NPU_MAX_PROMPT_LEN = 8192


# --- prefix caching and context ---------------------------------------------

# None = decide per device (the default): ON for CPU, OFF for GPU. The
# continuous-batching backend that provides prefix caching is a 47x win on the
# 285K CPU and a ~10x *loss* on the B390 GPU, where it also hangs outright on a
# repeated prefix — i.e. on exactly what an agent client sends every turn
# (measured 2026-08-10, see TODONT.md). A default that is right on one device
# and catastrophic on another has to be a per-device default, not a constant.
# --prompt-cache / --no-prompt-cache force it either way.
PROMPT_CACHE = None

# KV-cache pool size in GB. **The pool is the usable context window** — overrun
# it and the request hard-fails (#21) — which is why --context-tokens exists, to
# set it in the unit users actually have.
PROMPT_CACHE_GB = 2

KV_PRECISION = None    # --kv-precision: "u8" halves KV bytes/token, or None
CONTEXT_TOKENS = None  # --context-tokens: int, "auto", or None (use the GB flag)

# Claude Code's own system prompt measured ~35k tokens, and it requests up to
# 32k of output; the KV pool must hold both at once.
CODE_MIN_CONTEXT = 70000


# --- model loading ----------------------------------------------------------

# OpenVINO CACHE_DIR: compiled-model blobs, so a restart reloads instead of
# recompiling. Measured on Qwen3-8B/NPU: 65.1 s cold → 9.3 s cached.
# --no-model-cache disables it.
MODEL_CACHE_DIR = None
# --model-cache-gb: upper bound on .ov-cache. It had none, and an unbounded
# compile cache measured 41.9 GB here (36.1 GB of it untouched for 14 days).
# Evicting an entry costs one cold compile of that model, nothing more.
MODEL_CACHE_GB = 12.0

# Percentage of MoE expert weights streamed from disk on GPU, or "auto".
# Needs an XMX-capable GPU (Arc / Lunar Lake+) — a silent no-op without one.
# Measured on Arc 140V: 30B-A3B int4 runs in 2.35 GB resident at ratio 90.
OFFLOAD_RATIO = "auto"

# System-RAM headroom withheld from an integrated GPU's usable budget. This is
# mutable because main() resolves --gpu-reserve-gb after argument parsing; all
# consumers must read config.GPU_RESERVE_BYTES rather than importing the name.
GPU_RESERVE_BYTES = 3 * 2 ** 30


# --- prewarm ----------------------------------------------------------------

# Path (--prewarm) to a saved prompt, prefilled at startup and auto-captured
# while serving, so even the first agent turn is a cache hit rather than a cold
# prefill that can trip a client's idle watchdog.
PREWARM_FILE = None
PREWARM_MIN_CHARS = 4000   # only capture agent-sized system prompts, not chat


# --- tool calling -----------------------------------------------------------

AGENT_TOOLS = None   # --agent-tools: set of names to keep, or None (all)

# Measured on Qwen3-8B-int4-cw's own tokenizer, 2026-08-29, against the 8192
# cap (scripts/measure-tool-budget.py reproduces it):
#
#   assistant tool set,  8 tools    735 tokens    9.0% of the window
#   trimmed to 5 tools              544 tokens    6.6%
#   coding agent,       30 tools   4553 tokens   55.6%
#
# That spread is the whole argument. The NPU is excluded from a client's tools
# on two grounds — the prompt cap and multi-step planning — and the first is
# arithmetic nobody had done. An assistant tool set costs less than a tenth of
# the window; a coding agent's costs more than half. Refusing both throws away
# the case that fits in order to prevent the case that does not.
#
# 1200 leaves the 8-tool set 40% headroom while rejecting a 30-tool catalogue
# outright, and it caps the SCHEMA BLOCK only — the conversation still has to
# fit under NPU_MAX_PROMPT_LEN, which is checked where it always was.
NPU_TOOL_BUDGET = 1200


# --- server-side Python tool ------------------------------------------------

# Off by default: this runs code. The caps are defence-in-depth against a small
# model's mistakes and against prompt-injected pages, NOT a hostile-code
# sandbox on Windows.
PYTHON_TOOL_TIMEOUT = 5.0             # hard wall-clock limit per child process
PYTHON_TOOL_OUTPUT_BYTES = 64 * 1024  # stdout/stderr cap, per stream
PYTHON_TOOL_MAX_CODE_BYTES = 32 * 1024


# --- Odysseus ---------------------------------------------------------------
# Odysseus is the assistant layer (docs/ODYSSEUS.md); locally is the engine it
# talks to. Both run on the same box, so locally can bring it up — but only on
# request. Autostart is OFF by default because `docker compose up -d` starts
# four containers holding the user's calendar, mail and notes; an inference
# server does not get to do that because it happened to be launched.

ODYSSEUS_AUTOSTART = False
ODYSSEUS_PORT = 7000     # Odysseus's own APP_PORT default
ODYSSEUS_DIR = None      # --odysseus-dir; None means search (env, sibling, ~)

# A first run builds images from source (the compose service is `build: .`,
# there is no tag to pull), which is minutes, not seconds. The autostart runs
# on a background thread precisely so this number cannot delay serving chat.
ODYSSEUS_START_TIMEOUT = 180

# --container-engine: pin 'docker' or 'podman'. None means take whichever is
# on PATH, Docker first. Odysseus's compose file declares
# extra_hosts: host.docker.internal:host-gateway explicitly, so the one
# Docker-shaped dependency in this architecture is honoured by Podman too.
CONTAINER_ENGINE = None


# --- coding mode ------------------------------------------------------------
# While on, the utility and audio paths are refused rather than quietly
# competing for the device holding the coder.
CODING_MODE = False
