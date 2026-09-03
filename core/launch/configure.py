"""Apply persisted setup and command-line runtime configuration."""

import os

import flask.cli

from core import config, odysseus
from core.hardware.devices import _gpu_has_xmx
from core.models.describe import scan_models
from core.system.setup import _read_setup_config
from core.web import search as web_search_mod


def configure(args):

    # First-run choices live outside the repository and only fill values the
    # caller did not explicitly provide. Command-line flags remain the final
    # authority, while the launcher can become one-click after onboarding.
    saved_setup = _read_setup_config()
    saved_assistant = saved_setup.get("assistant") or {}
    default_model = os.path.abspath(os.path.join(config.SCRIPT_DIR, "model"))
    using_default_model = (os.path.abspath(os.path.expanduser(args.model_dir)) ==
                           default_model)
    if not args.proxy_url and using_default_model:
        if saved_assistant.get("backend") == "ollama":
            args.proxy_url = saved_assistant.get("url") or "http://127.0.0.1:11434"
            args.proxy_model = saved_assistant.get("model") or args.proxy_model
        elif (saved_assistant.get("backend") == "openvino" and
              os.path.isdir(saved_assistant.get("model_dir") or "")):
            args.model_dir = saved_assistant["model_dir"]
            args.device = saved_assistant.get("device") or args.device
    saved_voice = saved_setup.get("voice") or {}
    if not args.whisper_dir and os.path.isdir(saved_voice.get("stt_dir") or ""):
        args.whisper_dir = saved_voice["stt_dir"]
    if not args.tts_dir and os.path.isdir(saved_voice.get("tts_dir") or ""):
        args.tts_dir = saved_voice["tts_dir"]
    if not args.vad_dir and os.path.isdir(saved_voice.get("vad_dir") or ""):
        args.vad_dir = saved_voice["vad_dir"]
    if not args.speaker_dir and os.path.isdir(saved_voice.get("speaker_dir") or ""):
        args.speaker_dir = saved_voice["speaker_dir"]
    saved_web = saved_setup.get("web") or {}
    if saved_web.get("check_updates"):
        args.check_updates = True
    if not args.search_url and saved_web.get("searxng"):
        args.search_url = saved_web.get("url") or "http://127.0.0.1:8080"
        if not args.searxng_root:
            args.searxng_root = saved_web.get("root")

    # --scan is a report, not a server: no ports, no devices, no model load.
    if args.scan is not None:
        print(flush=True)
        scan_models(args.scan)
        return None

    model_dir = os.path.expanduser(args.model_dir)
    config.SERVER_PORT = args.port
    config.CHECK_UPDATES = bool(args.check_updates)
    config.AGENT_TOOLS = ({t.strip() for t in args.agent_tools.split(",") if t.strip()}
                   if args.agent_tools else None)
    config.MAX_IMAGE_DIM = args.max_dim
    config.DEBUG_REQUESTS = args.debug
    config.VSCODE_COMPAT = args.vscode_compat
    # Tri-state: explicit flags win, otherwise per-device (see config.PROMPT_CACHE).
    config.PROMPT_CACHE = (False if args.no_prompt_cache
                    else (True if args.prompt_cache else None))
    config.KV_PRECISION = args.kv_precision
    if args.npu_prompt_len:
        if args.npu_prompt_len > 8192:
            print(f"WARNING: --npu-prompt-len {args.npu_prompt_len} is above the "
                  f"measured vpux ceiling of 8192 and will fail to compile; "
                  f"using 8192", flush=True)
        config.NPU_MAX_PROMPT_LEN = min(8192, max(1024, args.npu_prompt_len))
    config.SEARXNG_IDLE = max(0, int(args.searxng_idle))
    root = (args.searxng_root or "").strip()
    if not root:
        # Convention over configuration: a sibling checkout is the layout
        # scripts/searxng.ps1 -Install produces.
        for guess in (os.path.join(os.path.dirname(config.SCRIPT_DIR), "searxng-src"),
                      os.path.join(config.SCRIPT_DIR, "searxng-src")):
            if os.path.isdir(guess):
                root = guess
                break
    config.SEARXNG_ROOT = root if root and os.path.isdir(root) else None
    # Assign the MODULE attribute, never a local copy. core/web/search.py owns
    # this value: it is what `web_search_status()` reads and what /health
    # reports. locally.py used to keep its own module-level WEB_SEARCH_URL and
    # set that instead, so `--search-url` was accepted, printed at startup, and
    # then silently ignored -- /health said "start with --search-url to enable
    # web search" on a server that had been started with exactly that flag.
    # This is the split trap CLAUDE.md records, in its mirror form: the
    # variable moved into core/ and the assignment stayed behind.
    # NOT `web_search` -- that name is the Flask view function defined
    # below, so importing the module under it was silently shadowed and
    # the assignment set an attribute on a function object. The symptom
    # was identical to having made no fix at all.
    web_search_mod.WEB_SEARCH_URL = (args.search_url or "").strip() or None
    if web_search_mod.WEB_SEARCH_URL:
        print(f"  Web search via {web_search_mod.WEB_SEARCH_URL} "
              f"(the only outbound network path this server has)", flush=True)
    config.CONTAINER_ENGINE = args.container_engine
    config.ODYSSEUS_PORT = int(args.odysseus_port or config.ODYSSEUS_PORT)
    config.ODYSSEUS_DIR = (os.path.abspath(os.path.expanduser(args.odysseus_dir))
                           if args.odysseus_dir else None)
    # None means the flag was not given, so fall back to what the settings
    # toggle saved. An explicit --odysseus-autostart / --no- always wins, the
    # same precedence locally.ini.example states for every other setting.
    config.ODYSSEUS_AUTOSTART = (bool(args.odysseus_autostart)
                                 if args.odysseus_autostart is not None
                                 else odysseus.load_autostart())
    config.PYTHON_TOOL_ENABLED = bool(args.python_tool)
    config.PYTHON_TOOL_TIMEOUT = max(0.1, min(60.0, float(args.python_timeout)))
    config.PYTHON_TOOL_OUTPUT_BYTES = max(1024, min(16 * 1024 * 1024,
                                              int(args.python_output_bytes)))
    if config.PYTHON_TOOL_ENABLED:
        print(f"  Python calculations enabled (child timeout "
              f"{config.PYTHON_TOOL_TIMEOUT:g}s, output cap "
              f"{config.PYTHON_TOOL_OUTPUT_BYTES} bytes)", flush=True)
    config.PROMPT_CACHE_GB = args.cache_size_gb
    if args.context_tokens is not None:
        raw = str(args.context_tokens).strip().lower()
        if raw == "auto":
            config.CONTEXT_TOKENS = "auto"
        else:
            try:
                config.CONTEXT_TOKENS = int(raw.replace("k", "000").replace(",", ""))
            except ValueError:
                print(f"WARNING: --context-tokens '{args.context_tokens}' is not "
                      f"a number or 'auto' — ignoring.", flush=True)
    raw_offload = str(args.offload_ratio).strip().lower()
    if raw_offload == "auto":
        config.OFFLOAD_RATIO = "auto"
    else:
        try:
            config.OFFLOAD_RATIO = max(0, min(99, int(raw_offload)))
        except ValueError:
            print(f"WARNING: --offload-ratio '{args.offload_ratio}' is not a "
                  f"number or 'auto' — using auto.", flush=True)
            config.OFFLOAD_RATIO = "auto"
    # A pinned ratio on a GPU without XMX is silently ignored by the plugin —
    # say so up front instead of letting the user believe their model got
    # smaller (see TODONT.md). 'auto' needs no warning: it checks for XMX
    # itself and simply stays off.
    if config.OFFLOAD_RATIO not in ("auto", 0) and not _gpu_has_xmx():
        print("WARNING: --offload-ratio set, but this GPU has no XMX "
              "(OPTIMIZATION_CAPABILITIES lacks GPU_HW_MATMUL). MoE disk "
              "offload will silently do nothing — the model must fit in "
              "GPU memory.", flush=True)
    if args.prewarm:
        config.PREWARM_FILE = os.path.expanduser(args.prewarm)
    elif args.idle_timeout == 0 and not args.no_prewarm and config.PROMPT_CACHE:
        # Agent/server mode (--idle-timeout 0 keeps models loaded forever) is
        # exactly where prewarm pays off — and the idle unload that would
        # throw the warmed cache away can't happen, so turn it on. Port-scoped
        # filename so two instances on one install don't overwrite each other.
        config.PREWARM_FILE = os.path.join(config.SCRIPT_DIR, f"prewarm-{args.port}.json")
    else:
        config.PREWARM_FILE = None

    # Quiet Flask/Werkzeug startup noise: kills the "Serving Flask app" /
    # "Debug mode: off" / "Running on http://..." / "Press CTRL+C to quit"
    # block (printed twice — once per Flask app), the dev-server warning,
    # and per-request access logs that would otherwise flood the console
    # in normal use. locally.py has its own app-level request logging.
    import logging
    import flask.cli
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    flask.cli.show_server_banner = lambda *a, **k: None

    print(flush=True)

    return model_dir
