"""Load slots in dependency order and run both Flask servers."""

import logging
import os
import threading
import time

from core import config, odysseus, runtime
from core.startup import _idle_watchdog, _load_in_background
from core.launch.quick_chat import start_shift_enter_chat


def _autostart_odysseus(api_port):
    """Bring Odysseus up (or adopt it) and hand it this server's address.

    Module-level rather than a closure so it can be tested without standing up
    two Flask apps and every slot.

    **The api_port argument is the fix, not a tidy-up.** This path used to call
    `start(timeout=...)` and nothing else, so `_sync_running_endpoint` hit its
    `api_port is None` early return, answered "adopted", and changed nothing --
    the endpoint repair existed only for a human typing `odysseus.ps1`. An
    Odysseus that was already running against a moved WSL gateway therefore
    stayed pointed at a dead address, which is exactly the state the user
    reported ("odysseus cant reach my endpoint").

    The probe at the end is the only honest confirmation: a host-side request
    proves locally can call itself, while this crosses the Compose network from
    inside the container the user actually runs.
    """
    found = odysseus.find_dir(config.ODYSSEUS_DIR)
    if not found:
        print("  [odysseus] no checkout found - nothing to start "
              "(set --odysseus-dir or $ODYSSEUS_DIR)", flush=True)
        return
    endpoint = odysseus.api_base_url(api_port)
    print(f"  [odysseus] starting {found}; its endpoint for this server is "
          f"{endpoint} (a first run builds images and takes minutes)",
          flush=True)
    began = time.time()
    ok, detail = odysseus.start(timeout=config.ODYSSEUS_START_TIMEOUT,
                                api_port=api_port)
    took = time.time() - began
    if not ok:
        print(f"  [odysseus] not started ({took:.1f}s): {detail}", flush=True)
        return
    print(f"  [odysseus] {detail} - {odysseus.url()} ({took:.1f}s)", flush=True)
    link = odysseus.probe_api(api_port)
    if link.get("reachable"):
        print(f"  [odysseus] container reached {endpoint}: "
              f"{len(link.get('models') or [])} model(s)", flush=True)
    else:
        print(f"  [odysseus] container could NOT reach {endpoint}: "
              f"{link.get('detail')} - run odysseus.ps1, which repairs the "
              f"scoped firewall rule", flush=True)


def serve(app, ollama_app, args, model_dir, devices, setup_only, all_slots):
    # 6. Start Flask, load models in background
    ports_msg = f"port {args.port}"
    if args.ollama_port:
        ports_msg += f" + Ollama on {args.ollama_port}"
    print(f"  Starting server on {ports_msg}...", flush=True)

    threads = []

    # --- Load order: the chat model first, the utilities after --------------
    #
    # Every slot used to start at once, and they are not equal: the user is
    # waiting for the CHAT model and nothing else, while OCR compiling on two
    # engines competes with it for the same cores and the same memory
    # bandwidth. Measured on the 358H with a warm compile cache, same machine,
    # back to back: everything concurrent reached "locally ready" in **15.43 s**;
    # with the utility slots held back it was **9.97 s**. The utilities cost
    # 5.5 s of the wait for a model they have nothing to do with.
    #
    # They are only DEFERRED, never dropped -- the wait is capped so a model
    # that loads slowly, or fails outright, cannot leave the Tools tab dead
    # forever, and the event is set in a `finally` so an exception releases it
    # too. On a proxy slot "loading" is one HTTP probe, so the wait is
    # negligible there, which matters because the utilities are the whole
    # reason locally runs on the NVIDIA machine.
    _primary_ready = threading.Event()
    _UTIL_DEFER_CAP = 120.0

    def _load_primary_then_release(*load_args):
        try:
            _load_in_background(*load_args)
        finally:
            _primary_ready.set()

    def _load_after_primary(*load_args):
        if not _primary_ready.wait(timeout=_UTIL_DEFER_CAP):
            print("  [util] chat model still loading after "
                  f"{_UTIL_DEFER_CAP:.0f}s; loading utilities anyway", flush=True)
        _load_in_background(*load_args)

    if args.proxy_url or not setup_only:
        t = threading.Thread(
            target=_load_primary_then_release,
            args=(runtime.primary, model_dir, devices, args.port, args.ollama_port, all_slots),
            daemon=True,
        )
        threads.append(t)
    else:
        print(f"  API: http://localhost:{args.port}/v1", flush=True)
        # Nothing is going to set this, and the utilities are the only thing
        # this mode can actually offer.
        _primary_ready.set()

    if runtime.secondary:
        t2 = threading.Thread(
            target=_load_in_background,
            args=(runtime.secondary, args.gpu_model_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        )
        threads.append(t2)

    if runtime.whisper_slot:
        tw = threading.Thread(
            target=_load_in_background,
            args=(runtime.whisper_slot, args.asr_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        )
        threads.append(tw)

    if runtime.tts_slot:
        tt = threading.Thread(
            target=_load_in_background,
            args=(runtime.tts_slot, args.tts_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        )
        threads.append(tt)

    if runtime._vad_slot:
        threads.append(threading.Thread(
            target=_load_in_background,
            args=(runtime._vad_slot, args.vad_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        ))

    if runtime._speaker_slot:
        threads.append(threading.Thread(
            target=_load_in_background,
            args=(runtime._speaker_slot, args.speaker_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        ))

    for util in (runtime.util_npu, runtime.util_gpu, runtime.util_cpu):
        if util:
            threads.append(threading.Thread(
                target=_load_after_primary,
                args=(util, runtime.UTIL_DIR, devices, args.port,
                      args.ollama_port, all_slots),
                daemon=True,
            ))

    for t in threads:
        t.start()

    # Odysseus autostart — on its own thread, never inline.
    #
    # `docker compose up -d` on a cold checkout builds the image from source
    # (the compose service is `build: .`), which is minutes. Doing that before
    # app.run() would mean locally refuses chat while waiting on an unrelated
    # program, so the whole thing — the search, the Docker probe, the build —
    # happens behind the server that is already answering.
    if config.ODYSSEUS_AUTOSTART:
        threading.Thread(target=_autostart_odysseus, args=(args.port,),
                         daemon=True).start()

    # Idle watchdog — unload models after inactivity
    if args.idle_timeout > 0:
        print(f"  Idle unload after {args.idle_timeout}s of inactivity", flush=True)
        if config.PREWARM_FILE:
            # The prefix cache lives in the pipeline; unload drops both and the
            # reload path doesn't re-warm (a synchronous re-warm would stall
            # the triggering request for the whole prefill).
            print(f"  WARNING: --prewarm + idle unload: the warmed cache is lost "
                  f"when the model idle-unloads and not rebuilt until restart. "
                  f"Use --idle-timeout 0 to keep it.", flush=True)
        watchdog = threading.Thread(
            target=_idle_watchdog,
            args=(all_slots, args.idle_timeout),
            daemon=True,
        )
        watchdog.start()
    elif config.PREWARM_FILE and not args.prewarm:
        print(f"  Prewarm auto-enabled (--idle-timeout 0): "
              f"{os.path.basename(config.PREWARM_FILE)} (--no-prewarm to disable)", flush=True)

    # Suppress Flask's default "Serving Flask app" banner — we have our own
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    # Start Ollama API on separate port in background thread
    if args.ollama_port:
        print(f"  Ollama API on port {args.ollama_port}", flush=True)
        def _run_ollama():
            try:
                ollama_app.run(
                    host=args.host, port=args.ollama_port, threaded=True,
                )
            except Exception as e:
                print(f"  WARNING: Ollama API failed to start: {e}", flush=True)
        ollama_thread = threading.Thread(target=_run_ollama, daemon=True)
        ollama_thread.start()

    # OpenAI API on main thread
    print(f"  OpenAI API on port {args.port}", flush=True)
    # Say the posture out loud. The failure this replaces was a firewall rule
    # that had silently stopped being in force, with nothing on screen to say so.
    from core import netguard
    print(f"  {netguard.describe(args.host)}", flush=True)
    print(f"  Optional chat: python locally.py chat --port {args.port}", flush=True)
    start_shift_enter_chat(args.port)
    print("  Ctrl+C stops this server and releases its models · "
          "Shift+Enter opens terminal chat", flush=True)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
