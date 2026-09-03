"""Load slots in dependency order and run both Flask servers."""

import logging
import os
import threading
import time

from core import config, odysseus, runtime
from core.startup import _idle_watchdog, _load_in_background


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
        print(f"  Setup UI: http://localhost:{args.port}", flush=True)
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
            args=(runtime.whisper_slot, args.whisper_dir, devices, args.port,
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
        def _start_odysseus():
            found = odysseus.find_dir(config.ODYSSEUS_DIR)
            if not found:
                print("  [odysseus] no checkout found - nothing to start "
                      "(set --odysseus-dir or $ODYSSEUS_DIR)", flush=True)
                return
            print(f"  [odysseus] starting {found} (first run builds images; "
                  f"this can take minutes)", flush=True)
            began = time.time()
            ok, detail = odysseus.start(timeout=config.ODYSSEUS_START_TIMEOUT)
            took = time.time() - began
            if ok:
                print(f"  [odysseus] {detail} - {odysseus.url()} ({took:.1f}s)",
                      flush=True)
            else:
                print(f"  [odysseus] not started ({took:.1f}s): {detail}",
                      flush=True)
        threading.Thread(target=_start_odysseus, daemon=True).start()

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
                    host="0.0.0.0", port=args.ollama_port, threaded=True,
                )
            except Exception as e:
                print(f"  WARNING: Ollama API failed to start: {e}", flush=True)
        ollama_thread = threading.Thread(target=_run_ollama, daemon=True)
        ollama_thread.start()

    # OpenAI API on main thread
    print(f"  OpenAI API on port {args.port}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True)
