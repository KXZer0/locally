"""Validate paths, discover devices, and construct runtime slots."""

import os
import sys

from core import cachetrim, config, runtime
from core.routes import audio as audio_routes
from core.slots.device import DeviceSlot
from core.slots.proxy import ProxySlot
from core.slots.speaker import SpeakerSlot
from core.slots.tts import TtsSlot
from core.slots.util import UtilSlot
from core.slots.vad import VadSlot
from core.slots.whisper import WhisperSlot
from core.startup import check_port, detect_devices


def build_slots(args, model_dir):
    # 1. Check ports
    if not check_port(args.port):
        print(f"ERROR: Port {args.port} is already in use.")
        print(f"Use --port <number> to pick another port.")
        sys.exit(1)
    if args.ollama_port and not check_port(args.ollama_port):
        print(f"WARNING: Ollama port {args.ollama_port} is in use. "
              f"Ollama API disabled. (Is Ollama already running?)")
        args.ollama_port = 0
    config.OLLAMA_COMPAT_PORT = args.ollama_port

    # 2. Detect devices
    devices = detect_devices()
    runtime.DEVICES.update(devices)   # runtime placement can then move a slot to any device
    print("  Devices:", flush=True)
    for kind, info in devices.items():
        suffix = f" [{info['id']}]" if info['id'] != kind else ""
        print(f"    {kind}{suffix}: {info['name']}")
    print()

    def _id_of(kind):
        return devices.get(kind, {}).get("id", kind)

    # 3. Resolve primary device
    device = args.device.upper()
    if device == "AUTO":
        if args.gpu_model_dir:
            # Dual mode: primary goes on NPU (or CPU if no NPU)
            device = "NPU" if "NPU" in devices else "CPU"
        elif "NPU" in devices:
            device = "NPU"
        elif "GPU" in devices:
            device = "GPU"
        else:
            device = "CPU"

    if device not in devices and device != "CPU":
        print(f"ERROR: Device {device} not available. Found: {list(devices.keys())}")
        sys.exit(1)

    # 4. Verify model directories
    setup_only = not args.proxy_url and not os.path.isdir(model_dir)
    if setup_only:
        print(f"  No assistant model found at {model_dir}.")
        print("  API available without a chat model. Use --model-dir, "
              "POST /v1/models/load, or /load in terminal chat.", flush=True)
    if args.gpu_model_dir and not os.path.isdir(args.gpu_model_dir):
        print(f"ERROR: GPU model directory not found: {args.gpu_model_dir}")
        sys.exit(1)
    if args.whisper_dir and not os.path.isdir(args.whisper_dir):
        print(f"ERROR: Whisper model directory not found: {args.whisper_dir}")
        sys.exit(1)
    if args.tts_dir and not os.path.isdir(args.tts_dir):
        print(f"ERROR: TTS model directory not found: {args.tts_dir}")
        sys.exit(1)
    if args.util_models_dir:
        if not os.path.isdir(args.util_models_dir):
            print(f"ERROR: --util-models-dir not found: {args.util_models_dir}")
            sys.exit(1)
        runtime.UTIL_DIR = os.path.abspath(args.util_models_dir)
    elif not args.no_util and args.auto_util:
        # Find the utility models without being told where they are. An
        # explicit flag meant every existing launch script silently omitted it,
        # so the models sat on disk while the UI reported "not installed" —
        # a setup error dressed up as a missing download. Detection is cheap
        # and unambiguous: the OCR pair is what the Util tab needs at minimum.
        for cand in (os.path.join(os.path.expanduser("~"), "models", "util"),
                     os.path.join(config.SCRIPT_DIR, "models", "util"),
                     os.path.join(config.SCRIPT_DIR, "util-models")):
            if all(os.path.isfile(os.path.join(cand, sub, "inference.onnx"))
                   for sub in ("ocr-det", "ocr-rec")):
                runtime.UTIL_DIR = os.path.abspath(cand)
                print(f"  Utilities: found models in {runtime.UTIL_DIR}", flush=True)
                break
    if args.models_dir:
        if not os.path.isdir(args.models_dir):
            print(f"ERROR: --models-dir not found: {args.models_dir}")
            sys.exit(1)
        runtime.MODELS_DIR = os.path.abspath(args.models_dir)
    elif setup_only:
        # The setup downloader needs one stable destination even before a
        # first model exists. Do not create it until the user clicks Install.
        runtime.MODELS_DIR = os.path.abspath(os.path.expanduser("~/models"))

    if not args.no_model_cache:
        config.MODEL_CACHE_DIR = os.path.abspath(
            args.model_cache_dir or os.path.join(config.SCRIPT_DIR, ".ov-cache"))
        try:
            os.makedirs(config.MODEL_CACHE_DIR, exist_ok=True)
        except OSError as e:
            print(f"  WARNING: can't use model cache ({e}); compiling every start")
            config.MODEL_CACHE_DIR = None
        # Trim BEFORE anything loads: evicting a blob mid-compile would turn a
        # cache miss into a failure.
        if config.MODEL_CACHE_DIR:
            config.MODEL_CACHE_GB = float(args.model_cache_gb or 0)
            cachetrim.trim(config.MODEL_CACHE_DIR, config.MODEL_CACHE_GB)

    # 5. Create device slots
    if args.proxy_url:
        runtime.primary = ProxySlot(args.proxy_url, model=args.proxy_model,
                            api_key=args.proxy_key)
        print(f"  Primary: proxied to {args.proxy_url}"
              + (f" ({args.proxy_model})" if args.proxy_model else ""), flush=True)
    else:
        runtime.primary = DeviceSlot(device, _id_of(device))
    all_slots = [runtime.primary]
    runtime.slots[:] = all_slots

    if args.gpu_model_dir:
        if "GPU" not in devices:
            print("WARNING: --gpu-model-dir given but no GPU detected. Ignoring.")
        else:
            runtime.secondary = DeviceSlot("GPU", _id_of("GPU"))
            all_slots.append(runtime.secondary)

    if args.whisper_dir:
        whisper_device = args.whisper_device.upper()
        if whisper_device not in devices and whisper_device != "CPU":
            print(f"WARNING: Whisper device {whisper_device} not available, falling back to CPU.")
            whisper_device = "CPU"
        runtime.whisper_slot = WhisperSlot(whisper_device, _id_of(whisper_device))
        all_slots.append(runtime.whisper_slot)

    if args.tts_dir:
        tts_device = args.tts_device.upper()
        if tts_device not in devices and tts_device != "CPU":
            print(f"WARNING: TTS device {tts_device} not available, falling back to CPU.")
            tts_device = "CPU"
        runtime.tts_slot = TtsSlot(tts_device, _id_of(tts_device))
        all_slots.append(runtime.tts_slot)

    if args.vad_dir:
        vad_device = args.vad_device.upper()
        if vad_device not in devices and vad_device != "CPU":
            print(f"WARNING: VAD device {vad_device} not available, falling back to CPU.")
            vad_device = "CPU"
        runtime._vad_slot = VadSlot(vad_device, _id_of(vad_device))
        all_slots.append(runtime._vad_slot)
        if not audio_routes.stream_available:
            print("WARNING: --vad-dir given but flask-sock is not installed, so "
                  "auto turn-taking has no socket to run over. "
                  "`pip install flask-sock`. Voice falls back to push-to-talk.")

    if args.speaker_dir:
        spk_device = args.speaker_device.upper()
        if spk_device not in devices and spk_device != "CPU":
            print(f"WARNING: speaker device {spk_device} not available, "
                  f"falling back to CPU.")
            spk_device = "CPU"
        runtime._speaker_slot = SpeakerSlot(spk_device, _id_of(spk_device),
                                    threshold=args.speaker_threshold)
        all_slots.append(runtime._speaker_slot)
        if not args.vad_dir:
            # Verification runs on the utterance the VAD delimits. Without the
            # VAD there are no server-side turns to gate -- push-to-talk is
            # already the user deciding when to speak.
            print("WARNING: --speaker-dir without --vad-dir does nothing: "
                  "speaker ID gates the VAD's turns, and push-to-talk has none.")

    if runtime.UTIL_DIR:
        # "auto" means every engine this box actually has. It is the default
        # because the tier map now contains GPU-only models (the vpux compiler
        # rejects Swin2SR), and with an NPU-only default that tier had no slot
        # to run on — the capability existed and nothing could reach it. Slots
        # are cheap: only OCR loads eagerly (~160 MB), everything else compiles
        # on first use and idle-unloads afterwards.
        auto = args.util_engines.strip().lower() == "auto"
        # Under "auto", CPU is the fallback rather than an addition: it is only
        # taken when the box has no Intel accelerator at all. That is the
        # machine the proxy slot exists for -- an NVIDIA desktop where Ollama
        # generates and locally supplies the UI and the utilities. Adding a CPU
        # slot alongside an NPU would cost real memory to duplicate models that
        # already have a faster home.
        accel = [e for e in ("NPU", "GPU") if e in devices]
        wanted = ((accel or ["CPU"]) if auto else
                  [e.strip().upper() for e in args.util_engines.split(",") if e.strip()])
        for engine in wanted:
            if engine not in ("NPU", "GPU", "CPU"):
                print(f"WARNING: unknown --util-engines value {engine!r}, ignoring.")
                continue
            if engine not in devices:
                # Silent under "auto" — asking for what the machine has cannot
                # be a mistake worth warning about on every start.
                if not auto:
                    print(f"WARNING: --util-engines asked for {engine} but none "
                          f"was detected. Utilities will not be available there.")
                continue
            slot = UtilSlot(engine, _id_of(engine))
            if engine == "NPU":
                runtime.util_npu = slot
            elif engine == "GPU":
                runtime.util_gpu = slot
            else:
                runtime.util_cpu = slot
            all_slots.append(slot)

    return devices, setup_only, all_slots
