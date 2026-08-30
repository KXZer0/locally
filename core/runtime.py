"""What the server currently HAS: the loaded slots, and where things live.

Distinct from core/config.py, which is what the user ASKED FOR. Config is
settled once when the command line is parsed; this changes while the server
runs -- a model swap re-points `primary`, the idle watchdog empties slots, and
the setup wizard can fill in a directory that was not known at startup.

**Import the module, never the names.** `from core.runtime import primary`
captures None at import time and never sees the model that loads a second
later:

    from core import runtime
    if runtime.primary and runtime.primary.status == "ready":
        ...
"""

# --- generative slots -------------------------------------------------------
# One model at a time is the default: the NPU has no memory of its own, so a
# second resident model costs real RAM and heat for a model you cannot talk to
# concurrently anyway. Measured: two models left ~2 GB free of 31.5; one leaves
# ~14 GB.
primary = None      # main model (NPU, GPU, CPU, or REMOTE via --proxy-url)
secondary = None    # optional second model (GPU, for vision or a bigger LLM)

# --- audio ------------------------------------------------------------------
whisper_slot = None    # speech to text
tts_slot = None        # text to speech
_vad_slot = None       # Silero: is anyone speaking
_speaker_slot = None   # ECAPA: is it the enrolled user

# --- utilities --------------------------------------------------------------
# Under --util-engines auto, CPU is a FALLBACK rather than an addition: it is
# taken only when no Intel accelerator is present, which is the machine
# --proxy-url exists for.
util_npu = None
util_gpu = None
util_cpu = None

# --- discovered at startup --------------------------------------------------
# detect_devices() result, so runtime placement can consider every device the
# machine has rather than only the ones that already hold a slot.
DEVICES = {}
UTIL_DIR = None      # where the utility models were found
MODELS_DIR = None    # --models-dir: where POST /v1/models/load looks
