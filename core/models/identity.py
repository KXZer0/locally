"""What a model directory IS: vision or text, generative or audio, and what to
call it.

The directory name is authoritative -- it is the web-UI label and the model
ID clients request -- so a symlink is followed only when that name carries no
information. Calling realpath unconditionally is what once discarded a
deliberate rename (#19)."""

import json
import os
from pathlib import Path


def is_vlm(model_dir):
    """Detect if a model is a VLM.

    The definitive signal is structural: OpenVINO exports a VLM as a
    multi-component model with a separate vision encoder alongside the language
    model, whereas a text-only LLM is a single openvino_model.xml. Match *any*
    vision-encoder component file rather than one exact name — the filenames
    vary across generations (Qwen3.5 ships three:
    openvino_vision_embeddings_model.xml, ..._merger_model.xml, ..._pos_model.xml;
    LLaVA-style exports use image_encoder), and a single hard-coded name would
    miss a future variant. Architecture-name sniffing misses new generations
    too — e.g. Qwen3.5 reports Qwen3_5ForConditionalGeneration / qwen3_5,
    matching none of the keys below — so check the files first and fall back to
    the config keys.
    """
    try:
        for fn in os.listdir(model_dir):
            low = fn.lower()
            if low.endswith(".xml") and ("vision" in low or "image_encoder" in low):
                return True
    except OSError:
        pass
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        arch = cfg.get("architectures", [""])[0].lower()
        model_type = cfg.get("model_type", "").lower()
        return any(
            k in arch or k in model_type
            for k in ("vl", "vision", "llava", "qwen2vl", "internvl", "minicpm",
                      "multimodal", "image_text", "got_ocr")
        )
    except Exception:
        return False


# Suffixes describing the *export* rather than the model, dropped from the
# display name (and so from the model ID clients configure).
_NAME_SUFFIXES = ("-ov", "-openvino", "-int8", "-int4")


# Directory names carrying no information. install.ps1 links model/ -> the
# real model directory, so one of these means "look through the link".
# Deliberately excludes "whisper-model": it's a real directory name people
# download into, and treating it as generic would rename that model's API ID
# from "whisper-model" to "whisper" for no gain.
_GENERIC_DIR_NAMES = ("model", "models", "gpu-model", "npu-model", "")


def _strip_name_suffixes(name):
    for suffix in _NAME_SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name


def resolve_display_name(model_dir):
    """Return (name, why) for the name locally shows and clients request.

    The directory name as given wins. Only when it carries no information
    (install.ps1 links a generic model/ at the real directory) do we follow
    the link to find a real name. Resolving symlinks unconditionally — what
    this did before #19 — silently threw away a deliberate rename, so
    renaming a model folder appeared to have no effect at all. Renaming the
    directory is the one naming interface that needs no documentation, so it
    has to work.
    """
    given = _strip_name_suffixes(
        os.path.basename(os.path.normpath(os.path.abspath(model_dir))))
    if given.lower() not in _GENERIC_DIR_NAMES:
        return given, "directory name"

    target = _strip_name_suffixes(
        os.path.basename(os.path.normpath(os.path.realpath(model_dir))))
    if target.lower() not in _GENERIC_DIR_NAMES:
        return target, "link target (directory name is generic)"

    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            model_type = json.load(f).get("model_type")
        if model_type:
            return model_type, "config.json model_type (no usable directory name)"
    except Exception:
        pass
    return "unknown", "nothing identifiable found"


def model_display_name(model_dir):
    """Human-readable model name — see resolve_display_name()."""
    return resolve_display_name(model_dir)[0]


def _is_model_dir(path):
    return any(os.path.isfile(os.path.join(path, f)) for f in
               ("openvino_model.xml", "openvino_language_model.xml",
                "openvino_encoder_model.xml"))


def _is_generative_dir(path):
    """True for chat/vision models — false for the audio ones.

    Whisper and Kokoro live in the same models folder and look like model
    dirs, but they belong in the ASR/TTS slots. Offering them as chat models
    would just be a load that fails a minute later.
    """
    if os.path.isdir(os.path.join(path, "voices")):
        return False                                   # Kokoro-style TTS
    try:
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
    except Exception:
        # The VAD ships as a bare .onnx with no config.json, so "can't tell"
        # would otherwise offer it as a chat model.
        if any(Path(path).glob("silero*")):
            return False
        return True                                    # can't tell — let it try
    if cfg.get("model_type") in ("whisper", "speecht5", "kokoro"):
        return False
    archs = " ".join(cfg.get("architectures") or []).lower()
    return not any(k in archs for k in
                   ("whisper", "speecht5", "kmodel", "texttospeech"))
