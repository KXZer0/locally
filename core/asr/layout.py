"""Which speech-to-text engine a directory holds.

`--asr-dir` takes either engine and this decides, so the user names a folder
rather than learning which flag goes with which model family:

  * Parakeet: the FluidInference OpenVINO export, read from FluidInference's
    own runtime (eddy-audio, `PARAKEET_STANDARD_FILES`):

        parakeet_melspectogram.xml/.bin   audio -> log-mel (upstream spelling)
        parakeet_encoder.xml/.bin         FastConformer encoder
        parakeet_decoder.xml/.bin         LSTM predictor
        parakeet_joint.xml/.bin           joint: token + duration heads
        parakeet_vocab.json               SentencePiece pieces

    Other exporters name the parts differently, so the match is by role
    (`*encoder*.xml`, `*decoder*.xml`, `*joint*.xml`) and the exact names are
    only a preference. The preprocessor is optional (core/asr/mel.py).
  * Whisper: an optimum-intel export, recognised by config.json or by the
    openvino_encoder_model + openvino_decoder_model pair.

The fused `decoder_joint-model.onnx` layout (onnx-asr, sherpa) is recognised
only to refuse it by name: its predictor and joint are one graph with a
different contract, and loading it as if it were split would fail somewhere
far less legible than here.
"""

import json
import os
from pathlib import Path

VOCAB_NAMES = ("parakeet_vocab.json", "vocab.json", "tokens.json", "vocab.txt",
               "tokens.txt")


def _pick(d, exact, role):
    """The exact upstream name if present, else the one file playing `role`."""
    if (d / exact).is_file():
        return d / exact
    hits = sorted(p for p in d.glob(f"*{role}*.xml")
                  if not (role == "decoder" and "joint" in p.name))
    return hits[0] if hits else None


def parakeet_files(path):
    """Role -> Path for a Parakeet export, or None when it is not one."""
    d = Path(path)
    if not d.is_dir():
        return None
    parts = {
        "encoder": _pick(d, "parakeet_encoder.xml", "encoder"),
        "decoder": _pick(d, "parakeet_decoder.xml", "decoder"),
        "joint": _pick(d, "parakeet_joint.xml", "joint"),
    }
    if not all(parts.values()):
        return None
    pre = d / "parakeet_melspectogram.xml"
    if not pre.is_file():
        hits = sorted([*d.glob("*melspec*.xml"), *d.glob("*preprocess*.xml")])
        pre = hits[0] if hits else None
    parts["preprocessor"] = pre
    parts["vocab"] = next((d / n for n in VOCAB_NAMES if (d / n).is_file()), None)
    parts["config"] = next((d / n for n in ("config.json", "metadata.json")
                            if (d / n).is_file()), None)
    return parts


def is_fused_transducer(path):
    d = Path(path)
    return d.is_dir() and any(d.glob("decoder_joint*.onnx"))


def is_whisper_dir(path):
    d = Path(path)
    try:
        with open(d / "config.json") as f:
            cfg = json.load(f)
        if cfg.get("model_type") == "whisper":
            return True
        if any("whisper" in a.lower() for a in cfg.get("architectures") or []):
            return True
    except (OSError, ValueError, AttributeError):
        pass
    return ((d / "openvino_encoder_model.xml").is_file()
            and (d / "openvino_decoder_model.xml").is_file()
            and not (d / "openvino_language_model.xml").is_file())


def asr_engine(path):
    """'parakeet', 'whisper', or None (not a speech-to-text model we know)."""
    if not path or not os.path.isdir(path):
        return None
    if parakeet_files(path):
        return "parakeet"
    if is_whisper_dir(path):
        return "whisper"
    return None


def describe_unknown(path):
    """A sentence for a directory --asr-dir could not classify."""
    if is_fused_transducer(path):
        return (f"{path} holds a fused decoder_joint ONNX export (onnx-asr/"
                "sherpa layout). locally's Parakeet engine reads the split "
                "OpenVINO export: FluidInference/parakeet-tdt-0.6b-v3-ov.")
    return (f"{path} is neither a Whisper export nor a Parakeet export "
            "(expected parakeet_encoder.xml, parakeet_decoder.xml, "
            "parakeet_joint.xml and parakeet_vocab.json).")
