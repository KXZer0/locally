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
  * Nemotron family: parakeet-unified-en-0.6b and the Nemotron streaming
    models, in FluidInference's export, read from eddy-audio
    (`NEMOTRON_FILES`):

        nemotron_encoder.xml/.bin    streaming FastConformer encoder
        nemotron_decoder.xml/.bin    LSTM predictor
        nemotron_joint.xml/.bin      joint: token scores only (plain RNN-T)
        nemotron_vocab.json          SentencePiece pieces
        metadata.json                geometry (core/asr/geometry.py)

    `nemotron_preprocessor.xml` may be present and is deliberately unused:
    it is dynamic-shaped (no NPU) and eddy replaces it with a native
    featurizer, as core/asr/mel.py does here. Checked BEFORE Parakeet, whose
    role-based match (`*encoder*.xml`) would otherwise claim these folders.
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


NEMOTRON_PARTS = {"encoder": "nemotron_encoder.xml",
                  "decoder": "nemotron_decoder.xml",
                  "joint": "nemotron_joint.xml",
                  "vocab": "nemotron_vocab.json",
                  "metadata": "metadata.json"}


def nemotron_files(path):
    """Role -> Path for a Nemotron-family export, or None when it is not one.

    The encoder and metadata.json are what make it one; a folder holding
    those but missing a part is still returned (with None for that part) so
    the slot can name what is missing instead of calling it "unknown"."""
    d = Path(path)
    if not d.is_dir():
        return None
    if not ((d / NEMOTRON_PARTS["encoder"]).is_file()
            and (d / NEMOTRON_PARTS["metadata"]).is_file()):
        return None
    return {role: (d / name if (d / name).is_file() else None)
            for role, name in NEMOTRON_PARTS.items()}


def nested_nemotron(path):
    """Subfolders (two levels down) holding a Nemotron-family export: the HF
    repos keep several builds side by side (streaming/fp16, fp16, en/int8)."""
    d = Path(path)
    if not d.is_dir():
        return []
    hits = [*d.glob("*/nemotron_encoder.xml"), *d.glob("*/*/nemotron_encoder.xml")]
    return sorted(str(p.parent.relative_to(d)) for p in hits)


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
    """'nemotron', 'parakeet', 'whisper', or None (not one we know).

    'nemotron' names the engine (eddy-audio's name for the backend); the
    model in it may well be called Parakeet -- parakeet-unified is."""
    if not path or not os.path.isdir(path):
        return None
    if nemotron_files(path):
        return "nemotron"
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
    nested = nested_nemotron(path)
    if nested:
        return (f"{path} holds Nemotron-family builds in subfolders; point "
                "--asr-dir at one of them: " + ", ".join(nested) + ".")
    return (f"{path} is not a speech-to-text export locally knows. Expected "
            "parakeet-unified / Nemotron streaming (nemotron_encoder.xml + "
            "metadata.json), Parakeet TDT (parakeet_encoder.xml, "
            "parakeet_decoder.xml, parakeet_joint.xml, parakeet_vocab.json) "
            "or Whisper (config.json).")
