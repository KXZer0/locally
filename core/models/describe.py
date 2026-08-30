"""Everything knowable about a model directory without loading it.

Backs --scan: display name and where it came from, LLM/VLM/Whisper,
architecture, MoE shape, geometry, integrity and real weight precision.
No server, no device init, no model load."""

import json
import os

from .geometry import (_kv_bytes_per_token, _model_max_context,
                       _text_config)
from .identity import (_is_generative_dir, _is_model_dir, is_vlm,
                       resolve_display_name)
from .integrity import _dir_size_bytes, _verify_weights_integrity
from .irinfo import read_ir_rt_info, weight_precision

from core import config


def describe_model(model_dir):
    """Everything locally can determine about a model directory from its own
    files, with no user input.

    The one thing the files do NOT record is the variant: config.json has no
    _name_or_path, and e.g. Qwen3-Coder-Next and Qwen3-Next-Instruct are
    identical in architecture and geometry. So the directory name stays the
    carrier of the variant — which is why resolve_display_name() must respect
    a rename (#19).
    """
    name, why = resolve_display_name(model_dir)
    rt = read_ir_rt_info(model_dir)
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    try:
        # Geometry lives under text_config on a VLM.
        geo = _text_config(model_dir)
    except Exception:
        geo = cfg

    if cfg.get("model_type") == "whisper":
        kind = "Whisper (speech-to-text)"
    elif is_vlm(model_dir):
        kind = "VLM (vision + text)"
    else:
        kind = "LLM (text)"

    return {
        "path": os.path.abspath(model_dir),
        "name": name,
        "name_source": why,
        "kind": kind,
        "architecture": (cfg.get("architectures") or [None])[0],
        "model_type": cfg.get("model_type"),
        "layers": geo.get("num_hidden_layers"),
        "context": geo.get("max_position_embeddings"),
        "experts": geo.get("num_experts") or geo.get("num_local_experts"),
        "experts_active": geo.get("num_experts_per_tok"),
        "precision": weight_precision(model_dir, rt),
        "size_bytes": _dir_size_bytes(model_dir),
        "kv_per_token": _kv_bytes_per_token(model_dir),
        "integrity": _verify_weights_integrity(model_dir),
        "openvino_version": rt.get("Runtime_version"),
        "optimum_intel_version": rt.get("optimum/optimum_intel_version"),
        "transformers_version": rt.get("optimum/transformers_version"),
    }


def _model_dirs_under(path, depth):
    """Model directories at or below `path`, searching `depth` levels down."""
    if not os.path.isdir(path):
        return []
    if _is_model_dir(path):
        return [path]
    if depth <= 0:
        return []
    found = []
    try:
        for entry in sorted(os.listdir(path)):
            sub = os.path.join(path, entry)
            if os.path.isdir(sub) and not entry.startswith("."):
                found.extend(_model_dirs_under(sub, depth - 1))
    except OSError:
        pass
    return found


def scan_models(paths):
    """Print what locally actually sees in each model directory.

    The alternative to this was a --model-name override flag, which needs
    knowledge a user shouldn't have to have (#19). This answers "what have I
    got, and what will it be called?" from the files on disk, so the answer
    comes from the machine rather than from documentation.
    """
    searched = [os.path.normpath(os.path.expanduser(p)) for p in
                (paths or [config.SCRIPT_DIR, "~/models"])]
    # One model reached by several paths (install.ps1 links model/ at a
    # directory in ~/models) is one model — report it once, listing the
    # aliases, rather than twice as if there were two copies.
    dirs, aliases = [], {}
    for path in searched:
        for d in _model_dirs_under(path, depth=2):
            real = os.path.realpath(d)
            if real in aliases:
                if d not in aliases[real]:
                    aliases[real].append(d)
                continue
            aliases[real] = []
            dirs.append(d)

    print("  locally model scan\n")
    if not dirs:
        print("  No OpenVINO models found in:")
        for path in searched:
            print(f"    {path}")
        print("\n  A model directory is one holding openvino_model.xml + .bin.")
        print("  Fetch one with:  .\\download-model.ps1 <hf-repo-id>")
        return

    for directory in dirs:
        info = describe_model(directory)
        print(f"  {info['path']}")
        for alias in aliases.get(os.path.realpath(directory), []):
            print(f"    (also reachable as {alias})")
        print(f"    Name in API/UI : {info['name']}"
              f"      (from {info['name_source']})")
        print(f"    Kind           : {info['kind']}")
        arch = info["architecture"] or info["model_type"] or "unknown"
        if info["model_type"] and info["architecture"]:
            arch += f" / {info['model_type']}"
        print(f"    Architecture   : {arch}")
        print(f"    Weights        : {info['precision']}", end="")
        if info["size_bytes"]:
            print(f"   {info['size_bytes'] / (1 << 30):,.1f} GB on disk")
        else:
            print()
        if info["experts"]:
            active = info["experts_active"] or "?"
            print(f"    MoE            : {info['experts']} experts, "
                  f"{active} active per token")
        geometry = []
        if info["layers"]:
            geometry.append(f"{info['layers']} layers")
        if info["context"]:
            geometry.append(f"{info['context']:,}-token context")
        if info["kv_per_token"]:
            geometry.append(f"{info['kv_per_token'] / 1024:,.0f} KB/token KV")
        if geometry:
            print(f"    Geometry       : {', '.join(geometry)}")
        built = [v for v in (
            f"OpenVINO {info['openvino_version']}" if info["openvino_version"] else None,
            f"optimum-intel {info['optimum_intel_version']}" if info["optimum_intel_version"] else None,
            f"transformers {info['transformers_version']}" if info["transformers_version"] else None,
        ) if v]
        if built:
            print(f"    Exported with  : {', '.join(built)}")
        if info["kind"].startswith("LLM"):
            print(f"    Agent mode     : tool calling on GPU/CPU; never on NPU "
                  f"(hard prompt cap)")
        if info["integrity"]:
            print(f"    PROBLEM        : {info['integrity']}")
        else:
            print(f"    Integrity      : weights complete")
        print()

    print("  To change the name shown in the UI and requested by clients,")
    print("  rename the model directory — that name is what locally uses.")
