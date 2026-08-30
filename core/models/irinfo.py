"""The IR's own account of itself: the model-level <rt_info> block.

Read from the TAIL of the .xml -- the graph is tens of MB on a large model and
the block sits after <edges>. This is how the real weight precision is known,
rather than trusting a folder name that can lie."""

import json
import os
import xml.etree.ElementTree as ET


def _flatten_rt_info(elem, prefix=""):
    """Flatten <a><b value="x"/></a> into {"a/b": "x"}."""
    out = {}
    for child in elem:
        key = prefix + child.tag
        if child.get("value") is not None:
            out[key] = child.get("value")
        out.update(_flatten_rt_info(child, key + "/"))
    return out


def read_ir_rt_info(model_dir):
    """Model-level <rt_info> from the IR .xml — the authoritative record of
    how a model was exported: nncf weight-compression mode and group size,
    plus the OpenVINO / optimum-intel / transformers versions that built it.
    Believe this over the directory name, which can say anything.

    Read from the tail of the file. The .xml holds the graph (tens of MB on a
    large model) and the model-level block is the last <rt_info> in it —
    per-node ones live inside <layers>, which precedes <edges> and the
    trailing block.
    """
    for base in ("openvino_model", "openvino_language_model"):
        xml = os.path.join(model_dir, base + ".xml")
        if not os.path.isfile(xml):
            continue
        try:
            with open(xml, "rb") as f:
                f.seek(max(0, os.path.getsize(xml) - 262144))
                tail = f.read()
            start = tail.rfind(b"<rt_info>")
            end = tail.rfind(b"</rt_info>")
            if start == -1 or end <= start:
                return {}
            fragment = tail[start:end + len(b"</rt_info>")]
            return _flatten_rt_info(ET.fromstring(fragment))
        except (OSError, ET.ParseError):
            return {}
    return {}


def weight_precision(model_dir, rt=None):
    """Human-readable weight precision, read from the IR's own nncf record.

    A directory called -int4-ov can contain anything; this is what the
    weights actually are.
    """
    rt = read_ir_rt_info(model_dir) if rt is None else rt
    mode = rt.get("nncf/weight_compression/mode")
    if not mode:
        try:
            with open(os.path.join(model_dir, "config.json")) as f:
                cfg = json.load(f)
            dtype = cfg.get("dtype") or cfg.get("torch_dtype")
            return f"{dtype} (weights not compressed)" if dtype else "unknown"
        except Exception:
            return "unknown"

    bits = ("INT4" if "int4" in mode else
            "INT8" if "int8" in mode else
            "FP8" if "f8" in mode or "fp8" in mode else mode)
    detail = []
    if mode.endswith("_sym"):
        detail.append("symmetric")
    elif mode.endswith("_asym"):
        detail.append("asymmetric")
    group_size = rt.get("nncf/weight_compression/group_size")
    if group_size == "-1":
        detail.append("channel-wise")
    elif group_size:
        detail.append(f"group size {group_size}")
    label = bits + (f" ({', '.join(detail)})" if detail else "")

    ratio = rt.get("nncf/weight_compression/ratio")
    try:
        # ratio < 1 means only that fraction of layers got the low-bit
        # treatment and the rest fell back to backup_mode — a mixed model
        # that a folder name would report as plain "int4".
        if ratio and float(ratio) < 1.0:
            backup = rt.get("nncf/weight_compression/backup_mode", "int8")
            label += f", {float(ratio) * 100:.0f}% of layers (rest {backup})"
    except ValueError:
        pass
    if rt.get("nncf/weight_compression/awq") == "True":
        label += " +AWQ"
    if rt.get("nncf/weight_compression/scale_estimation") == "True":
        label += " +scale-estimation"
    return label
