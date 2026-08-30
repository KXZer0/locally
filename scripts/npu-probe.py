#!/usr/bin/env python3
"""npu-probe.py — can a candidate utility model actually run on the NPU?

Phase 0 gate for the Util tab. Nothing gets built into locally.py until it
passes here, because the NPU's constraints are unusually easy to get wrong:

  * **Static shapes are mandatory.** Measured on a Core Ultra X7 358H
    (2026-08-12, OpenVINO 2026.3): a `[?,3,32,320]` input fails to compile
    with `[NPU_VCL] Compiler returned msg: Missing upper bound for one or
    more nodes`. The identical model reshaped to `[4,3,32,320]` compiles in
    0.58 s. Every model here therefore gets an explicit reshape() first.
  * **Intel publishes no list of NPU-validated non-LLM models.** The official
    NPU device doc states only "only models with static shapes are supported"
    plus a dtype table, and names no vision models at all. So there is no
    authority to defer to — measurement is the only source of truth.
  * **int8 is not a speedup on NPU.** Compute is fp16-native; quantization
    buys memory, not latency. At these model sizes (10-80 MB) it buys nothing
    worth the accuracy, so everything runs fp16.

The probe deliberately does NOT import the Flask server; it reuses the same
load conventions standalone, like test_npu_multimodal.py.

Usage:
    python scripts/npu-probe.py --all                  # every candidate
    python scripts/npu-probe.py ocr-det ocr-rec        # named candidates
    python scripts/npu-probe.py --all --device GPU     # compare engines
    python scripts/npu-probe.py --ocr page.png         # end-to-end read test
    python scripts/npu-probe.py ocr-det --shapes 640,736,960
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

import openvino as ov

try:
    import yaml
except ImportError:
    yaml = None

# Where models land. Mirrors the plan's ~/models/util/<engine>/<task> layout,
# except the probe keeps one copy per task (the IR is device-independent; only
# the compiled blob in the OV cache is per-device).
DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser("~"), "models", "util")
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".ov-cache"
)

# ImageNet normalisation, as specified by PP-OCRv6_det's own inference.yml
# (NormalizeImage: mean/std/scale). Do not guess these — a wrong mean shows up
# as "the model runs but finds no text", which reads like an NPU failure.
DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --- candidate registry ----------------------------------------------------
#
# shape is what we reshape the (dynamic) ONNX to before compiling. For the
# detector it is a letterbox canvas; for the recogniser it is the fixed
# 3x48x320 line image its own inference.yml declares.

CANDIDATES = {
    "ocr-det": {
        "title": "PP-OCRv6 medium — text detection",
        "repo": "PaddlePaddle/PP-OCRv6_medium_det_onnx",
        "files": ["inference.onnx", "inference.yml"],
        "shape": [1, 3, 736, 736],
        "tier": "npu",
    },
    "ocr-rec": {
        "title": "PP-OCRv6 medium — text recognition",
        "repo": "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        "files": ["inference.onnx", "inference.yml"],
        "shape": [1, 3, 48, 320],
        "tier": "npu",
    },
    "ocr-det-small": {
        "title": "PP-OCRv6 small — text detection (fallback tier)",
        "repo": "PaddlePaddle/PP-OCRv6_small_det_onnx",
        "files": ["inference.onnx", "inference.yml"],
        "shape": [1, 3, 736, 736],
        "tier": "npu",
    },
    "ocr-rec-small": {
        "title": "PP-OCRv6 small — text recognition (fallback tier)",
        "repo": "PaddlePaddle/PP-OCRv6_small_rec_onnx",
        "files": ["inference.onnx", "inference.yml"],
        "shape": [1, 3, 48, 320],
        "tier": "npu",
    },

    # Text tiers. Both are OpenVINO IR rather than ONNX, and both take three
    # (or two) int64 ports instead of one float image — which is what the
    # model_file / dict-shape support above exists for. 1x256 matches the
    # sequence length the served embed() already reshapes to
    # (utility_pipeline.py), so the probe measures what production would run.
    #
    # Note the two signatures differ: bge-base-en-v1.5 is BERT-derived and
    # takes token_type_ids; bge-reranker-base is XLM-RoBERTa and does not.
    # Naming the ports rather than positioning them is what keeps that from
    # being a silent mismatch.
    "embed-bge": {
        "title": "BGE base en v1.5 int8 — embeddings (replaces MiniLM-L6)",
        "repo": "OpenVINO/bge-base-en-v1.5-int8-ov",
        "files": ["openvino_model.xml", "openvino_model.bin", "config.json",
                  "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.txt"],
        "model_file": "openvino_model.xml",
        "shape": {"input_ids": [1, 256], "attention_mask": [1, 256],
                  "token_type_ids": [1, 256]},
        "tier": "npu",
    },
    "rerank-bge": {
        "title": "BGE reranker base int8 — cross-encoder search reranking",
        "repo": "OpenVINO/bge-reranker-base-int8-ov",
        "files": ["openvino_model.xml", "openvino_model.bin", "config.json",
                  "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "sentencepiece.bpe.model"],
        "model_file": "openvino_model.xml",
        "shape": {"input_ids": [1, 256], "attention_mask": [1, 256]},
        "tier": "npu",
    },

    # Document stack. Every shape below comes from the model's own
    # inference.yml rather than from guesswork -- the phase-0 write-up records
    # that a wrong preprocessing constant shows up as "runs fine, finds
    # nothing", which reads like an NPU failure and is not one.
    "layout": {
        # DETR-derived, declaring use_dynamic_shape: false and target_size
        # [800, 800]. THREE ports, not the two its inference.yml mentions:
        # im_shape, image, scale_factor. Missing im_shape leaves a [?,2] with
        # an INT64_MAX upper bound, and the NPU compiler does not report that
        # as an error -- it takes the process down with an access violation
        # (0xC0000005). So every port gets a bound here, and the probe is worth
        # running alone: a segfault kills the whole --all run with it.
        "title": "PP-DocLayoutV3 — document layout (regions + reading order)",
        "repo": "PaddlePaddle/PP-DocLayoutV3_onnx",
        "files": ["inference.onnx", "inference.yml"],
        "shape": {"im_shape": [1, 2], "image": [1, 3, 800, 800],
                  "scale_factor": [1, 2]},
        "tier": "npu",
    },
    "table": {
        # ResizeTableImage max_len 512, size [512, 512]. Its postprocess is
        # TableLabelDecode over a character dict, so the structure comes out as
        # a token sequence -- the probe will show whether that decode is fixed
        # length (NPU-viable) or open-ended (GPU only).
        "title": "SLANeXt wired — table structure recognition",
        "repo": "PaddlePaddle/SLANeXt_wired_onnx",
        "files": ["inference.onnx", "inference.yml"],
        "shape": [1, 3, 512, 512],
        "tier": "npu",
    },
    "dewarp": {
        # UVDoc publishes a range rather than one size: [1,3,128,64] min,
        # [1,3,256,128] opt, [8,3,512,256] max. The opt shape is what its own
        # config nominates, so that is what gets probed first.
        "title": "UVDoc — dewarp photographed / curved pages before OCR",
        "repo": "PaddlePaddle/UVDoc_onnx",
        "files": ["inference.onnx", "inference.yml"],
        "shape": [1, 3, 256, 128],
        "tier": "npu",
    },

    "birefnet": {
        # General-purpose matting, unlike MODNet which is portrait-only. Same
        # architecture as RMBG-2.0 but MIT rather than non-commercial, which is
        # the deciding factor here on the YOLO11n precedent. Preprocessing is
        # fully specified by preprocessor_config.json (1024x1024, /255, ImageNet
        # mean/std) — the thing UVDoc lacked.
        #
        # 1024x1024 is 4x MODNet's pixel count and the file is ~6x its size, so
        # whether the NPU keeps it interactive is the actual question, not
        # whether it compiles.
        "title": "BiRefNet lite fp16 — general background removal (MIT)",
        "repo": "onnx-community/BiRefNet_lite-ONNX",
        "files": ["onnx/model_fp16.onnx", "config.json",
                  "preprocessor_config.json"],
        "model_file": "onnx/model_fp16.onnx",
        "shape": [1, 3, 1024, 1024],
        "tier": "npu",
    },

    "upscale-swin2sr": {
        # Replaces OMZ single-image-super-resolution-1033 (2019, 0.1 MB, fixed
        # 3x, and a hard 640x360 input ceiling that made feeding it a 1080p
        # image a net quality LOSS).
        #
        # Chosen over converting Real-ESRGAN locally: the reason to convert was
        # that every Real-ESRGAN ONNX on the hub is published by an individual.
        # This is Apache-2.0 at onnx-community AND at upstream caidas, which is
        # the provenance the conversion was meant to buy — without the
        # conversion.
        #
        # Preprocessing is fully specified: rescale 1/255 and NO mean/std
        # (unlike BiRefNet, which does normalise). Pad to a multiple of 8 to
        # match window_size. 256x256 tiles out to 1024x1024 at upscale 4.
        "title": "Swin2SR real-world x4 fp16 — upscaling (Apache-2.0)",
        "repo": "onnx-community/swin2SR-realworld-sr-x4-64-bsrgan-psnr-ONNX",
        "files": ["onnx/model_fp16.onnx", "config.json",
                  "preprocessor_config.json"],
        "model_file": "onnx/model_fp16.onnx",
        "shape": [1, 3, 256, 256],
        "tier": "npu",
    },

    # NOT LISTED: formula -> LaTeX. PaddlePaddle/PP-FormulaNet_plus-L_onnx
    # exists but contains only .gitattributes and README.md -- no weights at
    # all. Every other FormulaNet variant (plus-L/M/S, PP-FormulaNet-S/-L) is
    # Paddle or safetensors format with no ONNX export published. Converting
    # one is a separate piece of work, not a probe. See NPU-UTIL-PHASE0.md.
}


# --- model shape / input helpers -------------------------------------------
#
# The first four candidates were all single-input float ONNX files named
# inference.onnx, so the probe assumed all three of those things. The embedding
# and reranker tiers are none of them: OpenVINO IR (.xml + .bin), three inputs,
# int64. These helpers absorb the difference without changing what an OCR spec
# has to say.

def _model_path(model_dir, spec):
    """The file to hand read_model. Defaults to the OCR convention."""
    return os.path.join(model_dir, spec.get("model_file", "inference.onnx"))


def _disk_mb(path):
    """Size on disk. An IR is a pair, and the .bin holds ~all of the weight."""
    total = os.path.getsize(path)
    if path.endswith(".xml"):
        weights = path[:-4] + ".bin"
        if os.path.isfile(weights):
            total += os.path.getsize(weights)
    return round(total / 1e6, 1)


def _shape_tag(shape):
    """Label for one shape spec: '1x3x736x736', or 'input_ids=1x256,...'."""
    if isinstance(shape, dict):
        return ",".join(f"{k}={'x'.join(str(d) for d in v)}"
                        for k, v in shape.items())
    return "x".join(str(d) for d in shape)


def _reshape_map(model, shape):
    """Build the dict reshape() wants.

    A list applies to the first input (the original single-input behaviour); a
    dict is keyed by input name, which is the only way to express a model whose
    inputs differ in shape.
    """
    if isinstance(shape, dict):
        by_name = {}
        for port in model.inputs:
            names = port.get_names() or {port.get_any_name()}
            for name in names:
                if name in shape:
                    by_name[port] = ov.PartialShape(shape[name])
                    break
        missing = set(shape) - {n for p in by_name for n in p.get_names()}
        if missing:
            raise ValueError(f"no such input(s): {sorted(missing)}")
        return by_name
    return {model.inputs[0]: ov.PartialShape(shape)}


def _dummy_inputs(compiled):
    """Plausible input tensors, typed per port.

    Random floats are meaningless for a tokenizer's int64 ports and can be
    out of range, so integers get fixed values instead: an all-zero attention
    mask makes the softmax degenerate (and on some builds produces NaN, which
    reads as a broken model rather than a bad probe), so masks are ones and
    every other integer port is zeros — token id 0 is [PAD] in every vocab
    here, and token_type 0 is always valid.
    """
    feed = {}
    for port in compiled.inputs:
        shape = [int(d) for d in port.shape]
        name = (port.get_any_name() or "").lower()
        if port.get_element_type().is_integral():
            fill = np.ones if "mask" in name else np.zeros
            feed[port] = fill(shape, dtype=np.int64)
        else:
            feed[port] = np.random.rand(*shape).astype(np.float32)
    return feed


# --- fetching --------------------------------------------------------------

def fetch(key, spec, models_dir):
    """Download a candidate's files if missing. Returns its directory.

    Uses the curl recipe from CLAUDE.md rather than huggingface_hub: on this
    network ~75% of HTTPS connections to huggingface.co are reset mid-handshake,
    and hf_hub does not retry connection resets hard enough.
    """
    dest = os.path.join(models_dir, key)
    os.makedirs(dest, exist_ok=True)

    for name in spec["files"]:
        path = os.path.join(dest, name)
        # Some repos nest weights (onnx/model_fp16.onnx); mirror the layout
        # rather than flattening it, so the local path matches the remote one.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        url = f"https://huggingface.co/{spec['repo']}/resolve/main/{name}"
        print(f"    fetching {name} ...", end="", flush=True)
        cmd = [
            "curl", "-L", "-s", "-o", path, "-C", "-",
            "--retry", "30", "--retry-all-errors", "--retry-delay", "1",
            "--speed-limit", "2000", "--speed-time", "20",
            url,
        ]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0 or not os.path.isfile(path):
            raise RuntimeError(f"download failed: {url} ({r.stderr.decode()[:200]})")
        mb = os.path.getsize(path) / 1e6
        print(f" {mb:.1f} MB in {time.perf_counter() - t0:.1f}s")
    return dest


# --- the probe itself ------------------------------------------------------

def probe(key, spec, models_dir, device, shapes=None, cache=True):
    """Fetch, convert, reshape, compile and infer one candidate.

    Returns a result dict; never raises, so --all reports every candidate
    instead of stopping at the first failure.
    """
    res = {"key": key, "title": spec["title"], "device": device, "ok": False}
    print(f"\n[{key}] {spec['title']}")

    try:
        model_dir = fetch(key, spec, models_dir)
    except Exception as e:
        res["error"] = f"download: {e}"
        print(f"    FAIL {res['error']}")
        return res

    src_path = _model_path(model_dir, spec)
    res["disk_mb"] = _disk_mb(src_path)

    core = ov.Core()
    if device not in core.get_available_devices():
        res["error"] = f"device {device} not available"
        print(f"    SKIP {res['error']}")
        return res

    # ONNX and IR are both native OpenVINO frontends — read_model handles either
    # directly, so there is no onnx/onnxruntime dependency and no separate
    # conversion step.
    try:
        t0 = time.perf_counter()
        model = core.read_model(src_path)
        res["read_s"] = round(time.perf_counter() - t0, 2)
    except Exception as e:
        res["error"] = f"read_model: {type(e).__name__}: {e}"
        print(f"    FAIL {res['error']}")
        return res

    # Report every input, not just the first: a model that takes ids + mask +
    # token_type is exactly the case where the first port tells you least.
    res["onnx_shape"] = ", ".join(
        f"{p.get_any_name()} {p.get_partial_shape()}" for p in model.inputs)
    print(f"    inputs: {res['onnx_shape']}  ({res['disk_mb']} MB)")

    for shape in shapes or [spec["shape"]]:
        tag = _shape_tag(shape)
        r = dict(res, shape=tag)
        try:
            m = core.read_model(src_path)
            m.reshape(_reshape_map(m, shape))
        except Exception as e:
            r["error"] = f"reshape: {type(e).__name__}: {str(e)[:160]}"
            print(f"    [{tag}] FAIL {r['error']}")
            continue

        cfg = {"CACHE_DIR": CACHE_DIR} if cache else {}
        try:
            t0 = time.perf_counter()
            compiled = core.compile_model(m, device, cfg)
            r["compile_s"] = round(time.perf_counter() - t0, 2)
        except Exception as e:
            # The message that matters is the compiler's, not the wrapper's.
            msg = str(e).strip().splitlines()
            r["error"] = "compile: " + " | ".join(x.strip() for x in msg[-3:])[:300]
            print(f"    [{tag}] FAIL compile on {device}")
            print(f"          {r['error']}")
            continue

        feed = _dummy_inputs(compiled)
        try:
            compiled(feed)                                # warm the first run
            t0 = time.perf_counter()
            runs = 5
            for _ in range(runs):
                out = compiled(feed)
            r["infer_ms"] = round((time.perf_counter() - t0) / runs * 1000, 1)
            r["out_shape"] = str(list(out.values())[0].shape)
        except Exception as e:
            r["error"] = f"infer: {type(e).__name__}: {str(e)[:160]}"
            print(f"    [{tag}] FAIL {r['error']}")
            continue

        r["ok"] = True
        print(f"    [{tag}] OK  compile {r['compile_s']}s  infer {r['infer_ms']}ms"
              f"  out {r['out_shape']}")
        res = r

    return res


# --- end-to-end OCR (correctness, not just "it compiles") ------------------

def _say(text):
    """Print text the console encoding may not accept.

    The recognition dict spans 50 languages and includes emoji, so a faithful
    read of some pages is unprintable on a cp1252 Windows console — and the
    UnicodeEncodeError would land on the *reporting* line, making a successful
    OCR run look like a crash.
    """
    enc = sys.stdout.encoding or "utf-8"
    print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def read_image(path, models_dir, device, det_size=736):
    """Full det -> rec pass on a real image. The correctness check.

    Runs the SAME code the server runs, via ocr_pipeline.OcrEngine. This file
    used to carry its own copy of the letterbox/DB-postprocess/CTC logic, which
    silently went stale the moment the served pipeline gained width bucketing,
    a real unclip and confidence filtering — so the gate was passing on output
    the server would never produce. A gate that tests a different
    implementation is not a gate.
    """
    from PIL import Image
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ocr_pipeline import OcrEngine

    for key in ("ocr-det", "ocr-rec"):
        d = os.path.join(models_dir, key)
        if not os.path.isfile(os.path.join(d, "inference.onnx")):
            fetch(key, CANDIDATES[key], models_dir)

    print(f"\n=== reading {path} on {device} ===")
    engine = OcrEngine(models_dir, device=device, cache_dir=CACHE_DIR,
                       det_size=det_size)
    t0 = time.perf_counter()
    engine.load()
    print(f"load: {time.perf_counter() - t0:.2f}s")

    result = engine.read(Image.open(path))
    t = result["timings_ms"]
    kept = [b for b in result["blocks"] if b["kept"]]
    print(f"detect: {result['regions']} regions in {t['detect']:.0f} ms")
    print(f"recognise: {len(kept)} lines in {t['recognise']:.0f} ms "
          f"({result['dropped_low_conf']} dropped below the confidence floor)\n")

    for b in kept:
        _say(f"  [{b['conf']:.2f}] {b['text']}")
    if not result["text"]:
        print("  (no text recognised)")
    _say("\n--- assembled text ---")
    _say(result["text"])
    return result


# --- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Probe utility models on the NPU (Phase 0 gate).")
    ap.add_argument("keys", nargs="*", help="candidates to probe")
    ap.add_argument("--all", action="store_true", help="probe every candidate")
    ap.add_argument("--device", default="NPU", help="OpenVINO device (default NPU)")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--shapes", help="comma-separated square sizes to try (det only)")
    ap.add_argument("--no-cache", action="store_true", help="skip the OV compile cache")
    ap.add_argument("--ocr", metavar="IMAGE", help="end-to-end read test on an image")
    ap.add_argument("--list", action="store_true", help="list candidates and exit")
    args = ap.parse_args()

    if args.list:
        for k, s in CANDIDATES.items():
            print(f"  {k:16s} {s['title']}")
        return 0

    print("=== locally util-model probe ===")
    print(f"openvino  {ov.__version__}")
    core = ov.Core()
    print(f"devices   {core.get_available_devices()}")
    try:
        print(f"target    {args.device} — {core.get_property(args.device, 'FULL_DEVICE_NAME')}")
    except Exception:
        pass

    if args.ocr:
        read_image(args.ocr, args.models_dir, args.device)
        return 0

    keys = list(CANDIDATES) if args.all else args.keys
    if not keys:
        ap.error("name at least one candidate, or pass --all (see --list)")
    unknown = [k for k in keys if k not in CANDIDATES]
    if unknown:
        ap.error(f"unknown candidate(s): {', '.join(unknown)}")

    shapes = None
    if args.shapes:
        shapes = [[1, 3, int(s), int(s)] for s in args.shapes.split(",")]

    results = []
    for k in keys:
        spec = CANDIDATES[k]
        use = shapes if (shapes and "det" in k) else None
        results.append(probe(k, spec, args.models_dir, args.device,
                             shapes=use, cache=not args.no_cache))

    print(f"\n=== Summary ({args.device}) ===")
    width = max(len(r["key"]) for r in results)
    for r in results:
        if r["ok"]:
            print(f"  {r['key']:<{width}}  PASS  {r['shape']:>14}  "
                  f"compile {r['compile_s']:>5}s  infer {r['infer_ms']:>7} ms")
        else:
            print(f"  {r['key']:<{width}}  FAIL  {r.get('error', 'unknown')[:110]}")

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\n{len(failed)} of {len(results)} failed. Record each in TODONT.md "
              f"with the compiler message above before working around it.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
