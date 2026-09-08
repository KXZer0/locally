"""Local OpenVINO utility pipelines used by locally's Utilities sidebar.

These models are intentionally lazy-loaded. OCR has its own pipeline because it
is also used by the standalone probe; this file covers image matting,
super-resolution, object detection, embeddings, and optional image generation.
No user input is written to disk.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image, ImageDraw, ImageFont


class UtilityUnavailable(RuntimeError):
    """A utility cannot run: its model is missing, or the task is disabled.

    Distinct from a genuine failure so the HTTP layer can answer 503 ("not
    offered") instead of 500 ("we broke").
    """


# Tasks that are switched off on purpose, with the reason the user sees.
_DISABLED_TASKS = {
    "generate": "no Stable Diffusion pipeline passed the local compile gate "
                "on this hardware (see TODONT.md)",
}


# Per-device model tiers.
#
# The NPU gets the largest model that passes the probe gate: it is the low-power
# engine and running work there is the point of the project. The GPU is for what
# the NPU genuinely cannot run -- dynamic shapes that will not reshape,
# unsupported ops, autoregressive decoders -- and is picked for efficiency
# rather than maximum accuracy.
#
# A bare string means "same model on every device", which is the honest default:
# most of these were probed on both and the NPU was fast enough that a second
# copy would buy nothing. A dict is per-device, and `None` means this device
# cannot serve this task at all -- a third state distinct from "not installed"
# and "deliberately disabled", and the UI needs all three to say anything true.
TIERS = {
    # MODNet stays: BiRefNet is the better model and MIT-licensed, but it
    # neither compiles on this NPU nor executes on this GPU (TODONT.md).
    "background": "background/modnet-fp16.onnx",
    # The first genuine per-device split. Swin2SR is x4 with no input ceiling,
    # but the vpux compiler rejects it, so the NPU keeps the old OMZ model —
    # limited (fixed 3x, 640x360 input cap) but real, and it runs where the
    # better model cannot.
    "upscale": {
        "npu": "upscale/single-image-super-resolution-1033.xml",
        "gpu": "upscale-swin2sr/onnx/model_fp16.onnx",
        # CPU takes the GPU's model, not the NPU's: what rules Swin2SR out on
        # the NPU is the vpux compiler rejecting it, which says nothing about
        # x86. Slower there, but a true 4x with no input ceiling beats a fixed
        # 3x capped at 640x360.
        "cpu": "upscale-swin2sr/onnx/model_fp16.onnx",
    },
    "detect": "detect-rfdetr/rfdetr_small.xml",
    # all-MiniLM-L6 stays: BGE-base was probed and did NOT beat it on this
    # workload (docs/archive/NPU-UTIL-PHASE0.md, phase 4). The reranker is where the
    # retrieval gain actually came from.
    "search": "embed/openvino_model_qint8_quantized.xml",
    "rerank": "rerank-bge/openvino_model.xml",
    "layout": "layout/inference.onnx",
    "generate": {"npu": None, "gpu": "image-gen/model_index.json", "cpu": None},
}


# PP-DocLayoutV3's classes, in the order its inference.yml declares them. The
# index into this list is the class id the model emits, so the order is load
# bearing -- do not sort it.
_LAYOUT_LABELS = (
    "abstract algorithm aside_text chart content display_formula doc_title "
    "figure_title footer footer_image footnote formula_number header "
    "header_image image inline_formula number paragraph_title reference "
    "reference_content seal table text vertical_text vision_footnote"
).split()

# Its own inference.yml's draw_threshold.
_LAYOUT_THRESHOLD = 0.5

# Swin2SR tiling. The tile is the static shape the model is compiled for; the
# overlap is trimmed from inner edges because a windowed transformer's output
# near a tile border is conditioned on padding that isn't really there.
_SWIN_TILE = 256
_SWIN_OVERLAP = 32
_SWIN_SCALE = 4


# Why a device cannot serve a task, when the answer is not "download a model".
# Keyed (task, device) because the reason is specific to the pairing.
_DEVICE_UNSUPPORTED = {
    ("generate", "npu"):
        "the NPU has no working diffusion path — Stable Diffusion's text "
        "encoder has two dynamic output-bound dimensions, and the forced-static "
        "compile ran over 22 minutes without producing a model (see TODONT.md)",
}


def _tier_path(root, task, device):
    """Resolve a task's model path for one device, or None if unsupported.

    A bare string means "the same model on every engine"; a dict is per-device
    and a missing or None entry means that engine cannot serve the task -- a
    third state, distinct from "not installed" and "deliberately disabled".
    CPU is spelled out in every dict rather than falling back to the GPU's
    entry, because a silent fallback would have quietly given the NPU-only
    tiers to a CPU that may not want them.
    """
    spec = TIERS.get(task)
    rel = spec.get(device.lower()) if isinstance(spec, dict) else spec
    return (root / rel) if rel else None


_COCO_LABELS = (
    "__background__ person bicycle car motorcycle airplane bus train truck boat "
    "traffic_light fire_hydrant N/A stop_sign parking_meter bench bird cat dog "
    "horse sheep cow elephant bear zebra giraffe N/A backpack umbrella N/A N/A "
    "handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat "
    "baseball_glove skateboard surfboard tennis_racket bottle N/A wine_glass cup "
    "fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog "
    "pizza donut cake chair couch potted_plant bed N/A dining_table N/A N/A toilet "
    "N/A tv laptop mouse remote keyboard cell_phone microwave oven toaster sink "
    "refrigerator N/A book clock vase scissors teddy_bear hair_drier toothbrush"
).split()


def _resize(image: Image.Image, size: tuple[int, int], resample=Image.Resampling.BILINEAR):
    return image.resize(size, resample)


def _compile(core, path, device, cache_dir=None, reshape=None):
    model = core.read_model(str(path))
    if reshape:
        model.reshape(reshape)
    props = {"CACHE_DIR": str(cache_dir)} if cache_dir else {}
    return core.compile_model(model, device, props)


def _box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def _class_nms(items, threshold=0.65):
    kept = []
    for item in sorted(items, key=lambda row: row["score"], reverse=True):
        if any(item["class_id"] == prior["class_id"] and
               _box_iou(item["box"], prior["box"]) > threshold
               for prior in kept):
            continue
        kept.append(item)
    return kept


class UtilityEngine:
    """Lazy utility models for one OpenVINO device."""

    def __init__(self, model_dir, device="NPU", cache_dir=None):
        self.root = Path(model_dir)
        self.device = device
        self.cache_dir = cache_dir
        self.core = ov.Core()
        self._models = {}
        # One tokenizer per task: the embedder is BERT-derived and the reranker
        # is XLM-RoBERTa, so they are not interchangeable.
        self._tokenizers = {}
        self._generator = None
        # Paths are per-device now; a None entry means this engine cannot serve
        # that task, which availability and _require report differently from a
        # missing file.
        self.paths = {task: _tier_path(self.root, task, self.device)
                      for task in TIERS}

    def unsupported_reason(self, name):
        """Why this device cannot serve this task at all, or None."""
        if self.paths.get(name) is not None:
            return None
        return _DEVICE_UNSUPPORTED.get(
            (name, self.device.lower()),
            f"not available on {self.device.upper()} in this build")

    @property
    def unavailable_reasons(self):
        """task -> why it is off, for every task that is not available here.

        Three causes need three different sentences: a missing file is fixed by
        downloading, a deliberate disable cannot be fixed that way at all, and
        a device that cannot run the model is fixed by switching engine. Collapsing
        them into "model not installed" sends people to fetch a model that would
        not help — the mistake `_require` already exists to avoid.
        """
        reasons = {}
        for name, available in self.availability.items():
            if available:
                continue
            if name in _DISABLED_TASKS:
                reasons[name] = _DISABLED_TASKS[name]
            elif self.paths.get(name) is None:
                reasons[name] = self.unsupported_reason(name)
            else:
                reasons[name] = (f"the {name} model is not installed "
                                 f"(fetch it with scripts/npu-probe.py)")
        return reasons

    @property
    def availability(self):
        available = {name: bool(path) and path.is_file()
                     for name, path in self.paths.items()}
        # Image generation is DISABLED, not missing — the distinction matters
        # because the files may well be present. Stable Diffusion 1.5's text
        # encoder is dynamic: direct NPU compile fails, forcing static pipeline
        # shapes spent >22 minutes at full CPU without producing a model, and
        # the Intel GPU fallback exited during compile on this machine. The
        # endpoint and UI are kept so a working pipeline can be dropped in, but
        # nothing here may advertise the task merely because files exist —
        # a click must not be able to repeat that 22 minutes.
        #
        # Re-enable by deleting this line, only after a replacement pipeline
        # passes both compile and one real generation.
        available["generate"] = False
        return available

    @property
    def loaded(self):
        names = list(self._models)
        if self._generator is not None:
            names.append("generate")
        return names

    def _require(self, name):
        """Raise UtilityUnavailable if a task can't run, saying which it is.

        "Not installed" and "installed but disabled" need different words: the
        first is fixed by downloading a model, the second cannot be fixed that
        way and telling someone to download it sends them in circles. Callers
        catch UtilityUnavailable and report 503 — a task we deliberately do not
        offer is not a server fault, and reporting it as one (HTTP 500,
        "Image generation failed") makes a known limitation look like a crash.
        """
        if self.availability.get(name):
            return
        raise UtilityUnavailable(
            f"The {name} utility is unavailable on "
            f"{self.device.upper()}: {self.unavailable_reasons[name]}")

    def _model(self, name):
        self._require(name)
        if name in self._models:
            return self._models[name]
        reshape = None
        if name == "background":
            reshape = {"input": [1, 3, 512, 512]}
        elif name == "search":
            reshape = {
                "input_ids": [1, 256],
                "attention_mask": [1, 256],
                "token_type_ids": [1, 256],
            }
        elif name == "rerank":
            # bge-reranker-base is XLM-RoBERTa: two ports, no token_type_ids.
            # Naming them rather than positioning them is what stops that
            # difference from becoming a silent mismatch against the embedder.
            reshape = {"input_ids": [1, 256], "attention_mask": [1, 256]}
        elif name == "upscale" and self._upscale_is_swin:
            # Exported fully dynamic ([?,?,?,?]); pinned to the tile size the
            # tiling loop feeds it.
            reshape = {"pixel_values": [1, 3, _SWIN_TILE, _SWIN_TILE]}
        elif name == "layout":
            # THREE ports. im_shape is not mentioned in the model's own
            # inference.yml, and leaving it unbounded does not raise — the NPU
            # compiler exits the process with an access violation (TODONT.md).
            reshape = {"im_shape": [1, 2], "image": [1, 3, 800, 800],
                       "scale_factor": [1, 2]}
        compiled = _compile(self.core, self.paths[name], self.device,
                            self.cache_dir, reshape)
        self._models[name] = compiled
        return compiled

    def remove_background(self, image: Image.Image):
        image = image.convert("RGB")
        original_size = image.size
        sample = np.asarray(_resize(image, (512, 512)), dtype=np.float32)
        sample = ((sample / 127.5) - 1.0).transpose(2, 0, 1)[None]
        model = self._model("background")
        t0 = time.perf_counter()
        alpha = np.asarray(model({"input": sample})[model.output(0)])[0, 0]
        infer_ms = round((time.perf_counter() - t0) * 1000, 1)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8), "L")
        alpha_img = _resize(alpha_img, original_size, Image.Resampling.BILINEAR)
        result = image.convert("RGBA")
        result.putalpha(alpha_img)
        return result, {"infer": infer_ms}

    @property
    def _upscale_is_swin(self):
        """Is this device's upscaler Swin2SR rather than the old OMZ model?"""
        path = self.paths.get("upscale")
        return bool(path) and "swin2sr" in str(path).lower()

    def upscale(self, image: Image.Image):
        """x4 (Swin2SR) or 3x (OMZ) upscale, depending on the device's tier."""
        if self._upscale_is_swin:
            return self._upscale_swin2sr(image)
        return self._upscale_omz(image)

    def _upscale_swin2sr(self, image: Image.Image):
        """x4 upscale by tiling. No input-size ceiling.

        The OMZ model this replaces fixed its input at 640x360 and its output at
        1920x1080, so anything larger was *downscaled on the way in* — feeding it
        a 1080p image returned a blurrier 1080p image, because the round trip
        cannot restore what the downscale removed. Tiling removes the ceiling:
        output is 4x the source at any size, bounded by time and memory rather
        than by a hardcoded canvas.

        Cost is roughly linear in pixels — ~0.8 s per 256x256 tile on the B390 —
        so a 1080p source is tens of seconds. That is the price of 7680x4320.
        """
        image = image.convert("RGB")
        width, height = image.size
        model = self._model("upscale")
        tile, overlap, scale = _SWIN_TILE, _SWIN_OVERLAP, _SWIN_SCALE
        step = tile - overlap
        out = Image.new("RGB", (width * scale, height * scale))

        t0 = time.perf_counter()
        tiles = 0
        for top in range(0, height, step):
            for left in range(0, width, step):
                right, bottom = min(left + tile, width), min(top + tile, height)
                patch = np.asarray(image.crop((left, top, right, bottom)),
                                   dtype=np.float32)
                ph, pw = patch.shape[:2]
                # Edge-replicate to the compiled static shape. Zero padding
                # would darken the border, the same reason the OMZ path pads
                # with 'edge' rather than with black.
                patch = np.pad(patch, ((0, tile - ph), (0, tile - pw), (0, 0)),
                               mode="edge")
                # rescale 1/255 and NO mean/std — per Swin2SRImageProcessor,
                # which unlike BiRefNet's config does not normalise.
                x = (patch / 255.0).transpose(2, 0, 1)[None]
                y = np.asarray(model({model.input(0): x})[model.output(0)])[0]
                tiles += 1
                y = np.clip(y.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
                up = Image.fromarray(y, "RGB").crop((0, 0, pw * scale, ph * scale))
                # Trim the seam margin on inner edges only: the outer border of
                # the image has no neighbouring tile to disagree with.
                tx = 0 if left == 0 else overlap // 2
                ty = 0 if top == 0 else overlap // 2
                out.paste(up.crop((tx * scale, ty * scale, up.width, up.height)),
                          ((left + tx) * scale, (top + ty) * scale))
        infer_ms = round((time.perf_counter() - t0) * 1000, 1)
        return out, {
            "infer": infer_ms,
            "tiles": tiles,
            "input_width": width, "input_height": height,
            "output_width": out.width, "output_height": out.height,
            "scale": float(scale),
        }

    def _upscale_omz(self, image: Image.Image):
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, 640 / width, 360 / height)
        fit_w = max(1, round(width * scale))
        fit_h = max(1, round(height * scale))
        fitted = _resize(image, (fit_w, fit_h), Image.Resampling.LANCZOS)

        # Edge padding avoids a black border influencing the convolution near
        # the crop while keeping the model's required static landscape shape.
        arr = np.asarray(fitted)
        pad_x = 640 - fit_w
        pad_y = 360 - fit_h
        left, top = pad_x // 2, pad_y // 2
        right, bottom = pad_x - left, pad_y - top
        padded = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="edge")
        bicubic = _resize(Image.fromarray(padded), (1920, 1080),
                          Image.Resampling.BICUBIC)

        # The Open Model Zoo model uses BGR inputs in the 0..255 range and
        # returns RGB-like values near 0..1; forgetting the output *255 makes
        # a technically valid but nearly black image.
        low = padded[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None]
        high = np.asarray(bicubic)[:, :, ::-1].astype(np.float32)
        high = high.transpose(2, 0, 1)[None]
        model = self._model("upscale")
        t0 = time.perf_counter()
        output = np.asarray(model({model.input(0): low, model.input(1): high})[
            model.output(0)])[0]
        infer_ms = round((time.perf_counter() - t0) * 1000, 1)
        output = np.clip(output.transpose(1, 2, 0)[:, :, ::-1] * 255.0,
                         0, 255).astype(np.uint8)
        crop = output[top * 3:(top + fit_h) * 3,
                      left * 3:(left + fit_w) * 3]
        result = Image.fromarray(crop, "RGB")
        return result, {
            "infer": infer_ms,
            "input_width": width,
            "input_height": height,
            "output_width": result.width,
            "output_height": result.height,
            "scale": round(result.width / width, 3),
        }

    def detect(self, image: Image.Image, threshold=0.99):
        image = image.convert("RGB")
        width, height = image.size
        sample = np.asarray(_resize(image, (512, 512)), dtype=np.float32) / 255.0
        sample = sample.transpose(2, 0, 1)[None]
        model = self._model("detect")
        t0 = time.perf_counter()
        result = model({"images": sample})
        infer_ms = round((time.perf_counter() - t0) * 1000, 1)
        boxes = np.asarray(result[model.output("bboxes")])[0]
        labels = np.asarray(result[model.output("labels")])[0]
        scores = np.asarray(result[model.output("scores")])[0]
        detections = []
        for box, class_id, score in zip(boxes, labels, scores):
            score = float(score)
            class_id = int(class_id)
            if score < threshold or class_id <= 0:
                continue
            x1, y1, x2, y2 = np.clip(box.astype(float), 0.0, 1.0)
            if (x2 - x1) * (y2 - y1) < 0.0002:
                continue
            detections.append({
                "class_id": class_id,
                "label": (_COCO_LABELS[class_id] if class_id < len(_COCO_LABELS)
                          else f"class_{class_id}"),
                "score": round(score, 4),
                "box": [round(float(x1 * width), 1), round(float(y1 * height), 1),
                        round(float(x2 * width), 1), round(float(y2 * height), 1)],
            })
        detections = _class_nms(detections)
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()
        line = max(2, round(min(width, height) / 260))
        for item in detections:
            x1, y1, x2, y2 = item["box"]
            color = (48, 199, 224)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line)
            label = f"{item['label']} {item['score']:.0%}"
            label_box = draw.textbbox((x1, y1), label, font=font)
            text_h = label_box[3] - label_box[1] + 6
            text_w = label_box[2] - label_box[0] + 8
            y_text = max(0, y1 - text_h)
            draw.rectangle((x1, y_text, x1 + text_w, y_text + text_h), fill=color)
            draw.text((x1 + 4, y_text + 3), label, fill=(8, 18, 22), font=font)
        return annotated, detections, {"infer": infer_ms}

    def layout(self, image: Image.Image, threshold=_LAYOUT_THRESHOLD):
        """Segment a page into labelled regions, in the model's reading order.

        Returns (regions, timings). Each region carries label, score, a box in
        original page pixels, and `order` — PP-DocLayoutV3 emits a reading-order
        value alongside every box, and that is what makes this useful for
        assembling text rather than merely drawing rectangles: sorting by it
        recovers document order even in multi-column layouts, where sorting by
        y alone interleaves the columns.

        Two constants here were established by measurement, not from the
        model's config, and both fail silently if wrong:

        * **The image must be scaled to [0, 1].** `inference.yml` says
          `norm_type: none` with mean 0 / std 1, which reads as "feed raw
          pixels". Measured on a synthetic page: with /255 the model returns
          five regions at 0.81-0.97 confidence; without it, zero above 0.02.
          "Runs fine, finds nothing" is exactly the failure the OCR notes warn
          about.
        * **The output is `[class_id, score, x1, y1, x2, y2, order]`.**
          PaddleDetection DETR usually emits six columns; the seventh was
          identified by checking it against the y-coordinates on a known page,
          where it came out perfectly monotonic top-to-bottom.

        No NMS: this is a DETR set prediction, so boxes do not duplicate the
        way an anchor model's do (`detect()` needs `_class_nms`; this does not).
        """
        image = image.convert("RGB")
        width, height = image.size
        sample = np.asarray(_resize(image, (800, 800)), dtype=np.float32) / 255.0
        model = self._model("layout")
        inputs = {
            "image": sample.transpose(2, 0, 1)[None],
            "im_shape": np.array([[800.0, 800.0]], dtype=np.float32),
            # Maps detections in the 800x800 canvas back to page pixels.
            "scale_factor": np.array([[800.0 / height, 800.0 / width]],
                                     dtype=np.float32),
        }
        t0 = time.perf_counter()
        raw = np.asarray(model(inputs)[model.output(0)])
        infer_ms = round((time.perf_counter() - t0) * 1000, 1)

        regions = []
        for row in raw:
            score = float(row[1])
            if score < threshold:
                continue
            class_id = int(row[0])
            x1, y1, x2, y2 = (float(v) for v in row[2:6])
            regions.append({
                "label": (_LAYOUT_LABELS[class_id]
                          if 0 <= class_id < len(_LAYOUT_LABELS)
                          else f"class_{class_id}"),
                "class_id": class_id,
                "score": round(score, 4),
                "box": [round(max(0.0, x1), 1), round(max(0.0, y1), 1),
                        round(min(float(width), x2), 1),
                        round(min(float(height), y2), 1)],
                "order": float(row[6]),
            })
        regions.sort(key=lambda r: r["order"])
        return regions, {"infer": infer_ms}

    def _tokenizer_for(self, task):
        """Lazy tokenizer for a text task, cached per task.

        Loaded from the model's own directory rather than a hardcoded name, so
        it cannot drift out of sync with TIERS.
        """
        if task not in self._tokenizers:
            from transformers import AutoTokenizer
            self._tokenizers[task] = AutoTokenizer.from_pretrained(
                str(self.paths[task].parent), local_files_only=True)
        return self._tokenizers[task]

    def _token_feed(self, model, encoded):
        """Encoded text -> the int64 ports this model actually declares.

        Built from the compiled model's own inputs because the two text models
        disagree: the embedder takes token_type_ids and the reranker does not.
        Feeding a port that does not exist is an error, and silently omitting
        one that does is worse.
        """
        feed = {}
        for port in model.inputs:
            name = port.get_any_name()
            value = encoded.get(name)
            if value is None:
                value = np.zeros((1, 256), dtype=np.int64)
            feed[name] = np.asarray(value).astype(np.int64)
        return feed

    def embed(self, texts):
        self._require("search")
        tokenizer = self._tokenizer_for("search")
        model = self._model("search")
        vectors = []
        t0 = time.perf_counter()
        for text in texts:
            encoded = tokenizer(
                text, padding="max_length", truncation=True, max_length=256,
                return_tensors="np")
            inputs = self._token_feed(model, encoded)
            hidden = np.asarray(model(inputs)[model.output(0)])[0]
            mask = inputs["attention_mask"][0, :, None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=0) / max(float(mask.sum()), 1.0)
            norm = np.linalg.norm(pooled)
            vectors.append(pooled / max(float(norm), 1e-12))
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        return np.asarray(vectors, dtype=np.float32), {"infer": elapsed}

    def rerank(self, query, passages):
        """Score (query, passage) pairs with a cross-encoder. Higher is better.

        Not an embedding model: it reads the pair jointly and emits one logit,
        which is why it can order candidates a bi-encoder cannot. Measured on
        this project's own corpus it took top-1 from 7/9 to 8/9 and MRR from
        0.889 to 0.944, while swapping the *embedder* for a bigger one changed
        nothing (docs/archive/NPU-UTIL-PHASE0.md, phase 4).

        Scores are raw logits — comparable to each other for one query, not
        across queries, and not probabilities. Callers should sort, not
        threshold.
        """
        self._require("rerank")
        tokenizer = self._tokenizer_for("rerank")
        model = self._model("rerank")
        scores = []
        t0 = time.perf_counter()
        for passage in passages:
            encoded = tokenizer(
                query, passage, padding="max_length", truncation=True,
                max_length=256, return_tensors="np")
            out = model(self._token_feed(model, encoded))[model.output(0)]
            scores.append(float(np.asarray(out).ravel()[0]))
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        return scores, {"infer": elapsed}

    def generate(self, prompt, steps=12, seed=0):
        self._require("generate")
        if self._generator is None:
            import openvino_genai as ovg
            props = {"CACHE_DIR": str(self.cache_dir)} if self.cache_dir else {}
            # Construct, reshape, then compile. Passing the device to the
            # constructor compiles the dynamic text encoder immediately, which
            # the NPU rejects (two dynamic output-bound dimensions).
            # The pipeline takes the directory; TIERS names model_index.json
            # inside it, so that availability can test for a real file.
            self._generator = ovg.Text2ImagePipeline(
                str(self.paths["generate"].parent))
            self._generator.reshape(1, 512, 512, 7.5)
            self._generator.compile(self.device, **props)
        kwargs = {"num_inference_steps": int(steps), "width": 512,
                  "height": 512, "rng_seed": int(seed)}
        t0 = time.perf_counter()
        result = self._generator.generate(prompt, **kwargs)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        data = np.asarray(result.data)[0]
        if data.ndim == 3 and data.shape[0] in (3, 4):
            data = data.transpose(1, 2, 0)
        if data.dtype != np.uint8:
            multiplier = 255.0 if float(data.max()) <= 1.5 else 1.0
            data = np.clip(data * multiplier, 0, 255).astype(np.uint8)
        return Image.fromarray(data[:, :, :3], "RGB"), {"infer": elapsed}

    def unload(self):
        self._models.clear()
        self._tokenizers.clear()
        self._generator = None
        gc.collect()
