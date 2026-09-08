#!/usr/bin/env python3
"""ocr_pipeline.py — PP-OCRv6 text detection + recognition on one OpenVINO device.

Lives outside locally.py deliberately, against the project's one-file default:
`scripts/npu-probe.py` is the gate that decides whether this pipeline may be
served at all, so the probe and the server must run *the same* code. Inlining
this in locally.py would leave the gate validating a copy that drifts from what
actually runs.

Everything here is shaped by one hardware fact, measured on a Core Ultra X7 358H
(2026-08-12, OpenVINO 2026.3): **the NPU rejects dynamic shapes.** A `[?,3,?,?]`
input fails to compile with `[NPU_VCL] ... Missing upper bound for one or more
nodes`. Both PP-OCRv6 ONNX files ship fully dynamic, so each is reshaped to a
fixed size before compile — the detector to a square letterbox canvas, the
recogniser to its native 3x48x320 line image.

Measured on that NPU: detection 26 ms/page, recognition 5 ms/line, a full page
in ~80 ms. See docs/archive/NPU-UTIL-PHASE0.md for the full gate results.

Preprocessing constants are read from each model's own inference.yml rather than
hardcoded from a paper — a wrong mean shows up as "runs fine, finds no text",
which is easy to misread as an NPU failure.
"""

import os
import time

import numpy as np
import openvino as ov
from PIL import Image

try:
    import yaml
except ImportError:                                     # pragma: no cover
    yaml = None

try:
    from scipy import ndimage
except ImportError:                                     # pragma: no cover
    ndimage = None


# Detection normalisation, from PP-OCRv6_det's inference.yml NormalizeImage op.
DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Recognition input height is fixed by the model (RecResizeImg: 3 x 48 x 320).
# The width is not: the ONNX ships dynamic (`[?,3,48,?]`) and PaddleOCR's own
# config ranges to 48x3200, i.e. wide lines are meant to use a WIDER input, not
# be cut into 320px pieces.
#
# The NPU can't take a dynamic width, so compile one static model per bucket and
# route each line to the smallest that fits it at native scale. This replaced a
# splitting approach that was strictly worse: cutting a line produced doubled
# text at the seams ("probe" -> "probbe"), and cutting at whitespace instead
# traded that for spurious spaces and stray edge glyphs ("Cod ler"). Bucketing
# has no seams at all.
#
# Verified on NPU (2026-08-12): 320 -> 5.5 ms, 640 -> 9.3 ms, 1280 -> 16.0 ms;
# cold compile 0.07/6.9/18.9 s, ~0.1 s once the OV cache is warm.
REC_H = 48
REC_BUCKETS = (320, 640, 1280)

# DBPostProcess defaults, also from the detector's own inference.yml.
DB_THRESH = 0.2
DB_BOX_THRESH = 0.45
DB_UNCLIP_RATIO = 1.4

# Recognition confidence floor. Measured on a terminal screenshot with powerline
# glyphs: every correct read scored >= 0.96, every garbage read <= 0.68. 0.8
# splits them cleanly. Tune with real pages before trusting it further.
MIN_CONF = 0.8


class OcrError(RuntimeError):
    """Raised for anything a caller should surface verbatim to the user."""


def _require_deps():
    if yaml is None:
        raise OcrError("pyyaml is required for OCR (pip install pyyaml)")
    if ndimage is None:
        raise OcrError("scipy is required for OCR (pip install scipy)")


# --- geometry --------------------------------------------------------------

def _letterbox(img, size):
    """Fit img into a size x size canvas, preserving aspect. -> (canvas, scale).

    The detector is fully convolutional, so any square canvas works; padding
    rather than stretching keeps the aspect ratio the recogniser expects when
    the crops come back out.
    """
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img.resize((nw, nh), Image.LANCZOS), (0, 0))
    return canvas, scale


def _unclip(x0, y0, x1, y1, ratio, bounds):
    """Expand a box the way DBPostProcess does, without pulling in pyclipper.

    DB shrinks text regions at training time, so every detected box is smaller
    than the glyphs it covers and must be grown back or the recogniser reads
    clipped characters. The real implementation offsets the polygon by
    `area * ratio / perimeter`; for the axis-aligned rectangles this pipeline
    produces, that distance has a closed form, so the same rule applies exactly
    with no polygon library.
    """
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    dist = (w * h * ratio) / (2.0 * (w + h))
    W, H = bounds
    b = (max(0, int(x0 - dist)), max(0, int(y0 - dist)),
         min(W, int(x1 + dist)), min(H, int(y1 + dist)))
    return b if b[2] > b[0] and b[3] > b[1] else None


def _db_postprocess(prob, scale, orig_size, thresh=DB_THRESH,
                    box_thresh=DB_BOX_THRESH, unclip_ratio=DB_UNCLIP_RATIO,
                    min_area=12):
    """Probability map -> boxes in ORIGINAL image coordinates.

    Threshold, label connected components, score each by its mean probability,
    unclip, and map back through the letterbox scale.
    """
    mask = prob > thresh
    lab, n = ndimage.label(mask)
    if not n:
        return []

    boxes = []
    for idx, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        if (x1 - x0) * (y1 - y0) < min_area:
            continue
        region = lab[sl] == idx
        score = float(prob[sl][region].mean())
        if score < box_thresh:
            continue
        b = _unclip(x0 / scale, y0 / scale, x1 / scale, y1 / scale,
                    unclip_ratio, orig_size)
        if b:
            boxes.append((b, score))
    return boxes


# --- reading order ---------------------------------------------------------

def _group_lines(items, tol=0.6):
    """Group boxes into text lines, then sort each line left-to-right.

    Sorting purely by y is wrong the moment a page has two columns or a box
    sits a few pixels higher than its neighbour. Grouping by vertical overlap
    relative to box height handles both: two boxes share a line when they
    overlap vertically by more than `tol` of the shorter one.
    """
    if not items:
        return []
    rest = sorted(items, key=lambda it: it["bbox"][1])
    lines = []
    for it in rest:
        y0, y1 = it["bbox"][1], it["bbox"][3]
        placed = False
        for line in lines:
            ly0 = min(b["bbox"][1] for b in line)
            ly1 = max(b["bbox"][3] for b in line)
            overlap = min(y1, ly1) - max(y0, ly0)
            if overlap > tol * min(y1 - y0, ly1 - ly0):
                line.append(it)
                placed = True
                break
        if not placed:
            lines.append([it])
    for line in lines:
        line.sort(key=lambda b: b["bbox"][0])
    lines.sort(key=lambda line: min(b["bbox"][1] for b in line))
    return lines


# --- CTC -------------------------------------------------------------------

def _load_charset(model_dir):
    """Read the recognition charset from inference.yml.

    The dict holds 18708 entries but the model emits 18710 classes, and the gap
    is not padding — it is PaddleOCR's CTCLabelDecode layout:

        0            CTC blank
        1..18708     character_dict[i-1]
        18709        SPACE  (appended by use_space_char, absent from the dict)

    Miss that last row and the model reads perfectly except every space
    disappears ("locallyutilprobe"), which looks like a tokenizer bug rather
    than an off-by-one. Verified on this exact model 2026-08-12.
    """
    path = os.path.join(model_dir, "inference.yml")
    if not os.path.isfile(path):
        raise OcrError(f"missing recognition config: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        chars = list(cfg["PostProcess"]["character_dict"])
    except (KeyError, TypeError):
        raise OcrError(f"no PostProcess.character_dict in {path}")
    return chars + [" "]


def _join_line(line):
    """Join the boxes of one text line, restoring the spacing between them.

    Detection returns each text region separately, so concatenating them loses
    every gap: a terminal status bar reads "tommyllocallymain33.10.11" instead
    of separate fields. The horizontal distance between boxes carries that
    information — measured against glyph height, since that is what sets the
    width of a space in any font.
    """
    if not line:
        return ""
    out = [line[0]["text"]]
    for prev, cur in zip(line, line[1:]):
        gap = cur["bbox"][0] - prev["bbox"][2]
        height = max(1, prev["bbox"][3] - prev["bbox"][1])
        if gap > 1.2 * height:
            out.append("   ")          # a deliberate column break
        elif gap > 0.25 * height:
            out.append(" ")
        out.append(cur["text"])
    return "".join(out).strip()


def _cut_columns(crop, piece_w):
    """Choose split columns for a too-wide line, avoiding cutting any glyph.

    Overlapping the cuts and stitching the seam does NOT work here: the two
    crops read the shared strip *differently*, because a half-visible glyph at
    a boundary is misrecognised. Exact suffix/prefix stitching then can't fire
    and the doubling survives ("Intel" -> "IrIntel", "can" -> "caran").

    So cut where there is no glyph to damage. Take the per-column ink profile
    and, near each target boundary, pick the emptiest column within a window.
    Pieces then tile the line with no overlap and concatenate directly.

    Returns [(x0, x1, gap_before), ...] where gap_before marks a cut made at a
    genuinely blank column — a word space the cut would otherwise swallow.
    """
    w, h = crop.size
    g = np.asarray(crop.convert("L"), dtype=np.float32)
    # Ink per column, normalised: 1.0 = darkest column, 0.0 = blank.
    ink = g.max() - g.mean(axis=0) if g.max() > g.min() else np.zeros(w)
    if ink.max() > 0:
        ink = ink / ink.max()
    blank = float(np.percentile(ink, 10)) + 0.02

    cuts, x = [], 0
    while x < w:
        target = x + piece_w
        if target >= w:
            cuts.append((x, w))
            break
        win = max(4, int(piece_w * 0.15))
        lo, hi = max(x + 1, target - win), min(w - 1, target + win)
        best = int(lo + np.argmin(ink[lo:hi])) if hi > lo else target
        cuts.append((x, best))
        x = best

    return [(a, b, (i > 0 and ink[a] <= blank)) for i, (a, b) in enumerate(cuts)]


def _ctc_decode(logits, charset):
    """Greedy CTC: argmax, collapse repeats, drop blank. -> (text, mean conf)."""
    ids = logits.argmax(axis=-1)[0]
    conf = logits.max(axis=-1)[0]
    out, kept, prev = [], [], -1
    for i, c in zip(ids, conf):
        if i != prev and i != 0:
            j = int(i) - 1
            out.append(charset[j] if 0 <= j < len(charset) else "")
            kept.append(float(c))
        prev = i
    return "".join(out), (float(np.mean(kept)) if kept else 0.0)


# --- the engine ------------------------------------------------------------

class OcrEngine:
    """Compiled PP-OCRv6 detector + recogniser on one device.

    Not thread-safe on its own — the caller (a slot) owns the lock, matching how
    DeviceSlot/WhisperSlot serialise access to their pipelines.
    """

    def __init__(self, models_dir, device="NPU", cache_dir=None,
                 det_size=736, min_conf=MIN_CONF):
        self.models_dir = models_dir
        self.device = device
        self.cache_dir = cache_dir
        self.det_size = det_size
        self.min_conf = min_conf
        self.det = None
        self.rec = None
        self.charset = None

    # -- lifecycle --

    def load(self):
        _require_deps()
        core = ov.Core()
        cfg = {"CACHE_DIR": self.cache_dir} if self.cache_dir else {}

        det_dir = os.path.join(self.models_dir, "ocr-det")
        rec_dir = os.path.join(self.models_dir, "ocr-rec")
        for d in (det_dir, rec_dir):
            if not os.path.isfile(os.path.join(d, "inference.onnx")):
                raise OcrError(
                    f"OCR model not found: {os.path.join(d, 'inference.onnx')}. "
                    f"Fetch it with: python scripts/npu-probe.py --all")

        # Reshape to static BEFORE compile — the NPU cannot take the dynamic
        # shapes these ONNX files ship with.
        det = core.read_model(os.path.join(det_dir, "inference.onnx"))
        det.reshape({det.inputs[0]:
                     ov.PartialShape([1, 3, self.det_size, self.det_size])})
        self.det = core.compile_model(det, self.device, cfg)

        rec_path = os.path.join(rec_dir, "inference.onnx")
        self.rec = {}
        for width in REC_BUCKETS:
            rec = core.read_model(rec_path)
            rec.reshape({rec.inputs[0]: ov.PartialShape([1, 3, REC_H, width])})
            self.rec[width] = core.compile_model(rec, self.device, cfg)

        self.charset = _load_charset(rec_dir)

    def unload(self):
        self.det = self.rec = self.charset = None

    @property
    def ready(self):
        return self.det is not None and bool(self.rec)

    # -- inference --

    def _detect(self, img):
        canvas, scale = _letterbox(img, self.det_size)
        a = np.asarray(canvas, dtype=np.float32) / 255.0
        a = (a - DET_MEAN) / DET_STD
        x = np.ascontiguousarray(a.transpose(2, 0, 1)[None])
        prob = list(self.det(x).values())[0][0, 0]
        return _db_postprocess(prob, scale, img.size)

    def _recognise_crop(self, crop):
        """Read one line image, routed to the narrowest bucket that fits it.

        A line only gets cut when it exceeds the widest bucket (aspect > ~26:1,
        e.g. a full-width line on a 4K screenshot). Those cuts are made at
        whitespace columns, and at 1280px pieces they are rare enough that the
        seam artefacts which made 320px splitting unusable don't accumulate.
        """
        w, h = crop.size
        if h < 1 or w < 1:
            return "", 0.0
        full_w = max(1, int(w * REC_H / h))

        widest = REC_BUCKETS[-1]
        if full_w <= widest:
            return self._rec_once(crop, full_w)

        piece_w = max(8, int(widest * h / REC_H))
        parts, confs = [], []
        for lo, hi, _gap in _cut_columns(crop, piece_w):
            if hi - lo < 2:
                continue
            piece = crop.crop((lo, 0, hi, h))
            pw = max(1, int(piece.size[0] * REC_H / h))
            t, c = self._rec_once(piece, min(widest, pw))
            if t:
                parts.append(t)
                confs.append(c)
        return "".join(parts), (float(np.mean(confs)) if confs else 0.0)

    def _rec_once(self, crop, target_w):
        """Run one crop through the smallest bucket that holds it."""
        bucket = next((b for b in REC_BUCKETS if target_w <= b), REC_BUCKETS[-1])
        target_w = max(1, min(target_w, bucket))
        buf = np.zeros((REC_H, bucket, 3), dtype=np.float32)
        resized = np.asarray(
            crop.resize((target_w, REC_H), Image.LANCZOS),
            dtype=np.float32) / 255.0
        buf[:, :resized.shape[1]] = resized[:, :bucket]
        buf = (buf - 0.5) / 0.5                       # RecResizeImg normalisation
        x = np.ascontiguousarray(buf.transpose(2, 0, 1)[None])
        return _ctc_decode(list(self.rec[bucket](x).values())[0], self.charset)

    def read(self, img):
        """Detect and read all text. -> dict(text, markdown, blocks, timings).

        `blocks` keeps every read with its confidence and box, including ones
        below the confidence floor, flagged rather than dropped — a caller that
        wants everything can have it, but `text`/`markdown` are filtered.
        """
        if not self.ready:
            raise OcrError("OCR engine not loaded")
        if img.mode != "RGB":
            img = img.convert("RGB")

        t0 = time.perf_counter()
        found = self._detect(img)
        det_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        blocks = []
        for bbox, det_score in found:
            text, conf = self._recognise_crop(img.crop(bbox))
            if not text.strip():
                continue
            blocks.append({
                "text": text,
                "conf": round(conf, 3),
                "det_conf": round(det_score, 3),
                "bbox": list(bbox),
                "kept": conf >= self.min_conf,
            })
        rec_ms = (time.perf_counter() - t0) * 1000

        kept = [b for b in blocks if b["kept"]]
        lines = [_join_line(line) for line in _group_lines(kept)]
        lines = [ln for ln in lines if ln]
        text = "\n".join(lines)

        return {
            "text": text,
            "markdown": text,          # layout model (Phase 2) upgrades this
            "blocks": blocks,
            "regions": len(found),
            "dropped_low_conf": len(blocks) - len(kept),
            "timings_ms": {"detect": round(det_ms, 1),
                           "recognise": round(rec_ms, 1)},
        }
