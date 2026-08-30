# Util tab — Phase 0 probe results (2026-08-12)

Gate for the Util tab (OCR/vision utilities on the NPU). Nothing goes into
`locally.py` until it passes `scripts/npu-probe.py`. **All four OCR candidates
passed on all three devices.**

Machine: Core Ultra X7 358H (Panther Lake) / Arc B390 iGPU / Intel AI Boost NPU,
OpenVINO 2026.3, genai 2026.3.

## Why a probe at all

Intel publishes **no list of NPU-validated non-LLM models** — the official NPU
device doc states only "only models with static shapes are supported" plus a dtype
table, and names no vision models. There is no authority to defer to, so every
candidate is a measurement.

The constraint that drives the design, measured here rather than assumed:

```
[?,3,32,320] on NPU -> [NPU_VCL] Compiler returned msg:
                       Missing upper bound for one or more nodes
model.reshape([4,3,32,320]) -> compiles in 0.58 s
```

Both PP-OCRv6 ONNX files ship **fully dynamic** (`x [?,3,?,?]` for detection,
`x [?,3,48,?]` for recognition), so the reshape is mandatory, not optional.

## Results

`python scripts/npu-probe.py --all --device {NPU,GPU,CPU}`

| model | ONNX | static shape | NPU | GPU | CPU |
|---|---|---|---|---|---|
| `ocr-det` PP-OCRv6 medium | 62.0 MB | 1×3×736×736 | **26.2 ms** | 11.3 ms | 114.5 ms |
| `ocr-rec` PP-OCRv6 medium | 76.6 MB | 1×3×48×320 | **4.8 ms** | 4.0 ms | 18.1 ms |
| `ocr-det-small` | 9.9 MB | 1×3×736×736 | **12.6 ms** | 6.5 ms | — |
| `ocr-rec-small` | 21.2 MB | 1×3×48×320 | **3.3 ms** | 3.1 ms | — |

**Compile caching works on the NPU** (it advertises `EXPORT_IMPORT`), and it
matters: detection **31.9 s cold → 0.39 s cached**, recognition **7.25 s → 0.03 s**.
Pass `CACHE_DIR` for these models exactly as the LLM slots do.

## The GPU is faster than the NPU here — keep the NPU default anyway

Detection is **~2.3× faster on the GPU** (11.3 vs 26.2 ms). That is a real result
and the docs should not imply otherwise. The NPU stays the default for reasons that
have nothing to do with throughput:

- Both are *instant* at human timescale — a full page is ~80 ms on the NPU.
- The GPU is the scarce resource: it is holding a 15 GB coder model. OCR that runs
  on the NPU costs the LLM nothing.
- The NPU is the low-power engine and has its own DMA path
  (`.claude/memory/npu_memory_path_isolation.md`).

**Still unverified**: that NPU OCR is genuinely *free* while the GPU serves an LLM.
That is the feature's premise and it needs the concurrent measurement described in
the plan. If NPU OCR degrades GPU tok/s, the premise is wrong.

CPU is 4–10× slower than the NPU and is not a useful engine tier for this.

## Accuracy

**Synthetic page (known ground truth), NPU — 4/4 lines byte-exact:**

```
[1.00] locally util probe
[0.97] Total: $42.99
[0.99] PP-OCRv6 on Intel NPU
[1.00] invoice 2026-08-12
```

Detection 46 ms, recognition 34 ms (8 ms/line).

**Real terminal screenshot** (`screenshots/openclaw-1-*.png`, powerline glyphs and
box drawing — a deliberately hard case), NPU, 20 regions in 46 ms:

| read | conf | verdict |
|---|---|---|
| `.\start-openclaw.ps1` | 1.00 | exact |
| `locally ready.` | 1.00 | exact |
| `session agent:main:main` | 1.00 | exact |
| `Device: GPU` | 0.98 | exact |
| `Launching OpenCLAW(chat)` | 0.96 | exact |
| `S]` | 0.68 | garbage (decorative glyph) |
| `HIda?` | 0.56 | garbage |
| `73` | 0.43 | garbage |

**Every wrong read is low-confidence; every correct read is ≥0.96.** So the server
should filter recognition output at roughly **conf ≥ 0.8** — that alone removes the
decorative-glyph noise without touching real text. Worth re-checking on a
photographed page before hard-coding the number.

## Implementation notes discovered here

**The character dict is off by two, and it matters.** `inference.yml` carries 18708
entries but the model emits 18710 classes. The layout is PaddleOCR's
`CTCLabelDecode`:

```
index 0          CTC blank
index 1..18708   character_dict[i-1]
index 18709      SPACE  (appended by use_space_char — NOT in the dict)
```

Miss the last row and the model reads perfectly *except every space vanishes*
(`locallyutilprobe`), which reads like a tokenizer bug rather than an off-by-one.
`_load_char_dict` appends the space for this reason.

**Preprocessing comes from each model's own `inference.yml`, not from guesswork** —
a wrong mean shows up as "runs fine, finds no text", which looks like an NPU
failure:
- detection: BGR, `scale 1/255`, mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, HWC→CHW
- recognition: `RecResizeImg` to 3×48×320, `(x/255 - 0.5)/0.5`
- detection post: `DBPostProcess`, `thresh 0.2`, `box_thresh 0.45`, `unclip_ratio 1.4`

**Detection reference shape is 736×736, not the paper's 640×640** — that is the
midpoint of the `trt_dynamic_shapes` range in `inference.yml` (32² to 4000²).

**The probe's box extraction is a stand-in.** It thresholds the probability map and
uses `scipy.ndimage.label` for connected components, padding boxes 25% to
approximate the unclip. Real `DBPostProcess` offsets polygons (`pyclipper`). Good
enough to prove text is found and read; the server should do the real thing.

## Phase 1 results (server integration)

`UtilSlot` + `POST /v1/util/read` are in. Verified with Qwen3-8B-int4-cw **and**
PP-OCRv6 both resident on the NPU:

```
locally ready
  NPU  : Qwen3-8B-int4-cw (LLM) -- Intel(R) AI Boost
  NPU  : PP-OCRv6 (UTIL)        -- Intel(R) AI Boost
```

| check | result |
|---|---|
| multipart `file=@page.png` | 200, `X-Device: NPU`, text exact |
| JSON `{"image": "data:image/png;base64,…"}` | 200, same text |
| `engine=gpu` when not loaded | 503 naming `--util-models-dir` |
| unknown engine / missing image | 400 with a usable message |
| `/health` → `util.npu` | `ready`, `last_ms` populated |
| `POST /v1/models/unload` | drops it generically: `PP-OCRv6 162 MB` next to `Qwen3-8B 4529 MB` |
| request after unload | auto-reloads and answers in 2.3 s total |

**Two models on the NPU cost 162 MB on top of the 4529 MB LLM** — the residency
claim holds.

### The bug this phase found: concurrent NPU inference kills the device

Hammering `/v1/util/read` *while* `/v1/chat/completions` was generating produced
`ZE_RESULT_ERROR_DEVICE_LOST ... device hung, reset, was removed` and took the
whole server process down. Residency was never the problem — the same two slots
had already served interleaved requests happily; the crash landed on the first
pair that actually *overlapped*.

Flask runs `threaded=True`, so this was reachable by any user typing in Chat
while the Util tab read an image. Fixed by `_device_lock()`: every NPU slot
shares one lock, GPU/CPU keep one per slot. Full write-up in `TODONT.md`.

**The fix is free.** Identical stress after it — three threads hammering OCR
across three chat turns — gives 12 concurrent OCR requests, **0 errors, chat
7.89 s/turn vs 7.93 s solo (−1%, noise)**. OCR takes ~150 ms and fills the gaps
between turns.

This also settles part of the parallel-device question, though not all of it:
NPU utilities do not slow an **NPU-resident** LLM. The original premise — that
they don't disturb a **GPU-resident** LLM — still needs its own measurement.

## Phase 2 results (Util UI + document reader)

The web UI now has Chat / Voice / Util tabs. Util keeps its file and result when
switching tabs, defaults to NPU OCR, disables unavailable engines from `/health`,
and can copy/download Markdown or send it to Chat with a question. Verified in
the real browser at desktop and 390×844 mobile widths: paste image → OCR → copy
→ Ask in Chat completed end to end, keyboard tab navigation worked, there was no
horizontal overflow, and the browser console stayed clean.

`/v1/util/read` now distinguishes two fundamentally different jobs:

- images and PDF pages without a text layer → PP-OCRv6 on NPU/GPU;
- PDF/DOCX/PPTX/XLSX/XLS/HTML/EPUB/CSV/text with embedded content → local
  MarkItDown stream conversion, plugins disabled, upload held in memory.

Native parsing returns `engine: null`, `source: native`, and `X-Device: LOCAL`.
That label matters: parsing ZIP/XML/plain text is not NPU inference and the UI
must not turn accelerator use into marketing fiction. In-memory DOCX, PPTX,
XLSX, HTML, CSV, and TXT fixtures all converted correctly. An image-only PDF
was detected as scanned and OCRed through the real NPU endpoint (4 regions,
correct title/amount/date). PDFium renders scan pages at 144 DPI; only the first
50 scanned pages are rendered/OCRed, and the response reports truncation.

## Phase 3 results (remaining utilities + sidebar)

The app now uses one persistent left sidebar. Chat, Voice, Read document,
Remove background, Upscale, Detect objects, Generate image, and Search files
are direct peers. The previous large centered Util header and nested utility
list were removed. Copy is literal and task-oriented; phrases such as “Useful
work, off the cloud” and “Local models, named silicon” were removed.

Four more pipelines were run on the real NPU through the same
`utility_pipeline.py` used by Flask:

| task | selected model | fixed input | measured result |
|---|---|---|---|
| portrait matting | Xenova MODNet FP16 (Apache-2.0) | 1×3×512×512 | 91.2 ms; alpha 0…255; visual cutout correct |
| super-resolution | OMZ single-image-super-resolution-1033 (Apache-2.0) | 640×360 + 1920×1080 | 110.1 ms; 135×180 → 405×540; visual output correct |
| detection | OpenVINO RF-DETR Small INT8 (Apache-2.0) | 1×3×512×512 | 46.5 ms; bus + four people correct at 0.99 |
| embeddings | MiniLM-L6 INT8 (Apache-2.0) | three 1×256 tensors | 27.5 ms for three strings; related 0.514 vs unrelated 0.022 |

The Flask test client then exercised `/background`, `/detect`, `/index`, and
`/search` through a real `UtilSlot@NPU`. All returned 200. Semantic search put
the OCR document first (cosine 0.684) and the banana document second (−0.030).

Implementation details:

- non-OCR models compile lazily; startup only checks files and loads OCR;
- every NPU utility uses the same global inference lock as NPU chat;
- image results return in memory without writing uploads to disk;
- search indexes are in-memory only, bounded to 8 indexes, 20 files per index,
  and 500 chunks per index;
- `/health.util.<engine>.tasks` drives availability and disables missing or
  rejected models in the UI.

## Phase 4 results (text tiers: embeddings and reranking, 2026-08-13)

Both candidates are OpenVINO IR rather than ONNX and take int64 token ports
rather than one float image, so `npu-probe.py` grew `model_file`, dict-keyed
reshape, and per-port dummy inputs. The four OCR candidates were re-probed
unchanged afterwards (`ocr-rec-small` 2.7 ms, `ocr-det-small` 9.5 ms).

| model | disk | shape | NPU compile | NPU infer | output |
|---|---|---|---|---|---|
| `OpenVINO/bge-base-en-v1.5-int8-ov` (MIT) | 110.2 MB | 3 × 1×256 | 5.40 s | **12.7 ms** | (1, 256, 768) |
| `OpenVINO/bge-reranker-base-int8-ov` (MIT) | 280.0 MB | 2 × 1×256 | 6.65 s | **13.0 ms** | (1, 1) |

Both compile and run on the NPU, so by the "largest model the NPU can take"
rule both belong there. Note the signatures differ — bge-base is BERT-derived
and takes `token_type_ids`, the reranker is XLM-RoBERTa and does not.

### The embedding upgrade was measured and rejected

`all-MiniLM-L6` was *not* replaced. Nine queries over twelve passages drawn from
this project's own domain, same input to every configuration:

| config | top-1 | MRR | recall@5 |
|---|---|---|---|
| MiniLM-L6 (current) | 7/9 | 0.889 | 1.000 |
| BGE-base (candidate) | 7/9 | 0.870 | 1.000 |
| **BGE + reranker (top-5)** | **8/9** | **0.944** | 1.000 |

BGE-base did not beat MiniLM-L6 while costing ~5× the disk (110 vs 23 MB) and
~40% more latency per string. **The reranker is where the entire gain is.**

`recall@5 = 1.000` for both embedders is the structural explanation, and it is
the number to remember: with indexes bounded to 500 chunks
(`_UTIL_SEARCH_MAX_CHUNKS`), the correct passage is *always* already in the
candidate pool. Retrieval is not the bottleneck at this scale — **ordering is** —
so a cross-encoder over the top-k buys what a bigger bi-encoder does not.

Caveat, so this is not over-read: 9 queries is an eval, not a benchmark, and
BGE-base does outrank MiniLM-L6 on published MTEB results. The claim here is
narrow and specific to this workload — at a 500-chunk ceiling, on this hardware,
the swap does not pay for itself. Revisit if the index bound is ever raised, as
a bigger corpus is exactly where a stronger bi-encoder starts to matter.

An earlier 4-passage version of this test could not separate the two models at
all (margins 0.180 vs 0.168). That said the test was too weak, not that the
models were equal — worth remembering before concluding "no difference" from a
small sample.

### End-to-end through the real endpoints

`/v1/util/index` then `/v1/util/search` via the Flask test client, four
documents, query *"What hardware does expert offloading need?"*:

| mode | top-1 | cosine | cross-encoder |
|---|---|---|---|
| `rerank: false` | **wrong** (`memory.txt`) | +0.3321 vs +0.2740 | — |
| `rerank: true` | **correct** (`offload.txt`) | +0.2740 (lower!) | −0.628 vs −10.088 |

The bi-encoder ranked the wrong document first; the cross-encoder separated them
by **9.5** in the right direction. Reranking cost 177 ms for four passages
including first-call warmup (~13 ms/passage steady-state, per the probe).

Two design consequences fell out of this and are worth keeping:

- **Retrieve wide, then reorder.** `_UTIL_RERANK_POOL = 20` candidates go to the
  cross-encoder regardless of `top_k`, because recall was never the problem.
- **Return both scores.** Once reranked, results come back with a *lower* cosine
  ranked above a higher one — visibly contradictory unless the number the sort
  actually used is present. `score` stays the embedding cosine in both modes and
  `rerank_score` appears alongside it, with a `reranked` flag on the response.

Reranking is an improvement, not a dependency: `UtilityUnavailable` is caught and
search falls back to embedding order rather than 503-ing on a feature the caller
never asked for by name.

## Phase 5 results (document stack, 2026-08-13)

Four candidates probed; **two passed, two are blocked upstream**. All four are
Apache-2.0. Shapes came from each model's own `inference.yml`, per the rule
above.

| model | disk | static shape | NPU compile (cold→cached) | NPU | GPU |
|---|---|---|---|---|---|
| `PP-DocLayoutV3_onnx` | 130.5 MB | im_shape 1×2, image 1×3×800×800, scale_factor 1×2 | 38.75 s → **0.96 s** | **81.7 ms** | 43.0 ms |
| `UVDoc_onnx` | 31.7 MB | 1×3×256×128 | 7.05 s → **1.02 s** | **18.9 ms** | 3.4 ms |
| `SLANeXt_wired_onnx` | — | — | — | **fails `read_model`** | same |
| `PP-FormulaNet_plus-L_onnx` | — | — | — | **no weights published** | same |

Both survivors run on the NPU, so both stay there — the GPU is faster (1.9× on
layout, 5.6× on dewarp), which is the same pattern OCR showed, and the same
answer applies: instant is instant at human timescale, and NPU work leaves the
GPU free. Compile caching matters here more than anywhere yet: layout is a
**40× cold-to-cached** difference.

Full write-ups of the two failures are in TODONT.md. In short: SLANeXt is
autoregressive and its paddle2onnx `while` export cannot be imported by
OpenVINO at all (not a device problem — it fails before device selection), and
the FormulaNet ONNX repo contains only a README.

**PP-DocLayoutV3 has three input ports, not the two its `inference.yml` names.**
Leaving `im_shape` unbounded did not raise — the NPU compiler took the process
down with an access violation. Bound every port the *model* declares, not every
port the config mentions.

### Layout: two constants that had to be measured, not read

Both fail *silently* if wrong, which is the whole reason the gate exists.

- **The image must be scaled to [0, 1].** `inference.yml` says
  `norm_type: none`, mean 0, std 1 — which reads as "feed raw pixels". On a
  synthetic page: **with `/255`, five regions at 0.81–0.97; without it, zero
  above 0.02.** Same "runs fine, finds nothing" signature as a wrong OCR mean.
- **The output is `[class_id, score, x1, y1, x2, y2, order]`.** PaddleDetection
  DETR normally emits six columns. The seventh was identified by checking it
  against a known page: its values (28, 70, 85, 101, 138) came out perfectly
  monotonic with the regions' y-coordinates. It is the reading order, and it is
  the most valuable column — sorting text by y alone interleaves the columns of
  a two-column page, while this does not.

No NMS is applied: DETR is a set prediction, so boxes do not duplicate the way
`detect()`'s anchor-based output does.

Verified end to end through `/v1/util/read` on a page with known ground truth:
5/5 regions, correct reading order, table localised to (89,612)-(1109,864)
against an actual (90,612)-(1110,860). Structured Markdown marks headings and
flags table/chart/figure/formula regions instead of presenting their contents as
prose — honest, since without SLANeXt the cell order inside a table really is
scrambled. Layout is strictly additive: any failure falls back to the plain
reading rather than costing the caller text OCR already read correctly.

### UVDoc passed the probe but is NOT integrated

It compiles and runs (18.9 ms NPU), but its `inference.yml` contains **only**
TensorRT shape hints — no `Preprocess` block, no postprocess, nothing. Every
other model here had its constants read off its own config, and the two constants
above are a live demonstration of what guessing costs. Two unknowns remain:
the real input resolution (the published optimum is 1×3×**256×128**, an odd
aspect for a page) and whether the 3-channel output is a dewarped image or a
sampling grid to be applied with `remap`. Integrating it means determining both
against warped pages with known ground truth — a separate piece of work, not a
line of wiring.

## Phase 6 results (image tiers, 2026-08-13)

| model | NPU | GPU | outcome |
|---|---|---|---|
| `onnx-community/swin2SR-realworld-sr-x4-64-bsrgan-psnr-ONNX` fp16, 28 MB (Apache-2.0) | **compile rejected** | compile 36.8 s, **808 ms** per 256² tile | **GPU tier** |
| `onnx-community/BiRefNet_lite-ONNX` fp16, 114 MB (MIT) | compile never finished (killed at 9.5 min) | compiles, **cannot execute** | **blocked — MODNet stays** |

Both failures are written up in TODONT.md.

### Swin2SR replaced Real-ESRGAN as the plan's upscaler

The plan called for converting `xinntao/Real-ESRGAN` locally, because every
Real-ESRGAN ONNX on the hub is published by an individual rather than an org.
Swin2SR is **Apache-2.0 at `onnx-community` and at upstream `caidas`**, which is
the provenance the conversion existed to buy — so the conversion was dropped.
Its preprocessing is fully specified (`rescale 1/255`, **no** mean/std, pad to
multiples of 8, `upscale: 4`, `window_size: 8`).

### This is the tier system's first real per-device split

`upscale` now resolves differently per engine, and the difference is not
cosmetic — measured through `UtilityEngine`:

| source | NPU (OMZ 1033) | GPU (Swin2SR) |
|---|---|---|
| 320×200 | 960×600 (3.00×) | 1280×800 (**4.00×**) |
| 960×640 | 1620×1080 (**1.69×**) | 3840×2560 (**4.00×**) |

The 1.69× is the old ceiling made visible: `_upscale_omz` downscales anything
larger than 640×360 *on the way in*, so a bigger source buys less magnification,
not more. `_upscale_swin2sr` holds a true 4× at any size — 960×640 becomes
3840×2560 in 15 tiles / 15.5 s, roughly 1 s per tile and linear in pixels.

Tiling correctness was checked visually across a tile boundary (stride 224
source px → 896 output px): a curved edge crosses the seam continuously with no
brightness step. Inner edges trim `_SWIN_OVERLAP // 2`; outer image edges do
not, having no neighbour to disagree with. Tiles are edge-replicated to the
static shape rather than zero-padded, for the same reason the OMZ path pads with
`edge` — black padding darkens the border.

## Rejected candidates

- MODNet INT8 compiled but returned an almost constant alpha mask (`0.123`) on
  NPU. The FP16 model produced the same correct mask as CPU.
- YOLO11n INT8 was fast (~11.5 ms) but was removed because its AGPL license is
  a poor default. RF-DETR Small is Apache-2.0.
- Stable Diffusion 1.5 INT8 image generation is disabled. Direct NPU compile
  failed because the text encoder had two dynamic output-bound dimensions.
  Calling `reshape(1, 512, 512, 7.5)` first avoided that error but compiled for
  more than 22 minutes without producing a model or image. The Intel GPU
  fallback exited during compile. The downloaded 1.08 GB snapshot was removed.

## Still worth probing later

Document layout recovery (PP-DocLayoutV3), formula-to-LaTeX, non-portrait
background removal, and a newer/static OpenVINO diffusion pipeline. Each still
needs compile, inference, and correctness checks on the target NPU before its UI
status can change to ready.

## Model and runtime references

- [OpenVINO NPU inference constraints](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)
- [OpenVINO GenAI image-generation API](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai.html)
- [MODNet model card](https://huggingface.co/Xenova/modnet)
- [Open Model Zoo super-resolution model](https://docs.openvino.ai/2023.3/omz_models_model_single_image_super_resolution_1033.html)
- [RF-DETR Small INT8 OpenVINO model](https://huggingface.co/OpenVINO/rfdetr_small-int8-ov)
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Stable Diffusion 1.5 INT8 OpenVINO model](https://huggingface.co/OpenVINO/stable-diffusion-v1-5-int8-ov)
