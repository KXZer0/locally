# locally — Device Plan

## Vision

An OpenAI-compatible local LLM server that works on Intel hardware
that nobody else supports well. NPU-first, GPU optional.

**Primary target:** Any Intel laptop with an NPU (Core Ultra series).
No NVIDIA required. No ARC required. Just an NPU and `install.ps1`.

**Bonus:** If you also have an ARC GPU, load a VLM for vision tasks
alongside the NPU chat model. Both run simultaneously.

---

## Device priority

```
1. NPU  — primary device. Every Core Ultra laptop has one.
           LLM chat with streaming. This is the product.

2. GPU  — optional. ARC iGPU or discrete. Adds vision (VLM)
           or runs a larger LLM if no NPU model configured.

3. CPU  — last resort. Slow but always available.
           Useful for testing on machines without NPU/GPU.
```

### Who uses what

| User has | What happens |
|---|---|
| NPU only (most laptops) | LLM on NPU. Chat with streaming. Done. |
| NPU + ARC GPU | LLM on NPU, VLM on GPU. Chat streams, images work. |
| ARC GPU only (no NPU) | Single model on GPU. Current behavior. |
| Nothing (CPU only) | LLM on CPU. Slow but works. |

---

## Proven by test-dual-device.py (2026-04-12)

| Test | Result |
|---|---|
| NPU alone (LLM text) | 1.44s for 50 tokens |
| GPU alone (VLM image) | 2.58s |
| Both simultaneous | NPU 1.85s, GPU 2.81s — minimal interference |
| NPU load time | ~23s |
| GPU load time | ~13s |

Both devices coexist. No crashes, no contention.

---

## CLI

```powershell
# NPU only (the common case)
python arc-server.py --device NPU --model-dir model

# GPU only (current behavior, backwards compatible)
python arc-server.py --device GPU --model-dir model

# Dual: NPU for chat + GPU for vision
python arc-server.py --model-dir model --gpu-model-dir gpu-model

# Auto-detect (picks best available device)
python arc-server.py --model-dir model
```

### Arguments

| Flag | Default | Purpose |
|---|---|---|
| `--model-dir` | `model/` | Primary model (loaded on best available device) |
| `--device` | auto | Force device: `NPU`, `GPU`, or `CPU` |
| `--gpu-model-dir` | none | Secondary GPU model (VLM). Enables dual mode. |
| `--port` | 8000 | Server port |
| `--max-dim` | 768 | Max image dimension for VLM |

### Auto-detect logic

When `--device` is not specified:
1. If `--gpu-model-dir` given → primary on NPU, secondary on GPU
2. Else if NPU available → primary on NPU
3. Else if GPU available → primary on GPU
4. Else → CPU

---

## Startup sequence

1. Check port availability
2. Detect available devices (NPU, GPU, CPU)
3. Start Flask (background loading, /health available immediately)
4. Load primary model on chosen device
5. Load secondary GPU model if `--gpu-model-dir` given
6. Warmup all loaded models
7. Mark ready, print banner

### Banner examples

NPU only:
```
================================================
  locally ready
    NPU : Qwen3-8B (LLM) — Intel AI Boost
    URL : http://localhost:8000
================================================
```

Dual device:
```
================================================
  locally ready
    NPU : Qwen3-8B (LLM) — Intel AI Boost
    GPU : Qwen2.5-VL-3B (VLM) — Intel Arc 140V
    URL : http://localhost:8000
================================================
```

GPU only:
```
================================================
  locally ready
    GPU : Qwen2.5-VL-3B (VLM) — Intel Arc 140V
    URL : http://localhost:8000
================================================
```

---

## Request routing

### Single device mode (NPU or GPU)

Everything goes to the one loaded model. Simple.
- If it's an LLM and images are sent → 400 error
- If it's a VLM → images and text both work

### Dual device mode (NPU + GPU)

**GPU has a VLM (Option A):**
```
POST /v1/chat/completions
     │
     ├── Has images? ──► GPU (VLMPipeline)
     │                    No streaming, short answers
     │
     └── Text only? ──► NPU (LLMPipeline)
                         Streaming, chat quality
```

**GPU has a bigger LLM (Option B):**
```
POST /v1/chat/completions
     │
     ├── Has images? ──► 400 error (no VLM loaded)
     │
     └── Text only? ──► GPU (LLMPipeline)
                         Streaming, best quality
                         NPU sits idle (or use as fallback
                         if GPU is busy)
```

When the GPU has a big LLM, it becomes the primary text device
(better quality, streaming). The NPU is still there as a backup
or for lighter tasks. The user picks by model name:

```json
{"model": "qwen3-14b", ...}     → GPU (big LLM)
{"model": "qwen3-8b", ...}      → NPU (lighter LLM)
```

Without explicit model selection, default to GPU (better quality).

Explicit model selection always overrides auto-routing.

### Streaming behavior

| Device | Pipeline | stream:true |
|---|---|---|
| NPU | LLMPipeline | Yes, SSE token-by-token |
| GPU (LLM) | LLMPipeline | Yes, SSE token-by-token |
| GPU (VLM) | VLMPipeline | Silently returns full response (no error) |

---

## /health endpoint

```json
{
  "status": "ready",
  "devices": {
    "npu": {
      "status": "ready",
      "model": "qwen3-8b",
      "type": "llm",
      "device": "Intel(R) AI Boost"
    },
    "gpu": {
      "status": "ready",
      "model": "qwen2_5-vl-3b",
      "type": "vlm",
      "device": "Intel(R) Arc(TM) 140V GPU (16GB)"
    }
  }
}
```

Single device:
```json
{
  "status": "ready",
  "devices": {
    "npu": {
      "status": "ready",
      "model": "qwen3-8b",
      "type": "llm",
      "device": "Intel(R) AI Boost"
    }
  }
}
```

---

## /v1/models endpoint

Returns all loaded models:

```json
{
  "object": "list",
  "data": [
    {"id": "qwen3-8b", "object": "model", "owned_by": "local-npu"},
    {"id": "qwen2_5-vl-3b", "object": "model", "owned_by": "local-gpu"}
  ]
}
```

---

## Response headers

```
X-Device: npu
X-Model: qwen3-8b
```

Useful for debugging. Not required by any client.

---

## Console logging

```
09:15:03 <- [NPU] 3 msgs, 120 chars, max_tokens=500 (stream)
09:15:08 -> [NPU] ~85 tokens in 4.8s (17.7 tok/s)

09:15:12 <- [GPU] 1 image, 47 chars, max_tokens=200
09:15:15 -> [GPU] 56 chars in 2.8s
```

---

## Threading

```python
npu_lock = threading.Lock()
gpu_lock = threading.Lock()  # only if dual mode
```

NPU and GPU process requests independently. NPU streaming runs in
a background thread while Flask yields SSE chunks.

Flask runs with `threaded=True`. Concurrency is controlled by
per-device locks (`DeviceSlot.lock`), not Flask's threading model.
This ensures `/health` and the web UI always respond, even during
long inference runs. Two requests to the same device queue on the
lock — second waits, eventually gets answered.

---

## install.ps1 changes

### NPU-first flow

```
=== locally Install ===

[OK] venv created
[OK] Dependencies installed

Detected devices:
  [+] NPU: Intel(R) AI Boost
  [+] GPU: Intel(R) Arc(TM) 140V GPU (16GB)

=== Step 1: Chat Model (NPU) ===

  NPU-optimized models (text chat with streaming):
    1. Qwen3 8B (INT4-CW)                  (~5 GB)  Recommended. Best quality.
    2. Phi 3.5 Mini (INT4-CW)              (~2 GB)  Smaller, faster.
    3. DeepSeek R1 Distill 7B (INT4-CW)    (~4 GB)  Reasoning specialist.
    4. DeepSeek R1 Distill 1.5B (INT4-CW)  (~1 GB)  Tiny. Testing only.
    5. Mistral 7B v0.3 (INT4-CW)           (~4 GB)  General purpose.

  Pick a chat model [1-5]: 1

  Downloading Qwen3 8B...
  [OK] Chat model ready.

=== Step 2: GPU Model (optional) ===

  You also have an Intel ARC GPU. What do you want to use it for?

    A. Vision model   — image understanding alongside NPU chat
    B. Bigger LLM     — much smarter chat than NPU can run
    C. Skip           — NPU only

  [A/B/C]: A

  --- Option A: Vision models ---
    1. Gemma 3 4B Vision (INT4)          (~3 GB)  Fast, good quality.
    2. Qwen2.5-VL 7B (INT4)             (~5 GB)  Better quality.
    3. Qwen2.5-VL 3B (INT8, convert)    (~4 GB)  Proven in production.

  --- Option B: Large LLMs (much smarter than NPU model) ---
    1. Qwen3 14B (INT4)                  (~8 GB)  Great reasoning.
    2. Qwen3 30B-A3B MoE (INT4)          (~17 GB) Best quality. Tight fit.
    3. Phi 4 (INT4)                       (~8 GB)  Strong reasoning.
    4. Phi 4 Reasoning (INT4)             (~8 GB)  Chain-of-thought.
    5. DeepSeek R1 Distill 14B (INT4)     (~8 GB)  Reasoning specialist.

  Pick a model or press Enter to skip:
```

### No NPU detected

```
Detected devices:
  [!] NPU: not found
  [+] GPU: Intel(R) Arc(TM) 140V GPU (16GB)

No NPU detected. Selecting a GPU model instead.

=== Select Model (GPU) ===
  (shows full model list — VLMs and LLMs)
```

### No GPU either

```
Detected devices:
  [!] NPU: not found
  [!] GPU: not found
  [+] CPU: Intel(R) Core(TM) i7-...

No NPU or GPU detected. Models will run on CPU (slower).

=== Select Model (CPU) ===
  (shows LLM list only — VLMs too slow on CPU)
```

### Generated start script

```powershell
# start.ps1 (auto-generated by install.ps1)
.\venv\Scripts\Activate.ps1
python arc-server.py --device NPU --gpu-model-dir gpu-model
```

Or for NPU-only:
```powershell
.\venv\Scripts\Activate.ps1
python arc-server.py --device NPU
```

### File layout

```
arc-server/         (future: locally/)
  model/            <-- primary model (NPU chat model)
  gpu-model/        <-- secondary model (GPU vision, optional)
  venv/
  start.ps1         <-- auto-generated
  models.json       <-- curated model registry
  arc-server.py     <-- the server
  install.ps1
```

All model dirs and venv are gitignored.

---

## models.json structure

Three categories: `npu`, `gpu`, `cpu`

```json
[
  {"type": "npu", "name": "Qwen3 8B (INT4-CW)", "hf_id": "OpenVINO/Qwen3-8B-int4-cw-ov", ...},
  {"type": "npu", "name": "Phi 3.5 Mini (INT4-CW)", ...},
  ...
  {"type": "gpu", "name": "Gemma 3 4B Vision (INT4)", ...},
  {"type": "gpu", "name": "Qwen2.5-VL 7B (INT4)", ...},
  ...
  {"type": "gpu", "name": "Qwen3 14B (INT4)", ...},
  ...
]
```

install.ps1 filters by type based on detected devices:
- NPU detected → show `npu` models for step 1
- GPU detected → show `gpu` models for step 2
- Neither → show `gpu` models (they work on CPU too) for step 1

---

## Implementation order

1. **Add `--device` flag** to arc-server.py. Support NPU, GPU, CPU,
   auto. Load LLMPipeline on the chosen device. Test NPU alone.

2. **Add `--gpu-model-dir`** for dual mode. Load second pipeline
   on GPU. Add routing logic.

3. **Update /health, /v1/models** for multi-device.

4. **Update logging** with device tags, X-Device headers.

5. **Update install.ps1** — NPU-first two-step flow, device
   detection, generated start.ps1.

6. **Update models.json** — add NPU category.

7. **Test all combinations:**
   - NPU only (text chat, streaming)
   - GPU only (VLM, current behavior)
   - Dual (text → NPU, images → GPU)
   - CPU fallback

---

## What this does NOT include (yet)

- GUI ("locally" frontend — separate plan)
- Hot-swapping models without restart
- Automatic model recommendations based on available RAM
- NPU for VLM (no VLMPipeline on NPU in current OpenVINO)
- Multiple NPU models loaded simultaneously
