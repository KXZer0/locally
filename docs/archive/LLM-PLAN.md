# LLM-PLAN: Model download & conversion at install time

## Goal

Remove the 3.9 GB Git LFS model blob from the repo. Instead, let the
user pick a model during install. `install.ps1` presents a curated
list of compatible models, downloads from HuggingFace, and converts
to OpenVINO IR — all in one step.

The repo ships code only. No multi-GB blobs. Normal git clone.

---

## Current state

- `model/` contains a pre-converted Qwen2.5-VL-3B (3.9 GB, Git LFS)
- install.ps1 creates venv, installs deps, verifies model exists
- No model download or conversion capability

## Proposed state

- `model/` is empty or absent in the repo (gitignored)
- install.ps1 creates venv, installs deps (including export tools),
  presents model menu, downloads + converts the chosen model
- First run takes longer (download + conversion) but repo is tiny

---

## Model menu

Curated list of models known to work with OpenVINO 2026.1+ on Intel
ARC GPUs. Grouped by type, sorted by size.

```
=== Available models ===

  VLM (vision + text):
    1. Qwen3-VL-2B-Instruct      (~2 GB, fastest)
    2. Qwen3-VL-4B-Instruct      (~4 GB, good balance)
    3. Qwen3-VL-8B-Instruct      (~9 GB)
    4. Qwen2.5-VL-3B-Instruct    (~4 GB, proven in production)
    5. Qwen2.5-VL-7B-Instruct    (~7 GB)

  LLM (text only):
    6. Qwen3-8B                   (~5 GB INT4)
    7. Qwen3-14B                  (~8 GB INT4)
    8. Qwen3-30B-A3B              (~17 GB INT4, MoE)

  Pick a number [1-8]:
```

### Model registry

The list is a simple Python dict or JSON file in the repo:

```python
MODELS = {
    "1": {
        "name": "Qwen3-VL-2B-Instruct",
        "hf_id": "Qwen/Qwen3-VL-2B-Instruct",
        "type": "vlm",
        "task": None,  # autodetect
        "weight_format": "int8",
        "trust_remote_code": True,
        "est_size_gb": 2,
    },
    "2": {
        "name": "Qwen3-VL-4B-Instruct",
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "type": "vlm",
        "task": None,
        "weight_format": "int8",
        "trust_remote_code": True,
        "est_size_gb": 4,
    },
    # ...
}
```

Easy to extend — add a new entry when a new model is verified working.
Could also be a `models.json` file for easier editing without touching
Python.

### Fallback task flag

Some models fail with autodetect. The registry includes an optional
`task` field (e.g. `"image-text-to-text"`) for models that need it.
This captures the hard-won knowledge from previous conversion battles.

---

## install.ps1 changes

### New flow

```
.\install.ps1
  1. Create venv (if not exists)
  2. Install runtime deps (flask, openvino, openvino-genai, etc.)
  3. Install export deps (optimum-intel, transformers, accelerate, etc.)
  4. Check if model/ already has a converted model
     - If yes: "Model already present: qwen3-vl-4b. Re-download? [y/N]"
     - If no: show model menu
  5. User picks a model
  6. Download from HuggingFace + convert to OpenVINO IR in model/
  7. Verify conversion succeeded (check .bin files exist)
  8. Done — print startup instructions
```

### Conversion command (generated from registry)

```powershell
optimum-cli export openvino `
  --model "Qwen/Qwen3-VL-4B-Instruct" `
  --weight-format int8 `
  --trust-remote-code `
  model\
```

### Progress indication

The conversion takes 5-20 minutes depending on model size and network.
optimum-cli prints progress bars for download and conversion. Let them
flow through to the terminal — don't suppress with Out-Null.

### HuggingFace login

Some models are gated (require accepting a license on HF). If the
download fails with a 401/403, print:

```
ERROR: Download failed. This model may require HuggingFace login.
Run: huggingface-cli login
Then re-run install.ps1
```

---

## requirements.txt changes

Split into two files, or use optional groups:

### Option A: Two files

```
requirements.txt          # runtime only (flask, openvino, etc.)
requirements-export.txt   # adds optimum-intel, transformers, etc.
```

install.ps1 installs both. arc-server.py only needs the runtime deps.

### Option B: Single file with everything

Keep it simple. The export deps add ~200 MB but only run once.
On a dev machine this is fine.

**Recommendation: Option B.** Simpler. One file. The extra deps sit
idle after conversion but don't hurt anything.

### Updated requirements.txt

```
# Runtime
flask
openvino>=2026.1.0
openvino-genai>=2026.1.0.0
openvino-tokenizers>=2026.1.0.0
pillow>=10.3.0
numpy>=1.26.4

# Model export (used by install.ps1)
optimum-intel[openvino]>=1.27.0
transformers>=4.57.0
accelerate>=1.0.0
huggingface_hub>=0.36.0
```

---

## .gitignore changes

```
model/
venv/
__pycache__/
```

The `model/` directory is fully gitignored. No LFS, no blobs.
Remove `.gitattributes` LFS rules.

---

## Migration from current setup

1. Remove `model/` from Git LFS tracking
2. Add `model/` to `.gitignore`
3. Update `install.ps1` with model menu + conversion
4. Update `requirements.txt` with export deps
5. Update `README.md` — install instructions change from
   "git lfs pull" to "run install.ps1 and pick a model"
6. Delete `.gitattributes` (no longer needed)

---

## Open questions

1. **INT8 vs INT4**: The menu currently defaults to INT8 for VLMs
   (proven) and could offer INT4 for LLMs (smaller, faster, slight
   quality loss). Should this be user-configurable or hardcoded per
   model in the registry?

2. **Pre-exported models on HF**: Some models have pre-exported
   OpenVINO versions on HuggingFace (e.g. OpenVINO/gemma-3-4b-it-int4-ov).
   These skip the conversion step entirely — just download and go.
   The registry could prefer these when available.

3. **Model switching after install**: Currently `--model-dir` points
   at `model/`. Should install.ps1 support multiple models in
   `models/model-name/`, or is one model at a time enough?

---

## Depends on

- Successful Qwen3-VL-4B conversion with OpenVINO 2026.1.0
  (currently being tested)
