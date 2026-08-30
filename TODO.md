# TODO

## Load GGUF directly — openvino-genai already can (2026-08-06)

openvino-genai 2026.1 ships a **GGUF reader** we don't expose at all
(`gguf_modeling.cpp`, `gguf_quants.cpp`, `gguf_tokenizer.cpp`, `load_gguf`,
`GGUFAdapterImpl` in the shipped `openvino_genai.dll`). If locally passed a
`.gguf` path to `LLMPipeline`, users could skip `optimum-cli export` — and
skip the RAM wall that makes conversion the worst part of the project.

Scope limit, read out of the binary's dispatch table (the three names sit
immediately before the `Unsupported model architecture '` string):

    llama    qwen2    qwen3

Probed and **absent**: `qwen3next`, `qwen3moe`, `qwen2moe`, `deepseek2`,
`glm4`, `granite`, `bitnet`. So dense Qwen3/Qwen2.5-Coder and Llama GGUFs
are in scope; **MoE is not** — Qwen3-Coder-Next GGUF will not load, which
kills the obvious hope that GGUF routes around the 400 GB conversion.

Next step is a 10-minute experiment, not a design: point `LLMPipeline` at a
small Qwen3 GGUF on GPU and see whether it loads and generates. Everything
above is inferred from strings in a stripped DLL — verify before writing it
down as fact anywhere user-facing. Expect NPU not to work (it needs its own
compiled blob path).

Worth knowing even if we don't ship it: it's the answer to "why do I have to
convert when Ollama just pulls?" for a chunk of the model space.

---

## Make more memory available to the iGPU (2026-08-04)

By default Windows budgets the iGPU ~half of system RAM (the Arc 140V on a
32 GB laptop reports a 16.5 GB budget via `GPU_DEVICE_TOTAL_MEM_SIZE`).
Intel's driver 32.0.101.6987+ added **"Shared GPU Memory Override"** in the
Intel Graphics Software app (Core Ultra series 1/2): default ~57%, up to
~87% of RAM — explicitly marketed for local AI. That's the difference
between "30B model won't fit" and "fits with a fat KV pool".

Status / open questions:

- The memory preflight (`_preflight_memory`) already reads the driver's
  reported budget and names the override in its "will NOT work" hint.
- **Verify on this laptop** that flipping the override actually moves
  `GPU_DEVICE_TOTAL_MEM_SIZE` (expect 16.5 → ~27 GiB at 87% of 32 GB).
  Blocked so far: Intel Graphics Software crashes when toggling the
  override (2026-08-04); retry after a reboot.
- Is there a programmatic/registry way to set it (so install.ps1 could
  offer it, or at least link the setting)? Driver-version detection +
  a pointer in the install summary may be the realistic scope.
- Document in README once verified (currently one line in the KV-pool
  section).

---

## Spinoff idea: claude-code CLI as Ollama backend (2026-05-26)

**Not part of locally.** Separate repo if pursued. locally is local-Intel
inference; this is the opposite (cloud Anthropic via local CLI). Captured
here so it doesn't get lost.

### The idea

`claude-code -p "<prompt>"` runs in non-interactive print-mode: prompt
in via argv/stdin, response out via stdout, no REPL. So in principle a
small bridge could:

1. Listen on `localhost:11434` speaking Ollama API
2. Translate each `/api/generate` or `/api/chat` request into a
   `claude-code -p` invocation
3. Stream stdout back as NDJSON

Result: any Ollama-aware tool (Open WebUI, Continue, mobile Ollama
clients, etc.) gets Claude as the model, using whatever auth the
local `claude-code` install already has.

### IPC choice

- **Anonymous pipes** (`subprocess.Popen(stdin=PIPE, stdout=PIPE)`) —
  simplest, cross-platform, one-shot per request. Probably the right
  default.
- **Named pipes** — Windows has them too (`\\.\pipe\<name>`), and
  `System.IO.Pipes.NamedPipeServerStream` is cross-platform via .NET.
  Worth it only if the bridge wants a long-lived `claude-code` process
  serving many requests (would need claude-code to support a
  daemon/streaming-stdin mode, which `-p` doesn't currently).

### Practical concerns before pursuing

- **claude-code is interactive-oriented.** `-p` mode works but isn't
  the supported "stable backend" surface. Behavior might change between
  releases.
- **Session state.** Ollama clients expect stateless or
  client-managed history. claude-code might carry conversation context
  across invocations in ways that surprise an Ollama client.
- **Per-request startup cost.** Spawning claude-code per request adds
  latency. For chat that's fine; for batch agents it's not.
- **Auth model mismatch.** claude-code uses the user's logged-in
  account; that's a single tenant. Fine for personal use, doesn't
  scale to a shared server.
- **Already exists upstream?** Worth checking if Anthropic ships a
  similar bridge or if anyone in the community has one — don't reinvent.

### If pursued

One Python file (`claude_ollama_bridge.py`), uses subprocess.PIPE,
maps `POST /api/chat` and `POST /api/generate` to `claude-code -p`,
ignores `/api/pull`/`/api/delete` (stubs returning success), forwards
stdout as NDJSON streaming response. Reusable shape from locally's
existing `ollama_app` Flask blueprint.

---

## Suppress OpenVINO native chatter on model load — low priority (2026-05-26)

openvino_genai 2026.1 prints model-property dumps (`Model: OV
Tokenizer / NETWORK_NAME / NUM_STREAMS / INFERENCE_NUM_THREADS / …`)
plus `[INFO] pruning_ratio` and `[XAttention] DISABLED` lines from
native C++ during pipeline construction and warmup. Roughly 25 lines
per loaded model, only at startup; per-request inference is silent.

### What didn't work

- `OPENVINO_LOG_LEVEL=0` env var (the dump shows `LOG_LEVEL: LOG_NONE`
  already — it's a deliberate print, not a log statement).
- A `contextlib.contextmanager` that did `dup2(devnull, 1/2)` around
  pipeline construction (`657f4bb`, reverted in `8824eca`). Works
  single-threaded but races when both NPU and GPU loaders run
  concurrently — both touch process-global fd 1/2 at the same time
  and the "save original stdout" step can land *after* the other
  thread already redirected. Models silently went to `status=error`
  because their own error prints were lost.

### Path forward

Thread-safe version: global `threading.Lock` around the dup2
sequence. Cost: NPU and GPU model loads serialize through the
suppress block (~30s + 30s instead of ~30s parallel). Acceptable —
loading already mostly serial inside OpenVINO's plugin machinery.

Or: find an upstream-supported flag. Search openvino_genai 2026.1
source for the property-dump call site; there may be a builder
option, runtime property, or env var we missed.

### Why low priority

- Only visible at startup, not during use.
- Doesn't affect correctness.
- The interim "broken suppress" cost was much higher than the
  chatter itself — better to accept the chatter than ship a fix
  that breaks model loading.

---

## CPU as primary on NPU/GPU systems — settled non-goal (2026-05-26)

`install.ps1` already offers CPU as the primary slot when no NPU
and no GPU are detected (line ~450). On NPU- or GPU-equipped
systems, NPU > GPU > CPU is the install-time default and stays so.

### Context that briefly suggested otherwise

`0bbb948` benchmarked Qwen3-8B (text LLM) on Arrow Lake desktop and
found **CPU > iGPU > NPU** — decode is memory-bandwidth-bound, and
DDR5 + many CPU cores beat the 4-core Xe-LPG iGPU and the NPU's
power-sipping memory path. That made it tempting to expose CPU as
a deliberate choice for desktop users.

### Why we're not adding the install prompt

The "CPU wins" rule turned out narrower than first thought:

1. **VLM flips the result.** 2026-05-26 QA: same desktop, same
   Qwen3-VL-8B-INT4 model, image-bearing prompts ran ~2.2x **slower**
   on CPU than on Xe-LPG iGPU (15.29s vs 6.93s avg). VLM prefill is
   compute-bound on the vision encoder; iGPU wins. Text-only on the
   same model: CPU only ~10-30% slower than GPU.

2. **NPU has its own memory path.** Intel AI Boost uses dedicated
   DMA, separate from CPU/GPU memory controllers. Benchmark numbers
   are best-case for CPU/GPU (idle system) and unchanged for NPU.
   Under real load (browser open, build running, game rendering),
   CPU and iGPU lose bandwidth they share; NPU keeps its own.
   For "always-on assistant" / "while-I-work" workloads — which is
   what most users actually have — NPU is undervalued by idle
   benchmarks.

3. **Adding a prompt adds friction for the 95%.** The "Keep it
   simple" preference in CLAUDE.md argues against interactive
   choices for niche power-user scenarios.

4. **The runtime override already exists.** Anyone who's measured
   and wants CPU can do `python locally.py --device CPU --model-dir
   .\model` — discoverable from `--help`.

### Settled position

NPU > GPU > CPU stays as the install default on Ultra hardware.
CPU is only offered when no NPU and no GPU are present. The
benchmark data and NPU memory-path nuance live here as context for
any future "why not let users pick CPU?" question.

Intel now ships pre-exported, pre-quantized Whisper models on
[huggingface.co/OpenVINO](https://huggingface.co/OpenVINO). Our
`models.json` whisper entries still convert from `openai/whisper-*`
to FP16 (slower install, larger files). Worth benchmarking the
pre-exported variants and replacing the entries if they're competitive.

Candidates to test:

- **`OpenVINO/distil-whisper-large-v3-int8-ov`** — most-downloaded
  whisper variant in the OpenVINO org (9k+ downloads). Distilled
  large-v3 is reportedly ~6× faster than the original at similar
  accuracy. INT8 quantization on top should be a real win.
- **`OpenVINO/whisper-large-v3-int4-ov`** — best accuracy if size
  fits. INT4 makes large-v3 viable on 16 GB GPUs.
- **`OpenVINO/whisper-medium-int8-ov`** and **`-int4-ov`** — direct
  upgrade path for our current "Whisper Medium" FP16 entry.
- **`OpenVINO/whisper-small-int4-ov`** — smallest viable multi-language.

What to measure (extend `benchmark.py --backend whisper`?):
- WER on Norwegian + English samples (you have local audio)
- Wallclock per second-of-audio
- Cold-load time
- Memory footprint

Replace `models.json` whisper entries with whatever benchmarks best.
Tag survivors as proven; drop the rest. Don't add untested ones to
the install menu.

---

## NVIDIA support — deliberate non-goal (settled 2026-05-21)

Settled: locally will **never** support NVIDIA GPUs, even though there
is now a working path.

**The path exists.** OpenVINO 2026 ships an experimental NVIDIA plugin
via `openvino-extensibility`. It's possible to run inference on an
RTX through OpenVINO — but it drags CUDA/cuDNN into the stack, lives
in contrib/plugin land, and is a developer backend rather than a
drop-in user feature. Docs:
https://docs.openvino.ai/2026/documentation/openvino-extensibility/openvino-plugin-library/plugin.html

**Why we won't.** Ollama already does NVIDIA inference excellently.
Anyone with an NVIDIA card should use Ollama; locally's whole reason
to exist is the Intel NPU + ARC story that Ollama doesn't cover.
Supporting both would dilute the project's identity, multiply the
test matrix, and compete with a much better tool on its home turf.

**What changed.** `0bbb948` filtered non-Intel GPUs out of
`detect_devices()` in `locally.py` so the RTX 5090 wouldn't be offered
as a footgun (compile errors + `CL_INVALID_VALUE` at warmup). On
2026-05-21, the same filter was added to `install.ps1`, plus
multi-GPU enumeration handling (`GPU.0`/`GPU.1` → canonical `GPU` with
the actual OpenVINO id tracked separately). Multi-GPU desktops (iGPU +
non-Intel dGPU) now detect the Intel GPU correctly.

---

## locally on NVIDIA GPUs — verified does NOT work (2026-05-03, historical)

Kept for context. On a desktop with both Intel iGPU and an RTX 5090,
`python locally.py --device GPU.1` (when GPU.1 was the RTX) failed:

- Model compile: 144 errors generated by the `intel_gpu` plugin's kernels.
- Warmup crashes with `CL_INVALID_VALUE` from `clEnqueueMapBuffer`.

Root cause: OpenVINO's stock `intel_gpu` plugin enumerates any
OpenCL-capable device. NVIDIA's driver provides OpenCL, so the 5090
shows up — but the plugin's kernels use Intel-specific GPU intrinsics
that NVIDIA's OpenCL runtime doesn't support. Enumeration ≠ executable.
The new NVIDIA-specific plugin (above) is a separate story, but we've
chosen not to pursue it.

---

## Text-to-Speech (TTS) — `/v1/audio/speech`

`openvino_genai.Text2SpeechPipeline` exists. Only SpeechT5 supported so far.

**Export:**
```bash
optimum-cli export openvino \
  --model microsoft/speecht5_tts \
  --weight-format int4 \
  --model-kwargs '{"vocoder":"microsoft/speecht5_hifigan"}' \
  speecht5_tts
```

**What's needed:**
- `--tts-dir` flag, similar to `--whisper-dir`
- `POST /v1/audio/speech` endpoint (OpenAI-compatible)
- Speaker embedding files (512×float32 `.bin`), map OpenAI voice names (`alloy`, `echo`, etc.) to them
- CPU or GPU only — no NPU support for encoder-decoder models

**Caveats:**
- SpeechT5 is serviceable but clearly first-gen neural TTS, not ElevenLabs quality
- English-centric — Norwegian output would be rough
- Voice selection via embedding files, not named presets — UX is awkward
- Small model (~few hundred MB), fast on CPU

**Verdict:** Clean API surface, completes the OpenAI compatibility story. Worth adding
once STT (Whisper) is proven. Low priority until then.

---

## Spinoff project idea: Ollama API wrapper for any OpenAI-compatible server

**Not part of locally.** Separate repo if pursued.

### Honest assessment (verified 2026-04-13)

Initial motivation was "tools that speak Ollama but not OpenAI." This
turned out to be weaker than hoped:

- **Major tools support both.** Continue.dev, Zed, Cursor, Open WebUI,
  VS Code extensions — all take custom OpenAI-compatible base URLs.
- **Walled-garden tools don't help either way.** Android Studio's AI
  (Gemini-only) and JetBrains AI Assistant (their own backend) won't
  accept any local endpoint, Ollama or OpenAI.
- **Genuinely Ollama-only tools are niche**: Llama Coder (VS Code),
  Enchanted (macOS), Maid, various mobile clients. Real but small audience.

### The narrower valid case

- Protocol quirks: `/api/tags` vs `/v1/models` have different shapes.
  Some tools nominally "OpenAI" still call Ollama-specific endpoints
  (`/api/show` for metadata).
- Ollama NDJSON vs OpenAI SSE framing trips tools tested against only one.
- Dev ecosystems built around `ollama` CLI expect a real Ollama server.

### If pursued

| Ollama endpoint | Upstream call | Translation |
|---|---|---|
| `GET /api/tags` | `GET /v1/models` | reshape model list |
| `POST /api/show` | `GET /v1/models` | pick one, reshape |
| `POST /api/chat` | `POST /v1/chat/completions` | SSE → NDJSON |
| `POST /api/generate` | `POST /v1/chat/completions` | wrap prompt as user msg |
| `POST /api/pull`/`delete`/`copy` | stub — return success | |

One Python file, single config (`--upstream http://ovms:8080 --port 11434`).
Reusable chunks already exist in locally.py's `ollama_app` and
`_ollama_stream_*` functions.

**Verdict**: Interesting afternoon project, but the audience is smaller
than the initial Reddit comment suggested. Not urgent.
