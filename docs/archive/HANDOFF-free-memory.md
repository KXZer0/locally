# locally — context + next task (free-memory control)

Hand this to Claude Code with the repo open at `C:\Projects\locally`.

---

## 0. Read this first

**Before writing any code, read the repo.** In this order:

- `CLAUDE.md` — the project's own conventions. It already records most of what's
  below, in more detail. Where this document and the repo disagree, **the repo wins** —
  say so rather than silently working around it.
- `TODONT.md` — approaches already tried and rejected, each with the reason. Read it
  before proposing anything structural. Add an entry whenever an approach is abandoned.
- `locally.py` — the whole server, one file (~3.5k lines). That's deliberate; don't
  split it into modules.
- `static/js/app.js`, `templates/index.html`, `static/css/style.css` — the web UI.
- `scripts/locally-launch.ps1` + `scripts/locally-key.ps1` — the hotkey launcher.
  `locally-key.ps1` is the hot path (raise an existing window, ~720 ms) and hands off
  to `locally-launch.ps1` only when the server actually needs starting.

**Working style:** one change, run it, verify it, then continue. Don't batch several
changes into one unverified pass. Measure claims — this project's docs quote real
numbers, and that convention is worth keeping.

---

## 1. What this app is

**locally is a local OpenAI-compatible LLM/VLM server for Intel hardware, NPU-first.**
It's the thing you run instead of Ollama when your machine has an Intel NPU and iGPU,
because Ollama has no OpenVINO backend and can't touch the NPU at all.

It serves:

- `POST /v1/chat/completions` — OpenAI shape, streaming via SSE, images supported
- `POST /v1/audio/transcriptions` — Whisper ASR, OpenAI shape
- `POST /v1/audio/speech` — Kokoro/SpeechT5 TTS, returns WAV
- An Ollama-compatible shim on `:11434` so VS Code Copilot Chat works (`--vscode-compat`)
- A web UI at `:8000` with two tabs: **Chat** and **Voice**

The distinguishing feature — the reason it exists rather than being a wrapper — is that
it runs on **specific pieces of silicon and tells you which one answered**. Device
provenance is the visual spine of the UI: amber = NPU, cyan = GPU, slate = CPU. Every
reply's hairline rule, badge, and the voice orb take the colour of the engine that
produced it.

### The machine it runs on

| | |
|---|---|
| CPU | Intel Core Ultra X7 358H (Panther Lake, 4P + 12E) |
| iGPU | Intel Arc B390 (Xe3, XMX confirmed — `GPU_HW_MATMUL` present) |
| NPU | Intel AI Boost (NPU5) |
| RAM | 32 GB LPDDR5X — **unified, shared across CPU/GPU/NPU** |
| GPU budget | 18.5 GB (raised from 16.4 via Intel Graphics Software → Shared GPU Memory Override) |
| OS | Windows 11, PowerShell 7, Python 3.14, venv at `.\venv` |
| OpenVINO | 2026.3 (`openvino`, `openvino-genai`, `openvino-tokenizers`) |

**The constraint that matters: bandwidth, not capacity.** ~68 GB/s shared across all
three engines. Two models resident don't run in parallel usefully — they compete for
one bus. This is why the app is deliberately **one model at a time**.

### Running it

Press the Copilot key (bound to `C:\Projects\locally\locally.lnk` → System32's
`powershell.exe` running `scripts/locally-key.ps1`; there used to be a small C# exe
here, and Smart App Control now blocks it — see TODONT.md), or:

```
pwsh -File scripts\locally-launch.ps1
```

Default: Gemma 4 E4B on GPU (vision + tool calling), Whisper on GPU, Kokoro on CPU,
`--idle-timeout 900`.

---

## 2. State as of 2026-08-10 (all measured on this machine)

Working and verified:

- **One model at a time**, with **dynamic placement** (`_choose_device`). `device` on
  `/v1/models/load` is a preference, not an order. Preference order **NPU → GPU → CPU**.
  The slot *moves* between devices — after unloading, `device_name`/`device_id` are
  re-pointed, so two models are never resident at once. Capability is checked before
  memory: NPU ruled out for VLMs and for group-quantized int4 (`group_size != -1`
  crashes the vpux compiler), read from the IR's `rt_info`, not the folder name.
  Verified: Qwen3-8B → NPU; Gemma E4B (VLM) and Qwen2.5-Coder-14B (gs=128) → GPU,
  each with a `placement` string explaining why.
- **Runtime model swap**, no restart: `GET /v1/models/available`, `POST /v1/models/load`.
  ~20-45 s depending on model. Retries 3× with a gc between attempts (dropping a
  pipeline releases native resources on the GC's schedule; building a new one too soon
  intermittently fails to re-register `openvino_tokenizers.dll`).
- **Compile caching** (`MODEL_CACHE_DIR`, default `.ov-cache/`): **Qwen3-8B on NPU
  65.1 s cold → 9.3 s cached**, for 659 MB of disk. `--no-model-cache` disables.
- **Voice**: Whisper ASR **0.33 s on GPU** vs 1.05 s on CPU. Kokoro TTS ~1.3 s/sentence
  on CPU, 24 kHz, 54 voices. TTS has **no streaming** — so `splitSpeakable` peels
  finished sentences off the token stream and `makeSpeechQueue` synthesizes clip N+1
  while N plays. First audio lands after the first *sentence*, not the whole message.
- **Voice and chat get different prompts.** Markdown read aloud is noise; audio can't
  be skimmed. Voice replies are capped (`VOICE_MAX_TOKENS`) and instructed to plain
  2-3 sentences that point at the Chat tab for tables/code. Measured: same question,
  chat answer rambles, voice answer 196 chars with no markdown.
- **Hotkey launcher**: native 6 KB exe, **33 ms** to raise an existing window (was
  494 ms via PowerShell). Does a 1 ms TCP check on :8000 — a stale window over a dead
  server restarts the server instead of showing "failed to fetch".

### Gotchas discovered the hard way

- **This network resets ~75% of HTTPS connections to huggingface.co.** `hf download`
  does not retry hard enough. Use `scripts/`-adjacent tooling with `curl -C -
  --retry 30 --retry-all-errors --speed-limit 2000 --speed-time 20`, or parallel range
  requests with per-chunk retry (measured 6.6 MB/s single-stream vs 13.5 MB/s at 16
  connections, with some ranges failing outright at that concurrency). **Run one
  download at a time.**
- **Gemma 4 does not work on any NPU we own** — see `TODONT.md`. Garbage on the older
  NPU, 0.1 tok/s on NPU5. GPU/CPU only.
- **Gemma 4 12B was a dead end**: exported fine (22.3 GB bf16 → 7.3 GB int4,
  `transformers==5.10.0` required for `Gemma4UnifiedForConditionalGeneration`) but is
  slower than E4B and its vision path is broken. Deleted. Don't redo it.
- Windows hides `.lnk` extensions — a shortcut named `locally.lnk` displays as `locally`.

---

## 3. The task: a real "free memory" control

### The problem

I want to run a heavy program (a game, a build, a VM) without killing locally, and get
the RAM back on demand. Today that doesn't work, and the reason is subtle.

**`POST /v1/models/unload` genuinely unloads the models** — all slots report
`idle_unloaded`, and OpenVINO frees its allocations. **But the process RSS does not
drop at all.** Measured: 1,195 MB before unload, 1,195 MB after unloading all three
models. The C allocator keeps the freed pages in its heap instead of returning them to
Windows.

So `--idle-timeout` and the existing unload endpoint free memory *for reuse inside the
process*, but Task Manager shows no change and Windows doesn't get the pages back until
it's under pressure. For "I'm about to launch something heavy", that's not good enough.

Also worth knowing, so the numbers aren't confusing: **RSS is a bad proxy for footprint
here.** The same process measured 1,195 MB with models idle (Windows had trimmed the
working set) and **7,475 MB** immediately after actually using them. NPU/GPU allocations
go through the driver and aren't fully attributed to `python.exe`. The stack floor
(Python + numpy + Flask + OpenVINO + the three device plugins) is **162 MB**; everything
above that is model weights.

### What to build

**1. Make unload actually return memory to the OS.**

After unloading, trim the working set so Windows reclaims the physical pages. On Windows
that's `EmptyWorkingSet` (psapi) or `SetProcessWorkingSetSizeEx(handle, -1, -1)` via
`ctypes`. Guard it behind a platform check — this repo runs on Linux too (issue #6), so
it must be a no-op elsewhere, not a crash.

Verify by measuring RSS **and** system-wide available RAM before and after. Report both
in the response. If trimming turns out not to help (measure it — don't assume), say so
plainly and fall back to option 3 below.

**2. A button in the UI.**

Settings already has a "Free memory" button wired to `/v1/models/unload`. It should:

- Report what it actually freed, in MB, not just "ok" — e.g. "Unloaded Qwen3-8B, Whisper,
  Kokoro — 6.2 GB returned to the system."
- Make clear the models reload automatically on the next message (they do —
  `ensure_loaded` handles chat, ASR and TTS).
- Not lie. If the pages weren't actually returned, don't claim they were.

**3. Consider: an explicit "stop server" path.**

If trimming the working set doesn't reliably return the memory, the honest answer is
that only exiting the process does. That's now cheap: the hotkey restarts in ~1 s and
the compile cache makes the model reload fast (9.3 s for Qwen3-8B on NPU). A
"Shut down locally" button plus the key to bring it back may beat any amount of
allocator fighting.

**Decide this on measurements, and write the result into `TODONT.md` if the trimming
approach loses.**

### Constraints

- Don't break `--idle-timeout` or the existing unload endpoint; extend them.
- Audio slots participate in idle-unload and must keep doing so. There's history here:
  `WhisperSlot` originally had no `last_used`/`unload()`, so `--whisper-dir` with the
  default 1800 s timeout crashed the watchdog thread on its first pass and silently
  disabled idle-unload for *every* slot. Don't regress that.
- Never yank a pipeline out from under a running generation — the unload path uses
  `lock.acquire(blocking=False)` and skips busy slots. Keep that.
- Keep it in `locally.py`. One file is fine.

---

## 4. Ideas after this one

- **Voice**: only VAD/barge-in remain unverified by me, because the dev browser blocks
  `getUserMedia`. Everything downstream of the recorded audio blob is measured. Worth a
  real-microphone pass — turn-taking thresholds are in `VAD` in `app.js` and are
  deliberately generous (1.5 s silence, adjustable 0.6-4 s).
- **Qwen3-Coder-30B-A3B** (~17 GB, MoE, tool-capable) as the coding model — the B390 has
  XMX so `--offload-ratio` genuinely works here (unlike the non-XMX desktop; see TODONT).
- **Gemma 4 26B-A4B** is downloaded and verified loadable (14.3 GB, INT4+AWQ, 128
  experts). Not yet benchmarked against E4B on this machine.
- `/v1/models/available` currently filters audio models out of the chat list
  (`_is_generative_dir`). If more model kinds appear (embeddings, rerankers), that
  filter is where they'd be classified.

---

## 5. Don't

- Don't upgrade `transformers` in the main venv — it breaks the working Qwen exports.
  Use a throwaway venv outside the repo for conversions.
- Don't download through `install.ps1` on this network. One transfer at a time.
- Don't put Whisper, TTS, or Gemma on the NPU.
- Don't assume TTS streams. It doesn't.
- Don't add a CDN dependency for fonts, icons, or JS — fonts are self-hosted in
  `static/fonts/` and icons are inline SVG. This must work offline.
- Don't mark anything `Verified` in `models.json` without a real measurement.
- Don't claim memory was freed without measuring both process RSS and system available RAM.
