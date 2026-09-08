# locally

OpenAI-compatible LLM/VLM server for Intel hardware. NPU-first.

## Current direction — headless API and optional terminal chat

The user retired the custom web UI. Do not rebuild it. `README.md` is the current
operating guide; old UI checkpoints and the rebuild plan are historical.
`locally.py` starts the foreground API; `chat` attaches an optional terminal client,
and `chat --start` owns a server until exit. Markdown copy/save preserve source.
No browser setup settings are read. Audio/util models require explicit opt-in.
Windows entry points: `api.ps1` uses the installer's `start.ps1`; `chat.ps1`
attaches to an existing API or owns a temporary one when the port is free.
The shortcut/Copilot-key launcher defaults to API mode; `-Mode chat` opens chat.
Keep API compatibility for external harnesses. Do not launch or stop a user's
separate harness when attaching/detaching terminal chat. Existing integration
source is retained but browser/control routes are not mounted.

## Architecture

- `locally.py` — command entry point; `core/app.py` builds APIs, `core/slots/device.py` owns inference, `core/terminal.py` is an HTTP client
- **The terminal client is an HTTP client and nothing else** (`core/terminal.py`,
  `locally.py chat` / `status`). It imports no Flask and no OpenVINO -- attaching
  to a running server must not pay a native import -- and the one thing it does
  take from `core/` is `_ThinkFilter`, so reasoning is hidden by the same code
  the Anthropic path uses rather than by a second copy that can drift. Four
  behaviours are deliberate and were each found by running it: the prompt waits
  while `/health` says `loading`, because the socket binds seconds before a model
  can answer and the first turn was otherwise refused outright (a piped turn was
  simply lost); Ctrl+C posts `/v1/cancel` before giving up on the stream, since
  the spinner promises a cancel; a `length` finish keeps the answer and labels it
  incomplete instead of raising it away (on the 1024-token default that is the
  ordinary end of a long answer); and an unclosed `<think>` falls back to showing
  the reasoning, because an empty answer reads as a hang. Verified on
  Qwen3-8B-abliterated/NPU: closed think block -> reasoning hidden, answer alone;
  200-token budget -> answer kept, "Stopped at the 200-token budget".
- **`finish_reason` was `"stop"` for every streamed turn** (fixed 2026-09-04).
  Both `stream_llm` and `stream_vlm` hard-coded it, so an answer cut off at
  `max_new_tokens` arrived labelled exactly like one that finished and no client
  -- terminal or harness -- could tell them apart. `extract_finish_reason` reads
  `DecodedResults.finish_reasons` (the pipeline's own verdict) and falls back to
  `"stop"` when the field is missing, because claiming a truncation that did not
  happen is worse than the silence it replaces. Do not count streamer callbacks
  instead; TODONT.md has the measurement.
- **Nothing outside `core/models/describe.py` keeps a list of model names.**
  `--list-models` prints `path<TAB>name<TAB>llm|vlm` using the same discovery
  and the same `_is_generative_dir` filter as `/v1/models/available` (depth 1,
  so a diffusion pipeline's `unet` is not offered as a chat model; `--scan`
  keeps depth 2 because a report about the disk should show the disk). The
  terminal's `/load` completes from the live endpoint, and
  `scripts/locally-launch.ps1` takes a PREFERENCE ORDER of names and falls
  through to `--list-models` when none of them exist -- deleting a model is
  the user's business and must not decide which model starts. The launcher
  also stopped naming a device: `--device auto` lets `_choose_device` read the
  IR's `rt_info`, which a shell script cannot do.
- **The terminal client is where the retired UI's non-chat surface went.**
  `/think on|off` sends Qwen3's `/no_think` on a COPY of the last user message
  (a control token kept in history is re-sent every turn and lands in `/save` --
  and the first cut did exactly that, caught by a test rather than by reading
  it). `/read`, `/index`, `/find` and `/upscale` are the utility endpoints, with
  `/read` putting the extracted Markdown into the conversation. `/util off`
  needed a new server input, `POST /v1/models/unload {"scope": "util"}`: `device`
  cannot say "the OCR engines but not the model answering on the same device",
  and on a one-model box that took the conversation down with the utilities.
  Utility commands wait while engines compile -- they load after the chat model,
  so the first seconds of a server's life are exactly when a client asks, and
  the server used to answer "Start with --util-models-dir" when it was already
  set. Verified on this box: README.md read in 9,542 chars, CLAUDE.md + TODONT.md
  indexed to 245 chunks with the right passage top-ranked, 320x200 upscaled 3.0x
  on the NPU.
- **Voice in the terminal is push-to-talk, and the audio dependency is optional**
  (`core/terminal_voice.py`, `/voice`). The server already owns Whisper, Kokoro
  and Silero, so the only missing pieces were a microphone and a speaker:
  `sounddevice` is imported when `/voice` runs and never on the chat path, since
  attaching a terminal to a running server must not require an audio stack. A
  terminal cannot see a key being RELEASED, which is why Enter starts and stops
  a take rather than a held key. Two behaviours are carried over from the web UI
  because they were measured there: speech starts before the answer finishes
  (clips are peeled off the token stream a sentence at a time, and synthesis of
  clip N+1 runs on its own thread while N plays), and each line is printed when
  its AUDIO starts, so the text cannot run ahead of the voice. Spoken turns
  append the voice directive last, cap at 220 tokens and force no-think --
  reasoning must never be spoken -- and keep neither the directive nor the token
  in the conversation, which voice and text share. Measured through the client:
  Kokoro 3.2 s of audio in 1.9 s, Whisper read it back word for word.
- NPU: LLMPipeline with MAX_PROMPT_LEN=8192 (raised from 4096 on 2026-08-19;
  8192 is the vpux compiler's ceiling, not a memory limit — 9216/10240/12288
  all fail graph legalisation, and KV compression does not move it. See
  TODONT.md and scripts/npu-context-probe.py), streaming via SSE
- GPU: VLMPipeline (images) or LLMPipeline (text). Both stream as of openvino-genai 2026.1 — verified on Arc 140V iGPU.
- **Prefix caching is a CPU win and a GPU trap** (2026-08-10, B390 +
  Qwen3-Coder-30B-A3B): the CB backend prefills **~10x slower** than the plain
  pipeline (187 vs ~2,000 tok/s) and **hangs** when consecutive prompts share a
  near-total prefix — i.e. exactly what an agent client sends. `--no-prompt-cache`
  took Codex's turn from 75 s to **8.7 s** and pi's from 53.6 s to **1.8 s**.
  Full numbers in TODONT.md. The bullet below describes the CPU behaviour, where
  the 47x figure still holds. **The default is now per-device** (`PROMPT_CACHE
  = None` -> on for CPU, off for GPU); `--prompt-cache` / `--no-prompt-cache`
  force it. A default that is a 47x win on one device and a hang on another
  cannot be a constant.
- **`--kv-precision u8`** halves KV bytes/token (96 KB -> ~48 KB on the 30B),
  which is what makes a 100k-token session fit a 24 GB GPU budget: f16 KV needs
  9.6 GB at 100k on top of 15.2 GB of weights, u8 needs 4.8. Verified on the
  B390: loads, warms in 1.8 s, answers a 9k-token prompt correctly, and
  **prefills at 2,088 tok/s — no penalty vs f16**. The memory halving is by
  construction (element size); a real 100k-token session has not been run yet.
  Quality cost is far below that of the int4 weights already in use.
- Per-request logs now carry **prefill tok/s** (`prompt_tokens / TTFT`), because
  an agent turn is dominated by prefill and nothing reported it — which is how a
  backend running 10x slow read as "the model is slow" for a full day.
- Prefix (KV) caching: **default on** for GPU/CPU **LLM** slots — they load via the
  continuous-batching backend (`LLMPipeline(..., scheduler_config=SchedulerConfig(
  enable_prefix_caching=True, cache_size=PROMPT_CACHE_GB))`). A repeated prompt prefix (an
  agent's fixed system prompt + tool schemas, identical every turn) is prefilled once, not
  every turn — measured ~47× faster on a cached turn (24.4s→0.5s for a ~2k-token prefix on
  the 285K CPU). Auto-invalidated by any prefix change (no staleness). `--no-prompt-cache`
  disables it; `--cache-size-gb N` sizes the pool (default 2). **The pool is the usable
  context window** — overrun it and the request hard-fails (#21) — so `--context-tokens N`
  takes the unit users actually have ("I need 90k for this agent") and converts it with
  the model's own KV geometry; `--context-tokens auto` fills what the device has left
  after weights. Both are capped at the model's real `max_position_embeddings` (a pool
  larger than the model can address is memory nobody can use) and resolved **per model in
  `load()`**, not once per process: KV bytes/token vary ~3× across models one slot holds
  over its life (84 KB gemma-4-E4B, 144 Qwen3-8B, 240 gemma-4-26b), so a fixed GB figure
  silently means a different context after every swap. The resolved figure is in
  `/health` as `context_tokens` + `kv_pool_gb`. Measured: `--context-tokens 30000` on
  Qwen3-8B/GPU → 5 GB pool holding 36k; `auto` → 6 GB, capped at the model's 40k. NPU and VLM slots keep the
  plain pipeline (NPU has no CB path; it keeps MAX_PROMPT_LEN). Falls back to the plain
  pipeline with a warning if a device can't build the CB backend. `--prewarm <file>`
  prefills a saved agent prompt at startup (the file auto-captures the first big prompt
  served via `_maybe_capture_prewarm` — on both the OpenAI and Ollama chat paths — so: run
  once → restart with `--prewarm`) so even the first turn is a cache hit instead of a cold
  prefill that can trip a client's idle watchdog. Prewarm is auto-enabled as
  `prewarm-<port>.json` when `--idle-timeout 0` (opt out: `--no-prewarm`); combining
  `--prewarm` with idle unload warns, since unload discards the warmed cache and the reload
  path deliberately does NOT re-warm (a synchronous re-warm would stall the triggering
  request pre-SSE and trip the client watchdogs the heartbeat exists to defeat).
- **Anthropic Messages API** (`POST /v1/messages`, `/v1/messages/count_tokens`) — the
  same trick as the Ollama shim, for Codex: it speaks Anthropic, so locally does
  too, and the real CLI drives a local NPU/iGPU model with `ANTHROPIC_BASE_URL` +
  `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_MODEL` and **no proxy process**
  (`Codex-router` is the alternative and is no longer needed). Translation is
  edge-only: the request is converted to the OpenAI shape `_prepare_turn` already
  handles, and the SSE stream is adapted by *consuming* the existing OpenAI frames
  (`_anthropic_stream`) rather than teaching DeviceSlot a second frame format — one
  tested streaming path (cancel, heartbeat, tool buffering), not two that can drift.
  Anthropic's protocol is block-structured where OpenAI's is a flat delta list, so text
  lives in an explicitly opened/closed content block and each tool call is its own
  block; since tool turns are buffered anyway, each tool block is emitted whole. Verified
  on Qwen3-8B/GPU: non-stream, stream, tool call + `tool_result` round trip, count_tokens.
  `<think>` blocks are **stripped here but not on the OpenAI path** — the web UI renders
  them as collapsible, an API client has nowhere to put them. `_ThinkFilter` does it
  token-by-token (tags arrive split as `<`,`think`,`>`), and if a small model burns its
  whole budget reasoning without closing the tag, the reasoning is returned rather than
  an empty message.
- **Driving Codex from a local model** (verified 2026-08-10, Arc B390):
  `scripts/Codex-local.ps1` sets `ANTHROPIC_BASE_URL`/`_AUTH_TOKEN`/`_MODEL`
  and launches the CLI — which ships *inside the Codex desktop app*
  (`%APPDATA%\Codex\Codex\<version>\Codex.exe`), so it is normally not
  on PATH and nothing needs installing. The settings panel has a button
  (`POST /v1/code/launch`, localhost-only, directory-only input) that runs the
  same script. Two things decide whether this is usable, and both are
  measurable: **tool count** (Codex sends 30 tool schemas = 68% of a
  141,959-char request; `Codex --tools "Bash,Edit,Read,Write,Glob,Grep"`
  cuts it to 48,747; `--agent-tools "Bash,Edit,Read,Write,Glob,Grep"` does the
  same server-side — 30 tools -> 6, rendered prompt 141,959 -> 55,735 chars —
  which is the only option for clients with no such flag, e.g. the VS Code
  extension) and **offload ratio** (any offload punishes prefill far
  harder than decode — agents are prefill-bound). With both handled:
  **75 s cold / 66 s warm per turn**, vs 369 s before. Tool calls execute
  correctly; answer *quality* is a separate question — a 30B int4 miscounted
  files in a Glob result on the first try.
- Observability: per-request log lines include TTFT (streaming: wall-clock to first token;
  non-streaming: `perf_metrics` via `extract_perf`) — a prefix-cache hit is sub-second vs a
  cold multi-second/minute prefill, so hits/misses are visible without instrumentation.
  `/health` has `prompt_cache_info` (pool size, prewarm file) + per-slot `last_ttft_ms` and
  `prewarmed`; `prompt_cache` stays a bare bool (start-openclaw.ps1 truth-tests it).
- Memory preflight at load (`_preflight_memory`): warns (never blocks) when weights + KV
  pool exceed the device budget (GPU: `GPU_DEVICE_TOTAL_MEM_SIZE`, which reflects Windows'
  ~half-RAM iGPU policy and Intel's "Shared GPU Memory Override" driver setting; CPU: total
  RAM) and logs the KV pool's token capacity from `config.json` geometry (~56 KB/token for a
  7B coder, ~96 KB for 30B — a too-small pool hard-fails generation with
  `Got unfinished GenerationStatus`, issue #21; `explain_genai_error` annotates that error
  with a `--cache-size-gb` hint wherever it surfaces).
- Whisper: WhisperSlot + WhisperPipeline for STT, `POST /v1/audio/transcriptions`, CPU or GPU
- VAD: `VadSlot` + Silero on plain OpenVINO (`--vad-dir`, `--vad-device`, default CPU —
  it is ~1M parameters and has no business on the NPU). Not a genai pipeline; there
  isn't one. Drives auto turn-taking over the `/v1/audio/stream` WebSocket
  (`flask-sock`, optional: without it the flag warns and voice stays push-to-talk).
- `POST /v1/audio/warm` brings Whisper/TTS/VAD resident before the first turn. Idle
  unload is doing its job when it evicts them, but the reload then lands *inside* the
  user's first spoken turn and reads as "voice is slow". The Voice tab calls it on
  entry — measured **3.0 s** of loading moved out of the first turn (1.96 s TTS +
  1.09 s Whisper, warm compile cache).
- Utilities: `UtilSlot` holds PP-OCRv6 plus a lazy `UtilityEngine` on NPU and/or
  GPU. Endpoints: `/v1/util/read`, `/background`, `/upscale`, `/detect`,
  `/generate`, `/index`, and `/search`. Verified NPU models are MODNet FP16
  portrait matting, OMZ single-image-super-resolution-1033, RF-DETR Small INT8,
  and MiniLM-L6 INT8 embeddings. Image generation remains disabled after the
  SD1.5 INT8 NPU pipeline failed dynamic compile, forced-static compile ran for
  >22 minutes without output, and the GPU fallback exited during compile.
  `POST /v1/util/read` accepts images plus PDF/DOCX/PPTX/XLSX/XLS/HTML/EPUB/
  CSV/text documents. Images and PDF pages without a text layer use OCR on the
  requested engine (NPU default). Text-bearing documents are converted in memory
  with MarkItDown's stream API and return `engine: null`/`X-Device: LOCAL` — do not
  pretend XML/text parsing used an accelerator. PDFium renders at 144 DPI and at
  most 50 scanned pages per request. MarkItDown plugins stay disabled so this
  private local path cannot grow cloud-backed converters by accident.
  Utility models compile on first use, report per-task availability in `/health`,
  and share the global NPU lock with chat. Search indexes are bounded and held
  only in process memory (8 indexes, 20 files/index, 500 chunks/index).
  The web UI uses one persistent left sidebar—Chat, Voice, and each utility are
  peers; do not add a second nested utility navigation or marketing slogans.
- TTS: TtsSlot + `Text2SpeechPipeline`, `POST /v1/audio/speech` (OpenAI shape, returns
  WAV; `response_format: pcm` also available), `--tts-dir` / `--tts-device` (default CPU,
  keeping the GPU for ASR and the NPU for chat). **Backend-agnostic**: genai 2026.3 drives
  both SpeechT5 and Kokoro through the same class, so the model directory decides. Kokoro
  is the better voice and is what's verified here (`OpenVINO/Kokoro-82M-int8-ov`, 24 kHz,
  54 voices, ~1.7 s warm for a short line on the 358H CPU). Kokoro **requires** an explicit
  speaker embedding — `TtsSlot._speaker_embedding` loads `voices/<name>.bin` and reshapes it
  to whatever `get_speaker_embedding_shape()` reports ({510,1,256} Kokoro, {1,512} SpeechT5),
  so one code path serves both; SpeechT5 needs no voice file (built-in cmu-arctic fallback).
  **There is no streaming** — generate() returns the whole utterance — so every caller must
  be written to wait for a complete waveform.
- Audio slots participate in idle-unload like generative ones. They previously did **not**:
  `_idle_watchdog` walks every slot and reads `last_used` / calls `unload()`, and WhisperSlot
  had neither, so `--whisper-dir` with the default 1800 s timeout killed the watchdog thread
  on its first pass and silently disabled idle-unload for *all* slots. Fixed 2026-08-09.
- OpenVINO GenAI may unify VLM/LLMPipeline — when that happens, simplify the dual-pipeline routing
- Routing: images go to GPU, text goes to NPU (or GPU if no NPU)
- **Runtime model swap** (no restart): `GET /v1/models/available` lists loadable model
  dirs (`--models-dir`, else the parents of what's loaded plus `~/models`; audio models
  are filtered out by `_is_generative_dir` so Whisper/Kokoro aren't offered as chat
  models). `POST /v1/models/load {"model": name, "device": "NPU|GPU"}` unloads the slot's
  current model **before** loading the new one, so peak memory stays one model — the
  Ollama-style one-at-a-time behaviour. Synchronous: it returns when the model can
  actually serve (20-40 s typical), rather than reporting success on a model that can't
  answer yet. The settings panel has the same control. The load is **retried up to 3×**
  with a gc between attempts: dropping a pipeline releases its native resources on the
  GC's schedule, and building a new one too soon intermittently fails to re-register the
  tokenizers extension (`Failed to load shared object: openvino_tokenizers.dll`) —
  observed once, then the identical request succeeded.
- **Dynamic placement** (`_choose_device`): `device` on `/v1/models/load` is a
  *preference*, not an order, and placement considers **every detected device**
  (`DEVICES`), not only ones that already hold a slot — so a single-slot
  "one model at a time" setup can still use the NPU. The slot itself **moves**:
  after unloading, `device_name`/`device_id` are re-pointed at the chosen device,
  so the two models are never resident at once.
  Preference order is **NPU → GPU → CPU** (NPU-first is the project's whole
  point, and it's the low-power engine; CPU last, see TODONT.md). Capability is
  checked before memory — the NPU is ruled out for VLMs (no working vision path)
  and for group-quantized int4 (`group_size` != -1 crashes the vpux compiler),
  read from the IR's `rt_info` rather than the folder name. Then fit (weights +
  KV pool vs `_device_mem_bytes`, skipped when `--offload-ratio` is on since
  offload keeps only part of the experts resident); the NPU reports no budget at
  all (it allocates from system RAM), which is normal, not a warning.
  Returns a `placement` string explaining the choice, e.g. "ruled out NPU: NPU
  has no working vision path; GPU: fits (6.6/18.5 GB)".
  Measured: Qwen3-8B (channel-wise, text) auto-lands on NPU; Gemma E4B (VLM) and
  Qwen2.5-Coder-14B (group_size 128) both auto-land on GPU with the reason given.
- **One model at a time is the default.** `scripts/locally-launch.ps1` starts a
  single generative slot, because the NPU has no memory of its own — it allocates
  from the same pool as the GPU — so a second resident model costs real RAM and
  heat for a model you cannot talk to concurrently anyway (decode is bandwidth-
  bound on a shared bus). Measured: two models left ~2 GB free of 31.5 GB; one
  leaves ~14 GB.
- **Compile caching** (`MODEL_CACHE_DIR`, default `.ov-cache/`): loading a model is
  mostly *compiling* it for the device, and the NPU rebuilds its whole graph every
  start. `CACHE_DIR` is passed to every pipeline (all three devices advertise
  `EXPORT_IMPORT`). Measured on Qwen3-8B/NPU: **65.1 s cold → 9.3 s cached**, for
  659 MB of disk. `--no-model-cache` disables it.
- **Voice and chat get different prompts.** `buildMessages(forceNoThink, forVoice)`
  appends a voice-only directive last (so it overrides the general prompt's
  formatting) and voice turns cap at `VOICE_MAX_TOKENS`. Markdown read aloud is
  noise — a table becomes gibberish and a list becomes a stream of "dash" — and
  audio can't be skimmed, so spoken replies are 2-3 plain sentences that point at
  the Chat tab when the real answer needs a table or code. Measured: same question,
  same model — chat answer rambles, voice answer is 196 chars with no markdown.
- **Thinking models on the voice path** (2026-08-15). Reasoning must never be spoken, and
  the old regex could only tell reasoning from answer once `</think>` existed — which is
  never true mid-stream. `ThinkStream` is the client-side twin of the server's
  `_ThinkFilter`: same 8-character hold-back so a tag split across tokens can't leak, but
  it decides from stream *state* rather than from a regex over a half-written document.
  Three things it has to get right, all found by running it:
  - `/no_think` is a real Qwen3 control token and the English sentence is not. Asked
    "what is an NPU?" with only the sentence, Qwen3-8B opened a think block and spent
    **all 220 voice tokens** inside it without answering. With the token appended to the
    last user message (on a copy — `chatHistory` must not grow control tokens):
    **210 tokens / 13.4 s → 31 tokens / 3.1 s**, correct answer.
  - An unclosed `<think>` does **not** mean the model was still reasoning. Under
    `/no_think` Qwen3 opens the tag, writes the answer inside it and stops without
    closing. What separates the cases is `finish_reason`: `length` means it never reached
    an answer, `stop` means what it wrote *is* the answer. Hence `consumeStream` now
    reports the finish reason.
  - When it genuinely ran out mid-thought, voice says one short line pointing at the chat
    tab. The server's "show the reasoning rather than nothing" call is right for an API
    client and wrong out loud: the reasoning is in the chat thread, collapsed, and reading
    a page of internal monologue aloud is worse than silence.
  Voice forces no-think by default; the **Let it think first** toggle
  (`locally-voice-think`) allows it, and the filter holds all output until `</think>`.
- **Speech starts before the answer finishes.** TTS can't stream, but it doesn't
  need the whole message: `splitSpeakable` peels completed sentences off the token
  stream and `makeSpeechQueue` synthesizes clip N+1 while clip N plays. Measured on
  a real turn — speaking began at 3.57 s while the LLM didn't finish until 3.75 s;
  first audio 4.5 s vs ~8 s waiting for the full message. The splitter won't break
  on decimals ("3.14159") and flushes over-long clauses so audio keeps up.
  `speechEpoch` + a set of abort controllers cancel the whole queue on barge-in.
  The **first** clip of a turn may be much shorter than the rest
  (`SPEAK_FIRST_MIN_CHARS` 4 vs `SPEAK_MIN_CHARS` 40): every character before the
  first clip is dead air, and "Sure." gets audio playing a whole clip earlier.
- **The text is revealed by playback, not by the token stream** (karaoke). `#voice-reply`
  is written from `makeSpeechQueue`'s `onSpeakStart`, which fires when `audio.play()`
  actually resolves, so what you read is what you are hearing. Painting it from the
  deltas instead put the text seconds ahead of the voice, which is what made the two
  feel unrelated. The **chat thread still streams live and unchanged** — Chat stays the
  complete record. Without a TTS model there is nothing to sync to, so it falls back to
  live text.
- **Two bugs this replaced, both silent** (fixed 2026-08-15, regression tests matter):
  the closing sentence of every answer was displayed but **never spoken** (the end-of-turn
  tail was recomputed as a slice that always evaluated to `''`, while the real remainder
  sat unread in `pendingTail`); and a thinking model had its **reasoning read aloud** —
  including the literal string `<think>` — after which the actual answer was never queued
  at all, because the visible text shrank when `</think>` arrived and the offset
  bookkeeping never recovered.
- `POST /v1/models/unload {"device": "GPU"}` (device optional — omit to free every
  slot) releases memory on demand; the model dir is remembered so the next request
  reloads it via `ensure_loaded`. The manual twin of `--idle-timeout`. The Whisper
  endpoint calls `ensure_loaded` too, so an unloaded ASR slot wakes instead of 503ing.
  **Unloading really does return the RAM to the OS** — measured +6.2 GB (GPU),
  +5.8 (NPU), +5.7 (CPU) of system-wide available memory, with no allocator
  trickery involved; `EmptyWorkingSet` on top adds nothing (see TODONT). What was
  missing was the measurement, so the reply now carries a `memory` block:
  `weights_mb` (ours, auditable — what was dropped) kept separate from
  `before`/`after` system availability (an observation about the whole machine,
  which under pressure moves by more than we released). **Do not measure this
  with RSS**: an idle process has already had its working set trimmed by Windows,
  which is how "unload frees nothing" was once concluded. `_mem_status` reads
  `ullAvailPhys`, `_process_memory` reads `PrivateUsage`, and `_settle_memory`
  polls because the GPU driver releases asynchronously (~2 s, ~3 GB late).
- **GPU memory ceiling is a Windows policy, not hardware.** On the 358H,
  `GPU_DEVICE_TOTAL_MEM_SIZE` is 16.4 GB — exactly 52% of 31.5 GB RAM. The NPU
  reserves nothing (it exposes no memory-budget property and allocates from system
  RAM on demand), so "the NPU is hogging memory" is never the explanation. Raise the
  iGPU share in Intel Graphics Software → System → "Shared GPU Memory Override" if a
  model doesn't fit; that is the alternative to `--offload-ratio`.
- **Turn-taking is a model, not a loudness gate** (rewritten 2026-08-15). The old VAD
  compared mic RMS against `max(absolute floor, noise floor × multiple)`, and no amount
  of tuning can make that work: a level says how loud the room is, not whether anyone
  is speaking. In any room with steady noise the gate sat open permanently — turns
  started by themselves and never ended, because the silence counter never got to run.
  `VadSlot` runs **Silero** on OpenVINO instead. Measured on this box, mean speech
  probability: digital silence 0.004, quiet white noise 0.013, **loud white noise 0.007,
  loud 220 Hz tone 0.006 — 0% of frames over threshold in both** — against **0.801 for
  real speech, 80% of frames over**. That separation is the whole feature.
  Use `silero_vad_openvino_16k.onnx`: the stock `silero_vad.onnx` carries a
  sample-rate `If` node that OpenVINO's ONNX frontend cannot convert, and that build
  also takes a **576-sample window** (512 new samples + 64 of carried context), which
  is why `VadSlot` reads the window width from the model rather than assuming 512.
  Without `--vad-dir` the Voice tab **disables auto turn-taking and says why** — it
  does not fall back to the loudness gate, because the gate is what was broken.
- Web UI: `templates/index.html` + `static/css/style.css` + `static/js/app.js`, three tabs
  (Chat / Voice / Util). Chat and Voice share **one** `chatHistory` — switching modes
  must never drop the conversation, and voice turns are mirrored into the chat thread.
  Util keeps its selected file/result independently and can hand extracted text to Chat.
- **Markdown is hand-rolled and escape-first** (`renderBody`): the body is HTML-escaped
  before anything else, so only tags the renderer constructs itself can reach the DOM —
  that property, plus `<think>` blocks and scheme-restricted links, is why marked.js
  isn't here (TODONT). The order is the design: anything whose interior must survive
  the inline rules — fenced/inline code, **math**, backslash escapes, link and image
  tags — is parked in a `\0`-delimited placeholder up front and restored last (`\0` is
  stripped from the input, so a placeholder can't be forged by model output). Covers
  GFM: nested and loose lists, task lists, table alignment, images, autolinks, setext
  `=` headings, `~~~` fences, `_`/`*` emphasis with a word-boundary rule so snake_case
  survives.
- **Math is TeX → MathML** (`static/js/tex.js`, 24 KB, no dependency): `$…$`, `$$…$$`,
  `\(…\)`, `\[…\]` and bare `\begin{aligned|cases|pmatrix|…}`. The browser lays it out —
  MathML Core is in every current engine — so there is no KaTeX and no math webfont.
  **The `font-family` on `.msg-body math` is load-bearing**: growing ∑ and stretching a
  fence needs the OpenType MATH table, which the UI's Inter doesn't have (measured: ∑
  14px → 31px, `(` 14px → 27px with `Cambria Math`/generic `math` in the stack). Single
  `$` is the ambiguous delimiter since it also writes money; the rule is no space just
  inside the delimiters and no digit after the closing `$`, which leaves "$5 and $10"
  alone. Unparsable TeX falls back to its own source in a `<code class="math-raw">`
  rather than vanishing.
- **The palette is monochrome, and that is load-bearing** (2026-08-24). The UI used to
  code every reply by engine — `--npu` amber, `--gpu` cyan, `--cpu` slate, set per element
  as `--device` from the `X-Device` header — so routing was readable without reading
  labels. It also meant two saturated hues from opposite sides of the wheel sat next to
  each other on every screen, on surfaces that were themselves blue-cast (`#0f1114`,
  `#171a1e`, `#1b1f24` — B is 4-5 points above R in all of them), which is how a
  "neutral dark" UI read as navy. The whole ramp is now pure neutral (R=G=B) in
  `style.css :root`: `--bg #0A0A0A`, ink as one white at four opacities, surfaces as
  white alphas. **Provenance did not go away** — it moved to the engine's NAME in the mono
  face plus a dot whose FILL says the same thing: solid NPU, ring GPU, hollow CPU. Three
  states that survive a monochrome palette, a colour-blind reader, and a grayscale
  screenshot. `--npu`/`--gpu`/`--cpu`/`--device` all still resolve, to `--ink`, because
  ~40 rules reference them; that keeps the reskin in one place instead of requiring every
  call site to be found. **The one hue left is `--alarm`** (`#C4544A`), for the live
  microphone and for things that actually broke — an error must not be mistakable for
  chrome. Anything else reaching for colour is a bug: a rendered-page audit of 589
  elements returns exactly one chromatic value, and that is the rule.
- **Motion vocabulary is three moves, and no more**: `enter` (opacity + `translateY` +
  `blur()` → 0, staggered by *duration* rather than delay so nothing sits visibly waiting
  its turn), a `-2px` hover lift on `cubic-bezier(.16,1,.3,1)`, and one `sweep` — a conic
  gradient masked to a 1.5px ring that travels round the brand mark while tokens stream,
  driven by `body[data-busy]` from `setGenerating()`. The mark is therefore the logo, the
  busy indicator and the product's idea at once. Everything is under 300ms except first
  paint, and `prefers-reduced-motion` collapses all of it. Backdrop blur stays off
  full-surface elements: the iGPU may be running inference, so compositing is not free.
- **The mark's geometry is measured from the artwork, not eyeballed.** It is a filled
  ring — both edges are three-lobed polar curves `r = a0 + a3*sin(3t) + a6*cos(6t)` —
  plus a free-floating circle. It is **not** a rounded triangle and is **not** stroked;
  an earlier cut was both, and TODONT.md records how that happened. Coefficients were
  fitted by least squares to the alpha channel of the original PNG (outer
  a0=35.06/a3=7.950/a6=-0.819, inner a0=22.01/a3=1.772/a6=+0.185, core r=11.43,
  97.1% IoU against the original). The inner edge lobes far less than the outer, which
  is what makes the band thick across each side and thin at the lobes, and `a6` flips
  sign between them. `scripts/mark.py` holds the geometry, `scripts/build_mark.py`
  emits the sprite path, `locally.svg`, the PNG tiles and a per-size `.ico`
  — **regenerate, never hand-edit the coordinates.** One symbol (`#i-mark`) serves every
  size now; the old small cut existed only because the superseded design had a middle
  ring that closed up below ~32px. Two things the SVG needs that the PNGs do not prove:
  the inner contour must wind opposite the outer (that is what cuts the hole under
  fill-rule nonzero), and the curves need 24 Bézier segments, not 12 — the 6th harmonic
  adds inflections that leave 12 segments 1.7px off at 512px.
- Fonts are **self-hosted** in `static/fonts/` (Inter + JetBrains Mono woff2, ~115 KB).
  No CDN anywhere — a local-first tool must work on a plane. The mono face is reserved for
  machine truth (device names, timings, token counts).
- Settings: a persisted **system prompt** (`localStorage`) composed in one place,
  `buildMessages()`, with the no-think directive — Qwen drifts into Chinese without an
  explicit language instruction. `null` means "never set" (use the default); `""` is a
  deliberate opt-out and must survive a reload. The default targets the failure modes of
  **4-30B** models (preamble, padding, invented APIs/citations, disclaimers on ordinary
  questions) in concrete terms, since small models follow "read text exactly as printed"
  and ignore "be accurate". It is **~220 tokens** (measured, Qwen3-8B and gemma-4-E4B) and
  must stay near that: on the NPU it competes with the user's own text inside
  MAX_PROMPT_LEN=8192, and it is re-sent every turn. A frontier agent's 21k-token prompt
  would eat half the NPU budget before anyone typed anything.
- Voice tab: IDLE → LISTENING → TRANSCRIBING → THINKING → SPEAKING → IDLE. Turn
  decisions come from the server's VAD over `/v1/audio/stream`; the rAF loop now only
  paints the orb and the meter from the local level, so the picture never stutters with
  the network. Push-to-talk is the fallback and the only mode without `--vad-dir`.
  **Barge-in cuts audio first** and only then tries to cancel generation — OpenVINO may
  be blocked in native code and ignore the cancel, so the user-visible interrupt must not
  depend on it. It is now also possible during TRANSCRIBING and THINKING, which
  previously had no interrupt path at all. Two traps found the hard way: `pause()`
  fires neither `ended` nor `error` (so playback promises must be settled explicitly), and
  an interrupt *during synthesis* must invalidate the pending request via a `speechEpoch`
  guard or the assistant starts talking after being interrupted.
- **Capture is one AudioWorklet** (`static/js/vad-worklet.js`) at a 16 kHz `AudioContext`,
  emitting 512-sample frames — verified 31 frames/s of 512 samples. It replaced
  `AnalyserNode` + `MediaRecorder` for three reasons: Silero needs fixed-size frames, the
  WebM/Opus blob had to be decoded and resampled after the fact, and a recorder started
  *after* speech was detected clipped the first word. Frames now feed a **500 ms pre-roll
  ring**, so a turn begins with the audio that opened it. The chat tab's mic button keeps
  its own `MediaRecorder` and its own stream — they shared one `mediaStream` global, which
  meant ending a chat recording tore down the voice graph's tracks.
- **Auto turn-taking uploads nothing.** The server already holds the turn's audio, so it
  transcribes it too and answers `speech_end` with a `transcript` event. Speculation
  starts at 400 ms of silence, before the 900 ms patience expires — measured **0.53 s of a
  2.31 s Whisper run happening during the pause**, with no upload afterwards. Speculation
  alone was not enough (the head start is smaller than the run), which is why the server
  owns the whole transcription rather than racing the client for it. Every run is
  sequence-tagged so a guess started before the user carried on can never overwrite the
  real result. PTT still uploads a WAV — there is no open socket to have pre-fed.
- Collapsible `<think>` blocks, "Just answer me, dammit!" button, temperature slider
- `threaded=True` on Flask, concurrency via per-device locks
- `models.json` — curated model registry (npu, gpu_vlm, gpu_llm, whisper categories)
- `install.ps1` detects devices, shows model menu, generates `start.ps1`. Agent setups get
  `--idle-timeout 0` (keeps the prefix cache alive; auto-enables prewarm). Local/cached
  models are validated before being offered or linked: the `.bin`+`.xml` pair must exist
  and the `.bin` must not be truncated (#17) — the IR `.xml` records each weight blob's
  offset+size, so max(offset+size) is the exact minimum `.bin` size (the IR has no
  checksum; truncation is the realistic failure, corruption-in-place is out of scope).
  `locally.py` re-checks the same invariant at load (`_verify_weights_integrity`) since
  models can arrive without install.ps1 — a truncated/missing model fails with a
  plain-English error, and the "Is another process using the NPU?" hint is suppressed
  for that class of failure.
- Model naming: the **directory name is authoritative** — it's the web-UI label and the
  model ID clients request. `resolve_display_name` uses the name as given and only follows
  a symlink/junction when that name is generic (`model/`, `gpu-model/`, which is what
  install.ps1 links). It previously called `realpath` unconditionally, which silently
  discarded a deliberate rename (#19). There is deliberately **no `--model-name` flag** —
  see `TODONT.md` for why the rename is the interface.
- `--scan` reports what each model directory actually holds — display name (and where it
  came from), LLM/VLM/Whisper, architecture, MoE shape, geometry, integrity, and the real
  weight precision read from the IR's model-level `<rt_info>` (`nncf/weight_compression/
  mode` + `group_size` + `ratio` + `awq`) rather than from the folder name, which can lie.
  `read_ir_rt_info` seeks the **tail** of the `.xml` (the graph is tens of MB on a large
  model; the model-level block is the last `<rt_info>`, after `<edges>`). No server, no
  device init, no model load. Note VLM configs nest geometry under `text_config` —
  `_text_config` handles that, which also fixed the KV half of the memory preflight
  silently no-op'ing on every VLM.
- `download-model.ps1` — fetch/convert any HF model. PowerShell-style flags
  (`-Convert -Weight int4 -Trust`), NOT GNU `--convert` (#19: the docs once showed
  `--` syntax and users copy-pasted it; a catch-all param now prints the corrected
  command when someone tries). **Conversion is RAM-bound, not disk-bound**: optimum-intel's
  Qwen3-Next patcher builds an fp32 copy of every expert weight (workaround for OpenVINO
  CVS-181449) — for Qwen3-Coder-Next that's `512 experts × 2048 × 512 × 4 B` = 2 GB per
  projection stack, ~288 GB across 48 layers × 3. Measured: **400 GB of Windows pagefile
  (on 128 GB RAM) succeeded**, 200 GB did not (#19, Dmitriy Teteruk). Weight format is
  irrelevant to this stage — the blowup is before quantization.
- Tool calling: **GPU/iGPU + CPU** (gated by `_tools_supported`, i.e.
  `device_name in ("GPU","CPU")`); the **NPU is excluded** — it has a hard prompt cap and
  small NPU-class models can't drive agent loops, so when the NPU serves the request we
  ignore `tools` and answer as plain chat. `/api/show` advertises the `tools` capability only
  for GPU/CPU slots (so Copilot won't offer NPU models for agent mode). CPU is viable for
  agents on strong desktops (e.g. Core Ultra 9, many cores) where prefill can beat a weak
  iGPU. Tool specs from the request `tools` array are rendered into a system prompt
  (Qwen3-Coder native format); the model's emitted call is parsed back into OpenAI/Ollama
  `tool_calls`. `parse_tool_calls` recognizes several native formats, since a model
  often ignores our prompt and falls back to what it was trained on: Qwen3-Coder XML, Hermes
  JSON-in-`<tool_call>`, **bare `<function=>` with no wrapper (Qwen2.5-Coder native)**, Mistral
  `[TOOL_CALLS]`, Llama `<|python_tag|>`, DeepSeek `<｜tool▁calls▁begin｜>` blocks, plus a
  bare-JSON fallback. See `render_tools_prompt` / `parse_tool_calls`. Copilot Chat 0.53+ hits
  `/v1/chat/completions` (delegates to `chat_completions`); `/api/chat` also handled.

## Environment

- Primary: Windows 11, Python 3.10+
- Cross-platform: scripts use `#requires -Version 7.0` and branch on
  `$IsWindows`. Linux + PowerShell 7 is confirmed working (user-reported
  on Core Ultra 7 258V with NPU + GPU, issue #6). There is no install.sh —
  Linux runs the same install.ps1 via pwsh. On Linux, NPU/GPU need the
  Intel userspace drivers installed or only CPU is detected; the Linux NPU
  stack (`intel-npu-driver`) is less mature than Windows.
- Intel Core Ultra (NPU) + Intel ARC 140V 16GB (GPU)
- OpenVINO 2026.1+ with openvino_genai
- venv in `venv/`, activate before running
- `venv-2026.3/` — OpenVINO 2026.3 runtime **plus the modern export stack**
  (optimum-intel 2.1.0, transformers pinned to 5.4 — the LFM2 exporter's cap —
  einops, nncf 3.3). Use it for exporting architectures optimum-intel 1.27
  can't load (EAGLE-3 drafts, new-family models); `venv/` keeps the old
  1.27/tf-4.57 stack that some exporters still need. 2026.3 passed the
  regression suite 2026-08-06.

## Development preferences

- Read `TODONT.md` before proposing anything structural — it records approaches
  already rejected, with the reason. Add an entry whenever one is abandoned.
- Keep it simple. One file (`locally.py`) is fine. Don't split into modules unless it gets unwieldy.
- PowerShell for install/launch scripts (Windows-native users).
- Runtime flags over hardcoded config (e.g. `--port`, `--device`).
- When testing, use small payloads / short prompts. Don't run full model loads unless needed.
- VLM prompts must be dead simple for small models (3B). One question, one answer, minimal JSON. All logic in Python, not in the prompt.
- Qwen3-VL is now pre-exported by Intel (OpenVINO/Qwen3-VL-8B-Instruct-int4-ov, May 2026) — not yet tested here. Earlier note about optimum-intel support is obsolete.

## Known issues

- **HF downloads on the 358H laptop's network need aggressive retry, not fewer workers.**
  Measured 2026-08-09: ~75% of HTTPS connections to huggingface.co are reset mid-handshake
  (`WinError 10054` / curl error 35). Controlled test, 12 sequential requests per client —
  python-urllib 2/12, curl 4/12, hf_hub-style UA 3/12, browser UA 1/12: the failure rate is
  **client- and User-Agent-independent** (an earlier UA theory looked right on two samples
  and did not survive testing). `hf download` fails because huggingface_hub doesn't retry
  connection resets hard enough, *not* because of parallelism — though genuinely parallel
  transfers do make it worse, so keep it serial too. What works:
  `curl -C - --retry 30 --retry-all-errors --retry-delay 1 --speed-limit 2000 --speed-time 20`
  (resume + stall detection). At ~24 MB/s between resets a 22 GB model still lands in ~15 min.
- NPU default prompt limit is 1024 tokens — we override to MAX_PROMPT_LEN=8192
  (measured ceiling; see TODONT.md)
- (resolved 2026-05-25) VLMPipeline gained streaming support in openvino-genai 2026.1; verified on Arc 140V iGPU at ~11 tok/s decode.
- Qwen3 thinking models can exhaust token budget on `<think>` before producing an answer
- Cancel (`/v1/cancel`) relies on OpenVINO invoking the streamer callback. If the native code blocks without yielding, cancel won't take effect — generation completes naturally.
- Chat history unbounded in web UI — user clears with Ctrl+N when long sessions approach MAX_PROMPT_LEN
- Tool-enabled turns are buffered, not token-streamed: we must see the whole tool-call block
  before emitting a structured `tool_calls` delta, so the full generation is collected before
  the result is sent (no incremental tokens that turn). To stop a slow prefill on a big agent
  prompt from tripping client idle watchdogs (Copilot/OpenClaw abort with no output after
  ~120s), the streaming tool path runs generation in a background thread and emits SSE
  keep-alive pings every `HEARTBEAT_SECS` (`_sse_tool_stream`); the plain stream path
  (`stream_llm`) pings the same way during a long prefill. True token streaming on tool turns
  (stream until a tool-call prefix appears) is still TODO.
- Big agent prompts (OpenClaw ships ~21k-token system prompts) prefill slowly on weak iGPUs
  (~6 min TTFT on the desktop 285K Xe-LPG). Mitigations: smaller coder model, CPU on strong
  desktops, trimming the client's tool set, and the keep-alive above so turns complete instead
  of aborting. OpenVINO can't cancel a blocked prefill, so an aborted client leaves the
  generation churning — another reason to keep clients connected via heartbeat.

## MoE disk offload (2026-08-06)

**`--offload-ratio` defaults to `auto`** and should stay there: the right ratio is
not a preference, it is arithmetic — (weights + KV pool + runtime) against the
device budget, divided by the share of the weights that are actually
offloadable (`_moe_expert_fraction`, computed from the config's expert
geometry: ~95% for Qwen3-Coder-30B-A3B, ~90% for gemma-4-26b-a4b). Auto picks
the smallest ratio that fits and stays off entirely when the model already
does, which matters because the cost is real and non-linear (140V, same model:
ratio 30 → 25.3 tok/s, ratio 90 → 5.1). It only engages on **GPU + XMX + MoE**,
the three conditions under which offload does anything at all — a pinned ratio
on a dense model or a non-XMX GPU is silently ignored by the plugin, which is
the most confusing failure in TODONT.md. `_device_fits` (dynamic placement)
applies the same three conditions, or it would place an unfittable dense model
on the GPU on the strength of an offload that can't help it.

`--offload-ratio PCT` pins a value instead; `0` disables. PCT% of MoE expert
weights are streamed from disk on GPU slots (OpenVINO 2026.3 `OFFLOAD_RATIO`). **Requires XMX** (Arc/Lunar Lake;
`GPU_HW_MATMUL` in OPTIMIZATION_CAPABILITIES) — silent no-op without, and
locally warns at startup. Verified on Arc 140V, Qwen3-30B-A3B int4
steady-state: ratio 30 → 10.8 GB resident @ 25.3 tok/s (interactive!);
90 → 2.35 GB @ 5.1. Pick the smallest ratio that fits. The expert LRU
needs ~60 tokens to warm — benchmark steady-state, not first-sentence.
Known upstream bug: a SECOND generate() on an offload-active PLAIN
pipeline hangs in native code (uninterruptible). locally's serving path is
unaffected — the CB backend it uses was verified with sequential requests
(140V, ratio 30: 12.5 then 15.9 tok/s, prefix cache TTFT 8.0s→1.9s). Non-XMX iGPUs can't
load big MoE at all (USM staging OOM) — full story in TODONT.md.

## NPU export rule (2026-08-06)

Models converted for the NPU **must be channel-wise** (`download-model.ps1
-Weight int4-cw` or `int8-cw`): default group-quantized int4 IRs crash the
NPU driver compiler ("Found N duplicated names", known vpux bug). int8-cw
halves decode vs int4-cw but keeps more quality — except on LFM2-family,
where no good int8 NPU variant exists (see TODONT.md). OFFLOAD_RATIO (2026.3
MoE disk offload, GPU-only) could not be validated on the desktop iGPU —
see TODONT.md before recommending it.

## Verified models

- Qwen3-8B (INT4-CW) on NPU — recommended, needs MAX_PROMPT_LEN=8192
  (verified: 8,160-token needle prompt recalled correctly)
- SmolLM3-3B (INT4-CW 23 tok/s, INT8-CW 12 tok/s) on 285K NPU — 2026.3, our export
- LFM2-1.2B / LFM2.5-1.2B-Instruct (INT4-CW, ~37-39 tok/s) on 285K NPU — NPU-only
  builds, old-stack exports fail CPU/GPU (see TODONT.md)
- MiniCPM5-1B (INT4) on GPU/CPU — 2026.3, no NPU support upstream
- Phi 3.5 Mini (INT4-CW) on NPU — smaller, faster
- DeepSeek-R1-1.5B (INT4-CW) on NPU — works but terrible quality (testing only)
- Gemma 3 4B Vision (INT4) on GPU — fast VLM
- Qwen2.5-VL-3B/7B (INT4/INT8) on GPU — proven for image tasks
- Qwen3-30B-A3B on GPU — needs >16GB VRAM, falls back to CPU silently on 16GB cards
- **Qwen3-Coder-30B-A3B-Instruct (INT4) on Arc B390** — the agent/coding model here,
  and the newest Qwen coder that exists (no 3.5/3.6/3.7 Coder has shipped). 15.2 GB,
  128 experts / 8 active, 262k model context, 96 KB/token KV. Auto offload is what
  makes it fit the 18.5 GB budget, and the ratio it picks trades directly against
  context: **32k → ratio 15% → 21.2 tok/s** (TTFT 1.15 s warm), **87k → ratio 49% →
  12.8 tok/s** (TTFT 5.8 s). Tool calls and sequential requests verified through the
  serving path (the upstream second-generate hang is plain-pipeline only). Raising
  the iGPU Shared Memory Override to ~25 GB takes the ratio to 0 and buys both.
