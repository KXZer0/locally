# locally

OpenAI-compatible LLM/VLM server for Intel hardware. NPU-first.

## Architecture

- `locally.py` — Flask server, DeviceSlot class per device, auto-detects VLM/LLM from config.json
- NPU: LLMPipeline with MAX_PROMPT_LEN=8192 (raised from 4096 on 2026-08-19;
  8192 is the vpux compiler's ceiling, not a memory limit — 9216/10240/12288
  all fail graph legalisation, and KV compression does not move it. See
  TODONT.md and scripts/npu-context-probe.py), streaming via SSE
- GPU: VLMPipeline (images) or LLMPipeline (text). Both stream as of openvino-genai 2026.1 — verified on Arc 140V iGPU.
- **Prefix caching is a CPU win and a GPU trap** (2026-08-10, B390 +
  Qwen3-Coder-30B-A3B): the CB backend prefills **~10x slower** than the plain
  pipeline (187 vs ~2,000 tok/s) and **hangs** when consecutive prompts share a
  near-total prefix — i.e. exactly what an agent client sends. `--no-prompt-cache`
  took Claude Code's turn from 75 s to **8.7 s** and pi's from 53.6 s to **1.8 s**.
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
  same trick as the Ollama shim, for Claude Code: it speaks Anthropic, so locally does
  too, and the real CLI drives a local NPU/iGPU model with `ANTHROPIC_BASE_URL` +
  `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_MODEL` and **no proxy process**
  (`claude-code-router` is the alternative and is no longer needed). Translation is
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
- **Driving Claude Code from a local model** (verified 2026-08-10, Arc B390):
  `scripts/claude-local.ps1` sets `ANTHROPIC_BASE_URL`/`_AUTH_TOKEN`/`_MODEL`
  and launches the CLI — which ships *inside the Claude desktop app*
  (`%APPDATA%\Claude\claude-code\<version>\claude.exe`), so it is normally not
  on PATH and nothing needs installing. The settings panel has a button
  (`POST /v1/code/launch`, localhost-only, directory-only input) that runs the
  same script. Two things decide whether this is usable, and both are
  measurable: **tool count** (Claude Code sends 30 tool schemas = 68% of a
  141,959-char request; `claude --tools "Bash,Edit,Read,Write,Glob,Grep"`
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
  MiniLM-L6 INT8 embeddings, **BGE-reranker-base INT8** (cross-encoder search
  reranking, 13 ms) and **PP-DocLayoutV3** (document layout, 82 ms).
- **Utility models are per-device tiers** (`TIERS` in `utility_pipeline.py`). A
  bare string means "same model on every engine"; a dict is per-device and
  `None` means that engine cannot serve the task — a third state distinct from
  "not installed" and "deliberately disabled", and `/health.util.<engine>.disabled`
  now carries the reason for all three, because each needs a different action
  from the user. The NPU gets the largest model that passes the probe gate (it is
  the low-power engine and the point of the project); the GPU exists for what the
  NPU genuinely cannot run. Only `upscale` currently differs: **OMZ 1033 on NPU,
  Swin2SR x4 on GPU**, because the vpux compiler rejects Swin2SR outright.
- **`--util-engines` defaults to `auto`** (every engine detected), not `npu`.
  It had to change the moment the tier map gained a GPU-only model: with an
  NPU-only default no GPU util slot existed, so `/v1/util/upscale?engine=gpu`
  503'd and the UI's GPU toggle stayed greyed out — the capability was built,
  documented, and unreachable. Slots are cheap (only OCR loads eagerly, ~160 MB;
  everything else compiles on first use and idle-unloads after). Verified live:
  NPU 480×320 → 1440×960 (3.00×, 138 ms), GPU → **1920×1280 (4.00×, 4.7 s)**.
  Under `auto` a missing engine is skipped silently — asking for what the box
  has cannot be a mistake worth warning about on every start.
- **Upscaling has no input ceiling on GPU.** `_upscale_omz` downscales anything
  above 640×360 *on the way in*, so a bigger source bought less magnification —
  measured 960×640 → 1620×1080, i.e. **1.69×**, a net quality loss on a 1080p
  source. `_upscale_swin2sr` tiles (256² tiles, 32 px overlap trimmed on inner
  edges only, edge-replicated rather than zero-padded) and holds a true **4.00×**
  at any size: 960×640 → 3840×2560. Cost is linear in pixels, ~1 s per tile.
- **Search retrieves wide, then reranks.** Embeddings pick 20 candidates
  (`_UTIL_RERANK_POOL`) and a cross-encoder reorders them. Measured on this
  project's own corpus: top-1 7/9 → **8/9**, MRR 0.889 → **0.944**. Swapping the
  *embedder* for a bigger one (BGE-base) changed nothing and was rejected —
  recall@5 was already 1.000, so ordering was the bottleneck, not retrieval.
  Responses carry `score` (embedding cosine, same meaning in both modes) *and*
  `rerank_score`, because a reranked list shows a lower cosine above a higher
  one and that reads as a bug without the number the sort actually used.
- **`/v1/util/read` returns structure, not a wall of text.** PP-DocLayoutV3
  segments the page and supplies a reading order (the 7th output column — sorting
  by y alone interleaves the columns of a two-column page). Headings become
  Markdown headings; table/chart/figure/formula regions are *flagged* rather than
  inlined as prose, which is honest — without a table-structure model their cell
  order really is scrambled. Strictly additive: any layout failure falls back to
  the plain reading rather than costing text OCR already read correctly.
- **`GET /v1/memory` + the bottom-right memory HUD.** Every figure it shows
  already existed server-side and none of it was visible, which is how a slot
  silently pinned at 90% offload read as "the model got slow". Three states —
  resident / offloading / critical — with thresholds from the same
  `_usable_gpu_bytes()` the offload decision uses, so the HUD cannot contradict
  the placement it describes. It always shows the driver's advertised ceiling
  **and** what the machine can actually back (23.6 GB vs 12.7 GB here), because
  showing the first alone is what makes "I have 24 GB of VRAM" a reasonable
  conclusion on a 31.5 GB machine. Every warning carries one concrete action.
- **One model control.** The settings panel had two dropdowns — one to pick a
  loaded model, one to load from disk — but with one model resident at a time
  the first always had exactly one entry, so it asked a question with a single
  possible answer. Now choosing a model *is* loading it. The chat request sends
  `loadedModelId` (what is resident) rather than the dropdown's value (what is on
  disk); conflating them would name a model that is not loaded yet. Image generation remains disabled after the
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
- **Speaker verification: Silero says *someone* is talking, this says *who***
  (2026-08-29). `SpeakerSlot` + `--speaker-dir`, a sibling of `VadSlot` on the plain
  OpenVINO runtime. It exists because nothing in the voice path had any notion of
  identity — a grep for `speaker_verif|ecapa|voiceprint|enroll|diariz` returned zero
  hits — so another person in the room opened a turn at **175 ms** of speech
  (`MIN_SPEECH_MS / 2`, even though `MIN_SPEECH_MS = 350` is commented "shorter than
  this is a cough"), got transcribed, and was sent as the user's own message; and
  `BARGE_MS = 400` of anyone's voice cut the assistant off. Both were reported as VAD
  bugs. The VAD was right and was being asked the wrong question.
  **Three verdicts, not two**: `match` / `reject` / **`abstain`**. Too little audio, no
  profile, a stale profile or a failed model all mean "cannot tell", and a system that
  cannot tell **lets the turn through** — an assistant that ignores its owner is a worse
  failure than one that occasionally hears the television. Gating is only in force when
  a slot, a loaded model, and a profile enrolled *by that model* with ≥6 s all hold.
  **Two input conventions, both handled.** Rank-2 is a raw waveform; **rank-3 is Kaldi
  fbank features, which is what every ONNX export people can actually download takes**
  (WeSpeaker, SpeechBrain, 3D-Speaker). `_fbank` computes them here — 25 ms/10 ms
  frames, NFFT 512, Povey window (hann^0.85), per-frame DC removal, 0.97 pre-emphasis,
  80 mel bins, log with Kaldi's floor, then CMN — rather than depending on torch for 40
  lines of arithmetic. **CMN makes the per-bin mean identically zero**, which is correct
  and is also why a mean-pooling test stub emits the same vector for every input; use
  the second moment. The profile is a duration-weighted running mean of L2-normalised
  embeddings, stamped with the model that made it (embeddings from two models are not
  comparable, and scoring against a stale profile would reject the real user with no
  visible cause) and stored in `speaker-profile.json`, which is **gitignored — it is
  biometric data**.
  **The threshold cannot be set from the outside**: it depends on model, microphone and
  room, so every verification logs its cosine and `POST /v1/audio/verify` scores a
  recording without gating anything. Record yourself, record someone else, put
  `--speaker-threshold` in the gap. Barge-in raises its bar from 400 ms to
  `MIN_BARGE_VERIFY_MS` (650) when gating is on and collects the candidate audio
  separately — the 500 ms pre-roll ring is below the verifier's minimum and is mostly
  the assistant's own voice coming back through the microphone. That is ~250 ms more
  latency before the assistant stops talking, in exchange for it being *you* that stops
  it. Endpoints: `GET/POST/DELETE /v1/audio/enroll` (additive; several short recordings
  beat one careful reading) and `POST /v1/audio/verify`.
  **Warming up on digital silence cannot work** — zeros are the one input guaranteed to
  produce the degenerate embedding `embed()` refuses, so warmup uses noise. Found by the
  first run of the test suite, and it would have disabled the feature for every model.
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
- **Startup: the chat model goes first, and /health stopped blocking** (2026-08-30).
  Two separate causes, both measured on the 358H with a warm compile cache.
  (1) **Every slot loaded at once**, and they are not equal: the user waits for the
  chat model, while OCR compiling on two engines competes for the same cores and
  bandwidth. Back to back, same machine: all-concurrent reached "locally ready" in
  **15.43 s**, chat-only in **9.97 s** — the utilities cost 5.5 s of the wait for a
  model they have nothing to do with. Utility slots now wait on a
  `threading.Event` the primary load sets in a `finally` (so a FAILED load
  releases them too), capped at 120 s so a slow model can never leave the Tools
  tab dead. Result: **11.24 s to chat-ready**, utilities finishing at 16.17 s off
  the critical path. Deferred, never dropped.
  (2) **`/health` cost 1.86 s per call**, on the endpoint the web UI polls —
  against 1.5 ms for `/v1/models`. The OpenCode and Odysseus liveness probes each
  did a real HTTP round trip (1508 ms + 360 ms, each its own configured timeout).
  The reason is not what you would guess: on this machine a connection to a CLOSED
  loopback port is **dropped, not refused**, so a "fast fail" that assumes
  ECONNREFUSED never fails fast. `core/portprobe.py` inverts it — readers get the
  last known answer instantly and a background thread refreshes it, while
  start()/stop() record what they already know for free. **1.86 s → 0.006 s**, and
  a state change still shows up immediately. Never put a network probe on
  `/health` again without caching it.
- **locally installs itself as a Windows app** (2026-08-30). `-CreateShortcut` wrote
  `locally.lnk` into the repo root and told the user to go wire it up by hand,
  while `locally-win32.ps1` was ALREADY looking for it under Start Menu\Programs —
  creation and lookup disagreed about where the app lives. It now also copies the
  shortcut to `%APPDATA%\...\Start Menu\Programs` (`-Desktop` adds a desktop
  copy), which is what Windows counts as installed: verified with `Get-StartApps`,
  which now returns `locally` with a generated AppID, so it is searchable from
  Start and pinnable to the taskbar. No binary is shipped and nothing is signed —
  the shortcut target is System32's powershell.exe — so Smart App Control never
  enters into it. **The Copilot key cannot be bound through Windows' own setting**:
  Settings -> Personalization -> Text input -> "Customize Copilot key" lists only
  MSIX-packaged signed apps, so it cannot target a .lnk or a plain .exe. A remapper
  pointed at the installed shortcut is the working route, and the script now says
  so instead of leaving it implied. **On this machine that remapper is NewPilot**
  (`GamerJagdish.NewPilot`, MSIX) — not PowerToys, which the docs used to name and
  whose Keyboard Manager holds no remaps here. NewPilot is itself MSIX-packaged,
  which is exactly why Windows accepts it as a Copilot-key target when a .lnk is
  refused; it then forwards the press to locally's shortcut. Windows Copilot is
  **not installed** here, so Win+C is free and already raises locally. Do not
  re-derive this and do not "fix" it by suggesting PowerToys.
- **First-run setup has no containers** (2026-08-30). It was a bordered, filled,
  shadowed dialog holding bordered, filled, rounded cards — boxes inside a box
  inside an overlay — and Camilo called it: "remove the containers". The shell is
  now edge-to-edge and the dialog draws nothing; only the ~920px measure survives,
  because the container going away is not a reason for a line of text to become
  1440px wide. Options are rows told apart by a single hairline, and the rule is
  the ONLY chrome: nothing is drawn around an item, just between items. Selection
  used to be a ring around a card, so with no card it became a 2px ink bar in the
  left margin plus a brighter title — two signals, both surviving monochrome and
  grayscale.
  **The softness is radial gradients, not `filter`/`backdrop-filter`.** That is
  the perf note above: blur on a full-surface element composites every frame and
  the iGPU may be loading a model while setup is open. A gradient is blurry by
  construction and costs one paint. Two fields, sized 120vmax so their edges never
  enter frame — an ellipse edge crossing the screen reads as a shape, and the point
  is that you cannot find where it begins.
  **Three alignment bugs, all found by measuring the rendered page, none by
  reading it.** (1) The dots sat in the middle column of a 3-column grid whose
  outer columns were `1fr` but sized by their CONTENT — Back on one side, a
  variable-length status string plus Continue on the other — so any status text
  pushed the "centre" off centre. Dots are now absolutely centred on the footer,
  which is independent of both sides. (2) Continue was measured at x=538 in a
  footer spanning 260..1020: Back is `display:none` on step 1, so auto-placement
  put the actions block in column 1. Both are now placed by explicit
  `grid-column`. (3) Head, content and footer had three different horizontal
  paddings, so the mark, the headline and the buttons each started on a different
  vertical line; they now share one `--setup-measure`, so the screen has exactly
  two vertical lines. Verified across all five steps: dots centre 640, Continue
  right edge 1020, Back left 260 — identical on every step. Below 680px the
  centred dots and the right-pinned button provably collide (measured at 375px),
  so there the dots take a row of their own.
- **The Code tab shows OpenCode's WEB interface, embedded** (2026-08-30). `opencode
  serve` already hosts the same UI that `opencode web` opens a browser at, so locally
  runs it headless (`core/opencode_web.py`) and frames it rather than spawning a
  window that competes with its own UI. It **pins the port** (4747) because
  `--port` defaults to **0**, i.e. random — a caller that does not pin it cannot
  link to, embed, or health-check what it just started. It **adopts** an
  already-listening OpenCode instead of spawning a duplicate, reports itself in
  `/health.opencode_web`, and only ever stops a server it started.
  Framing is allowed here and forbidden for Odysseus, and that is measured, not
  taste: OpenCode sends no `frame-ancestors`, no `X-Frame-Options` and no HSTS.
  The server is bound to **127.0.0.1 and that is not configurable** — it starts
  unauthenticated (`OPENCODE_SERVER_PASSWORD is not set; server is unsecured`) and
  runs commands, so giving it locally's 0.0.0.0 reach would hand a shell to the LAN.
  The consequence is that the Code tab's web view only works from the server's own
  machine; from a phone the button is disabled with that reason, verified over the
  LAN IP. **Stopping it must kill the process TREE**: `shutil.which("opencode")`
  resolves to `opencode.CMD`, so the direct child is a cmd.exe wrapper and the real
  `opencode.exe` is its grandchild — `terminate()` reaped the wrapper while the
  server kept answering 200, and the orphan would then be silently re-adopted on the
  next start. `stop()` taskkills the tree *before* reaping the wrapper (taskkill
  walks live parent links) and then verifies the port, because "stopped" has to mean
  stopped.
- **Odysseus is a sidebar LINK, and only when it answers** (2026-08-30). Address in
  Settings (`locally-odysseus-url`, default `http://localhost:7000`); the entry stays
  hidden until a probe succeeds, so it is never a control wired to nothing. The probe
  runs in the **browser**, not the server, because the browser is what follows the
  link — a server-side probe would light the entry up for a phone that cannot reach
  it. Never an iframe: HSTS, own-hostname cookies and a service worker each break an
  embed independently (docs/ODYSSEUS.md).
  **locally can also START it** (`core/odysseus.py`): `docker compose up -d` on a
  checkout found via `$ODYSSEUS_DIR` -> sibling `odysseus/` -> `~/odysseus`, the
  same order `_local_version()` already used, and only when a compose file is
  actually there. It **adopts** a stack already serving rather than racing it, and
  it stops with `docker compose stop`, never `down` -- `down` destroys containers
  and volumes, and this is the user's assistant with their data in it. Autostart
  runs on a **background thread**: a first run pulls images for minutes and chat
  must not wait for it. `docker_available` is reported alongside `docker_installed`
  because "Docker is unavailable" has two causes with two different fixes, and a UI
  that cannot tell them apart can only ever give one of them the wrong instruction.
  The **"start it with locally" toggle persists to `odysseus-autostart.json`**
  (gitignored, machine-local): the thing it controls happens at startup, so a
  preference that did not survive a restart could never take effect. An explicit
  `--odysseus-autostart` / `--no-` still wins, the precedence locally.ini.example
  already states.
- **Three more `core/`-split NameErrors, found by an AST sweep** (2026-08-30).
  `locally.py` referenced four names that had moved into `core/` and were never
  imported back: `_UTIL_SEARCH_MAX_CHUNKS` (`/v1/util/index` 500), `_UTIL_RERANK_POOL`
  (`/v1/util/search` 500), `_fetch_page_text` (every URL read), `_builtin_runs`
  (every builtin tool turn) -- plus `_markitdown_instance`, whose `global` moved to
  `core/documents/read.py` while the variable stayed behind, breaking EVERY
  text-bearing document read (pdf-with-text, docx, pptx, xlsx, html, epub, csv, txt).
  This is the trap the split rule above warns about, and it had already bitten once.
  Reading the code cannot find these -- `global` makes an absent name look defined --
  so the check is now mechanical: walk the AST, collect module-level bindings, and
  diff them against every `Name` loaded. It found all five in one pass and reports
  zero across `locally.py` and every module in `core/`. Run it before trusting a
  future split.
- Web UI: `templates/index.html` + **nine** stylesheets in `static/css/` + ES modules
  under `static/js/` (`main.js` is the entry point and does nothing but import and
  `init()`), four tabs (Chat / Voice / Code / Tools). Chat and Voice share **one**
  `chatHistory` — switching modes must never drop the conversation, and voice turns are
  mirrored into the chat thread. Util keeps its selected file/result independently and
  can hand extracted text to Chat. The nine sheets replaced 39 on 2026-09-03
  (`docs/handoff-css.md`) and are now the **source of truth**: `css_collapse.py` still
  regenerates them from the 39 originals in `c6a8c9d`, but four of them have been edited
  by hand since, so it checks `scripts/css-collapse.manifest.json` and refuses rather
  than silently deleting those rules.
- **The frontend is measured from the command line, not from a console** (2026-09-03).
  `scripts/uiserve.py` serves the page with stubbed boot endpoints, so CSS and DOM work
  does not wait 10–40 s for a model to compile onto a device, and `?strip=<id>` serves it
  with one element removed — the only way to ask "did adding this move anything else"
  and get an answer rather than an `nth-child` renumbering. `scripts/uidrive.mjs` drives
  the WebView2/Edge Chromium over CDP with **no npm dependency** (Node 21+ has a global
  `WebSocket`), sets a real layout viewport, seeds `localStorage` before the app's first
  line runs, and appends `Memory.getDOMCounters` and console errors to every result.
  This exists because a hidden browser pane reports `document.hidden === true`, where
  `requestAnimationFrame` **never fires** — so every rAF-batched measurement hangs
  instead of returning, which reads as a broken feature rather than a broken rig.
  Do not measure this UI in a hidden pane, and do not measure DOM cost with
  `performance.memory`: nodes do not live on the JS heap.
- **The sidebar's width is draggable and the thread is capped** (§2.2/§2.3,
  `docs/handoff-layout.md`). Width is one custom property (`--rail-open`), so the grid,
  the sidebar and everything measured off them move together; it applies only where the
  handle does (>780px, fine pointer), because a 348px rail carried over from a desktop
  left a 375px viewport with **27px** of app column. The thread keeps 40 message bodies
  mounted and empties the rest into a string, pinned to the height they had: **44.8%
  fewer elements** over a 100-turn session (1,380 vs 2,500; 8,621 vs 11,683 renderer
  nodes) with `scrollHeight` unchanged to the pixel. Two of §2.3's items were measured
  and **do not exist** — the listener count is flat across 80 tab switches (164 → 162),
  and every `createObjectURL` already has its `revokeObjectURL`.
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
  adds inflections that leave 12 segments 1.7px off at 512px. **Anything that rotates
  the mark must turn it about the emitted `<circle cx cy>`, not about the viewBox
  centre**: `normalise()` centres the shape's bounding box, so the curves' shared
  origin lands below centre (currently `50, 58.445` → `transform-origin: 50% 58.445%`).
  Measured, a 120° rotation about the box centre misses the shape by 14.63 viewBox
  units and about the core's centre by 0.26 — a mark that swings versus a mark that
  spins. See TODONT.md.
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
- Tool calling: **GPU/iGPU + CPU + REMOTE always; the NPU when the tool set fits a
  budget.** The gate is `_tool_capable` = `_tools_supported` (device) OR
  `_npu_tools_affordable` (this request's rendered schema block ≤ `NPU_TOOL_BUDGET`,
  1200 tokens). The old blanket NPU exclusion rested on two claims and only one
  survived measurement (`scripts/measure-tool-budget.py`, Qwen3-8B tokenizer):
  an **8-tool assistant set costs 735 tokens — 9.0% of the 8192 window**, while a
  **30-tool coding agent costs 4,553 — 55.6%**. Refusing both threw away the case
  that fits to prevent the case that doesn't. Two things were needed to make it
  actually work on the NPU, both measured 2026-08-29 on the 358H:
  **(1) suppress reasoning** — asked "Remind me to call the dentist tomorrow" with
  8 tools, Qwen3-8B spent its entire 400-token budget inside `<think>` and emitted
  **no tool call at all** (272 tokens, 23.2 s). `_suppress_think_for_tools` appends
  the `/no_think` control token on NPU tool turns: **23.2 s → 4.2 s, and a correct
  `create_task`**. Scoped to the NPU — on the GPU, reasoning before a tool call is
  affordable. **(2) give it the date** — with no clock the model called
  `get_calendar` with its training cutoff (`2023-10-11`), then told the user the
  results looked "way in the future". `render_tools_prompt` now carries today's
  date (~15 tokens, all devices). Full turn, tool call + `tool_result` → answer:
  **19.0 s + 23.2 s → 4.0 s + 4.7 s**, correct events, correct dates.
  `/api/show` still advertises `tools` only for `_tools_supported` slots — an advert
  is answered before any tool set exists, so claiming it would invite Copilot to
  pick an NPU model and then send it 30 schemas. Clients that just send `tools`
  (Odysseus does) get them honored. Over-budget sets are refused with the token
  count and a fix, not a bare "GPU-only feature". CPU is viable for
  agents on strong desktops (e.g. Core Ultra 9, many cores) where prefill can beat a weak
  iGPU. Tool specs from the request `tools` array are rendered into a system prompt
  (Qwen3-Coder native format); the model's emitted call is parsed back into OpenAI/Ollama
  `tool_calls`. `parse_tool_calls` recognizes several native formats, since a model
  often ignores our prompt and falls back to what it was trained on: Qwen3-Coder XML, Hermes
  JSON-in-`<tool_call>`, **bare `<function=>` with no wrapper (Qwen2.5-Coder native)**, Mistral
  `[TOOL_CALLS]`, Llama `<|python_tag|>`, DeepSeek `<｜tool▁calls▁begin｜>` blocks, plus a
  bare-JSON fallback. See `render_tools_prompt` / `parse_tool_calls`. Copilot Chat 0.53+ hits
  `/v1/chat/completions` (delegates to `chat_completions`); `/api/chat` also handled.
- Server-side exact calculations are a separate opt-in path: `--python-tool` enables
  `POST /v1/util/python` and the `python_tool: true` chat mode. It injects one fixed
  schema and owns at most two tool rounds, so it works on the NPU without putting
  Python in a client's `tools` array. Each run is a killed `sys.executable -I`
  child in a fresh temporary directory with capped stdout/stderr. This is
  defence-in-depth for small-model mistakes and prompt-injected pages, not a
  hostile-code sandbox on Windows; `/health` reports the caps and prompt cost.

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
- **The one-file rule is retired** (2026-08-30). It read "one file is fine, don't
  split unless it gets unwieldy" — that condition was met at 10,956 lines, and
  Camilo called it: "it gets annoying to look over it." Code now lives in
  `core/`, split by what each part is FOR, with every module small enough to
  read in one sitting (target 150-300 lines; nothing over ~300). `locally.py`
  stays the entry point and the Flask app.
  - **Import the module, not the name, for anything mutable.** `core/config.py`
    holds the runtime tunables and `main()` reassigns them after parsing the
    command line, so `from core.config import NPU_MAX_PROMPT_LEN` binds the
    default forever and silently ignores the user's flag. Use
    `from core import config` and read `config.NPU_MAX_PROMPT_LEN`.
  - **Module-level state moves with the function that owns it.** Leaving a
    lazy-import cache behind in `locally.py` while its function moved to
    `core/web/reader.py` broke every URL read with a `NameError` — and no
    static check caught it, because `global` makes an absent name look defined.
    Only an end-to-end test found it.
  - Splitting is mechanical enough to be done with an AST tool rather than by
    hand: line-slicing goes wrong the moment a constant sits between two
    functions you wanted together, which in this file it did, repeatedly.
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
- The server-side Python turn is deliberately buffered: the model must finish its
  calculation request before the answer can be returned with the code/output audit
  block. It is off by default, capped at two rounds, and its Windows child-process
  import block is not a security boundary; ctypes/native escape paths are outside
  what this local convenience feature can enforce.

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

- **The budget auto measures against is live free RAM, not the driver's
  advertised ceiling** (`_usable_gpu_bytes`, fixed 2026-08-13).
  `GPU_DEVICE_TOTAL_MEM_SIZE` is a *ceiling*, and on an integrated GPU
  (`DEVICE_TYPE == INTEGRATED`) it is a policy share of the same system RAM
  everything else is using — Intel's "Shared GPU Memory Override" raises it
  without adding a byte. Auto used to trust it and so computed
  `need 20.2 GB <= ceiling 23.6 GB -> ratio 0` on a 31.5 GB machine that had
  **0.9 GB free**: the model loaded fully resident, Windows paged it, and an
  A3B MoE touching a fresh expert set every token decoded at **0.5 tok/s** out
  of the pagefile. It fit the policy, not the machine. The budget is now
  `min(ceiling, available_RAM - 3 GB)` on shared-memory GPUs, so the same
  35B picks ratio 0 at 25 GB free (fully resident, which is the goal), 23 at
  20 GB, and 52 at 15 GB. Discrete GPUs keep the driver figure — there the
  VRAM really is ours. A low read (driver pages not yet handed back, ~2 s per
  `_settle_memory`) biases the ratio *up*, which is the safe direction: too
  much offload is merely slower, too little is the pagefile.

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
