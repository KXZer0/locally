# TODONT

Things we tried that didn't work, or that work but aren't worth doing. Each
entry explains *why not* so we don't re-litigate it in six months.

## KV-cache compression to lift the NPU's 4096-token cap (2026-08-19)

Idea (widely repeated online, and what a Google/Gemini answer recommended
verbatim): the NPU's prompt cap is a *memory* limit, so compressing the KV
cache to INT8 or INT4 shrinks each token's footprint and lets the same
allocation hold 2-3x more context -- "a 4096-token cap immediately scales to
8192" and "past 12,000 tokens" at INT4.

**Verdict: the premise is wrong on this hardware, and the measurement is
unambiguous.** `MAX_PROMPT_LEN` is a *static shape* the graph is compiled
against, not an allocation. The NPU rejects dynamic dimensions, so the cap is
whatever the vpux compiler will legalise -- and it is not short of memory in
the first place.

Measured with `scripts/npu-context-probe.py`, Qwen3-8B-int4-cw on Intel AI Boost:

| MAX_PROMPT_LEN | result |
|---|---|
| 4096 | compiles, 4,064-token needle prompt recalled correctly |
| **8192** | compiles (120 s cold / 4.8 s cached), **8,160-token prompt recalled correctly** |
| 9216 | **compile fails** |
| 10240 | **compile fails** |
| 12288 | **compile fails** |

Every failure is the same, and it is a shape legalisation error, not an OOM:

```
failed to legalize operation 'VPU.NCE.Reduce' that was explicitly marked illegal
Failed Pass EnsureNCEOpsSizeRequirements
```

**KV compression changes nothing.** 12288 fails *identically* with
`KV_CACHE_PRECISION=u8` -- same pass, same op. That is the whole disproof: if
memory were binding, halving bytes/token would move the boundary, and it does
not move it at all.

The arithmetic says the same thing. Qwen3-8B is 144 KB of KV per token, so
4096 tokens is **0.56 GB** -- not the ~4 GB the "single 4 GB allocation block"
story assumes. Even 8192 tokens is 1.1 GB. Nothing was ever close to a memory
wall, so no amount of compression could have been the lever.

What did work: **raising the number**. `NPU_MAX_PROMPT_LEN` is now 8192, a
straight 2x, verified by recall and not just by "it compiled". Only powers of
two get through, so the next stop would be 16384 and it is far outside what the
compiler accepts today.

Re-evaluate if: an Intel NPU driver / vpux compiler release notes larger static
shapes for LLM graphs. The probe script makes that a five-minute retest.

Do not re-litigate this by adding `--kv-precision u4` for the NPU. Beyond the
above, KV quantisation *costs* accuracy, so reaching for it to fix a model that
"hallucinates" is backwards: the fix for running out of context is to stop
sending more context than fits (auto-compaction), not to store the overflow
more cheaply.

## Shipping a locally-built .exe — and signing it to unblock (2026-08-17)

`scripts/locallyKey.exe` was a ~6 KB C# launcher for the hardware-key hot path,
built with the in-box `csc.exe` so there was no SDK dependency. It worked for a
week and then stopped, with no change to the binary, the caller, or Windows
updates.

**What actually happened: Smart App Control left evaluation mode and started
enforcing.** Every boot in `Microsoft-Windows-CodeIntegrity/Operational` from
2025-10-16 to 2026-08-15 loaded policy `{2678656c-05ef-481f-bc5b-ebd8c991502d}`;
both boots on 2026-08-17 loaded `{1678656c-…}` — same policy, leading nibble
`2` → `1`. The log reaches back ten months and holds exactly 14 `locallyKey`
block events (3033/3077), all of them that morning. In evaluation mode SAC logs;
enforced, it blocks. The exe was always unsigned and always would have been
rejected — SAC simply wasn't enforcing yet. It can make this transition on its
own, without asking.

**Do not try to fix this with a self-signed certificate.** SAC judges a binary
against the Microsoft trusted root program plus cloud reputation, and does not
consult local certificate stores — so installing your own root as trusted
changes nothing, and SAC has no exclusion list by design. We generated a
`CN=KX` cert and signed the exe: `Get-AuthenticodeSignature` still reported the
chain as untrusted, and even had it read `Valid`, CI would not have loaded the
file. The only things that clear SAC are an EV certificate (hardware token,
a few hundred a year) or turning SAC off — and turning it off is **one-way**,
with no path back except resetting or reinstalling Windows.

**Verdict: ship no binary. Let a Microsoft-signed host run the logic.** The exe
existed only because the PowerShell path cost ~4 s, and the bulk of that was
`Add-Type` invoking the C# compiler at runtime to reach three user32 calls.
`Reflection.Emit` declares the same P/Invokes in-process with no compiler.
`scripts/locally-key.ps1` is the replacement, run by System32's
`powershell.exe`, measured at **~720 ms** on the hot path.

Two traps found by testing it, both silent:

- **`DefinePInvokeMethod` needs `PreserveSig` set explicitly.** Without it the
  CLR treats each `BOOL` return as an HRESULT and rewrites the signature. On a
  genuinely minimised window, `Add-Type`'s `IsIconic` returned True and the
  emitted one returned **False** — so the restore step quietly never ran and the
  window stayed minimised while still being brought to the foreground.
- **`Start-Process -ArgumentList` validates each array element as non-empty**,
  so forwarding `-NpuModel ''` — a documented value meaning "fall back to the
  GPU model" — failed the entire launch. Build one quoted string instead. The
  exe was no better: `string.Join(" ", args)` dropped the empty argument
  silently rather than erroring.

Also fixed in passing: matching the window by title must loop over **every**
matching process, not take the first. A Chromium app window is reported by more
than one of the browser's processes, and "first match" picks a different handle
run to run — measured, the exe grabbed a handle whose `IsIconic` read False
while the real window sat minimised. `locally-launch.ps1` always looped; the exe
never did.

## An energy gate as voice-activity detection (2026-08-15)

The Voice tab's turn-taking was mic RMS against
`max(absolute floor, noise floor × multiple)`, with a 1-5 "mic gate" slider and an
adaptive noise floor. It was reported as "it always thinks I am talking in a
crowded space", and only push-to-talk was usable.

**Verdict: not a tuning problem. Loudness cannot answer the question, and no
setting of a loudness gate can.** A level tells you how loud the room is, not
whether anyone is speaking. Once steady background noise sits above the gate, the
gate is open permanently: turns start on their own and then never end, because the
silence counter never gets a chance to accumulate. The 60 s `maxUtteranceMs`
safety net was doing the actual endpointing.

Measured on the replacement (Silero on OpenVINO), mean speech probability over
3 s signals: digital silence 0.004, quiet white noise 0.013, **loud white noise
0.007, loud 220 Hz tone 0.006** — 0% of frames over threshold for either — versus
**0.801 for real speech, 80% of frames over**. A loud tone is the exact case the
energy gate had to open on and the model does not.

Don't reintroduce a level gate as a "fallback for when the VAD model is missing".
It would be a fallback to the broken behaviour, and it would be hard to tell from
a bug. Without `--vad-dir` the UI disables auto turn-taking and says why;
push-to-talk was always the reliable path and now it is the honest one.
The local RMS survives only to drive the orb and the meter.

## Silero VAD in the browser via ONNX Runtime Web (2026-08-15)

Considered when deciding where the new VAD should run. Browser-side inference
keeps turn decisions local, needs no Python dependency, and survives a dead
socket.

**Verdict: server-side over a WebSocket instead.**

- It means vendoring ~10 MB of ONNX Runtime WASM into `static/`, next to a
  115 KB font set that the README holds up as the reason this works on a plane.
  The inference runtime would be roughly 80× the size of everything else served.
- The latency argument for browser-side doesn't hold here: everything is on
  localhost, so the round trip is comparable to a function call. Measured
  turn-taking is indistinguishable.
- It leaves the audio in the wrong place. The socket has to exist for the
  server to hold the utterance, and holding the utterance is what lets
  transcription start **during** the end-of-turn pause — measured 0.53 s of a
  2.31 s Whisper run overlapped, with no upload afterwards. A browser-side VAD
  still has to ship the audio after the fact.
- Server-side makes the VAD an OpenVINO slot like every other model: it appears
  in `/health`, idle-unloads, and picks up the compile cache for free.

Cost of the choice: `flask-sock` is a new dependency, and if the socket dies
voice drops to push-to-talk. That is the same fallback as a missing model, and
the UI says so rather than going quiet.

**Use `silero_vad_openvino_16k.onnx`, not `silero_vad.onnx`.** The stock v5 ONNX
carries an `If` node switching on sample rate, and OpenVINO's ONNX frontend
refuses it outright: *"The input data tensor's rank has to be known (static)"* on
`Conv-16`/`ReduceMean-16`. The 16 kHz-only build from the same repo has no
branch. It also takes a **576-sample window** — 512 new samples plus 64 of
carried context — so anything that assumes Silero's window is 512 will feed it a
wrongly-shaped tensor. `VadSlot` reads the width from the model.

## BiRefNet as the background-removal upgrade (2026-08-13)

Idea: replace MODNet (18.7 MB, portrait-only, 512²) with
`onnx-community/BiRefNet_lite-ONNX` — same architecture as RMBG-2.0 but **MIT**
rather than RMBG's non-commercial licence, and general-purpose rather than
portrait-only. Its preprocessing is fully specified (1024², /255, ImageNet
mean/std), so this looked like the easy win of the batch.

**Verdict: blocked on this hardware, on both engines. MODNet stays.**

- **NPU: compile never completed.** Killed at 9.5 minutes with no output, having
  burned ~5,700 s of CPU across ~10 cores. Same shape as the SD1.5 entry below —
  and the elapsed time is itself disqualifying, because compile caching only
  helps the *second* load. A first run that costs ten minutes is not "the
  largest model the NPU can take", it is past it.
- **GPU: compiles, then cannot execute.**
  ```
  src\plugins\intel_gpu\src\graph\impls\onednn\primitive_onednn_base.h:550:
  could not execute a primitive
  ```
  That is the entire message — oneDNN gives no primitive name, no shape, no
  reason. Not enough to work around without bisecting the graph.

Not retried with the fp32 `onnx/model.onnx` (213.6 MB): it doubles the weight
of a model that already fails, and the fp16 failure is at execution rather than
at precision conversion. Worth one attempt if someone revisits this.

**Two duplicate probe processes** were found running the same compile
concurrently, from interrupted earlier attempts that had already launched.
Check for strays before concluding a compile is merely slow.

Re-evaluate if: a newer intel_gpu plugin names the failing primitive, or
onnx-community publishes a variant that is not 1024² (the resolution is the
prime suspect for both failures).

## Swin2SR on the NPU (2026-08-13)

Idea: NPU-first, so probe the new upscaler there before falling back.

**Verdict: the vpux compiler rejects it outright.**

```
compiler_impl.cpp:280: Compilation failed.
vclAllocatedExecutableCreate4 result: 0x78000004
[NPU_VCL] Compiler returned msg: Compilation failed
```

No node named, no reason given — unlike the "Missing upper bound" and
"duplicated names" messages elsewhere in this file, which at least point at
something. Swin2SR is a windowed transformer; that family has not compiled on
this NPU yet.

**This is the good outcome, not a failure of the plan**: it is exactly the
"GPU only if the NPU cannot handle it" case, and it gives the tier system its
first real per-device split — OMZ 1033 stays on NPU, Swin2SR serves GPU.

## Table structure and formula→LaTeX as document utilities (2026-08-13)

Idea: finish the document stack by adding PaddlePaddle's table-structure and
formula-recognition models next to PP-OCRv6, so `/v1/util/read` returns tables
as structure and formulas as LaTeX (which `static/js/tex.js` already renders)
rather than as scrambled text.

**Verdict:** both are blocked upstream, for two unrelated reasons. Layout
(PP-DocLayoutV3) and dewarping (UVDoc) from the same batch both passed and were
integrated — this entry is only about the two that did not.

**`SLANeXt_wired_onnx` cannot be imported by OpenVINO at all.** It fails in
`read_model`, before any device is chosen, so this is not a "GPU instead of NPU"
case:

```
While validating ONNX node '<Node(Loop): p2o.pd_op.while.0.0>':
The provided loop body graph canonical inputs size (1), does not match the sum
of loop carried dependencies and two mandatory inputs (11)
-- Conversion is failed for: Loop-17
```

The `p2o.pd_op.while` name is the tell: this is Paddle's `while` op as emitted
by paddle2onnx, and OpenVINO 2026.3's ONNX frontend rejects that Loop
construction. It also confirms the model is autoregressive — the structure
decoder emits tokens in a loop — which was the suspected NPU problem. The real
problem turned out to be worse and earlier. `SLANet`/`SLANet_plus` are the
previous generation and likely export the same way; not retested.

**`PP-FormulaNet_plus-L_onnx` is an empty repository.** It contains
`.gitattributes` and `README.md` and no weights. Every other FormulaNet variant
(`plus-L`, `plus-M`, `plus-S`, `PP-FormulaNet-S`, `PP-FormulaNet-L`,
`*_safetensors`) is Paddle or safetensors format with no ONNX export published.
Converting one is a separate piece of work with its own gate, not a probe.

Re-evaluate if: paddle2onnx changes how it lowers `while`, a newer OpenVINO
frontend accepts it, or PaddlePaddle publishes real ONNX weights for FormulaNet.
Check the repo has files before spending a download on it.

## The NPU compiler segfaults on an unbounded input dimension (2026-08-13)

Not a "don't do this" so much as a "this is what that crash was". Probing
PP-DocLayoutV3 with only two of its **three** input ports bounded took the whole
python process down with an access violation (`0xC0000005`, exit
`-1073741819`) instead of raising:

```
[IE::FrontEnd::importNetwork] Upper bounds are not specified for node 'Div.18'
(type 'Divide'): input '0' bounds are '[9223372036854775807, 2]'
```

That `9223372036854775807` is INT64_MAX — the unbounded dim of the `im_shape`
port, which the model's own `inference.yml` does not mention (it names only
`image` and `scale_factor`). Bounding all three compiled fine.

Two consequences worth keeping:

- **Reshape every port the model declares, not every port the config mentions.**
  The probe now prints all inputs for exactly this reason.
- **A segfault cannot be caught**, so it kills sibling candidates in the same
  `--all` run — which is how this first appeared as "three models failed".
  Probe a new multi-input model on its own the first time.

## Stable Diffusion 1.5 INT8 as an NPU utility (2026-08-12)

Idea: OpenVINO GenAI exposes `Text2ImagePipeline`, and the official docs list
NPU as a possible image-generation device, so add local 512×512 generation to
the Utilities sidebar using `OpenVINO/stable-diffusion-v1-5-int8-ov`.

**Verdict:** keep the UI/endpoint gated off. This pipeline did not become a
usable NPU or Intel-GPU utility on the Core Ultra X7 358H / OpenVINO 2026.3.

- Constructing directly on NPU fails in the CLIP text encoder:
  `expected exactly 1 dynamic output bound dimension, got 2`.
- Constructing without a device, then calling
  `reshape(1, 512, 512, 7.5)` before `compile("NPU")`, avoids the immediate
  dynamic-shape failure but spent **over 22 minutes** at full CPU without
  producing a compiled model or image. The exact probe processes were stopped.
- The equivalent one-step GPU probe exited during pipeline compile without an
  image. A model that can kill or silently exit a process during compile is not
  a utility feature.
- The 1.08 GB runtime snapshot downloaded for the probe was removed. The 1.2 GB
  safety-checker checkpoint was never downloaded because GenAI does not load it.

Do not advertise generation based on model-card device claims or successful
file download. Re-enable only after a replacement pipeline compiles in bounded
time and returns a visually valid image through `UtilityEngine.generate()`.

## MODNet INT8 on the NPU (2026-08-12)

Idea: quantize portrait matting to INT8 because it is an NPU utility.

**Verdict:** use MODNet FP16. INT8 compiled and inferred, but returned an almost
constant alpha mask around `0.123`; that is a correctness failure hidden behind
a successful API call. FP16 matched CPU output, produced alpha across 0…1, and
ran in ~71–91 ms. At this model size INT8 saves little and breaks the result.

## YOLO11n as the built-in detector (2026-08-12)

Idea: use the official Intel/OpenVINO YOLO11n INT8 export. It compiled on NPU,
ran in ~11.5 ms, and correctly found the bus and people in the probe image.

**Verdict:** do not ship it as the default because the model is AGPL-3.0. The
downloaded probe folder was removed. Use Apache-2.0 RF-DETR Small instead; it is
slower (~47 ms) but remains instant at UI scale and avoids imposing AGPL on the
application's utility path.

## Per-slot locks for two models on ONE NPU (2026-08-12)

Idea: the Util tab puts OCR models on the NPU alongside the chat LLM. Every
slot class already owns a `threading.Lock`, so a `UtilSlot` gets its own and
the two serve requests independently, like the GPU slots do.

**Verdict:** don't. **Concurrent inference on two NPU models kills the device.**
All NPU slots must share one lock — `_device_lock()` returns a single
module-level lock for `"NPU"` and a per-slot lock for GPU/CPU.

**What happened (Core Ultra X7 358H, OpenVINO 2026.3, genai 2026.3):** with
Qwen3-8B-int4-cw and PP-OCRv6 both resident on the NPU, hammering
`/v1/util/read` while `/v1/chat/completions` was generating produced

```
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:354:
L0 zeCommandQueueExecuteCommandLists result: ZE_RESULT_ERROR_DEVICE_LOST,
code 0x70000001 - device hung, reset, was removed, or driver update occurred
```

and **took the whole server process down**. The device itself recovered without
a reboot, but every in-flight request was lost.

- **Residency is fine; concurrency is not.** The same two slots had already
  served ~6 OCR reads and 4 chat turns interleaved *sequentially* with no
  trouble. The crash landed on the first pair of requests that actually
  overlapped (12:01:47 chat start, 12:01:48 OCR start).
- Flask runs `threaded=True`, so this was never hypothetical: any user typing
  in the Chat tab while the Util tab read an image would have hit it. It only
  stayed hidden because nothing else had ever put two models on the NPU.
- **The fix costs nothing measurable.** Re-run of the identical stress — three
  threads hammering OCR against three chat turns — gives 12 concurrent OCR
  requests, **0 errors, and chat unchanged at 7.89 s/turn vs 7.93 s solo
  (−1%, noise)**. OCR takes ~150 ms and slots into the gaps between turns, so
  serialising costs a chat request at most one OCR's latency.
- Sharing the slots' *existing* `lock` rather than adding a second one keeps a
  single locking discipline: nothing acquires two locks, so there is no order
  to get wrong, and `_idle_watchdog`'s non-blocking acquire still works.

**Do not "optimise" this back into per-slot locks on the NPU**, and treat
`NPU_RUN_INFERENCES_SEQUENTIALLY` with suspicion if it's ever proposed as the
replacement — it is a per-compiled-model property, and the failure here is
*across two separately compiled models*.

GPU and CPU keep per-slot locks. OpenVINO multiplexes them routinely and no
such failure has been seen there — but that is an absence of evidence, not a
measurement, so a second model on the GPU deserves the same stress test before
anyone assumes it is safe.

Re-evaluate if: an NPU driver release notes multi-context support (retest is
the three-thread stress above, which reproduces in under a minute).

## KaTeX/MathJax (or any markdown library) for the web UI (2026-08-11)

Idea: the UI now renders LaTeX. The obvious way to do that is KaTeX, and
while we're at it, replace the hand-rolled markdown renderer with marked.js
and stop maintaining regexes.

**Verdict:** don't. `static/js/tex.js` converts TeX to **MathML** and the
browser lays it out; `renderBody` in `app.js` stays hand-rolled.

**Why not:**
- **The CDN is not available to us and vendoring is expensive.** No CDN
  anywhere is a hard rule here — the fonts are self-hosted for the same
  reason. KaTeX vendored is ~280 KB of JS plus ~1 MB of its own woff2 faces,
  against 24 KB for tex.js and *zero* font bytes: MathML uses the system math
  font (Cambria Math on Windows), which is a local lookup, not a download.
- **The browser already does the hard part.** MathML Core ships in Chrome
  109+, Edge, Firefox and Safari 14+. Measured on the preview page: with the
  math font stack in place, ∑ grows 14px → 31px and a stretchy `(` 14px →
  27px — the layout KaTeX exists to reimplement, for free. Without that
  font-family the same MathML renders flat, which is the one trap here and
  is why the rule lives in `style.css` with the numbers attached.
- **Coverage we don't need.** KaTeX's value is the long tail of TeX. A chat
  model emits fractions, roots, scripts, sums/integrals, greek, matrices and
  `aligned` — that's what tex.js covers, and an unknown command degrades to
  its own name rather than throwing.
- **The markdown renderer is not a generic one and shouldn't be.** It escapes
  *first* and only emits tags it constructs itself, so model output cannot
  inject markup; it renders `<think>` blocks; it restricts link and image
  schemes. Swapping in marked.js means re-earning all of that through a
  sanitizer, which is a bigger dependency than the thing it replaces.

Re-evaluate if: a model here starts emitting TeX that tex.js visibly mangles
(the fallback shows raw source, so this is observable, not silent), or a
target browser drops MathML — neither is true today.

## gemma-4-31B on the 358H, by raising the iGPU memory override (2026-08-10)

Idea: Intel publishes `gemma-4-31B-it-int4-ov` (18.4 GB), and the Shared GPU
Memory Override can be pushed past the default ~half of RAM. So allocate more
and run the biggest Gemma there is.

**Verdict:** don't, and the blocker isn't the weights — it's the KV cache.
Run `gemma-4-26b-a4b-it-int4-ov` (already on disk) as the big Gemma.

**Why not (from the published config, before spending 18.4 GB of a lossy
network on it):**
- 60 layers x 16 KV heads x 256 head-dim = **960 KB per token of context**,
  10x the Qwen3-Coder-30B-A3B's 96 KB. So 8k of context costs **7.3 GB** of
  KV, 16k costs 14.6, 32k costs 29.3. On a 32 GB machine the arithmetic ends
  at 18.4 GB of weights + 7.3 GB of KV = ~26 GB for an **8k** window, leaving
  ~4 GB for Windows and the client. It "fits" and is unusable.
- Dense, confirmed (`enable_moe_block: false`, `num_experts: null`), which
  also means **`--offload-ratio` cannot help**: offload streams MoE *expert*
  weights, and there are none. All 18.4 GB stay resident.
- Decode reads every one of those bytes per token. At ~68 GB/s shared across
  CPU/GPU/NPU that is ~3 tok/s before overhead — against ~25 measured for a
  30B-A3B, which activates 3B.
- The MoE alternative is strictly better on this hardware: gemma-4-26b-a4b is
  14.3 GB, 240 KB/token, 4B active, and offload works on it.

The general rule this is an instance of: on unified memory, pick models by
*active* parameters and KV geometry, not by parameter count. A dense 31B and
an A3B MoE with more total parameters are not in the same class of usable.

Re-evaluate if: a machine with real dedicated VRAM (Arc dGPU, 32 GB+) is the
target, where both numbers stop competing with the OS for the same pool.

## Prefix caching (the CB backend) for agent workloads on the B390 (2026-08-10)

Prefix caching is default-on for GPU/CPU LLM slots because a repeated agent
prefix is prefilled once — measured 47x on a cached turn on the 285K **CPU**
(24.4s -> 0.5s). On the B390 iGPU serving Qwen3-Coder-30B-A3B it is the
opposite: it makes prefill **~10x slower** and it **hangs** on a repeated
prompt.

**Verdict:** run agent clients on this GPU with `--no-prompt-cache`. The
plain pipeline already reuses KV across turns, and does it far better here.

**Measured, same model / machine / prompt (35,839 chars from pi, ~9k tokens):**

| | prefix caching ON (CB) | `--no-prompt-cache` (plain) |
|---|---|---|
| request 1 TTFT | 33.6 s | **4.4 s** |
| request 2 TTFT | **hung** (>7 min, never returned) | **244 ms** |
| request 3 TTFT | — | **157 ms** |

- Implied prefill: **187 tok/s** through the CB backend vs **~2,000 tok/s**
  plain. That single ratio is what made local agent coding look impossible.
- End-to-end, Claude Code (`--tools` trimmed to 6, no offload):
  **75 s -> 8.7 s** first turn, 66 s -> 9.9 s second. pi: 53.6 s -> 6.4 s,
  then **1.8 s**.
- The hang reproduces whenever consecutive prompts share a near-total prefix
  (pi sends a byte-identical preamble; "ONE OK"/"TWO OK" differ only at the
  tail, and are the same length). Claude Code varies its prompts more, which
  is why it merely crawled instead of hanging — the same defect, hidden.
- Related to the known upstream "second generate() on an offload-active
  pipeline hangs", but distinct: offload was **0%** here (`fits 18.9/23.6 GB`).
  The common factor is the CB path, not offload.
- A wedged generation survives client disconnect *and* the process's own
  shutdown path — the server had to be killed by PID. OpenVINO cannot cancel
  a blocked native call, which is documented elsewhere in this file; this is
  another way to reach that state.

Re-evaluate if: a newer openvino-genai changes the CB prefill path — retest is
two identical `pi -p` runs and a look at TTFT. Keep the flag on **CPU** slots,
where the original 47x measurement stands and this defect has not been seen.

## Rewriting the client's system prompt to make agents fast (2026-08-10)

Idea: Claude Code's first turn against a local 30B took **369 s**, nearly all
of it prefill. Its request is ~142k characters, so replace that huge preamble
with a compact one at the `/v1/messages` edge (`--slim-agent-prompt`) and the
prefill collapses.

**Verdict:** built it, measured it, deleted it. The premise was wrong: the
system prompt is **5%** of the payload. The fix is a client flag, not server
code, and it is bigger than anything the rewrite could have won.

**Where the 141,959 characters of a Claude Code request actually are** (logged
at the endpoint under `--debug`, so this is the real request, not a guess):

| part | chars | share |
|---|---|---|
| system prompt | 7,332 | 5% |
| conversation | 38,291 | 27% |
| **tool schemas (30 tools)** | **99,044** | **68%** |

- Slimming the system prompt would have saved ~1.8k tokens of ~35k. It never
  even fired: 7,332 chars is below any sane threshold, which is how the
  feature announced its own pointlessness.
- `claude --tools "Bash,Edit,Read,Write,Glob,Grep"` cuts the request to
  **48,747 chars** (30 tools -> 6). That is the 66% nobody had to write code
  for. End-to-end, with the KV pool sized so offload stays at 0: **369 s ->
  75 s cold, 66 s warm** on Qwen3-Coder-30B-A3B / Arc B390.
- Generalizes past this one client: when an agent turn is slow on a small
  device, **measure the parts of the request before optimizing any of them**.
  Tool schemas are the default answer and the least suspected one.

Re-evaluate if: a client ships a genuinely huge *system* prompt with a small
tool set — then the rewrite becomes worth its complexity again. Check with
`--debug`, which now logs the three sizes on every Anthropic request.

## Trimming the working set to make unload "really" free memory (2026-08-10)

Idea: `POST /v1/models/unload` unloads the models, but process RSS was
observed not to move (1195 MB before, 1195 MB after), so the C allocator was
presumably sitting on the freed pages. Fix: call `EmptyWorkingSet` (psapi,
`SetProcessWorkingSetSizeEx(-1,-1)` is the same idea) after unloading so
Windows takes the physical pages back. Fallback if that failed: a "shut down
locally" button, since only process exit would return the RAM.

**Verdict:** neither. There was nothing to fix — **`unload()` already returns
the memory to Windows** — and `EmptyWorkingSet` adds nothing measurable on
top. What was missing was not a mechanism but a *measurement*, so the endpoint
now reports before/after numbers instead (`memory.returned_mb`).

**Why not (358H, 32 GB unified, genai 2026.3, all figures MB; ws = working
set, priv = private commit, avail = system-wide available):**

| after unload, before any trim | ws | priv | avail change |
|---|---|---|---|
| GPU, gemma-4-E4B-it int4 (VLM) | 6836 → 846 | 7282 → 1321 | **+6243** |
| NPU, Qwen3-8B int4-cw | 2361 → 92 | 1400 → 595 | **+5813** |
| CPU, gemma-4-E4B-it int4 | 5882 → 194 | 3879 → 673 | **+5673** |

- `EmptyWorkingSet` on top of those: **+9 / −2 / +6 MB**, i.e. noise. It drops
  the process's *own* working set to ~3 MB — which only means every page of
  Python and OpenVINO faults back in on the next request — while private
  commit, the thing the process still owes, does not move at all.
- Confirmed on the real serving path, three slots (Qwen3-8B@NPU +
  whisper@GPU + Kokoro@CPU), same workload either way: **6718 MB returned
  with the trim, 6684 MB without**. The only difference was the cosmetic
  working-set figure (4 MB vs 1211 MB).
- **The original "RSS didn't drop" reading was a measurement artifact.** It
  was taken from an already-idle process whose working set Windows had
  trimmed on its own — the model's pages were not in the working set to
  leave it. RSS is the wrong instrument here: measure `ullAvailPhys`
  (system, `_mem_status`) and `PrivateUsage` (process, `_process_memory`).
- Releasing is **asynchronous on the GPU**: priv fell 7282 → 4322 → 1321 MB
  over ~2 s *after* `unload()` returned. Sampling immediately under-reports
  by ~3 GB, which is why `_settle_memory` polls until availability stops
  climbing rather than snapshotting once.
- No shutdown button either: after unload the process holds ~1.2-1.8 GB, of
  which ~0.5 GB is the bare interpreter + OpenVINO + device plugins. Killing
  the server buys ~1 GB of 32 and costs a full model reload (measured today:
  NPU chat 10.2 s, TTS 8.7 s, ASR 5.9 s from `idle_unloaded` — and minutes
  for a cold GPU compile). Bad trade for the case the button exists for.

**And don't report the system-availability delta as "locally freed N GB"** —
that was the first version of the button and it lied within the hour. On a box
squeezed down to 368 MB free, unloading 4.8 GB of weights moved system
availability by **+22.8 GB**, because Windows dumped its standby list at the
same moment. The delta is a property of the machine, not of us. The button now
states the two facts separately: what was dropped (`weights_mb`, on-disk and
auditable) and what the machine reads before/after.

Re-evaluate if: a future OpenVINO or driver starts leaking on unload — the
tell is `memory.returned_mb` in the unload reply coming back far smaller than
the `weights_mb` of what was unloaded, and both numbers are now in every
response.

## Self-exporting Gemma 4 12B for the GPU slot (2026-08-09)

Idea: no Intel pre-export of `gemma-4-12B-it` exists (checked the whole
`OpenVINO/` org — E2B, E4B, 26B-A4B and 31B are published, 12B is not), so
export it ourselves and give the B390 a bigger, better Gemma than E4B.

**Verdict:** the export works and text generation is fine, but it is
**slower than the E4B it was meant to beat** and its vision path is broken.
Serve `gemma-4-E4B-it-int4-ov` instead. Only revisit if someone needs 12B
specifically for text quality and will accept half the speed.

**Measured (Arc B390 / Core Ultra X7 358H, 32 GB unified, genai 2026.3):**
- Export succeeded: transformers 5.10.0 (required — 12B is
  `Gemma4UnifiedForConditionalGeneration`, which 5.5.x cannot load) in a
  throwaway venv, `--task image-text-to-text --weight-format int4`.
  22.3 GB bf16 → **7.3 GB int4**, ~25 min, peaked ~14 GB RAM (no pagefile
  increase needed, unlike the MoE case in the entry below).
- **Speed: 11.1 tok/s decode, 4.08 s TTFT** vs E4B int4's **24.7 / 3.11**
  and Qwen3-8B-on-NPU's 18.6 / 4.70. The 12B is the slowest of the three.
- **Vision is broken**: any image request dies with `Tensor data with
  element type bf16, is not representable as pointer to f32`. Root cause is
  visible in the IR — our `openvino_vision_embeddings_model.xml` contains
  **5 bf16 tensors**; Intel's E4B vision tower has **zero** (f16/f32 only).
  `--weight-format` only governs the quantized LM; everything else inherits
  the source dtype, and `optimum-cli export openvino` has **no `--dtype`
  flag** to override it. Untested hypothesis if anyone retries: set
  `"dtype": "float16"` in the source `config.json` before exporting.

**The chat-template trap (this part generalizes — read it before exporting
any Gemma 4 yourself):** the model loads and then fails at *warmup* with
`Expected closing parenthesis in call args`. Gemma 4's template uses
adjacent string-literal concatenation (`"part one " "part two"` across
lines) inside `raise_exception(...)`; Jinja2 accepts it, OpenVINO's Minja
engine does not. Two things make it easy to lose an hour here:
- Fixing `chat_template.jinja` in the model dir does **nothing**. The
  template GenAI actually reads is baked into `openvino_tokenizer.xml`'s
  `rt_info`, in XML-escaped form.
- Intel's published exports are already patched (their E4B has 0 such
  joins, Google's 12B source has 2), so the failure only appears on
  self-exports and looks like a broken conversion.
The clean fix is to merge the literals in the **source**
`chat_template.jinja` *before* exporting, so the fixed template is baked in.
After the fact, patch the escaped copy in the tokenizer XML — in escaped
form a concatenation is `&quot;&#10;<spaces>&quot;`, and a genuinely
separate argument carries its comma *before* the newline, so merging that
exact pattern cannot join distinct arguments. Fixer lives at
`scripts/fix_gemma4_template.py` (backs up both files, `--check` to dry-run).

Re-evaluate if: Intel publishes a `gemma-4-12B-it-*-ov` (then none of this
applies), or optimum-intel gains a way to force fp16 on non-quantized
submodels.

## Whole-book (100k+) prompts on CPU serving (2026-08-09)

Idea: serve secondreader's whole-novel prompts (~113k tokens) from the 285K
CPU slot — decode is fine there (25 tok/s short-context), 64 GB RAM holds
weights + a 12 GB KV pool, and the prefix cache makes repeat artifacts cheap.

**Verdict:** don't. CPU prefill is the wall, and the client-retry dynamics
around it are actively destructive. Whole-book serving waits for XMX
(140V/B60 — protocol in `docs/LAPTOP-140V-BOOKRUN.md`).

**Why not (Qwen3-30B-A3B-Instruct-2507 int4, 285K CPU, genai 2026.1):**
- Cold prefill measured ~45 tok/s at 10.6k tokens (TTFT 234s) and
  superlinear beyond: 112,753 tokens produced NO first token in 90 minutes.
  Decode with 10k context: ~6 tok/s (not the 25 of the short-context bench).
- The client's timeout+retry then created a death spiral: identical requests
  at exactly timeout-interval (5400s ×3 observed), each entering the CB
  engine while the previous still ran — OpenVINO cannot cancel, a dead
  socket does not stop a sequence. Three ~113k sequences in a ~131k-token
  pool = permanent preemption, zero completions in 3h20m. Any client of an
  uncancellable backend must set timeout > worst-case total and attempts=1.
- Sizing rule that was missed: the KV pool must hold prompt + max_tokens
  (113k + 32k = 145k > the 12 GB pool's 131k), so even a single request can
  evict its own prefix during generation.

Flow itself is fine — the same stack completed a 2-chapter book end-to-end
(819s total, artifact + clean citation check). The failure is CPU prefill
compute at book scale, not the pipeline.

Re-evaluate if: OpenVINO's CPU plugin gains a dramatically faster prefill
path (AMX-heavy), or a future locally gains chunked-prefill progress
reporting + duplicate-request rejection, which would at least defang the
retry spiral.

## Gemma 4 on the NPU (2026-08-07)

Idea: Gemma 4 launched this week; the E-series (E2B/E4B) are edge-sized
multimodal models with Intel pre-exports — natural NPU candidates, and the
blog post promised we'd test them.

**Verdict:** no Gemma 4 on the NPU for now, on any precision. CPU (and
presumably XMX GPU) is the way to run them.

**Why not (285K NPU, driver 32.0.100.4778, genai 2026.3):**
- `gemma-4-E4B-it-int8-ov` (Intel's own export) **compiles** for NPU
  (103 s) but generates **garbage at 0.5 tok/s** — multilingual token
  salad, three identical runs. The same file on CPU: 13.0 tok/s, perfectly
  coherent. Export is sound; the NPU path is numerically broken for
  `Gemma4ForConditionalGeneration`.
- int4 variants are already documented (zenn.dev, 2026-08) to crash the
  vpux compiler with the duplicated-names bug; we did not re-prove that.
- Both failure modes differ from the LFM int8 traps (fast-garbage /
  slow-correct) — this is slow-AND-garbage, a distinct NPU-path defect.

The models themselves are good: `gemma-4-26b-a4b-it-int4-ov` (VLM MoE,
128 experts) does **21.0 tok/s steady-state on the 285K CPU**, coherent,
16 s load. E4B int8 does 13.0 on CPU. Gemma 4 belongs in the CPU/GPU
columns, not the NPU column.

**Update 2026-08-07 (VLM + offload premiere, same 140V):**
`gemma-4-26b-a4b` served through locally with `--offload-ratio 30` works —
7.9 tok/s on the cold first request, 12.5 on the second (LRU warming),
vs 26.6 resident. So VLMPipeline forwards OFFLOAD_RATIO and it engages;
the desktop 35B failures were XMX-only after all. Side-finding: VLM slots
use the PLAIN pipeline, and two sequential generates with offload active
did NOT hang — the second-generate hang is therefore LLMPipeline-specific,
not plain-pipeline-general. Vision also verified under offload: an image
request (XKCD strip) answered correctly at 8.9 tok/s — the vision encoder
is not expert weights, stays resident, works. (Entry below is Gemma-on-NPU.)

**Update 2026-08-07 (laptop NPU4, Arc 140V machine):** same E4B int8 on
the newer NPU generation produces **coherent** output — but at
**0.1 tok/s** (8 minutes per answer, three consistent runs). So two
separate defects: the numerical garbage is specific to the older NPU
arch/driver (3720 wrong, NPU4 right), while the speed is broken on BOTH
generations (0.1-0.5 tok/s smells like most of the graph falling back off
the NPU via NPUW partitioning). Verdict unchanged — no Gemma 4 on any NPU
we own — but the upstream report can now be precise: wrong-on-3720,
~100x-too-slow-everywhere. For comparison the same file does 16.4 tok/s
on the same laptop's GPU and 13.0 on the desktop CPU.

Re-evaluate if: an NPU driver or openvino release notes gemma4 fixes —
retest is `scripts/vlm-bench.py`, three minutes; or Intel ships a
`-int4-cw-ov` build of a gemma-4 (none exist today, unlike gemma-3).

## OFFLOAD_RATIO (2026.3 MoE disk offload) on the desktop 285K iGPU (2026-08-06)

Idea: OpenVINO 2026.3's MoE disk offload ("30B on 16 GB of memory") should
let big MoE models (Qwen3.6-35B-A3B, Qwen3-30B-A3B, and Dmitriy's 74 GB
Qwen3-Coder-Next from #19) run on this 33 GB-shared-memory iGPU.

**Verdict:** could not be made to work on this machine, on ANY model, at ANY
ratio, after a full day of controlled experiments. Do not recommend it to
users (incl. #19) as more than "exists upstream, unverified by us."

**What was measured (genai 2026.3.0, iGPU shared mem 33 GB, 64 GB RAM,
141 GB pagefile):**
- Qwen3.6-35B-A3B int4 VLM (2026.2 export): USM **Device** OOM (512 MB
  alloc) at ratio absent/40/90 — identical failure, ~9.5 min in. Compiling
  its language model directly with the property (no VLM wrapper) fails the
  same, so it is not a property-forwarding problem.
- Qwen3-30B-A3B-int4-ov, Intel pre-convert (2026.0 export): ratio 0 → USM
  **Host** OOM (384 MB); ratio 90 → USM Device OOM, one minute later and
  after staging ~120 GB of host commit. The offload machinery clearly
  *engages* — and still fails.
- LFM2-24B-A2B-int4-ov, Intel pre-convert (2026.2 export, **11.6 GB** —
  fits the 33 GB pool three times over): ratio 0 AND 90 → USM Host OOM
  (384 MB). An 11.6 GB model failing a 33 GB device on load is the smoking
  gun: the failure is in the GPU plugin's **weight-staging phase**, before
  any device-residency savings from offload can apply.
- Control that the pool itself works: Qwen3-8B int4 (~5 GB) and
  Qwen2.5-Coder-14B (~8 GB) load and generate fine on this iGPU. The
  practical ceiling on this box sits between ~8 and ~11.6 GB for MoE IRs.

**ROOT CAUSE (definitive, from source + device query):** the entire MoE
fusion path is gated in `transformations_pipeline.cpp`:

```cpp
// Gated on supports_immad (systolic-only) and oneDNN (required for expert GEMM dispatch).
if (device_info.supports_immad && config.get_use_onednn() && !config.get_moe_disable_fusion())
```

`supports_immad` = XMX/DPAS systolic hardware. The desktop 285K's Xe-LPG
iGPU has none (`OPTIMIZATION_CAPABILITIES` lists no `GPU_HW_MATMUL`;
verified 2026-08-06). No XMX → no TiledMoeBlock→MOECompressed fusion →
`OFFLOAD_RATIO` is a **silent no-op**, and experts stay as giant plain
constants — which is also why big-MoE loads OOM in staging on this device.
Proven end-to-end on a fusable IR: LFM2-8B-A1B exported fresh with the
2026.3 stack (tiled `u4 [32,1792,16,128]` expert constants confirmed in
the XML) loads fine and shows byte-identical device memory (14.91 GB) and
identical tok/s at ratio 0 and 90.

Intel's demos run on XMX-capable GPUs (Lunar Lake Arc 140V, Panther Lake,
Arc dGPUs). The release notes never mention the hardware gate.

**Consequence:** MoE disk offload is a hardware capability, not a software
setting, on this box. Raising the pagefile to re-export Qwen3-30B-A3B is
pointless *for offload on this machine* (the export itself would still be
useful only on an XMX-capable device). The Arc 140V laptop (original
locally dev machine) HAS XMX — that is the machine to validate offload on.

Re-evaluate if: (a) testing on an XMX GPU (Arc 140V laptop / any Arc dGPU)
— use a fresh-stack export, ratio 0 vs 90, `GPU_MEMORY_STATISTICS`;
(b) Intel lifts the immad gate for non-systolic GPUs in a future release
(watch `transformations_pipeline.cpp`); (c) recommending it to anyone —
ask for their GPU model first, `OPTIMIZATION_CAPABILITIES` containing
`GPU_HW_MATMUL` is the tell.

**Update 2026-08-06 (same evening):** condition (a) tested on the Arc 140V
laptop (Core Ultra 7 258V, XMX confirmed) — **offload works exactly as
advertised there**. LFM2-8B-A1B int4: 4.10 GB resident at ratio 0 →
0.70 GB at ratio 90 (−83%), SSD streaming visible. Qwen3-30B-A3B int4
(15.2 GB weights, Intel's 2026.0 pre-convert — so old IRs DO fuse on XMX;
the tiled layout was never the blocker): loads and generates at ratio 90
with **2.35 GB resident**, 2.5 tok/s (ratio was oversized; tuning the knee
is follow-up). The verdict above is thus purely about non-XMX hardware —
the feature itself is real, first reproduction outside Intel we know of.
locally grew `--offload-ratio` the same evening, with a startup warning on
non-XMX GPUs. install.ps1 surfaces XMX at device detection.

**Update 2026-08-07 (steady-state correction — the evening numbers above
were 2-5× too pessimistic):** the offload LRU needs ~60 tokens to warm,
and single-generate measurements reported cold-cache speed as the verdict.
Proper steady-state on the 140V, Qwen3-30B int4: ratio 30 → **25.3 tok/s**
(interactive — matches the 24-core desktop CPU running the same model
resident), 50 → 22.1, 90 → 5.1. Two benchmark bugs fixed the same morning:
warm-up contamination, and rate computed from ASSUMED token counts (a
4-token "Hello!" + EOS once reported 645 tok/s — real LFM2-8B GPU number
is 86.8). Also found: **a second generate() on an offload-active plain
pipeline hangs in native code, uninterruptible** (140V, 30B ratio 50) —
upstream-repro-worthy, and it means locally's own `--offload-ratio` serving
path (which reuses one pipeline across requests, though via the CB backend,
not the plain pipeline) MUST be verified with two sequential chat requests
before recommending the flag in production.

## int8 exports of LFM2 / LFM2.5 for the NPU (2026-08-06)

Idea: channel-wise int4 is the lossiest int4 variant and the NPU forces it,
so ship int8 builds of the LFM models for quality-sensitive use — it worked
for SmolLM3-3B (int8-cw-sym: coherent, 12.3 tok/s vs int4-cw's 23.3 on the
285K NPU, a fair trade).

**Verdict:** no publishable int8 variant exists for LFM2-family on NPU.
int4-cw is the only good configuration. The SmolLM3 result does NOT
generalize.

**Why not (both variants measured on 285K NPU, genai 2026.3):**
- `--weight-format int8 --sym --group-size -1` (mirroring the int4-cw
  recipe): compiles and runs FAST (32-33 tok/s) but generates garbage —
  LFM2-1.2B emits whitespace, LFM2.5-1.2B-Instruct emits "BY-AL-AN-AN-…"
  loops. Silent numerical breakage, not a crash: the worst failure mode.
- `--weight-format int8` (asymmetric, Intel's own recipe — their
  LFM2.5-350M-int8-ov uses it): output is coherent but decode is
  **1.4 tok/s** (119 tokens in 89 s). Intel's own 350M reference runs
  4.5 tok/s the same way — asymmetric zero-points evidently fall off the
  NPU fast path. Correct but unusable.
- SmolLM3-3B int8-cw-sym is fine (fast AND coherent), so this is
  LFM-architecture-specific (its short-conv/linear-attention blocks),
  not a general int8-on-NPU rule.

Re-evaluate if: a newer NPU driver or openvino release changes either half
(retest is two 5-minute benches with scratchpad `npu_bench.py`-style
timing), or Intel publishes a fast LFM int8 NPU build — read its rt_info
for the recipe before assuming ours was wrong.

## Qwen3.6-35B-A3B (Qwen3.5-MoE arch) on the NPU (2026-08-06)

Idea: with OpenVINO 2026.3 passing regression, put the new Qwen3.6-35B-A3B
INT4 export on the NPU — NPU coverage is the stated priority of the 2026.3
move, and an A3B MoE (3B active) looks NPU-sized on paper.

**Verdict:** doesn't load. Not a memory problem — an architecture-vs-plugin
incompatibility. Serve this model on GPU/CPU only until the NPU plugin
catches up.

**Why not:**
- Both `VLMPipeline` and `LLMPipeline` on NPU fail in ~3 s at shape
  inference, before compile, with
  `Check '!dim::is_empty(minus_one_dim)' failed ...
  reshape_shape_inference.hpp:357` on node
  `__module.model.model.language_model/aten::index/Reshape`
  ("Non-'-1' output dimensions do not evenly divide the input dimensions").
  The NPU's static-shape import can't reshape a boolean-mask `aten::index`
  in the Qwen3_5Moe language model. genai 2026.3.0.0-3277, 285K NPU
  ("AI Boost"), driver as of 2026-08-06.
- It is *not* the earlier commit failure: that was fixed (141 GB pagefile,
  33 GB iGPU shared-memory override) and this failure reproduces identically
  with memory to spare. Don't respond to this error by adding RAM/pagefile.
- Nothing locally can patch: the export is Intel-toolchain-fresh
  (OpenVINO 2026.2 export, optimum-intel 1.27.0.dev0) and the failure is in
  OpenVINO's NPU plugin shape inference, upstream of anything we configure
  (`MAX_PROMPT_LEN` etc. never comes into play).

Re-evaluate if: a later OpenVINO release notes NPU support for Qwen3.5-MoE /
`Qwen3_5MoeForConditionalGeneration` (retest is one `--scan`-verified dir +
a 3-second load attempt), or Intel publishes an NPU-targeted export of this
family.

## `--model-name` / `--model-description` override flags (2026-08-06)

Idea: let the user set the name shown in the web UI and reported as the
model ID, since renaming the model folder appeared to do nothing. Raised by
Dmitriy Teteruk (issue #19) after converting Qwen3-Coder-Next himself and
wanting it to show up as something sensible.

**Verdict:** don't add the flags. Fix the rename and add `--scan` instead.

**Why not:**
- **The bug was ours, not the interface's.** `model_display_name()` called
  `os.path.realpath()` unconditionally, so on a junction (which is what
  `install.ps1` creates) the name came from the *link target* and the user's
  rename was silently discarded. Renaming a directory is already the naming
  interface — it needed no documentation, no flag, and no knowledge. It just
  had to work. Now it does: the given name wins, and the link is only
  followed when the directory name is generic (`model/`, `gpu-model/`).
- **A flag puts the cost in the wrong place.** It has to be discovered in
  `--help`, then threaded through the generated `start.ps1`, then kept in
  sync per slot (primary, GPU, whisper). The person most likely to need it
  is the person least likely to be editing launch scripts — the exact user
  who reported it.
- **Most of what a description would say is already on disk, and more
  reliably.** The IR's model-level `<rt_info>` records the real nncf
  weight-compression mode, group size, ratio and AWQ flag; `config.json`
  gives architecture, layer count, context and MoE expert counts. `--scan`
  reports those as facts. A hand-typed description would just be an
  opportunity to be wrong — a folder named `-int4-ov` holding int8 weights
  is exactly the confusion the feature would have entrenched.

**The one thing detection genuinely cannot do:** recover the *variant*.
`config.json` in an OpenVINO export has no `_name_or_path`, and
Qwen3-Coder-Next vs Qwen3-Next-Instruct are identical in architecture and
geometry — indistinguishable from the files. That's precisely why the
directory name must stay authoritative for naming instead of being
second-guessed by a heuristic.

Re-evaluate if: someone needs two directories with the same basename served
under different IDs (two quantizations of one model in one process). That's
a real case a rename can't express — but nobody has asked for it, and the
dual-slot routing (`_route_request`) would need work first anyway.

## `--cpu-model-dir` — a third generative slot in one process (2026-08-03)

Idea: add a third `DeviceSlot` so one locally process could serve chat +
vision + coding at once (e.g. NPU chat, iGPU vision, CPU coder). Prompted by
a user question (Manuel Destouesse, email 2026-08-02) asking whether three
models could run simultaneously.

**Verdict:** works in principle, don't build it.

**Why not:**
- It adds a slot to serve the exact case we should be recommending *away*
  from locally. CPU is the one device where Ollama is unambiguously the
  better tool (see the next entry) — so the feature's whole purpose is to
  do badly what a `ollama serve` next door does well.
- `_route_request` (`locally.py:1142`) is built around exactly two
  generative slots: `for slot in (primary, secondary)` for explicit
  `model@DEVICE` selection, then a two-way heuristic (images → whichever
  slot is a VLM, text → the GPU if it holds an LLM, else primary). A third
  slot turns that heuristic into a policy question — with a coder model
  loaded, which slot gets an unlabelled text request? There's no good
  default, so it becomes config, which is the complexity we're avoiding.
- Memory-bound anyway on the target hardware. The NPU and iGPU both draw
  on system RAM, so three resident 4-bit models plus KV caches are
  competing for one pool on a 32 GB laptop. The device count was never the
  scarce resource.
- **Zero-code alternative already works:** two locally instances on
  different `--port`/`--ollama-port` values, or the recommended split
  (locally for NPU+iGPU, Ollama for the CPU model). Both are documented in
  README "When to use locally, and when to use Ollama".

Re-evaluate if: a single-device machine ever needs three models on that one
device (the two-slot cap, not the device count, would then be the real
blocker) — or if OpenVINO's CPU path decisively beats llama.cpp, which
would reverse the entry below and with it this one's first argument.

## Recommending / building out locally's CPU path (2026-08-03)

Recurring temptation: locally already runs on CPU, the `--device CPU` path
is tested and works, and the desktop 285K benchmarks are respectable
(17.8 tok/s on Qwen3-8B INT4, faster than that box's iGPU *and* NPU). So
it's tempting to present CPU as a first-class locally target and invest in
it — tool-calling on CPU is already enabled (`_tools_supported`).

**Verdict:** keep the CPU path as a working fallback, but recommend Ollama
for CPU-only users, and don't invest further in it.

**Why not:**
- Ollama's llama.cpp CPU backend is far more mature than our OpenVINO CPU
  path, `ollama pull` avoids the conversion/export problem entirely, and
  its tool calling uses per-model chat templates rather than our
  `render_tools_prompt` + `parse_tool_calls` regex approach. That parser
  already needs to recognize six native formats (Qwen3-Coder XML, Hermes,
  bare `<function=>`, Mistral, Llama, DeepSeek) precisely because models
  ignore our prompt — that's a maintenance treadmill Ollama doesn't have.
- ~~It contradicts the project's stated scope. locally exists for the Intel
  **NPU**; GPU/CPU are explicitly provisional (README "Roadmap note"),
  kept only while OpenVINO is meaningfully faster. Advertising CPU dilutes
  the one claim nothing else makes.~~ *(Update 2026-08-04: the provisional
  stance is reversed — GPU/CPU are committed long-term, since no
  OpenVINO-class Ollama Intel backend is coming and most users run agents
  (OpenClaw) on GPU/CPU. This argument no longer applies; the entry's
  verdict still stands on the ecosystem-maturity argument above.)*
- Coexistence is free: Ollama keeps 11434, and locally's port check
  (`locally.py:2193`) already detects that and disables its own Ollama
  shim rather than failing. There is no integration cost to pay.

**Caveat — this verdict is not measured.** We have benchmarked locally vs
Ollama on the Arc 140V iGPU (locally ~1.6× faster on decode, 2026-06-16)
but **never on CPU**. `bench-results/` has `cpu-qwen3-at-CPU-*.json` for
locally only. The recommendation above rests on ecosystem maturity and
scope, not on a throughput comparison.

Re-evaluate if: someone runs `benchmark.py --backend ollama` on CPU against
the same model/quantization and OpenVINO wins by a wide margin — that would
make CPU worth defending on the same measured grounds as the iGPU. Until
then, don't claim a speed verdict on CPU in docs or in replies to users.

## An accent colour for the revamped UI (2026-08-24)

The monochrome revamp was first drafted with a single warm accent — copper
`#D08658`, justified as "copper is the interconnect metal in silicon" and used
only where Claude uses its orange: the focused composer, the thinking
indicator, the primary action. It was rejected before any of it shipped.

The reason is worth keeping, because "one restrained accent" is the obvious
next suggestion anyone will make. The reference this UI was built against
(deepseek.com/harness) has **no accent at all**: its entire system is
`#0a0a0a`, pure white, and `rgba(255,255,255,.5)`, with surfaces built from
white alphas. Every affordance an accent would carry is carried by *light* —
the focused input goes to a full-strength hairline plus a 4px white-4% ring,
the primary button is a white plate with canvas-coloured text, the active tab
inverts. That is a complete system, and adding a hue to it does not make it
more legible, it makes it less neutral, which was the entire brief.

The only hue in the app is `--alarm` `#C4544A`, for the live microphone and
for genuine failure. It is not an accent and must not be reused as one: the
moment a second thing is red, red stops meaning "this broke". Two controls
were caught doing exactly that during the revamp and were neutralised — the
"remove attachment" button (detaching a file you just attached is an ordinary
action) and `.btn.stop` (stop is the inverse of send, a state change on a
control the user chose, not an error).

Verification, so this does not drift: walk every element on the rendered page,
compute `color`/`background`/`border*`/`fill`/`stroke`, and flag any value
whose max RGB channel minus its min exceeds ~6. 589 elements should return
exactly one finding, the alarm red. Anything else is a regression.

## Rebuilding the mark as a rounded triangle, from a picture of it (2026-08-25)

The `locally` mark was reconstructed by eye, from the logo pasted into a chat,
because the artwork was white-on-transparent and rendered as very nearly a
blank square. What got built was a rounded triangle: straight edges joined by
circular corner arcs, drawn as concentric hairline strokes. A great deal of
care then went into that shape — exact tangent points for the corner arcs, a
distance-field rasteriser so stroke width stayed constant, sizing chosen from
measured ink extent rather than circumradius. All of it correct, and all of it
solving a problem the mark does not have.

The real mark is a **filled ring whose two edges are three-lobed polar curves**,
plus a free-floating circle. It has no corners and no strokes. The tell was
visible in the original all along: the bottom edge curves *inward* between the
two lower lobes, which no rounded triangle does.

What settled it was measuring the file instead of looking at it. The shape
lives in the alpha channel, so a connected-component pass separates ring from
core from hole, and a least-squares fit of the radial profile returns the
curves directly:

    r(theta) = a0 + a3*sin(3*theta) + a6*cos(6*theta)

    outer edge  a0=35.06  a3=7.950  a6=-0.819   fit RMS 3.9px
    inner edge  a0=22.01  a3=1.772  a6=+0.185   fit RMS 1.0px

Reconstruction scores 97.1% IoU against the original raster; the residual is a
sub-pixel hairline at the boundary. A single 3rd harmonic manages only 94.7%
and reads visibly too round at the lobes — the 6th is what flattens the outer
sides while rounding the inner ones, and it flips sign between the two edges.

Three things worth keeping:

**Ask for the file.** One question would have skipped all of it. An image in a
conversation is a picture of an asset, not the asset, and a logo is exactly the
kind of thing where being 95% right is being wrong.

**Measure, do not eyeball.** Fitting harmonics to the radial profile took one
script and produced numbers that can be checked. Every earlier judgement about
this shape — "rounded triangle", corner radius, ring widths — was wrong in a
way that looked plausible in a contact sheet.

**Verify the thing that ships, not a proxy.** The PNGs rasterise the two
contours as separate fills; the SVG cuts its hole from *winding direction*
under fill-rule nonzero. A correct PNG therefore says nothing about whether the
SVG renders a ring or a solid blob. Signed area per contour, plus winding
numbers probed in the band, in the hole, and outside, is the check that
actually covers it. Likewise Bezier segment count: 12 Hermite segments left the
outer edge 1.7px off at 512px because the 6th harmonic adds three inflections
per lobe. 24 brings it to 0.16px.

## Reading animation state from a non-compositing browser pane (2026-08-24)

Chased a phantom bug for several rounds: the brand mark's `sweep` animation
reported `animation-name: sweep`, `animation-play-state: running`, and
`opacity: 0` forever, with a `CSSTransition` permanently "running" on the
pseudo-element. Nothing was wrong with the CSS.

When the browser pane is hidden the tab does not composite — `requestAnimationFrame`
fires once and stops, so transitions and animations never advance and
`getComputedStyle` keeps returning the *start* value of every in-flight
transition. Screenshots fail for the same reason.

To test animated state without a compositor, read the **target** rather than
the current value: inject `transition: none !important; animation: none
!important` for the selector, toggle the state, and read the computed style on
both sides. Here that returned `opacity: 0` idle and `opacity: 1` busy with the
ring mask resolving to `exclude`, which settled it in one call. The same
technique is the right way to assert any hover/state style in a headless or
hidden pane.

## Rotating the mark about the centre of its viewBox (2026-08-25)

The assistant's mark spins while a turn streams, and the spin looked wrong in a
way that was hard to name — the shape appeared to swing rather than turn, and
something seemed to trail it.

Two separate causes, one of them purely geometric:

**The mark's centre is not `50% 50%`.** `build_mark.py` normalises the sprite so
the artwork's *bounding box* fills the 100-unit viewBox. The three-lobed polar
curves' shared origin is not at the centre of that bounding box, so it lands
below it — at exactly the `<circle cx cy>` the sprite emits, currently
`50, 58.445`. `transform-origin: 50% 50%` therefore rotated the shape about a
point 8.4 units off its own axis, i.e. it orbited. Measured by rotating the path
120° and matching it back against itself: **14.63 viewBox units of mismatch about
the box centre, 0.26 about the core's centre** (0.26 is sampling noise). The
core circle is the authority — read it off the sprite, do not assume the box
centre, and if the mark is regenerated re-read it.

**"The loop has no seam, so the rotation may be eased" is true and was still the
wrong call.** The 3-fold symmetry does mean a 120° loop never visibly restarts,
which does mean the curve need not be `linear` — that reasoning is sound and it
is why the first cut eased through each third-turn and pulsed `scale` to 0.9 at
the midpoint. But it answers the wrong question. A spinner's job is to read as
steady progress, and three events (accelerate, shrink-and-grow, decelerate)
inside every 1.6s reads as a wobble. The freedom was real and not worth taking.
`linear` at 2.4s.

**Two animation periods do not need two elements.** To carry a second period the
first cut stacked a `<svg class="mark ghost">` behind the first, counter-rotating
and swelling to 1.75× at 42% opacity. Whatever that is in the abstract, on screen
at any moment mid-stream it is a half-transparent second copy of the logo sitting
behind the logo — which is exactly how it was reported. Put the periods on
different *properties* of different nodes instead: rotation on the `<svg>`,
brightness on its wrapper. One shape, two periods, no duplicate.

## Composer controls sharing a flex row with the draft (2026-08-25)

The composer had the attach/globe/mic/ring/send cluster and the textarea in one
`display: flex` row, with `align-items: flex-end` to pin the cluster to the
bottom edge as the draft grew. It was chosen so the cluster's gap to the box's
bottom edge would stay constant through a wrap — and it does — but typing past
the first line still felt unstable, because every wrapped line re-solves *every*
control's position against a container whose height is changing underneath them.
Nothing about `align-items` fixes that; the controls are simply in the wrong box.

Give the controls their own fixed-height row under the draft (the shape every
app in this class converged on). Then the only thing a wrap moves is text.
Measured 1–9 lines: the controls' gap to the box's bottom edge is 9px at every
height, and the row stays 36px.

The other half was arithmetic, not layout: `--fs-base` 15px at `line-height: 1.5`
is a **22.5px** line, so each added line rounds to a different sub-pixel and the
box grows in uneven steps. Integer metrics — 16px text, 24px line, 8px padding —
make one line exactly 40px, every further line exactly +24, and the 184px cap a
line boundary rather than mid-glyph. `app.js` derives all of it from the same
three constants.

Also removed the autosize heuristic that inferred growth from `input.value.length`
to avoid collapsing the box before measuring. It is wrong whenever an edit
changes the wrap without changing the character count — select a word, type a
longer one — and the box then holds a height that does not match its content.
One textarea's worth of forced layout per keystroke is not a cost worth a
heuristic that can disagree with the screen.

## Anchoring the streaming mark to the message's bottom edge with `position: absolute` (2026-08-25)

The assistant's mark was `position: absolute; left: 0; bottom: 1px` inside a
`position: relative` `.message.assistant`, on the reasoning that "while tokens
stream there is no meta line yet, so the mark rides the growing edge of the
answer and lands beside the timing line the moment the turn finishes." That
reasoning is half right and the other half is the bug: the timing line
(`addMeta()`) is appended as the message's **last child, after the mark**, so
finishing a turn grows the container and drags a `bottom`-anchored absolute
element down with it. The mark sat at one height while generating and jumped
to a lower one the instant the answer completed — reported as "thinking and
responded have different positions."

`position: absolute` computes its offset against the *whole padding box*,
which keeps changing as later siblings are appended — that is the general
shape of the trap, not something specific to this mark. Anything meant to sit
at a fixed spot **relative to one particular sibling** (here, directly under
`.msg-body`) should be a normal block placed right after that sibling in the
DOM, not an absolutely-positioned element anchored to the container it happens
to currently be the last thing in. In flow, appending a further sibling moves
nothing upstream of it.

Fixed by making `.msg-mark` a plain block between `.msg-body` and the (later
appended) `.meta` line. Verified by injecting a synthetic `.meta` node after
capturing the mark's rect and re-reading it: identical `top`/`left` before and
after.

## Recovering a lost NPU by reloading the model (2026-08-25)

Idea: the NPU slot came up `status: error` with `ZE_RESULT_ERROR_DEVICE_LOST`
at warmup. The slot already knows how to reload a model — `ensure_loaded`
drops the pipeline and builds a new one — so reloading should clear it.

**Verdict:** it cannot. Once a process sees `DEVICE_LOST`, every later NPU
inference **in that process** fails, including models compiled fresh
afterwards. Only restarting the process recovers the device.

Measured on the Core Ultra X7 358H, NPU driver 32.0.100.4512:

- Qwen3-8B-int4-cw failed warmup with `DEVICE_LOST`.
- A `/v1/util/read` call then reloaded the util slot from scratch —
  `[NPU] Loading utilities...` / `[NPU] Utilities ready`, so **compilation
  succeeded** — and the very next inference failed identically in 1.47 s.
  That is a different, much smaller model (PP-OCRv6), freshly compiled,
  failing the same way: the fault is the process's Level Zero context, not
  the model or its cached blob.
- Windows reported the device perfectly healthy throughout:
  `Get-PnpDevice` → `Status: OK`, `ConfigManagerErrorCode: CM_PROB_NONE`.
  Do not go looking for a device-manager fault; there isn't one.
- Killing the server and relaunching fixed it outright: warmup `done (1.6s)`,
  and OCR read the probe image at 0.966 confidence in ~1.05 s.

So: **compilation succeeding proves nothing** about whether the NPU can
execute. `explain_genai_error` only recognises `ZE_RESULT` when it appears
alongside `Compilation failed` (see the NPU compile branch), so a bare
runtime `DEVICE_LOST` currently falls through unexplained and leaves the slot
stuck at `status: error` forever, with the raw Level Zero string as the only
clue and no hint that a restart is the sole exit. Worth a branch that says so.

Not the same bug as the two-models-concurrently hang above — that one is
about *submitting* work to two NPU models at once and is fixed by the shared
`_NPU_LOCK`. This one was a single request with nothing else running.

## NPU_TURBO, and the rest of the NPU knobs (2026-08-25)

Idea: `locally.py` sets exactly three OpenVINO device properties (`CACHE_DIR`,
`KV_CACHE_PRECISION`, `OFFLOAD_RATIO`) and leaves ~10 more at plugin default.
`NPU_TURBO` reads **False** on this box — on an NPU-first project. That has to
be worth something.

**Verdict:** it is worth nothing measurable. Leave the NPU defaults alone.

`scripts/knob-sweep.py` on Qwen3-8B-int4-cw, 4 interleaved rounds, warm
steady state (rounds 2-4; round 1 is a cold NPU and a different regime):

| config | decode tok/s | vs baseline |
| --- | --- | --- |
| baseline | 14.54 | — |
| control (identical to baseline) | 14.98 | **+3.1%** |
| turbo | 14.51 | −0.2% |
| dynquant | 15.12 | +4.0% |
| qdq | 14.45 | −0.6% |
| turbo+dynquant | 13.58 | −6.6% |
| tiles-max | 14.22 | −2.2% |

`NPU_TURBO` is **−0.2% against a 3.1% noise floor**. `dynquant`'s +4.0% barely
clears that floor and is not worth a default. The only clearly resolvable
effect is `turbo+dynquant` at −6.6% — combining them is worse than either.

Two lessons about *measuring* this, which cost more than the result:

- **A do-nothing control config is mandatory.** The first sweep had no
  control and reported `cache-mode-size` — passed without `--cache-dir`, so
  its true effect is exactly zero — at **−25.7%**. That number was the
  harness's own noise, wearing a knob's name.
- **Blocked sampling cannot work on a thermally-drifting laptop.** Re-running
  one config three times in a row gave 5.63 s / 7.94 s / 7.39 s for identical
  work (~40% spread), and baseline lost 23% between round 1 and round 2 as
  the NPU heated. Hence interleaved rounds. Even so, configs run in the same
  order within each round, so late-position configs never get a cold-fast
  sample — if this is ever re-run, alternate the direction per round.
- `perf_metrics` is `None` unless you pass a `ChatHistory`; a bare prompt
  string returns a plain `str`. The first harness silently fell back to
  wall-clock and reported total generation time in a column labelled
  `ttft_ms`, mixing prefill into the decode rate.

## Removing a frameless window's edge strip with DWM attributes (2026-08-25)

Idea: `locally_app.py` runs pywebview with `frameless=True`, then restores
`WS_THICKFRAME | WS_SYSMENU | ...` because frameless strips them (leaving a
window that cannot resize, ignores `WM_CLOSE`, and has no title for the
launcher to match). That leaves a light strip around the page. The window
style is provably clean — measured `0x160F0000`, no `WS_CAPTION`, no
`WS_BORDER` — so it must be DWM compositing a frame, and DWM has attributes
for exactly this.

**Verdict:** no DWM attribute removes it, because it was never paint — it was
geometry. Three attempts, all rejected:

1. `DWMWA_NCRENDERING_POLICY = DWMNCRP_DISABLED` (+ `DWMWCP_DONOTROUND`):
   stops DWM compositing, which **exposes the raw resize border as a hard
   white stroke** and squares the corners. Strictly worse.
2. `DWMWA_WINDOW_CORNER_PREFERENCE = ROUND` + `DWMWA_BORDER_COLOR =
   COLOR_NONE`: strip still visible.
3. Adding `DWMWA_USE_IMMERSIVE_DARK_MODE` + `DWMWA_SYSTEMBACKDROP_TYPE =
   NONE`: still visible.

What settled it was measuring instead of looking: `GetWindowRect` against
`GetClientRect` + `ClientToScreen` showed the client area **inset 6/6/6/7 px**.
`WS_THICKFRAME` *is* that inset. The strip was the frame occupying real
pixels, so recolouring it was never going to help. Dropping the style takes
every edge to 0.

Cost, accepted deliberately by the owner: **the window no longer resizes by
dragging**. Maximise/restore still work (`WindowState`, not the sizing
border). If drag-resize is wanted back, **do not re-add `WS_THICKFRAME`** —
the answer is a `WM_NCCALCSIZE` window-proc subclass returning 0 for the
client-area adjustment, which keeps the style for hit-testing while giving
the frame zero pixels. That is how Electron does it, and it brings its own
traps (maximised windows covering the taskbar; `WM_NCHITTEST` needing manual
edge codes; the ctypes callback must be kept alive or Windows calls freed
memory).

Also worth knowing when debugging this class of bug: **a pale strip is
invisible against a dark desktop.** Two of the three failures above were
briefly believed fixed for that reason. Measure the inset; don't squint.

## Letting the model decide whether to search

Rejected: asking the model whether a question needs web search costs a whole
extra prefill on the NPU, with no prefix cache, just to answer a yes/no question
whose answer the user already knows. The UI makes the choice explicit instead:
off, search, or analyse the passages already retrieved.

## OmniRoute as a fallback gateway when credits run out (2026-08-29)

The want was legitimate — paid provider, then a free tier, then a local
model, without editing config each time. `diegosouzapw/OmniRoute` (48k
stars, "350 providers, 90+ free, 1.6 billion free Claude tokens") is what
turns up when you search for it, and it is not installable:

  * **CVE-2026-49352, CVSS 9.8** — a hardcoded secret allowing remote auth
    bypass, admin-panel access, and exfiltration of stored API keys.
  * Socket.dev flagged v3.8.5 with AI-detected potential malware in six
    locations and a supply-chain score of 48.
  * It **installs a custom root CA into the OS trust store** on Windows,
    macOS and Linux via privileged commands, manipulates DNS/hosts, and
    spawns a bundled MITM/TLS-interception server. Plus keychain harvesting,
    cloud-syncing of tokens, and obfuscated code.
  * Issue #2863 raised exactly this and was **closed with no maintainer
    comment** — seven direct questions, zero answers.
  * The advertised "AES-256-GCM encryption" applies only if
    `STORAGE_ENCRYPTION_KEY` is set. Default storage is plaintext.

The business model is the tell: "free Claude tokens" means pooled free-tier
access, which explains the incentive to route your traffic through
infrastructure that can decrypt it. **OpenCode already routes providers
natively** (75+, its own auth store), so the ladder needs no gateway at all.
If one is ever genuinely wanted, LiteLLM is the boring answer.

Note the academic **OmniRouter** (arXiv 2502.20576) is unrelated — a paper
on budget-constrained multi-LLM routing, not software. Same name, different
thing; do not let a search conflate them.

## Routing the NPU through Ollama or llama.cpp to avoid native OpenVINO models (2026-08-29)

Tempting, because it would make one model format serve every machine.
OpenVINO **is** now an upstream llama.cpp backend (`-DGGML_OPENVINO=ON`) and
it does list NPU among its targets, so the idea is not imaginary. It is
still a downgrade:

  * **No stateful KV cache.** Stateless execution only — which removes prefix
    caching outright, i.e. the 47× win on a repeated agent prefix.
  * `-np > 1` unsupported: no parallel sequences.
  * Context **defaults to the model's training context** (131k for Llama 3.2
    1B) and OOMs unless you pass `-c` by hand. `--context-tokens` already
    resolves this per model from real KV geometry.
  * Q4_0 primary, Q6_K auto-downconverted, embeddings dequantised to fp16.
  * Manual build: OpenVINO 2026.3.1, CMake, Ninja, VS2022 Build Tools.
  * The docs say accuracy validation and operator coverage are work in
    progress.

Meanwhile **stock Ollama cannot target an NPU at all** (`ollama#15917`, open
since May 2026) — which is the gap this project exists in, and searches for
"Ollama Intel NPU" now surface this repo as the answer. Keep the native
OpenVINO path for the NPU. The right way to serve other hardware is
`--proxy-url`, which puts locally in FRONT of Ollama instead of behind it.

## Installing liquid-gooey rather than porting the technique (2026-08-29)

`Jakubantalik/Libraries`' liquid-gooey is MIT and genuinely well-built —
the two-layer silhouette architecture is the right idea and was adopted.
The **package** was not: it is `React >= 18` as a peer dependency, and this
UI is five hand-written JS files with no npm, no bundler and no build step.
Installing it means adopting React and a toolchain to get a button effect.

The technique costs nothing to reimplement — an SVG layer under the content
with `feGaussianBlur` + `feColorMatrix`, about 40 lines. Take the idea, not
the dependency. See the orb rules in `style.css`.

## Rejecting rank-3 inputs in SpeakerSlot (2026-08-29)

The first cut of `SpeakerSlot` accepted only a rank-2 waveform input and
raised a clear error on rank-3, on the reasoning that we feed audio and a
model wanting features is misconfigured. That reasoning is backwards:
**every speaker-embedding ONNX export people can actually download takes
Kaldi fbank features** — WeSpeaker, SpeechBrain, 3D-Speaker. The "clear
error" would have fired on essentially every real model, shipping a slot
that loads nothing.

Computing fbank in-process is 40 lines of numpy and removes any reason to
depend on torch. The conventions are not free parameters — per-frame DC
removal, 0.97 pre-emphasis, a Povey window, Kaldi's log floor, then CMN —
because features that disagree with the ones a model trained on still
produce confident-looking embeddings that simply stop discriminating.

Two traps found while testing it. **CMN makes the per-bin mean identically
zero**, so a mean-pooling test stub emits the same vector for every input
and the fixture looks broken when it is the test that is wrong; pool the
second moment. And **warming up on digital silence cannot work** — zeros are
the one input guaranteed to produce a degenerate (zero-norm) embedding, so
warmup failed for every model and disabled the feature at startup.

## Fixing "it hears other people" by raising the VAD thresholds (2026-08-29)

The turn-start gate really is too low — 175 ms (`MIN_SPEECH_MS / 2`, while
`MIN_SPEECH_MS = 350` is commented "shorter than this is a cough") — and
raising it does help a little. It cannot fix the reported problem, because
the problem is not sensitivity. Silero answers *"is this speech?"*, and
another person talking **is** speech: it is returning the correct answer to
the wrong question. No threshold separates two speakers.

Tuning was tried first and abandoned. The fix is a second model that answers
the other question; see `SpeakerSlot`. Keep the threshold changes as a
mitigation, never as the solution.

## Mapping repetition_penalty onto frequency_penalty in the proxy slot (2026-08-29)

`ovg.GenerationConfig` carries `repetition_penalty`; the OpenAI request shape
has no such field, and `frequency_penalty` is right there looking similar.
They are different functions — one rescales logits for any token already
present, the other subtracts a count-weighted constant — so substituting one
for the other makes the identical request sample differently depending on
whether the slot happened to be local or proxied. That is a bug nobody can
see from the outside. `_gen_to_openai` drops it instead, and says so.

(Do keep the float32 rounding there: `GenerationConfig` stores 0.7 as
0.699999988079071, and forwarding that verbatim puts a number in the
upstream request that nobody typed.)

## An `AbortController` per view to tear down listeners (2026-09-03)

`docs/REBUILD-PLAN.md` §2.3 says "One `AbortController` per view; register
listeners through a helper that tears down on tab switch (today 127 added, 0
removed)". The figure is real and the conclusion does not follow: it is a
count of `addEventListener` calls in the source, and a listener bound once at
boot to an element that lives as long as the page has nothing to tear down.
12 of the 13 registrations on `document`/`window` are exactly that; the
thirteenth is the setup dialog's focus trap, which is the one
`removeEventListener` in the codebase and already removes itself.

The question a teardown helper answers is whether the count **grows**.
Measured with the renderer's own counter (`Memory.getDOMCounters` via
`scripts/uidrive.mjs`):

| | elements | DOM nodes | listeners |
|---|---|---|---|
| boot | 777 | 2,408 | **164** |
| after 80 tab switches | 777 | 2,373 | **162** |
| after 20 settings open/close | 777 | 2,377 | **162** |

Flat, and slightly down. **Verdict: do not build it.** It would be a
registration helper threaded through every module, a new indirection in front
of a browser API, and a thing to keep in step -- for a leak that is not there.
If a future view starts binding per-visit listeners to `document`, this table
is the test that will show it: re-run `scripts/probes/listeners.js`.

The same section's other two items were also already satisfied when it was
written: every `createObjectURL` in the app has a matching `revokeObjectURL`
(`voice/speech-queue.js`, `util/images.js`, `util/read.js`), and
`markdown/stream-painter.js` has coalesced its writes into one
`requestAnimationFrame` since it was written.

## Measuring the web UI in a hidden browser pane (2026-09-03)

`scripts/css-oracle.js` already warns that `innerWidth` is **0** in a hidden
pane, which makes every `max-width` query match. There is a second, worse
consequence: `document.hidden` is `true`, and Chromium **stops firing
`requestAnimationFrame`** for a hidden page. Every measurement that waits for
a frame -- which is most of them here, since the stream painter, the rail
drag and the message enter animation are all rAF-batched -- then hangs rather
than returning a wrong number. A 45-second timeout reads as a broken feature,
not a broken rig, and the natural next move is to "fix" working code.

**Verdict: the rig owns the browser.** `scripts/uidrive.mjs` launches the
WebView2/Edge Chromium headless over CDP and sets the viewport with
`Emulation.setDeviceMetricsOverride`; a headless page is a *visible* page as
far as the page can tell, so rAF runs, fonts load and `IntersectionObserver`
fires. Verified: `document.hidden === false`, `innerWidth` is whatever
`--width` says.

Two corollaries, both found the same day:

- **Do not measure DOM cost with `performance.memory`.** Dropping 1,120
  elements from the thread moved `usedJSHeapSize` by **0.06 MB**, because DOM
  nodes live in the renderer's C++ heap, not the JS heap. The same change is
  3,062 nodes in `Memory.getDOMCounters`. The first number invites the
  conclusion that the cap does nothing.
- **Do not benchmark a thread by appending it in a loop.** 400 messages
  appended in one task are never painted, so every box carries
  `content-visibility`'s `contain-intrinsic-size` estimate rather than a
  measured height, and scrolling through them grew `scrollHeight` by
  **17,945 px** -- with the thread cap removed entirely, so the estimate was
  the whole of it. The same 400 messages arriving one at a time, each painted
  before the next (which is what `addMessage()` does, since it auto-scrolls),
  grew it by **0 px**.
