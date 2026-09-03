# locally rebuild plan (2026-09-02)

Handoff for the implementing model. Every number below was measured in this
repo today unless marked "external". Read TODONT.md before changing anything
structural. Skills live in `C:\Projects\skills` (not installed; read their
SKILL.md directly).

## 0. Hard facts that bound the plan

| Item | Fact |
|---|---|
| Box | Core Ultra (Panther Lake, "358H"), Arc **B390** iGPU (XMX), NPU "AI Boost", OpenVINO + genai **2026.3.0** in `venv/` |
| GGUF in genai | Reader exists since 2025.2 (preview). Archs: **llama, qwen2, qwen3** dense only. Quants: Q4_0, Q4_K_M, Q8_0, FP16. **No MoE, no NPU**, tokenizer converted online is flaky. `enable_save_ov_model` writes IR next to the .gguf. (external: OpenVINO blog + strings in shipped DLL, see TODO.md) |
| Qwen3.5-9B | **Runs on GPU, not on NPU.** `OpenVINO/Qwen3.5-9B-int8-ov` (Intel, 2026-06-11, 9.5 GB) is the recommended GPU default — see §3.2f. **NPU is out**: hybrid Gated-DeltaNet cannot be legalized by the vpux compiler, and the MoE sibling fails shape inference (TODONT "Qwen3.6 on the NPU"). optimum-intel #1722 affects the **int4** export only; int8 sidesteps it. |
| GPU memory budget | **Live, not fixed.** `min(27.2 GB driver ceiling, free RAM − 3 GB)`, recomputed at every load. Ranged 13.5–25 GB on this box depending on what is open. Do not hardcode a verdict from one reading — see §3.2g. |
| NPU-validated LLMs | Qwen3-1.7B/4B/8B, SmolLM3-3B, LFM2/LFM2.5-1.2B, Phi-3.5-mini, Gemma-3-4b-it, Mistral-7B, DeepSeek-R1-distill, Granite-4.0-1B. That is Intel's *validated* list, not the limit of what compiles — see 3.3. NPU prompt ceiling 8192 (compiler, not memory). |
| NPU quantization | Repo rule says channel-wise only; **Intel's own NPU collection ships group-quantized builds** and its docs recommend g128 under 5B. Rule is unverified here and is costing quality. Test first, see 3.3 Experiment 1. |
| Frontend | 46 CSS files / 228 KB / 39 `<link>` tags; 3 stacked layers (`10-*` → `30-*` → `01-08` "revamp" overrides that win by load order); tokens defined twice (60 vars in `01-tokens` + 62 in `10-tokens`); 27 `!important`; 18 `backdrop-filter`; 59 `box-shadow`; 9 `blur()`; 13 `@keyframes`; **2 real width breakpoints** (720/900), no container queries, no `ResizeObserver`, no `content-visibility`, no cap on thread DOM; 127 `addEventListener` / 0 `removeEventListener`; `index.html` 1022 lines with every view inline. |
| Backend | `locally.py` still **5711 lines**, `main()` 540, `parse_args()` 228, `_prepare_turn` 170, `chat_completions` 163, `ollama_chat` 156 (two near-identical chat paths). 44 routes in one file. |

## 1. What "AI-ish" means here (research → applied)

External consensus (SmoothUI, 925studios, vibecodekit 2026 guides): AI-built
UIs share dark-by-default, Inter everywhere, glass/blur + glow, box-in-box
cards, six-identical-cards grids, hover bounce, gradient orbs, marketing copy,
and "everything animates". Fixes are decisions, not prompts: lock a DESIGN.md
(type, spacing scale, radius, motion tokens), one typeface with character,
copy in the owner's voice, motion only where it carries information.

What this repo actually exhibits (verified in CSS/HTML):
- **Layered overrides instead of a design**: three CSS generations coexist and
  later files cancel earlier ones. That is the single biggest "AI did this"
  signal in the codebase and the root of the RAM/paint cost.
- **Glass and blur** on 18 surfaces (CLAUDE.md already forbids full-surface
  blur; it crept back).
- **Inter as the only face**; JetBrains Mono is fine as the "machine truth" face.
- **Comment novellas in CSS** (history essays in every file header). Move
  history to TODONT/git; a stylesheet header states what the file owns, in 3 lines.
- **No layout system**: two breakpoints, no fluid type, no resizable panes.
- Monochrome palette is a strength, keep it (TODONT "An accent colour").

## 2. Frontend plan

### 2.1 CSS: collapse 46 files → ~9, one cascade
1. `DESIGN.md` (root) **is drafted** (2026-09-02, via ui-ux-pro-max, filtered
   through TODONT). It locks the monochrome ramp, type, the 4/8 scale, ONE
   radius, ONE shadow, the four motion durations (`--dur-instant/fast/base/slow`
   = 100/150/200/280 ms; §2.4a's `--dur-1/--dur-2` are `fast`/`base`), focus,
   status regions and copy rules. It also records what the generator proposed
   and was rejected (AI purple + cyan, CDN fonts, GSAP). Treat it as the source;
   every token in the new sheets must trace to it.
2. New files, loaded in one `<link>` each, wrapped in `@layer tokens, base,
   layout, components, views, utilities, overrides`:
   `tokens.css, base.css, layout.css, components.css, chat.css, voice.css,
   tools.css, motion.css, responsive.css`. `@layer` replaces load-order
   dependence, so "do not reorder" comments and all 27 `!important` go.
3. Method: for every selector that appears in >1 file (`.msg-body` 29×,
   `.orb` 28×, `.composer` 28×, `.message` 26×) compute the *final* resolved
   rule in the browser (`getComputedStyle` on the rendered page, one script)
   and write that once. Do not port by reading; port by measurement.
4. Delete `backdrop-filter` everywhere except ≤2 small overlays (menu,
   toast); replace with solid `--surface` alphas. Delete decorative
   `box-shadow`; keep one elevation token for overlays.
5. Acceptance: ≤10 stylesheets, ≤60 KB CSS, 0 `!important`, ≤2
   `backdrop-filter`, chromatic-value audit still returns exactly 1 hit
   (`--alarm`), no visual regression on the 5 screenshots taken before.

### 2.1a Component specs

`docs/COMPONENTS.md` holds worked specs for the three surfaces the user named
— **onboarding, update card, chatbox** — with the defects read out of the source
and vanilla replacement code tied to `DESIGN.md`. Highest-severity item found:
the setup dialog declares `aria-modal="true"` with **no focus trap**, and on
first run it cannot be dismissed, so a keyboard user can Tab into an app behind
a blocking dialog. Second: `#message-input` has no accessible name at all.
Implement those two before any restyling.

It also records why shadcn/ui + Tailwind were not adopted (React + Radix +
bundler against a no-build, self-hosted frontend), so the question does not get
reopened.

### 2.2 Layout and resizing
- Shell as CSS grid: `[sidebar] [main] [inspector?]` with `min-height:100dvh`.
- **Draggable sidebar width** (pointer events, persisted to `localStorage`,
  clamp 200-360px, double-click resets). Same for the Tools rail.
- **Container queries** on `.thread`, `.composer`, `.util-workspace`
  (`container-type: inline-size`) instead of viewport breakpoints; three
  sizes: <520 (phone), <900 (narrow), else.
- Fluid type: `font-size: clamp(14px, 0.9rem + 0.2vw, 16px)`; message measure
  `max-width: 72ch`.
- `ResizeObserver` only where JS must know size (composer textarea, orb canvas).
- Keyboard: focus rings visible, `Escape` closes overlays, tab order audited.

### 2.3 RAM and paint
- **Thread DOM cap**: keep ≤40 message nodes mounted; older ones collapse to a
  fixed-height placeholder holding the message id; re-hydrate on scroll
  (`IntersectionObserver`). Chat history data stays intact (Chat = complete record).
- `content-visibility: auto; contain-intrinsic-size: auto 240px` on every
  `.message` (cheap virtualization for the mounted 40).
- Stream painting already splits stable/tail (`stream-painter.js`); keep it,
  but batch DOM writes per `requestAnimationFrame`, not per token.
- Revoke `URL.createObjectURL` blobs (attachments, TTS clips) on removal.
- One `AbortController` per view; register listeners through a helper that
  tears down on tab switch (today 127 added, 0 removed).
- Image results (`.image-result` ×4) downscale to display size via
  `createImageBitmap` before insert; never keep full-res in DOM.
- `/health` poll stays 15 s idle; do not add pollers (portprobe lesson).
- Measure with `performance.memory`/DevTools heap after a 200-message
  session before and after. Target: flat heap after cap, <150 MB tab.

### 2.4 Motion (use skills: `animate`, `review-animations`, `improve-animations`, `apple-design`)
- Only `transform` and `opacity` animate. `filter: blur()` in `enter` goes
  (it forces a paint per frame on 3 elements per message).
- Vocabulary stays three moves (CLAUDE.md): enter, hover lift, sweep. Add
  **View Transitions API** for tab switches (one 180 ms cross-fade, zero JS).
- Springs only for gesture-driven things (sidebar drag release); everything
  else `--dur-2` + `--ease-out`. `prefers-reduced-motion` collapses all.
- Off-screen: pause the sweep when `document.hidden`; the orb rAF loop stops
  when the Voice tab is hidden.
- Run `review-animations` on the diff before merge; it defaults to flagging.

#### 2.4a View transitions, translated for vanilla JS

The React `<ViewTransition>` component does not apply here — there is no
React. The browser API it wraps does: `document.startViewTransition(mutate)`.
The app shell is WebView2 (Chromium), and Firefox 144+ / Safari 18.2+ cover
the LAN case; unsupported browsers just run `mutate()` with no animation.

**One helper, one gate.** Every transition goes through a single function so
the three reasons to skip are checked once:

```js
export function transition(mutate) {
    const skip = !document.startViewTransition
        || matchMedia('(prefers-reduced-motion: reduce)').matches
        || document.body.hasAttribute('data-busy');   // tokens streaming
    if (skip) { mutate(); return; }
    document.startViewTransition(mutate);
}
```

Gating reduced-motion in JS instead of CSS is deliberate: the skill's CSS
recipe needs `!important` on `::view-transition-*`, and §2.1 sets that count
to zero. The `data-busy` gate matters more: a view transition snapshots the
old and new document, and doing that over a 200-message thread while the iGPU
is running inference is exactly the compositing cost CLAUDE.md says to avoid.

**What gets one, decided by what it communicates** (the skill's rule: if you
cannot say what the motion means, do not add it):

| Surface | Kind | Treatment |
|---|---|---|
| Chat / Voice / Code / Tools tab switch | lateral | 180 ms cross-fade. **No directional slide** — tab-to-tab has no depth to communicate, and a slide falsely implies one |
| Tools rail view switch (read → upscale …) | lateral | same cross-fade |
| Settings panel, sources panel, palette | enter / exit | fade in ~160 ms, fade out ~120 ms |
| Empty state → first message | state change | welcome block fades out, thread fades in |
| Message thread during a turn | — | **never.** Streaming mutates the DOM every frame; each mutation would trigger a snapshot. The `data-busy` gate enforces this |
| Model picker menu, tooltips, HUD detail | — | too small; plain CSS `opacity` transition already there |

**CSS.** Each surface gets a `view-transition-name`, and the pseudo-elements
carry the motion using the tokens from `DESIGN.md`:

```css
#panel-chat  { view-transition-name: panel; }
/* one name shared by all four panels: only one is un-hidden at a time,
   so the old and new panel pair up as a cross-fade automatically */
::view-transition-old(panel) { animation: var(--dur-1) ease-in  fade reverse; }
::view-transition-new(panel) { animation: var(--dur-2) ease-out fade; }
```

Durations follow the skill's table: direct toggle 100–200 ms, panel change
150–250 ms. Nothing here exceeds `--dur-2`.

**Trap to write down.** A `view-transition-name` must be unique among
*rendered* elements at snapshot time. Hidden panels (`hidden` attribute,
`display:none`) do not count, which is what makes sharing one name across
the four tab panels safe — but the moment two are visible at once (a bug, or
a future split view) the transition throws and skips. Keep the name on the
panel, not on a child that might be duplicated.

This replaces the §2.4 bullet "Add View Transitions API for tab switches"
with the concrete recipe; that bullet's intent stands.

### 2.5 Template
- Split `index.html` (1022 lines) into Jinja includes per view; views that are
  `hidden` at boot (5 util views, setup, speaker) become `<template>` and are
  cloned on first open. Boot DOM shrinks and first paint is measurable.

## 3. Backend plan

### 3.1 Finish the split
- `locally.py` → `core/app.py` (Flask factory), `core/routes/{chat,models,
  audio,util,system}.py` (blueprints), `core/cli.py` (`parse_args` + `main`).
  Target: no file >400 lines. Run the AST name-sweep (CLAUDE.md) after each move.
- Merge `chat_completions` and `ollama_chat` into one `serve_turn(body,
  dialect)` with edge translators (the Anthropic path already does this; the
  Ollama one should too).

### 3.2 GGUF support (real scope: GPU/CPU, dense llama/qwen2/qwen3)
1. Experiment first (10 min): `LLMPipeline("~/models/gguf/Qwen2.5-Coder-7B-
   Instruct-Q4_K_M.gguf", "GPU", enable_save_ov_model=True)`; generate 20
   tokens. Record load time, tok/s, tokenizer sanity ("<|im_end|>" respected).
2. If it works: `_is_generative_dir` accepts `.gguf` files; `/v1/models/
   available` lists them; `load()` passes the path straight to the pipeline;
   `_choose_device` **rules out NPU for GGUF** with the reason "GGUF is
   group-quantized (Q4_0 = 32-wide blocks); NPU needs channel-wise IR". Saved IR
   goes to `.ov-cache/gguf/<name>/` so the second load skips conversion.
3. Tokenizer: if online conversion misbehaves, pair the GGUF with a
   pre-converted `openvino_tokenizer.xml` from the matching HF OpenVINO repo.
4. Refuse MoE GGUFs (`qwen3moe`, `deepseek2`, …) with a plain message and a
   pointer to the IR path. Document in README "GGUF: GPU/CPU only".
5. Prefix caching + `--context-tokens` must work unchanged (they sit on the
   pipeline, not the format). Verify with the CB backend.

### 3.2a The GPU model: Qwen3.6-35B-A3B (measured 2026-09-02)

**This is the "recent model that isn't useless" answer, and the arithmetic is
unusually good.** Config read from HF before downloading anything, the same
discipline that saved 18.4 GB on gemma-4-31B.

`jarvis-pet/Qwen3.6-35B-A3B-int4-ov` and `Morteza89/qwen3.6-35b-a3b-int4-ov`
are byte-identical in config. 18.3 GB, INT4 asymmetric g64 with INT8 backup,
ratio 1.0, VLM, 256 experts / 8 active (~3B active), 262,144-token context.
Requires openvino-genai ≥ 2026.2 — this box has 2026.3.

**KV geometry, the number that decides everything on unified memory:**

`full_attention_interval: 4` with `layer_types` alternating three
`linear_attention` layers to one `full_attention`. Only **10 of 40 layers keep
a growing KV cache**; the other 30 hold a constant-size recurrent state. So:

    10 layers × 2 kv_heads × 256 head_dim × 2 (K+V) × 2 B (f16) = 20 KB/token

| Model | KV per token | 100k context costs |
|---|---|---|
| gemma-4-31B (rejected, TODONT) | 960 KB | 96 GB |
| gemma-4-26b-a4b (**current GPU model**) | 240 KB | 24 GB |
| Qwen3-8B (current NPU model) | 144 KB | 14.4 GB |
| Qwen3-Coder-30B-A3B | 96 KB | 9.6 GB |
| **Qwen3.6-35B-A3B** | **20 KB** | **2.0 GB** |

**Twelve times lighter than the model currently in the GPU slot**, on a bigger
and much newer model. 18.3 GB of weights plus 2 GB of KV at 100k fits the
~18.5 GB budget with a modest `--offload-ratio`, and the expert fraction is
enormous (256 experts, 8 active) so offload has plenty to stream.

**The irony worth writing down:** the hybrid linear attention that makes
Qwen3.5/3.6 impossible on the NPU (§3.3 — the vpux compiler cannot legalize a
recurrent scan) is exactly what makes it *ideal* on the GPU. Constant-size
state instead of a growing cache is the whole point of Gated DeltaNet.

**Two risks, both cheap to test before committing the download:**
1. The card warns Intel GPU support is "conditional on compressed-weight
   operator support" and that the GPU path "may decompress weights to FP16".
   If int4 inflates to fp16 the 18.3 GB becomes ~35 GB and it is dead. Load it
   and read `/v1/memory` — this is the single go/no-go check.
2. Community uploads, not the official `OpenVINO/` org. Run `--scan` first;
   `_verify_weights_integrity` already checks the `.bin` against the IR's
   declared offsets, which is what a lossy network needs.

It is a reasoning model that emits think blocks and a VLM. Both are already
handled (`_ThinkFilter`, the VLM routing).

### 3.2b Models the user asked for, and why they are not the answer

**Gemma 4 12B — already tested here, and it lost.** TODONT (2026-08-09):
self-exported fine, then measured **11.1 tok/s against gemma-4-E4B's 24.7**,
i.e. slower than a much smaller model, with a broken vision path (bf16 tensors
in the vision tower that `--weight-format` does not govern). Intel publishes no
12B. More to the point, the GPU slot already holds **gemma-4-26b-a4b**, which
is larger, faster (4B active), and has working vision. Moving to 12B would be a
downgrade in every dimension.

**Qwen3.5-9B — RETRACTED. See §3.2f: `OpenVINO/Qwen3.5-9B-int8-ov` (published
2026-06-11) runs.** The claim below is wrong: it generalised the INT4 bug
(optimum-intel #1722) to the whole model. The int8 export sidesteps it. The
paragraph is kept only for its int4 and llama.cpp reasoning, which still hold.
Original text follows. Three paths, all closed:
OpenVINO export produces incoherent output (optimum-intel #1722, open — INT4
corrupts the Gated DeltaNet layers); **official llama.cpp does not support
Gated DeltaNet**, only a community fork; and openvino-genai's GGUF reader
handles llama/qwen2/qwen3 dense only. A GGUF exists
(`jc-builds/Qwen3.5-9B-Q4_K_M-GGUF`) but needs a forked runtime, which is not a
foundation for a server. **Qwen3.6-35B-A3B is newer than 3.5 anyway**, so this
is not a compromise.

**The rule these are all instances of** (TODONT, gemma-4-31B): on unified
memory, pick by *active* parameters and KV geometry, never by parameter count.

### 3.2c NPU = OpenVINO, GPU = GGUF: right instinct, wrong direction

The proposed split is half correct. **NPU = OpenVINO IR is forced** and there
is no alternative — agreed, keep it. **GPU = GGUF inverts where the leverage
is**, for four measured reasons:

1. **The GGUF reader cannot load any model in this conversation.** Its
   architecture list is llama, qwen2, qwen3 **dense**. Not Gemma 4, not
   Qwen3.5, not Qwen3.6, not any MoE. Routing the GPU through GGUF would
   restrict it to *older* models, which is the opposite of the goal.
2. **`--offload-ratio` has no GGUF equivalent.** It is an OpenVINO 2026.3
   feature and it is the only reason a 30B fits this GPU budget at all.
3. **llama.cpp's OpenVINO backend is already rejected** (TODONT, 2026-08-29):
   no stateful KV cache, so prefix caching goes — that is the 47× win on a
   repeated agent prefix. No parallel sequences. Manual build. Accuracy WIP.
4. **The plain llama.cpp backends are slower here.** README measured OpenVINO
   INT4 at ~1.6× Ollama's Vulkan decode on the Arc 140V.

**The split that gets the intent without the cost:**

| Slot | Format | Why |
|---|---|---|
| NPU | OpenVINO IR, channel-wise (g128 pending §3.3 Exp 1) | forced; no other path compiles |
| GPU **primary** | OpenVINO IR | offload, prefix cache, and the only path to current models |
| GPU **secondary** | GGUF, dense llama/qwen2/qwen3 | an on-ramp so a model someone already has runs with no conversion |

GGUF's value here is **convenience, not capability**: it removes the export
step and the 400 GB pagefile problem for models it supports. Ship it as §3.2
describes, advertise it honestly as a convenience path, and keep IR as the
GPU default.

### 3.2d Correction: `--kv-precision u8` was left out of the fit arithmetic

The flag exists (`core/config.py:52`, `locally.py:5144`) and halves KV
bytes/token. Every figure in §3.2a assumed f16. Redone against the **measured**
budget of 16.2 GB (`ullAvailPhys` 19.2 GB − 3 GB reserve, cap 27.2 GB):

| Model | Weights | KV/tok f16 → u8 | Total @8k u8 | Total @32k u8 |
|---|---|---|---|---|
| gemma-4-26b-a4b | 15.0 GB | 240 → **120 KB** | **15.96 GB ✓** | 18.9 GB ✗ |
| Qwen3-30B-A3B-2507 | 16.0 GB | 96 → 48 KB | 16.4 GB ~ | 17.5 GB ✗ |
| Qwen3.6-35B-A3B | 18.3 GB | 20 → 10 KB | 18.4 GB ✗ | 18.6 GB ✗ |
| **gpt-oss-20b** | 11.0 GB | 48 → **24 KB** | **11.2 GB ✓** | **11.8 GB ✓** |

**What this changes:** u8 KV is what keeps gemma-4-26b resident at 8k — it is
the model with the fattest cache, so it gains the most, and that is a real part
of why it "runs extremely well". **What it does not change:** Qwen3.6 is over
budget on *weights*, and no KV setting touches weights. KV precision buys
context, not headroom.

`--kv-precision u8` should therefore be the **default on GPU**, not an opt-in.
CLAUDE.md already records it costs no prefill speed (2,088 tok/s, same as f16)
and that the quality cost is far below the int4 weights already in use.

Second correction: int4 is already assumed everywhere above — these are all
`-int4-ov` exports. A GGUF `Q4_K_M` of the same model is typically **5-10%
smaller** than OpenVINO int4 g64 (finer block scales, no separate
zero-points), which is real but does not move an 18.3 GB model under a 16.2 GB
line.

### 3.2e Abliterated models change the GGUF call

Searched 2026-09-02. The abliterated/uncensored library is **GGUF-first and
community-published**: Qwen3.8-27B-Uncensored, Qwen3.6-27B-abliterated,
gemma-4-31B-heretic (~17 GB Q4_K_M), Qwen3-VL abliteration collections.
Intel's `OpenVINO/` org publishes none of these and never will, so **there is
no pre-made OpenVINO IR path for any of them**. That is a real argument for
GGUF that §3.2c did not account for, and it partly reverses that section's
conclusion.

**Three routes, in order of preference:**

1. **Self-export to OpenVINO IR** — the best option for anything dense and
   ≤14B, and the one the repo is already tooled for
   (`download-model.ps1 -Convert -Weight int4-cw`). The 400 GB pagefile horror
   was specific to Qwen3-**Next** MoE expert patching; an ordinary dense export
   is cheap (gemma-4-12B measured 25 min, ~14 GB peak RAM). Keeps prefix
   caching, offload, `--context-tokens` and the NPU option. **An abliterated
   dense Qwen3-8B/14B exported this way is the single best fit for this
   machine.**
2. **GGUF through openvino-genai** (§3.2) — zero conversion, keeps the
   OpenVINO GPU plugin, but the reader takes **llama / qwen2 / qwen3 dense
   only**. That covers abliterated Llama 3.x, Qwen2.5 and dense Qwen3. It does
   **not** cover Qwen3.6/3.8 (hybrid attention), gemma-4 (unsupported arch), or
   any MoE — which is most of the interesting large abliterated builds.
3. **llama.cpp proper** for the ones neither route reaches (Qwen3.8-27B,
   gemma-4-31B-heretic). This is outside locally and should stay outside:
   TODONT already rejected adopting llama.cpp's OpenVINO backend, and
   `--proxy-url` puts locally in *front* of another server rather than behind
   it. Point it at a llama.cpp server and it appears as a slot like any other.

**Revised §3.2c verdict:** GPU stays OpenVINO IR by default, GGUF is a
first-class secondary path (not merely "convenience"), and llama.cpp is reached
through `--proxy-url` rather than embedded. Route 1 is what to actually build
around.

### 3.2f 2026 models — corrected shortlist (measured 2026-09-02)

**Two earlier claims in this document were wrong and are retracted here.**

1. §3.2b said Qwen3.5-9B has "no working runtime". **False.** Intel published
   **`OpenVINO/Qwen3.5-9B-int8-ov` on 2026-06-11.** The optimum-intel #1722 bug
   is real but is specific to **INT4** corrupting the Gated DeltaNet layers;
   the **int8** export sidesteps it. This is the model originally asked for and
   it is the best fit on this machine.
2. §3.2a/§3.2d implied the runnable set was 2025-era. **False.** Intel's org
   carries **August 2026** exports.

All figures below are real: repo byte totals and KV computed from each
`config.json`, counting only `full_attention` layers where `layer_types` says
the rest are linear.

| Model | Repo | Size | KV/tok f16 → u8 | Published |
|---|---|---|---|---|
| **Qwen3.5-9B** | `OpenVINO/Qwen3.5-9B-int8-ov` | **9.5 GB** | 32 → **16 KB** | 2026-06-11 |
| gpt-oss-20b | `OpenVINO/gpt-oss-20b-int4-ov` | 12.6 GB | 24 → 12 KB | 2026-02-24 |
| Qwen3.8-27B | `OpenVINO/Qwen3.8-27B-int4-ov` | 16.0 GB | 64 → 32 KB | 2026-08-14 |
| Muse-Glimmer-30B | `OpenVINO/Muse-Glimmer-30B-int4-ov` | 18.2 GB | 13 → 7 KB | 2026-08-12 |
| Qwen3.6-35B-A3B | `circulus/…-int4-ov-flash` | 18.6 GB | 20 → 10 KB | 2026-08-15 |

**Fit against ONE measured 16.2 GB snapshot** (see §3.2g: the budget is live and rises when apps close) (`ullAvailPhys` − 3 GB, idle, no
Podman), with `--kv-precision u8`:

| Model | Weights | Left for KV | Context it can hold |
|---|---|---|---|
| **Qwen3.5-9B int8** | 9.5 GB | **6.7 GB** | **~420k tok — past the model's own max** |
| gpt-oss-20b | 12.6 GB | 3.6 GB | ~300k tok (max 131k) |
| Qwen3.8-27B | 16.0 GB | 0.2 GB | **~6k tok — at the line, will offload** |
| Muse-Glimmer-30B | 18.2 GB | — | over budget |
| Qwen3.6-35B-A3B | 18.6 GB | — | over budget (matches the slow run) |

**Recommendation: `OpenVINO/Qwen3.5-9B-int8-ov` as the GPU default.** June 2026,
official Intel, multimodal (`Qwen3_5ForConditionalGeneration`), fully resident
with 6.7 GB to spare — so it survives a browser, the Podman VM, *and* a long
context without ever touching offload. That headroom is the entire lesson of
§3.2d, and this is the only candidate that has it in quantity.

Second: **gpt-oss-20b** if more capability is wanted and the headroom can shrink.
**Qwen3.8-27B is the newest but lands exactly on the budget line with nothing
left for cache** — it is the Qwen3.6 mistake repeated 1.7 GB lower.

**NPU is unaffected.** All of these are hybrid linear attention
(`Qwen3_5…`, `MuseGlimmer…`), which §3.3 establishes the vpux compiler cannot
legalize. The NPU slot stays dense Qwen3.

---

### 3.2g The budget is a function, not a constant — and the reserve needs a flag

**Correction to §3.2f.** Every "fits / does not fit" verdict in this document
was computed from **one snapshot**: 19.2 GB available while a browser, the
Claude desktop app and other work were running. That is not a ceiling and
should never have been presented as one. `_usable_gpu_bytes()` re-reads live
availability at **every load**, so the same model fits or does not fit
depending on what else is open at that moment.

On a clear machine this box has far more room than §3.2f implied. Budget is
`min(27.2 GB ceiling, free − reserve)`; with the current 3 GB reserve:

| Free RAM | Budget | What fits fully resident |
|---|---|---|
| ~28 GB (clean boot) | 25.0 GB | **everything below, including Qwen3.6-35B and Muse-Glimmer** |
| ~24 GB (light use) | 21.0 GB | everything except nothing — all six |
| ~20 GB (normal) | 17.0 GB | Qwen3.5-9B, gpt-oss-20b, gemma-4-26b, Qwen3.8-27B |
| ~19 GB (**measured**) | 16.2 GB | Qwen3.5-9B, gpt-oss-20b, gemma-4-26b |
| ~16.5 GB (+ Podman VM) | 13.5 GB | Qwen3.5-9B, gpt-oss-20b |

So the user's objection is correct: **the machine can run the large models.**
What it cannot do is run them *while heavily loaded*. Qwen3.6 at 18.6 GB needs
roughly 22 GB free at load; it was tested well below that.

**The design consequence is not "pick a small model".** It is that the model
which fits *unconditionally* and the model that fits *on a clean boot* are
different choices, and both are valid:

- **`Qwen3.5-9B-int8` (9.5 GB) — the always-resident default.** Never offloads,
  survives Podman and a browser, holds more context than it can address.
- **`Qwen3.8-27B` (16.0 GB) or `Qwen3.6-35B-A3B` (18.6 GB) — the clean-boot
  models.** Genuinely usable when the machine is not loaded, which is what the
  27.2 GB Shared GPU Memory Override was raised for.

**Missing flag: `--gpu-reserve-gb N`.** `_OS_RESERVE_BYTES` is a hardcoded
`3 * 2**30` in `core/hardware/devices.py:17`, reported in `/health` as
`reserve_mb` but settable nowhere. That contradicts CLAUDE.md's own rule —
"runtime flags over hardcoded config" — and it is exactly the knob a user who
has closed their apps needs. On a 31.5 GB machine, 3 GB is ~10 % of RAM held
back permanently.

Add the flag, default unchanged at 3. Document the history in its help text so
lowering it is an informed choice, not a footgun: the reserve exists because an
18.3 GB model once "fit" a 23.6 GB *ceiling* on a machine with 0.9 GB actually
free and decoded at **0.5 tok/s** out of the pagefile. `0` restores exactly that
failure mode and should say so.

Pair it with the existing escape hatch: `--offload-ratio 0` already forces full
residency regardless of the arithmetic, for a user who knows what is free.

**What agents should build to, therefore:** treat the budget as live input, not
a fixed number. `/v1/models/available` should show each model's fit verdict
**recomputed at request time** (§3.7 already specifies this), so the UI says
"fits now" or "would offload now, close ~4 GB to fix" rather than baking a
verdict from whenever the catalog was written. That is the honest version of
"use my RAM for smart inference": the app should tell the user what is possible
*right now* and what freeing memory would buy.

### AGENT TASK A — abliterated 8B for the NPU slot

Self-contained. No repo files touched. Long-running. Local hardware required.

**Audit already done (2026-09-02):** `huihui-ai/Huihui-Qwen3-8B-abliterated-v2`
is safetensors-only, **zero** pickle weights, **zero** `.py` remote code,
Apache-2.0, 7,333 downloads/30d, unchanged since 2025-06-18. Do **not** pass
`-Trust`: it maps to `trust_remote_code=True`, there is no code in the repo to
trust, and Qwen3 is natively supported by transformers.

```bash
pwsh ./download-model.ps1 -Model huihui-ai/Huihui-Qwen3-8B-abliterated-v2 -Convert -Weight int4-cw
```

16.4 GB download (bf16). **Do not substitute the `mlabonne/*` repos** — they are
fp32, 2-6× the bytes, and identical after quantization.

Then, from the same source, produce the data-aware variant §5.0 asks for:

```bash
optimum-cli export openvino -m <local-source-dir> \
  --weight-format int4 --sym --group-size -1 --ratio 1.0 \
  --awq --scale-estimation --dataset wikitext2 Qwen3-8B-abliterated-int4-cw-awq-ov
```

Run in `venv-2026.3`. Report: load time on NPU, decode tok/s, and a 20-prompt
factual score against the current `Qwen3-8B-int4-cw`. Network drops ~75 % of HF
connections — use the documented `curl -C - --retry 30 --retry-all-errors
--retry-delay 1 --speed-limit 2000 --speed-time 20` recipe, serial.

### AGENT TASK B — Qwen3.5-9B on the GPU

```bash
pwsh ./download-model.ps1 -Model OpenVINO/Qwen3.5-9B-int8-ov
```

9.5 GB, already IR — no conversion. Load on GPU with `--kv-precision u8` and
`--offload-ratio 0`. **Go/no-go:** read `/v1/memory` and confirm resident bytes
are ~9.5 GB, not ~19 (int8 must not inflate to fp16). Then measure decode tok/s
and TTFT at 8k and at 64k. If it holds, it replaces `gpt-oss-20b` as the §3.2f
recommendation and becomes the default GPU model.

### AGENT TASK C — Podman-isolated tool sandbox

**Podman is installed and running** (v6.0.2, `podman-machine-default`).
Measured cost of `podman machine start` on this box: **1.44 GB** of host RAM,
taking free from 17.9 → 16.5 GB and the model budget from 16.2 → **13.5 GB**.
That is enough to push a 15 GB model out of residency, so the VM is not free and
must not be left running under an assistant profile.

Replace `core/sandbox/python_exec.py`'s subprocess with a container. CLAUDE.md
already concedes the current one "is not a security boundary" because ctypes and
native escapes defeat it; a container is a real one.

Requirements:
- `podman run --rm --network=none --read-only --memory=512m --pids-limit=64
  --cap-drop=ALL --security-opt no-new-privileges` with the workspace bind-mounted
  read-write and nothing else. `--network=none` is the point: a prompt-injected
  page cannot exfiltrate.
- **Start the machine on demand, stop it after.** `core/portprobe.py` already
  demonstrates the pattern for "expensive check, cached answer" — reuse it so a
  liveness probe never pays the VM cost.
- Report `podman_available` and `machine_running` separately in `/health`, for
  the same reason `docker_available`/`docker_installed` are separate: two causes,
  two different fixes.
- Fall back to the current subprocess path with a warning when Podman is absent,
  and say plainly in the UI which boundary is in force.

Stop it when idle: `podman machine stop`.

### 3.3 NPU models — what is actually possible (corrected 2026-09-02)

**The question "can I just make a model run on the NPU?" has a precise answer:
you control the export, you do not control the compiler.**

- **You control:** which model, weight format, symmetry, group size, AWQ /
  scale estimation, static shape config. Any model `optimum-intel` can load,
  you can export. Intel's published NPU list is what they *validated*, not the
  set of what *works* — it is a floor, not a ceiling.
- **You do not control:** whether the vpux compiler can legalize the resulting
  graph. The NPU compiles ahead-of-time to static shapes. A dense decoder-only
  transformer with standard GQA + RoPE + softmax attention is the shape the
  plugin is built for and generally compiles. Anything else fails at
  compile/shape-inference and **no export flag rescues it** (see TODONT:
  Qwen3.6-35B-A3B `aten::index` reshape failure; Swin2SR "Compilation failed"
  with no node named; Gemma 4 on NPU).

**Decision rule for any candidate model:**
`dense + standard softmax attention` → export it and try, likely works.
`linear attention / gated delta / Mamba-SSM / MoE routing` → will not compile,
do not spend time.

**Why 2026 models feel locked out.** The reason the new models are good is the
same reason they do not run: every 2026 flagship in this size class moved to
hybrid linear attention. Qwen3.5 is ~75% Gated DeltaNet + 25% full attention
plus sparse MoE. Granite 4 is 9:1 Mamba-2:transformer. Nemotron 3 Nano is
Mamba-2 hybrid MoE. These are recurrent scan operators, not matmul+softmax
graphs. Intel lists Qwen3.5/3.6 as CPU+GPU only for this reason, and
optimum-intel #1722 (INT4 export incoherent) is open on top of it.

**But the current model is also badly quantized, and that is fixable today.**
`--scan` on this machine reports:

| Model | Device | Quantization |
|---|---|---|
| Qwen3-8B-int4-cw | NPU | INT4 **symmetric, channel-wise, no AWQ, no scale estimation** |
| gemma-4-26b-a4b | GPU | INT4 asymmetric, **group size 64, +AWQ** |

Channel-wise means **one scale for an entire 4096-wide weight row** instead of
one per 64-128 values, with no data-aware compensation. Intel's own NPU docs
say group size 128 is *recommended for models up to 4-5B* and that channel-wise
"generally offers the best performance but **may reduce model accuracy**."
That quality cliff is a plausible large share of the observed hallucination —
it is a quantization artefact, not the base model.

**The repo's blanket rule is wrong and is steering every export to the worst
setting.** CLAUDE.md / AGENTS.md / README all state that group-quantized int4
crashes the vpux compiler ("Found N duplicated names"). There is **no TODONT
entry recording that experiment**, and Intel's own "LLMs optimized for NPU"
collection ships two group-quantized models: `OpenVINO/Phi-3.5-mini-instruct-
int4-gq-ov` and `OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-gq-ov`. The rule
is at best model-specific, at worst stale.

**Experiment 1 (highest value, ~20 min, do this first).** Download
`OpenVINO/Phi-3.5-mini-instruct-int4-gq-ov` and load it on NPU. It either
compiles — in which case the channel-wise rule is dead, group-128 becomes the
default for everything under ~5B, and quality jumps across the board — or it
reproduces "duplicated names", in which case write the TODONT entry that has
been missing and keep cw. Either outcome is worth more than any other item on
this list.

**Experiment 2: re-export the chat model properly.** Regardless of E1's
outcome, the current export is missing data-aware compression:
```bash
optimum-cli export openvino -m Qwen/Qwen3-4B-Instruct-2507   --weight-format int4 --sym --group-size 128 --ratio 1.0   --awq --scale-estimation --dataset wikitext2 Qwen3-4B-int4-g128-awq-ov
```
(swap `--group-size 128` for `-1` if E1 fails). Run in `venv-2026.3`. A
well-quantized 4B beats a badly-quantized 8B on hallucination — that is the
bet, and it is measurable with a fixed 20-prompt factual set scored by hand.

**Candidate models that can actually compile (dense, standard attention):**

| Model | Size | Note |
|---|---|---|
| **Granite-4.0-Micro** | 3B | 2026 generation. IBM shipped this **transformer-only** variant explicitly "for platforms not yet optimized for Mamba" — i.e. exactly this case. Best new candidate. Granite 4.0-1B already has NPU support in OpenVINO release notes. |
| **Phi-4-mini** | 3.8B | Dense, strong instruction following, Intel already ships Phi-3.5 NPU builds so the family compiles. |
| Qwen3-4B-Instruct-2507 | 4B | Dense GQA. The 2507 refresh is materially better than base Qwen3-4B. |
| Gemma-3-4b-it | 4B | Already in Intel's NPU collection (`-int4-cw-ov`). |
| Ministral / Mistral-7B-v0.3 | 7-8B | Dense; Intel ships NPU cw builds. |
| Qwen3-8B | 8B | Current model. Keep as the quality option **only after** re-export. |

**Strategic conclusion — stop treating the NPU as the chat device.** This
machine has a B390 with XMX doing 24.7 tok/s on a 26B. The NPU's honest role is
power efficiency and always-on work: utilities, embeddings, reranking, VAD,
OCR, and — worth investigating — as the **draft device for speculative
decoding**, which OpenVINO GenAI supports on NPU and for which Intel publishes
a FastDraft Phi-3-mini. NPU drafts, GPU verifies: that uses the NPU for what it
is good at without asking a 4B to be the answer. NPU-first should mean
NPU-for-what-fits, not NPU-for-chat-at-any-quality.

**Re-evaluate Qwen3.5 on NPU only when** a release note names
`Qwen3_5ForCausalLM` for NPU. The retest is a 3-second load attempt.

### 3.4 Small backend wins
- Per-request logs already have prefill tok/s; add `GET /v1/metrics` (JSON of
  the last 50 turns) so the HUD can show throughput without log scraping.
- `_settle_memory` and `/health` must never block on device queries (>50 ms).

### 3.5 Headless / CLI mode

`--headless` (no Flask routes for the web UI, no template render, no static
mounting) plus a `locally chat` REPL against the same slot machinery.

**Be honest about the saving.** The web UI's memory lives in the *browser*, not
the server — killing the UI saves the WebView2 process (~150-300 MB), not
server RAM. The genuine server-side wins are smaller and worth naming: the
utility slots never load (OCR alone is ~160 MB eager), Whisper/TTS/VAD stay
unloaded, and the icon sprite plus 46 stylesheets are never read from disk.
Realistic total: **300-500 MB**, nearly all of it the browser.

Design: `--headless` implies `--util-engines none` and skips the audio slots
unless explicitly asked for. The OpenAI/Ollama/Anthropic API routes stay up —
that is the whole point of a headless server. `/health` stays up.

### 3.6 OpenCode moves out of locally (user decision, 2026-09-02)

Explicitly requested: **OpenCode lives outside the locally environment.** This
supersedes §5.3's "OpenCode stays, scoped down".

Remove the embedding: `core/opencode_web.py` (spawn/adopt/port-pin/taskkill),
the Code tab and its iframe (`templates/index.html:938-974`),
`static/js/ui/code-tab.js`, the `/v1/code/launch` route, the settings
disclosure, and `/health.opencode_web`. Seven files carry the coupling
(`core/opencode_web.py`, `core/odysseus.py`, `static/js/ui/code-tab.js`,
`static/js/core/dom.js`, `static/js/updates.js`, `templates/index.html`,
`locally.py`).

What remains is the honest relationship: OpenCode is a separate program the
user runs, configured to point at locally's OpenAI-compatible base URL for
local models and at OpenRouter for datacenter models. locally keeps serving the
endpoint; it stops managing the process. This also deletes the whole class of
bugs the current code fights — the `.CMD` wrapper needing a tree-kill, adoption
of orphans, the LAN-unreachable disabled button.

### 3.7 Hugging Face token and in-app model fetching

**The token mechanism already exists** and is smaller work than it looks:
`download-model.ps1 -HfToken` and `install.ps1 -HfToken` both set `$env:HF_TOKEN`,
which `huggingface_hub` reads. What is missing is persistence and a UI.

1. **Persist it** to `hf-token.json` next to `odysseus-autostart.json`,
   **gitignored**. Same handling rules as the OpenRouter key in §5.3: never in
   `/health`, never in a log line, never echoed back to the browser. The
   settings field shows presence and last four characters, never the value, and
   writes are one-way.
2. **A model browser** — `GET /v1/models/search?q=` proxying the HF API,
   filtered to what this machine can actually run: `-int4-ov` / `-int8-ov` IR
   directories and `.gguf` files under a size cap. Show size, architecture and
   the fit verdict from `_choose_device` **before** the download starts, so a
   user learns "18.3 GB will offload on this box" without spending the
   bandwidth. That check is the single most valuable thing this UI can do and
   it is already implemented.
3. **Download with the retry recipe, not `hf download`.** Be straight in the UI
   copy: a token unlocks **gated repos and higher rate limits, it does not fix
   this network**. CLAUDE.md measured ~75% of HTTPS connections to
   huggingface.co reset mid-handshake here, client- and User-Agent-independent.
   The working invocation is the documented one:
   `curl -C - --retry 30 --retry-all-errors --retry-delay 1 --speed-limit 2000 --speed-time 20`,
   kept serial. Stream progress over the existing SSE plumbing.
4. Reuse `_verify_weights_integrity` on arrival — a resumed download on a lossy
   link is exactly the truncation case it was written for.

This is the "like Odysseus but fancier" ask, scoped to what actually helps: a
searchable catalog, an honest fit prediction, and a download that survives this
network.

### 3.8 Disk and git hygiene (measured 2026-09-02)

#### `.ov-cache` is 11 GB and ~9.7 GB of it is dead

The compile cache keys on model hash and device, so entries for deleted models
are never reused — they are dead weight, not a correctness risk. CLAUDE.md
measures one model/device pair at **659 MB** (Qwen3-8B/NPU, 65.1 s cold →
9.3 s cached). 11 GB is roughly sixteen such pairs; two generative models plus
the utility slots are resident now, so the great majority is for models no
longer on disk (Qwen3.6, Qwen3-Coder-30B and whatever else was trialled).

**Safe to delete outright.** It regenerates on next load. The only cost is one
slow compile per model afterwards — about 65 s for the NPU model, less on GPU.

```bash
rm -rf .ov-cache          # recovers ~11 GB; next load of each model is slow once
```

Do it when convenient, not mid-session. `--no-model-cache` disables the cache
entirely if it ever needs to stay small.

Also trivial and safe: `find . -name __pycache__ -type d -not -path './venv/*' -exec rm -rf {} +`
(12 directories). The Hugging Face cache is **547 KB** — nothing to reclaim.

#### `~/models` has nothing stale — do not delete anything there

Every directory is live:

| Path | Size | Status |
|---|---|---|
| `gemma-4-26b-a4b-it-int4-ov` | 15 GB | **live** — `gpu-model/` junction |
| `Qwen3-8B-int4-cw-ov` | 4.5 GB | **live** — `model/` junction |
| `gguf/` | 4.4 GB | needed for the §3.2 GGUF experiment |
| `util/`, `whisper-small`, `Kokoro-82M`, `silero-vad`, `speaker` | 1.7 GB | live slots |

Deleting the first two breaks the app: `model/` and `gpu-model/` are junctions
into this directory, and `resolve_display_name` follows them.

#### Committing to `main` is safe — the override worry does not apply

Investigated because of a concern that committing could override old work. It
cannot, and the reason is structural:

- `main` is an **orphan squash** — `git merge-base main <any branch>` returns
  nothing. It shares no ancestry with any development branch.
- `main` is also the **newest and most complete**: 2026-08-30, **212 files**,
  against the branches' 2026-08-25..29 and **93 files**.
- The "160-195 commits ahead" figure is an artifact of that orphaning. With no
  merge base, `rev-list main..branch` counts *every* commit on the branch, not
  unique work.
- Local `main` is 1 commit **behind** `origin/main`. Pull before committing.

So a commit on `main` cannot conflict with, override, or lose anything on those
branches. They are pre-squash history, superseded by the squash itself.

#### The six local branches: delete, but bundle first

`chore/pre-release-hygiene`, `delegate/30697716`, `delegate/4f0cb150`,
`delegate/6c80d6f9`, `ui/chat`, `worktree-agent-ae54001fb03f6c0c4`.

**Only `main` exists on `origin`.** These six are local-only, so deleting them
is irreversible — there is no remote copy to restore from. They are superseded
by the squash and hold no files `main` lacks, but they do hold the granular
commit history the squash discarded.

**Recommendation: archive to a bundle, then delete.** One file, seconds to
produce, and it makes the deletion reversible:

```bash
git bundle create ../locally-prehistory.bundle --branches --not main
git branch -D chore/pre-release-hygiene delegate/30697716 delegate/4f0cb150 \
               delegate/6c80d6f9 ui/chat worktree-agent-ae54001fb03f6c0c4
git worktree prune
```

Keep the bundle off the repo (`../`) so it is never committed. Restore later
with `git clone`/`git fetch` against the bundle file if the history is ever
wanted. If the pre-squash history is genuinely unwanted, skip the bundle — but
say so deliberately rather than by omission.

**Do not delete them before committing the current working tree**, since the
31 uncommitted files are new work present in neither `main` nor any branch.

## 4. Order of work and gates

0. **Pull `origin/main` (local is 1 behind), commit the working tree, then §3.8 cleanup.** Committing to `main` is safe — it is an orphan squash sharing no ancestry with the old branches.
1. **Screenshots + metrics baseline** (5 views × 3 widths, heap after 200
   msgs, CSS bytes, Lighthouse perf). Nothing merges without a before/after.
2. Backend split (3.1). Mechanical, testable, unblocks everything.
3. CSS collapse (2.1) + DESIGN.md. One PR, measured against the screenshots.
4. Layout/resizing (2.2) and RAM (2.3).
5. Motion pass (2.4) with `review-animations`.
6. Template split (2.5).
7. GGUF (3.2), then Qwen3-4B-cw export and registry entry (3.3).

Gates per PR: chromatic audit = 1 hit; `!important` = 0; heap flat over 200
messages; `prefers-reduced-motion` honoured; AST sweep = 0 unbound names.

## 5. Phase 2 — features (after the fixes land)

### 5.0 Correction on the chat model

**Qwen3-8B is still the right pick and the user is right to push back.** An 8B
beats a 3-4B on benchmarks broadly. Granite-4.0-Micro was flagged in 3.3 for
being *newer*, not better, and that distinction was not made clearly enough.
Do not swap the model.

The fix is the export, and it is nearly free. For an 8B, Intel recommends
channel-wise for *performance*, so keep `--group-size -1`. What is missing is
data-aware compression, which is **export-time only and costs nothing at
inference**:

```
optimum-cli export openvino -m Qwen/Qwen3-8B \
  --weight-format int4 --sym --group-size -1 --ratio 1.0 \
  --awq --scale-estimation --dataset wikitext2 Qwen3-8B-int4-cw-awq-ov
```

Same model, same speed, better-chosen scales. Run in `venv-2026.3`. Score it
against the current export on a fixed 20-prompt factual set before adopting.
Experiment 1 in 3.3 (group-quantized on NPU) still stands independently.

### 5.1 Quick assistant — a Copilot-style overlay

**Yes, this is possible, and most of the plumbing already exists in this repo.**
`locally_app.py` is a pywebview/WebView2 shell that already does ctypes/user32
window enumeration, DWM attribute work and AppUserModelID identity;
`locally-win32.ps1` already raises an existing window by title. A global hotkey
and a second hidden window are small additions to code that already speaks
Win32.

**The hotkey already exists, and that changes the build.** Win+C on this
machine already raises locally. Verified 2026-09-02: **no Copilot appx package
is installed**, so Win+C is not reserved here — the earlier claim that Windows
owns it was wrong for this box (it holds only on a stock install with Copilot
present). The binding is **NewPilot** (`GamerJagdish.NewPilot`, MSIX) — *not*
PowerToys, whose Keyboard Manager holds no remaps on this machine. NewPilot is
itself MSIX-packaged, which is precisely why Windows accepts it as a Copilot-key
target when a `.lnk` is refused; it forwards the press to locally's shortcut.
Re-point the key there, not in PowerToys.

What the key does today is run `scripts/locally-key.ps1`, which **raises the
main app window** — roughly 50 ms on the warm path, already carefully
optimised. That is a launcher, not a quick assistant: it captures nothing, it
opens the full app, and focus lands on locally before anything is read.

**The script is the right capture point, and this is the key insight.**
`locally-key.ps1` runs *while the source application still has focus*. That is
the only moment the selection is reachable. So the capture belongs there, ahead
of the raise:

1. `(New-Object -ComObject WScript.Shell).SendKeys('^c')`
2. `Get-Clipboard` (and `Get-Clipboard -Format Image` for pictures)
3. POST the captured context to locally
4. *Then* show the window

Both calls are plain cmdlets — **no `Add-Type`**, which matters because that
script's own notes measure `Add-Type` at ~336 ms and the whole budget is a few
hundred. Save and restore the prior clipboard around step 1.

The alternative is `RegisterHotKey` in-process, which avoids the focus change
entirely and is cleaner in principle. It is the *worse* fit here: the
script-based path exists specifically to dodge Smart App Control (TODONT,
2026-08-17), it is already fast, and it already solves the launch-or-raise
cases. Extend it rather than replacing it.

**So the remaining work is not "get a hotkey".** It is: (a) capture before the
raise, (b) point the key at a small overlay instead of the full app, and
(c) make that overlay appear from an already-created hidden window.

**Interaction:** press the hotkey anywhere in Windows. Whatever is selected (or
on the clipboard) is captured, a small frameless window appears near the cursor
already holding that context, and the answer streams in. `Esc` dismisses.
`Enter` follows up. One button promotes the exchange into the full Chat tab
with history intact.

**Mechanism, piece by piece:**

| Piece | How | Trap |
|---|---|---|
| Global hotkey | **Solved** — the key already runs `scripts/locally-key.ps1`. Only needed if the binding is ever moved in-process, where it is `RegisterHotKey` via ctypes on a dedicated thread with its own `GetMessage` loop | If moved in-process it must be its own thread; the WebView2 loop will not pump it |
| Which key | **Already bound to Win+C here via NewPilot** (Copilot is not installed, so the chord is free). Ship `Ctrl+Alt+Space` as the portable default for machines where Copilot still owns Win+C | Do not assume Win+C is free on a stock install, and do not send this user to PowerToys — NewPilot owns it. Windows' "Customize Copilot key" accepts only MSIX-signed apps, which is why an MSIX remapper is needed to target a `.lnk` at all |
| Selection capture | Synthesize `Ctrl+C` to the foreground window via `SendInput`, then read the clipboard | **Save and restore the previous clipboard.** Clobbering it is the fastest way to make this feel hostile. If no new content arrives within ~150 ms, fall back to what was already there |
| Image capture | `PIL.ImageGrab.grabclipboard()` handles both `CF_DIB` and file lists; Pillow is already a dependency | `Win+Shift+S` already puts a region on the clipboard, so screenshot-to-answer works with no capture UI of our own |
| Instant window | Pre-create the window **hidden at startup**, then `ShowWindow` / `SetForegroundWindow` on hotkey | Never create on hotkey. Creating a WebView2 window costs hundreds of ms and the whole feature is judged on that number |
| Its own page | New route `GET /quick` serving a **separate minimal bundle** | Must not load the main app's CSS. Even after §2.1 that is ~60 KB for a window showing one answer |
| Ordering | Capture the selection **first**, then show the window | Showing first moves focus, and the selection you wanted belongs to the app that just lost it |

**Latency budget:** hotkey to visible window under 100 ms. Achievable only if
the window is pre-created and the request fires *after* paint, not before. Show
the captured context immediately and stream the answer into it.

**The model must be resident.** This is unusable against a 20-40 s model load,
so it needs `--idle-timeout 0` plus prewarm (already the agent preset). If a
user runs with idle unload, the quick window must say "waking the model" rather
than appearing to hang.

**Images need a VLM.** If a text-only model is resident, an image query forces a
swap. Do not hide that: either keep a small VLM resident alongside, or show the
swap and its cost explicitly. Silently taking 30 s is worse than saying so.

**Build order:** the hotkey is done. Start with the hidden overlay window plus
clipboard text, end to end, measured to first paint. Add selection synthesis in
`locally-key.ps1` second. Images third. Each step is independently useful and
independently shippable.

### 5.2 Odysseus KEEPS its integration — do not delete it

**This section previously said "Remove Odysseus". That was reversed on
2026-09-02, later the same day. Agents must not act on the old text.**

The position now, in the user's own terms: Odysseus "seems good", is "better
than my locally" for everything except the NPU and the native web app, and its
features are wanted. `docs/ODYSSEUS.md` already states the standing
architecture and it is the one to build to — **Odysseus owns the assistant,
locally owns the silicon.** They meet at locally's OpenAI-compatible base URL.

Keep, unchanged: `core/odysseus.py`, `static/js/ui/odysseus.js`,
`docs/ODYSSEUS.md`, `scripts/odysseus-firewall.ps1`, `odysseus-autostart.json`,
the settings entries, `/health.odysseus`, its portprobe entry, and the
`--odysseus-autostart` flags. The sidebar link stays a link and never an
iframe — HSTS, own-hostname cookies and a service worker each break an embed
independently.

**The licensing constraint governs anything that looks like porting.**
Odysseus is **AGPL-3.0**; locally currently ships **no LICENSE file at all**.
Copying Odysseus source into locally would make locally AGPL-3.0 including the
network-use clause, which matters because it is published at
`github.com/KXZer0/locally`. Features may be **reimplemented independently**
from observed behaviour; code must not be copied. Any agent asked to "bring
over" an Odysseus feature must write it from scratch and say so in the commit.

**OpenCode is the opposite case and is still deleted** — §3.6. The user will
install it standalone to use its own API keys and OpenRouter. That removal
stands.

### 5.3 locally as the router; OpenCode for cloud models

**`core/slots/proxy.py` already does most of this.** `ProxySlot` implements
`DeviceSlot`'s surface against any OpenAI-compatible server and is wired to
`--proxy-url` / `--proxy-model`. OpenRouter is an OpenAI-compatible server. The
work here is not architectural.

What is missing:

1. **Several proxy slots, not one.** Today `--proxy-url` *replaces* the primary
   slot. Make remote providers additional entries in the model list, so a local
   model and a cloud model coexist and the picker chooses per turn.
2. **Auth.** An `Authorization: Bearer` header plus the `HTTP-Referer` /
   `X-Title` headers OpenRouter wants. Key from env or a gitignored file.
   **Never** in `/health`, never in a log line, never echoed to the UI.
3. **Remote model discovery.** `GET /v1/models` on the provider, cached, merged
   into `/v1/models/available` with a provenance field.
4. **Provenance must be unmistakable.** The UI already separates NPU/GPU/CPU by
   the mono-face engine name plus a dot whose fill carries the same information.
   Add a fourth state for remote. This is not decoration: in an app whose
   premise is local inference, a turn leaving the machine has to be visible
   without reading, and it has to survive the monochrome palette (§2.1).
5. **A hard local-only switch.** One setting that refuses every remote slot.
   Voice and the quick assistant should default to local regardless, since both
   capture ambient content the user did not deliberately paste.

**OpenCode moves OUT entirely — see §3.6, which supersedes this paragraph.** (Superseded text follows.) Keep the embedded Code tab
(`core/opencode_web.py`) and point OpenCode at OpenRouter for datacenter models
and at locally for local ones. Stop implying locally serves the Code tab's
models. Its 127.0.0.1 binding and the phone-cannot-reach-it limitation are
already documented and unchanged.

**Do not build a routing policy engine.** TODONT rejected OmniRoute. The
decision here is a model picker, not automatic fallback. The user chooses the
model; the app shows where it ran.

### 5.4 Feature order

1. §5.0 re-export. One command, immediate quality, and it unblocks judging
   everything else.
2. §3.6 OpenCode removal. Deletion, lowest risk, shrinks the surface before
   anything is added to it. (Odysseus is KEPT — see §5.2.)
3. §5.1 quick assistant, text-only path, measured to first paint.
4. §5.3 multi-proxy and provenance.
5. §5.1 selection synthesis, then images.

## 6. Skills to pull in (C:\Projects\skills)
`apple-design` (fluid interfaces, materials, reduced-motion), `animate`
(build), `review-animations` (gate), `improve-animations` (audit the 13
keyframes first), `emil-design-eng` (polish details), `pick-ui-library` (only
if a dependency is ever considered; today the answer is none, no CDN).

Sources (external): OpenVINO GenAI GGUF blog + feature update; OpenVINO 2026.3
release notes; optimum-intel issue #1722; SmoothUI / 925studios "AI slop"
guides; web.dev `content-visibility` and compositor-only properties.
