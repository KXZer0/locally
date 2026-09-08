# Checkpoint 4 — Consolidation and surface redesign

Work brief for Codex agents. Read this whole file before editing anything.
`docs/CHECKPOINT-3-UI.md` §1 (what "terminal-grade" means) and §2 (hard
constraints) still apply in full and are not repeated here.

## Checkpoint 3 verification — what actually landed

Audited 2026-08-19 against the running server. Codex's work is largely good and
the parts below are **verified fixed**, not assumed:

| Item | State |
|---|---|
| Audit finding 1 — mic meter animating `width` | **fixed** — `app.js:2656` now writes `transform: scaleX()` |
| Audit finding 2 — mem bar animating `width` | **fixed** — `app.js:3759/3772` now `scaleX()` |
| Audit finding 5 — reduced-motion zeroing everything | **fixed** — both blocks now target movement and keep opacity |
| Audit finding 6 — no hover gating | **fixed** — 3 `hover: hover` queries added |
| Palette must not animate | **held** — instant open/close preserved through `d96e1e4` |
| Launcher hot path | **254–344 ms** measured, target was sub-300 ms |

Two findings from that audit are **still open**: 10 and 11 (no leading or
tracking tokens). They are task 4e below.

## The one rule that was broken

`static/css/assistant.css` (938 lines) was added as a **second stylesheet
layered over `style.css`**, redefining `:root`. This is the parallel token
system the `animate` skill's Hard Rule 3 forbids outright — "extend the
codebase's tokens, don't fork them. Adding a parallel system is a defect" — and
it is now the single largest source of confusion in the UI.

Measured overlap between the two files:

- **85 selectors are declared in both.**
- **19** of those are *fully* superseded in `style.css` — every property is
  overridden, so the `style.css` copy is pure dead weight.
- **66** are *partially* overridden, which is the real hazard: to know what any
  element looks like you must read both files and mentally apply the cascade.

The device colours (`--npu`/`--gpu`/`--cpu`) survived correctly in `style.css`,
and `assistant.css` does use the easing and duration tokens, so the fix is a
merge rather than a rewrite.

**A concrete bug this caused** — removing a property from the winning sheet
resurrects the losing sheet's value. Found while de-containerising: dropping
`background`/`border` from `assistant.css .util-workspace` made
`style.css:848`'s version apply instead, including a `border-top: 2px solid
var(--device)`. *That* was the amber stripe around the utility card. You cannot
safely edit either file without reading both — which is the whole argument for
this merge.

### Two findings in an earlier draft of this brief were WRONG — corrected

Both came from reading a computed value without checking where it came from.
Recorded because the mistake is instructive, not to pad the document:

- ~~"the assistant message has three competing left accents"~~ — **false**.
  `assistant.css:350` already sets `.message.assistant::before { display: none }`.
  Computed style reported the pseudo-element's `content` and `background`, and I
  did not check `display`. There is one accent, and it is correct.
- ~~"`.util-heading` has a 138px right margin, a layout hack"~~ — **false**. That
  is `margin: auto` from `.util-content > * { width: min(920px, 100%); margin-left:
  auto; margin-right: auto }`, resolved at the current viewport. It is ordinary
  column centring.

The lesson for task 4a: **a computed value tells you what, never why.** Trace
every property to its rule before acting on it.

---

## Task 4a — merge the stylesheets  ← **Codex, do this first**

Everything else in this checkpoint is harder to review until this is done.

Fold `assistant.css` into `style.css` and delete `assistant.css` and its
`<link>` (`templates/index.html:14`). Target: **one stylesheet, one `:root`.**

Method, in this order — do not freestyle it:

1. Merge the two `:root` blocks. Where they conflict, **`assistant.css` wins**
   (it is the current design and it is what ships today). Keep every device
   colour and every motion/spacing/type token.
2. For each of the 19 fully-superseded selectors, delete the `style.css` copy.
3. For each of the 66 partial ones, merge into a single rule and verify with
   `getComputedStyle` that the element is byte-identical before and after.
4. Re-run the duplicate check until it reports zero:

```bash
python -c "
import re,io
def sels(p):
    s=re.sub(r'/\*.*?\*/','',io.open(p,encoding='utf-8').read(),flags=re.S)
    return {x.strip() for m in re.finditer(r'([^{}@]+)\{[^{}]*\}',s) for x in re.split(r',(?![^(]*\))',' '.join(m.group(1).split())) if x.strip()}
a=sels('static/css/style.css')
print('duplicate selectors:', len(a & sels('static/css/assistant.css')))"
```

**Verification is non-negotiable here.** Before deleting anything, snapshot
`getComputedStyle` for a representative element of every duplicated selector,
then compare after. A merge that "looks fine" and silently drops a property is
worse than the duplication.

## Task 4b — the assistant message  ← Codex

Measured today: body **13px** in a **760px** column. That is roughly 95–100
characters per line. The readable target is **65–75ch**, and the column is not
the problem — the type is too small for it.

| Property | Now | Target | Why |
|---|---|---|---|
| `.msg-body` font-size | 13px | **15px** (`--fs-base` becomes 15px, or use `--fs-lg`) | 760px ÷ 15px Inter ≈ 72ch, inside the target band, and 13px body is below comfortable reading size |
| `.msg-body` line-height | 19.5px (1.5) | 1.6 at the new size | apple-design: leading tracks size inversely; body wants looser |
| `.msg-role` font-size | **8px** | **11px** (`--fs-xs`) | 8px is below any legibility floor. It is a real label, not decoration |
| `.message.assistant::before` | already `display:none` | leave it | see the correction above |

On the card treatment: the assistant message currently has a background tint, a
border, an 11px radius and 11×13px padding — it reads as a boxed card. The
brief for this project is "dense but breathable". **Prefer flat**: drop the
background tint and the radius, keep the device border-left as the only
provenance mark, and spend the saved visual weight on vertical rhythm between
turns (`--sp-5` between messages rather than a 10px margin). Keep `pre`/code
blocks boxed — they should read as machine output against flat prose.

Do not touch `renderBody()` or the streaming path. This is CSS and markup only.

## Task 4c — voice  ← Codex

**The orb is not broken — do not "fix" it.** `.orb` is a 142px grid container
with `border-radius: 0`; the visible circles are its three children
(`orb-core` 78px, `orb-ring` 110px, `orb-ring-2` 138px), each at `border-radius:
50%`. This was checked. Leave the geometry alone.

What to change:

- `.voice-state` is **9px** with 1.26px tracking. Raise to **11px** (`--fs-xs`)
  and reduce tracking to ~0.08em. Uppercase mono at 9px is a texture, not a word.
- The panel is `display:flex` with `gap: normal` — i.e. no gap is set at all, so
  every space is coming from ad-hoc margins. Give it an explicit `gap` on the
  spacing scale and remove the margins it replaces.
- The state line, transcript, reply and timings should read as one vertical
  stack on a single rhythm. Right now they are four independently-spaced blocks.

Motion: the orb's ambient keyframes were verified GPU-safe (`transform`/`opacity`
only) and justified as state indication. Leave them.

## Task 4d — utilities  ← Codex

- `.util-panel` is `display:flex` with `gap: normal` — same problem as voice.
- `.util-nav-item` is 11px with a 7px radius and an amber-tinted background at
  `0.116` alpha. The tint is applied to the *resting* state, so every item looks
  half-selected. Selection should be the only thing wearing the device colour.
- The six util views are near-duplicate markup blocks (`index.html`). If the
  merge in 4a makes it obvious, collapse them to one template shape — but only
  if it does not grow `app.js` complexity. Do not force this.

## Task 4e — typography tokens (audit findings 10 & 11)  ← Codex

Still open from the Checkpoint 3 audit. From `apple-design` §15:

> Tracking is size-specific — never one value for all sizes. Large display text
> wants negative tracking; small text wants slightly positive. Leading tracks
> size inversely. Build hierarchy from weight + size + leading **as a set**.

Add paired tokens so a size never travels alone:

```css
--lh-tight: 1.15;   --ls-tight: -0.02em;   /* --fs-2xl, --fs-xl */
--lh-snug:  1.35;   --ls-normal: 0;        /* --fs-lg */
--lh-body:  1.6;                            /* --fs-base */
--lh-dense: 1.45;   --ls-wide: 0.06em;     /* --fs-sm, --fs-xs, mono labels */
```

Then apply them wherever a font-size is set. A heading at `--fs-2xl` with body
tracking is the specific defect this fixes.

**Not in scope:** converting px to rem. `apple-design` asks for it so OS text
scaling works, but the whole stylesheet is px and that is a project-wide call the
owner has not made. Leave it.

---

## Claude's task — context window tracking (functional, not visual)

Not assigned to Codex; it is a server change with a device-behaviour subtlety.

`contextTokenLimit()` (`app.js:314`) returns `null` for any slot that is not NPU
and has no `context_tokens`. The server sets `context_tokens` **only inside
`if kv_pool:`** (`locally.py:1660-1664`), which requires the continuous-batching
backend — and `scripts/locally-launch.ps1` passes `--no-prompt-cache` for GPU.
So GPU slots never get a denominator and the context ring has nothing to draw.
That is why only NPU tracks today.

The fix is that the context limit was never really a property of the KV pool.
Every model states its own ceiling in `config.json` as
`max_position_embeddings`, already read at `locally.py:1551`. Report that
always, and treat the KV pool as an **additional cap** when one exists:

- NPU → `min(MAX_PROMPT_LEN, max_position_embeddings)`
- GPU/CPU with prompt cache → `min(kv_pool_capacity, max_position_embeddings)`
- GPU/CPU without prompt cache → `max_position_embeddings`

`/health` should also say **which** constraint is binding, so a user who sees
32k on a 262k model can tell it is the pool and not the model.

## Verification

- **4a:** duplicate-selector count reaches 0; `assistant.css` deleted; computed
  styles unchanged for a sampled element of every merged selector.
- **4b/4c/4d:** measure the result. `getComputedStyle` on `.msg-body`,
  `.msg-role`, `.voice-state` must show the target sizes. Character measure
  between 65 and 75.
- **Offline:** DevTools Network shows zero external origins. Still the check that
  catches a CDN font arriving from a skill suggestion.
- **Motion:** nothing over 300ms; no Layout event in any animation frame.
- **App runs:** all three modes, a chat turn, a voice turn, one util task.

## A note on reviewing this work

I could not screenshot during this audit — the browser pane does not composite
in this environment, so every finding above comes from `getComputedStyle`,
stylesheet parsing, and the DOM. That catches cascade bugs, dead rules and type
sizes; it **cannot** tell you whether the result looks good. Someone has to
actually open it.

---

# Checkpoint 5 — Auto-compaction (added 2026-08-19)

Not UI work, but it lands in `app.js` so it belongs with the same owner.

## Read this first: the NPU window is 8192 and cannot go higher

Measured, not assumed — `scripts/npu-context-probe.py`, and the full table is in
`TODONT.md`. `NPU_MAX_PROMPT_LEN` is now **8192** (was 4096), verified by
recalling a needle from an 8,160-token prompt. 9216, 10240 and 12288 all fail
the vpux compiler's `EnsureNCEOpsSizeRequirements` pass.

**Do not try to raise it with KV-cache compression.** 12288 fails *identically*
with `KV_CACHE_PRECISION=u8`. Nothing here is short of memory — Qwen3-8B is
144 KB/token, so even 8192 tokens is 1.1 GB. It is a graph-shape limit.

That matters for the design below: on the NPU the window is small, hard, and
**not negotiable**, so the only remaining lever is sending less.

## The bug this fixes

The NPU pipeline **refuses** an over-long prompt — it does not truncate. Today
nothing trims history (`CLAUDE.md`: "Chat history unbounded in web UI — user
clears with Ctrl+N"), so a long chat walks straight into a hard failure, and the
reported symptom near the limit is the model repeating itself.

## Task 5a — fit the history before sending

`buildMessages()` (`app.js:223`) composes every turn. Add a fitting pass to it.

- Get the budget from `contextTokenLimit(slot)` — it is now populated on **every**
  device, not just NPU (`5fae084`).
- Compact at **~75% of the budget**, not at 100%. The reply needs room too, and
  a compaction that triggers only on failure has already failed.
- Count with the server: `POST /v1/messages/count_tokens` (`locally.py:5478`)
  uses the model's real tokenizer. Do not estimate with `length / 4` — being
  wrong by 15% at the boundary is the whole bug.
- **Summarise, do not just drop.** Take the oldest turns that must go, ask the
  model for a short summary of them, and replace them with one system-role note
  ("Earlier in this conversation: …"). Dropping turns silently is what makes a
  model contradict itself; a summary keeps the thread coherent.
- The system prompt and the most recent turns are never candidates.
- **Say it happened.** A visible line in the thread — "Compacted 14 earlier
  messages to stay within the 8k window." Silent memory loss is
  indistinguishable from the model being stupid, which is exactly how this
  currently reads.

## Task 5b — voice auto-forget

Voice shares `chatHistory` with chat, so it inherits the same overflow. But a
summarisation round-trip costs a model call, and voice is already latency-bound
— that is the wrong trade in the middle of a spoken turn.

- In voice, when compaction would be needed, **drop oldest turns outright**
  rather than summarising. Spoken exchanges are ephemeral; the chat thread keeps
  the full record, and `VOICE_MAX_TOKENS` (220) already caps each spoken reply.
- If a single turn alone exceeds the window, that is the one case compaction
  cannot solve: say so out loud in one short line and point at the Chat tab,
  matching the existing precedent for a model that runs out mid-thought.
- Never block a spoken turn on a summarisation call.

## Verification

- Drive a chat past 8192 tokens on NPU. It must compact and keep answering —
  no raw C++ assertion, no repetition. Before this change it hard-fails.
- Confirm the compaction notice appears and names a real number.
- Confirm a GPU model compacts at its own (much larger) limit, using
  `context_tokens` rather than a hardcoded figure.
- Confirm voice degrades by forgetting, never by erroring mid-turn.
