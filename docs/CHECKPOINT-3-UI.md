# Checkpoint 3 — Terminal-grade UI and motion

> **Status: complete and superseded for new UI work.** This document remains the
> historical record for Checkpoint 3. The full structural redesign is specified
> in `docs/CHECKPOINT-4-UI-REDESIGN.md`; agents implementing that redesign must
> follow Checkpoint 4 when the two briefs differ.

Work brief for Codex agents. Read this whole file before editing anything.

Checkpoints 1 and 2 are done: the launcher hot path, the PWA identity, the
streaming render fix (`b11c88e`), and the `/v1/ui/bootstrap` collapse (`3c3763a`).
This checkpoint is **presentation only** — no new endpoints, no new server
behaviour, no dependencies.

## Status — who has what

Split agreed 2026-08-19.

| Task | Owner | State |
|---|---|---|
| 3a tokens (`:root`) | Claude | **done** — `64c4b74` |
| 3a literal migration | Codex | **done** — `74fbbb6` |
| 3b motion | Codex | **done** — `83bf0cc` |
| 3c command palette | Claude + Codex | **done** — `9c0d7ce`, focus containment `d96e1e4` |
| 3d render cleanups | Codex | **done** — `8138219` |

The tokens moved to Claude because the palette depended on them. The ownership
split prevented overlapping edits while both agents were active. After the
palette landed, Codex changed only its missing Tab/Shift+Tab containment
(`d96e1e4`) to satisfy the focus-trap requirement below; the instant open/close
path remains unchanged.

### Correction: the palette does not animate

The first cut of this brief (and the first cut of the palette) faded it in over
150ms. `review-animations` puts a command-palette toggle in the 100+/day band,
whose rule is "no animation, ever". Fixed in both. §4 now leads with the
frequency table — **apply it before picking a curve**, or you will animate
things that should not move.

### A trap the palette work hit, which the migration will hit too

`.palette-input` (specificity 0,1,0) was silently losing to the shared control
rule `textarea, select, input[type="text"]` (0,1,1) — padding, border,
background, radius and font-size were all being overridden with no warning
anywhere. Source order does not save you; the fix was `input.palette-input`.

When migrating literals, **verify computed style, not the source**. A changed
declaration that loses the cascade looks correct in the file and wrong on
screen. `getComputedStyle(el)` in the console is the check.

## 0. Install the design skills first

```
npx skills@latest add emilkowalski/skills
```

Audited 2026-08-19: the installer is `vercel-labs/skills` (MIT, no
`preinstall`/`postinstall` hooks, dependencies `tar` + `yaml`); the content is
markdown-only with no executable files, scanned clean for injection and
exfiltration. Two of its skills carry an explicit anti-injection rule of their
own.

Use `animate`, `review-animations` and `apple-design`. **Ignore their stack
advice** — see §2.

## 1. What "terminal-grade" means here

Not a terminal skin. No scanlines, no CRT glow, no green-on-black, no fake
cursor blink on ordinary text. The qualities being borrowed are:

- **Instant.** Nothing waits on an animation to become usable.
- **Keyboard-first.** Every destination reachable without the mouse.
- **Dense but breathable.** Information-tight, on a consistent rhythm.
- **Machine truth in mono.** Device names, timings, token counts, byte sizes use
  `--mono`. Prose stays `--ui`. This rule already exists; apply it consistently.
- **Motion that reports state, never decorates.** If an animation does not tell
  the user what just changed, delete it.

The device-colour system (`--npu` amber / `--gpu` cyan / `--cpu` slate, applied
per-element as `--device` by `paintDevice()`, `static/js/app.js:256`) is the
visual spine and is **load-bearing information design**. Extend it. Do not
restyle, recolour, or replace it.

## 2. Hard constraints — these override any skill's recommendation

The skills will suggest Tailwind, component libraries, CDN fonts and a motion
library. All four are forbidden here.

1. **No CDN, ever.** Fonts are self-hosted in `static/fonts/` (`style.css:6-30`).
   The project's claim is that it works on a plane. One external URL breaks that.
2. **No frameworks, no build step, no bundler.** Vanilla DOM, plain `<script>`.
   `app.js` is 3,391 lines of it and stays that way.
3. **No motion library.** CSS transitions and `requestAnimationFrame` only.
   Rule 5 of the `animate` skill agrees: cheapest tool that works.
4. **Do not touch `renderBody()`** (`app.js:781`). It is escape-first — it
   escapes input before anything else, so only tags it constructs itself can
   reach the DOM. That property is why marked.js is not here. See `TODONT.md:335`.
5. **Do not touch the streaming render path.** It was just fixed. Any change that
   re-parses the full message per token is a regression.
6. **`.msg-body math { font-family }`** (`style.css:1188`) is load-bearing — it
   supplies the OpenType MATH table that makes sums grow and fences stretch. Do
   not "clean up" that font stack.
7. **`prefers-reduced-motion`** (`style.css:958`) already exists. Every new
   animation gets an entry there in the same commit, not as a follow-up.

## 3. Task 3a — the token system  ✅ DONE (Claude, `64c4b74`)

`:root` (`style.css:32`) currently defines colours, two radii, and nothing else.
Every padding, font-size and duration across 1,386 lines is a hardcoded literal,
and there are **zero `cubic-bezier` declarations** — all 9 transitions use
browser-default easing. That is the root of "clunky".

Add to `:root`, leaving every existing token unchanged:

```css
/* Easing — from the animate skill. Do not approximate these. */
--ease-out:     cubic-bezier(0.23, 1, 0.32, 1);
--ease-in-out:  cubic-bezier(0.77, 0, 0.175, 1);
--ease-drawer:  cubic-bezier(0.32, 0.72, 0, 1);

/* Duration */
--dur-instant: 100ms;   /* press feedback */
--dur-fast:    150ms;   /* tabs, toggles, hovers */
--dur-base:    200ms;   /* panels, dropdowns */
--dur-slow:    280ms;   /* settings, palette */

/* Spacing — 4px rhythm */
--sp-1: 4px;  --sp-2: 8px;  --sp-3: 12px; --sp-4: 16px;
--sp-5: 24px; --sp-6: 32px; --sp-7: 48px;

/* Type scale */
--fs-xs: 11px; --fs-sm: 12px; --fs-base: 14px;
--fs-lg: 16px; --fs-xl: 20px; --fs-2xl: 28px;

/* Elevation */
--shadow-sm: 0 1px 2px rgb(0 0 0 / 0.3);
--shadow-md: 0 4px 12px rgb(0 0 0 / 0.35);
--shadow-lg: 0 16px 48px rgb(0 0 0 / 0.45);
```

**Never `ease-in` on UI** — it starts slow at the exact moment the user is
watching. **Nothing over 300ms.**

Then migrate literals to tokens. Do this as a **separate commit** from the motion
work so a visual regression is bisectable to one or the other. Migrate by
section, confirming each renders identically before moving on. A value that does
not fit the scale is a signal that either the scale is wrong or the value was
arbitrary — decide which; do not add a one-off token.

## 4. Task 3b — motion  ✅ DONE (`83bf0cc`)

**Gate every animation on frequency before choosing a curve.** From
`review-animations/STANDARDS.md`:

| Frequency | Decision |
|---|---|
| 100+/day — keyboard shortcuts, command palette toggle | **No animation. Ever.** |
| Tens/day — hover, list navigation | Remove or drastically reduce |
| Occasional — modals, drawers, toasts | Standard animation |
| Rare — onboarding, celebrations | Can add delight |

"Never animate keyboard-initiated actions — they repeat hundreds of times daily;
animation makes them feel slow and disconnected." This is not a style
preference: motion on a high-frequency action reads as **lag**, which is the
opposite of the goal. Raycast ships no palette open/close animation.

An earlier revision of this brief told you to animate the palette at
`--dur-fast`. That was wrong and has been corrected — the palette now ships
with no animation at all.

| Surface | Frequency | Code | Spec |
|---|---|---|---|
| Command palette | 100+/day | `palette.js` | **none** — already correct, do not add any |
| Tab switch | tens/day | `setMode()` `app.js:1959` | drastically reduced: `opacity` **only**, `--dur-instant`, `--ease-out`. No `translateY` — it is reachable by arrow keys and by the palette, so it is a near-keyboard action |
| Util view switch | tens/day | `.util-nav-item` `style.css:737` | same as tab switch |
| Settings panel | occasional | `#settings-panel` `index.html:96` | standard: `--dur-slow` `--ease-drawer` |
| Message insert | occasional | thread append | `opacity 0→1` + `translateY(8px→0)`, `--dur-base` `--ease-out` |

Rules:

- `hidden` cannot be transitioned. Swap to a class driving `opacity`/`transform`
  and set `hidden` only after the transition ends — or use `@starting-style`
  with `transition-behavior: allow-discrete`.
- **Animate `transform` and `opacity` only.** Never `height`, `top`, `width` or
  `margin` — those run layout every frame.
- **Never animate the streaming message.** A per-token transition on a message
  repainting 40×/second is a stutter generator.
- Stagger only where several elements enter together: 30–80ms.
- Every one of these ships with a `prefers-reduced-motion` entry in the same
  commit.

## 5. Task 3c — command palette (Ctrl+K)  ✅ DONE (Claude, `9c0d7ce`)

The entire global shortcut set today is Ctrl+N and Escape (`app.js:3265`). The
palette is what makes this feel like a terminal, and it fixes a real structural
problem: 3 top-level tabs (`index.html:80`) and a 6-item Util rail
(`index.html:268`) with no shared navigation model.

Build a **command registry** — one array of
`{id, label, hint, group, run(), available()}` — and render from it. Do not
hardcode a list in the markup; every later feature registers itself here.

Initial commands: switch to Chat / Voice / Util; jump to each of the 6 util
tasks; new chat; open settings; load model; unload model; toggle no-think; focus
input.

Behaviour:

- `Ctrl+K` opens, `Escape` closes, `↑`/`↓` move, `Enter` runs.
- Substring match on label and group, ranked prefix-first. No fuzzy library.
- Filter on `input`, not `keydown`, so IME composition works.
- Commands whose `available()` is false render disabled **with the reason** —
  the same honesty rule the util buttons already follow via `/health`.
- Restore focus to the previously focused element on close.
- `role="dialog"`, `aria-modal`, `aria-activedescendant`, focus trap while open.
- **No enter/exit animation.** See the frequency gate in §4. Shipped this way;
  do not reintroduce a fade.

## 6. Task 3d — cheap wins in the same pass  ✅ DONE (`8138219`)

- `renderDevices()` (`app.js:360`) and `renderMemory()` (`app.js:3298`) wipe and
  rebuild their DOM on every poll regardless of whether anything changed. Gate
  both on a value signature and return early when identical.
- `escapeHtml()` (`app.js:998`) creates a throwaway DOM element per call and runs
  many times per render. Replace with a string replace over `& < > " '`.
- Textarea autosize forces two reflows per keystroke (set `height:auto`, read
  `scrollHeight`, write height). Cache the last computed height; skip the write
  when unchanged.
- `scrollToBottom()` (`app.js:275`) — confirm it is inside the rAF tick from
  checkpoint 1 and never called synchronously after a DOM write.

## 6b. Audit findings — run against the existing UI (2026-08-19)

`review-animations` over all 9 pre-existing transitions and 3 keyframes, and
`apple-design` over the token scale. **Nothing here was written for this
checkpoint** — these are pre-existing and none had been checked against anything.
Fold them into commits 2 and 3.

| # | Sev | Location | Standard | Finding |
|---|---|---|---|---|
| 1 | **HIGH** | `style.css:1135` `.mic-meter-fill` + `app.js:2240` | 7 — GPU-only | `transition: width 0.05s` **and** `style.width` written every rAF frame while the mic is open. A layout property animated continuously, with a 50ms transition that can never finish against ~16ms writes, so it retargets forever. Fix: `transform: scaleX()` + `transform-origin: left`. |
| 2 | **HIGH** | `style.css:1359` `.mem-bar > i` | 7, 4 | `transition: width .4s` — layout property, and 400ms is over the 300ms UI budget. Fix: `transform: scaleX()`. |
| 3 | MED | `style.css:641` `.orb-core` | 4 | 500ms on background/box-shadow. Over budget; needs a stated reason or reduction. |
| 4 | MED | `style.css:153` `.brand-mark` | 4 | 400ms. Same. |
| 5 | MED | `style.css:958` reduced-motion block | 8 | Sets `transition-duration: 0.001ms !important` on **everything**. The standard is "gentler, not zero — keep opacity/color, drop movement". This also kills the comprehension-aiding colour transitions. Scope it to transform-bearing motion. |
| 6 | MED | 13 `:hover` rules | 8 | Zero `@media (hover: hover) and (pointer: fine)` gating in the file. Touch fires false hovers on tap. |
| 7 | MED | all 9 transitions | 3 | Every one uses built-in `ease` or no easing at all. "Built-in CSS easings are too weak." The tokens now exist — commit 2 should swap them in. |
| 8 | LOW | `style.css:442` `.copy-btn` | 3 | `transition: opacity 0.15s` names no easing. |
| 9 | LOW | `style.css:1329` `.mem-hud-pill::after` | 3 | `transform .18s ease` on a chevron; `--ease-out` is the better curve. |

**Passing:** all three keyframes (`pulse`, `drift`, `churn`) animate only
`transform`/`opacity`, so they are GPU-safe, and as ambient state indicators on
the voice orb they are justified under standard 1. No `transition: all`
anywhere. No `scale(0)`.

### `apple-design` on the token scale (`64c4b74`)

| # | Sev | Finding |
|---|---|---|
| 10 | MED | **The type scale is size-only.** "Build hierarchy from weight + size + leading as a set, not size alone." There are no line-height or letter-spacing tokens, so `--fs-2xl: 28px` renders at body tracking and reads too loose. |
| 11 | MED | **Tracking must be size-specific** — "a fixed `letter-spacing` is wrong somewhere." Large text wants negative tracking (`-0.02em`), body near `0`. |
| 12 | LOW/decision | **Everything is px, not rem/em.** apple-design asks for spacing and type in `rem` so a larger OS text setting scales the layout rather than breaking it. The whole existing stylesheet is px, so this is a project-wide decision, not a token tweak — do not convert unilaterally. |

Findings 10 and 11 are additive and should land with commit 2. Finding 12 needs
a call from the project owner first.

## 7. Verification

- **No streaming regression.** Performance panel, one ~1000-token answer with a
  code block and an equation. Scripting time must not exceed the current build.
  Text selection inside a streaming message must survive.
- **Motion.** Every animated surface checked with reduced-motion on and off.
  Nothing over 300ms. Verify in the Performance panel that no animation frame
  shows a Layout event.
- **Palette.** Fully operable with the mouse unplugged. Every command runs.
  Disabled commands state why.
- **Tokens.** `grep -c cubic-bezier static/css/style.css` returns > 0. No
  hardcoded duration outside `:root`. Screenshot-diff all three tabs across the
  migration commit.
- **Offline.** DevTools Network — zero external origins. This is the check that
  catches a CDN font arriving via a skill suggestion.
- **The app still runs**: `.\scripts\locally-launch.ps1`, all three tabs, a chat
  turn, a voice turn, one util task.

## 8. Commit shape

Four commits, in this order, each independently working:

1. ~~`webui: design tokens`~~ — landed as `64c4b74`
2. ~~`webui: migrate hardcoded literals to tokens`~~ — landed as `74fbbb6`
3. ~~`webui: motion on tab, util, settings and message transitions`~~ — landed as `83bf0cc`
4. ~~`webui: command palette`~~ — landed as `9c0d7ce`

Follow-ups stayed separate: unchanged-render and textarea cleanup (`8138219`),
then the palette's missing focus containment (`d96e1e4`).

Do not squash 2 and 3. Commit 1 was additive and cannot regress a rendered
value; commit 2 changes existing values and is the one a visual regression must
stay bisectable to.
