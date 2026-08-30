# Checkpoint 4 — Full web UI redesign

Single source of truth for the redesign. Every worktree agent reads this file,
`AGENTS.md`, and nothing else from the conversation that produced it. The
checkpoint-planning commit is the common base for all five worktrees.

---

## 0. Why the last attempt failed

The 2026-08-24 reskin changed `:root` and left the structure alone. The evidence
is in the sheets themselves:

- **Two stylesheets fight.** `style.css` (1,747 lines) defines the design;
  `assistant.css` (1,785 lines) overrides it. `style.css` gives `.panel.entering`
  an opacity transition; `assistant.css:216` sets `.panel.entering { opacity: 1 }`.
  The animation is dead code that still ships. Same pattern for `.util-view`,
  `.settings-panel.entering`, `.message.user`.
- **Two spacing rhythms.** `style.css` declares a 4px scale (`--sp-1..8`).
  `assistant.css` then writes raw `5px 7px`, `14px`, `22px`, `9px`, `78px`.
  Nothing lines up with anything, which is exactly what "incorrectly spaced"
  feels like from the outside.
- **Legacy aliases hide the structure.** `--surface`, `--text`, `--border`,
  `--device`, `--npu/--gpu/--cpu` all resolve to the new ramp so ~200 old rules
  survive untouched. The reskin was one variable swap over the 2025 layout.
- **The documented layout was never built.** `CLAUDE.md` says "one persistent
  left sidebar — Chat, Voice, and each utility are peers". The shipped markup
  puts Chat/Voice/Tools in a top tab bar and shows the sidebar *only* under
  `body[data-mode="util"]`.

So: delete both sheets, keep the JavaScript, rebuild the layout.

---

## 1. Non-negotiables

| Rule | Why |
| --- | --- |
| **`static/js/app.js` is not edited.** 4,491 lines of working streaming, VAD, karaoke TTS, markdown and math. Rewriting it is a different project with real regression risk. | The DOM contract in §6 is what makes this possible. |
| **The palette does not change.** Values in §2 are copied from the current `:root`. | Monochrome is load-bearing and already correct — it is not what is broken. |
| **No CDN, no build step, no framework.** Plain CSS + the existing self-hosted fonts. | A local-first tool has to work on a plane; there is no bundler in this repo. |
| **No colour except `--alarm`.** | A rendered-page audit must still return exactly one chromatic value. |
| **Every spacing value comes from `--sp-*`.** A raw `px` in a padding, margin or gap is a review failure. | This is the single biggest cause of the current "off" feeling. |
| **No `ease-in` on any UI element.** | It delays the frame the user is watching. |

---

## 2. Tokens — `static/css/01-tokens.css`

### Surfaces, ink, lines — copy verbatim, do not retune

```css
--bg: #0A0A0A;  --bg-raised: #121212;  --bg-sunken: #070707;
--bg-overlay: #161616;  --chrome: rgb(10 10 10 / 0.86);

--ink: #FFFFFF;
--ink-2: rgb(255 255 255 / 0.62);
--ink-3: rgb(255 255 255 / 0.40);
--ink-4: rgb(255 255 255 / 0.26);

--line: rgb(255 255 255 / 0.10);
--line-soft: rgb(255 255 255 / 0.06);
--line-strong: rgb(255 255 255 / 0.20);
--fill: rgb(255 255 255 / 0.04);
--fill-hover: rgb(255 255 255 / 0.08);
--fill-active: rgb(255 255 255 / 0.12);

--solid: #FFFFFF;  --solid-hover: rgb(255 255 255 / 0.88);  --on-solid: #0A0A0A;
--alarm: #C4544A;  --alarm-dim: rgb(196 84 74 / 0.16);
```

**Legacy aliases are deleted.** `--surface`, `--surface-raised`, `--surface-sunken`,
`--text`, `--text-dim`, `--text-faint`, `--border`, `--border-soft`, `--error`,
`--success`, `--npu`, `--gpu`, `--cpu`, `--device` do not exist in the new sheets.
Every rule names `--bg-*` / `--ink-*` / `--line-*` directly. If a lane needs the
engine-provenance treatment, it uses `--ink` and the dot-fill convention in §5.4.

### Type

```css
--ui:   "Inter", "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif;
--mono: "JetBrains Mono", "Cascadia Code", Consolas, ui-monospace, monospace;

--fs-xs:   11px;   /* mono eyebrows, meta, timings — mono face only */
--fs-sm:   13px;   /* secondary copy, hints, sidebar meta          */
--fs-base: 15px;   /* all chrome: buttons, labels, settings        */
--fs-md:   16px;   /* reading text: thread body + composer input   */
--fs-lg:   20px;
--fs-xl:   28px;
--fs-2xl:  36px;
```

Tracking is size-specific (never one value for all sizes):

| Size | `letter-spacing` | `line-height` |
| --- | --- | --- |
| `--fs-2xl` 36px | `-0.021em` | `1.10` |
| `--fs-xl` 28px | `-0.018em` | `1.15` |
| `--fs-lg` 20px | `-0.011em` | `1.30` |
| `--fs-md` 16px | `0` | `1.60` |
| `--fs-base` 15px | `0` | `1.50` |
| `--fs-sm` 13px | `0` | `1.50` |
| mono `--fs-xs` | `+0.08em`, `text-transform: uppercase` | `1.40` |

The mono face stays reserved for machine truth: device names, timings, token
counts, byte sizes, percentages. It is never used for prose or button labels.

### Space, size, radius

```css
--sp-1: 4px;  --sp-2: 8px;  --sp-3: 12px; --sp-4: 16px;
--sp-5: 24px; --sp-6: 32px; --sp-7: 48px; --sp-8: 72px;

--rail: 260px;          /* sidebar width, collapses to 0            */
--header: 52px;         /* main-column header, not a global topbar  */
--measure: 720px;       /* the ONE content width, shared by all tabs */
--row: 36px;            /* sidebar item / small control height      */
--control: 40px;        /* standard button height                   */

--radius-pill: 999px;
--radius-composer: 24px;
--radius-bubble: 18px;
--radius-card: 16px;
--radius: 12px;
--radius-sm: 8px;
--radius-xs: 6px;
```

`--measure: 720px` is the spine. The thread, the tools workspace, the voice
stage and the composer all centre on it. Three different widths (760 / 900 /
centred-other) is what stops the current app reading as one product.

### Motion

```css
--ease-out:    cubic-bezier(0.16, 1, 0.30, 1);
--ease-std:    cubic-bezier(0.40, 0, 0.20, 1);
--ease-drawer: cubic-bezier(0.32, 0.72, 0, 1);

--dur-instant: 100ms;  --dur-fast: 150ms;
--dur-base:    200ms;  --dur-slow: 280ms;
```

`--dur-enter: 700ms` and `--ease-in-out` are **deleted**. Nothing in this app
takes 700ms and nothing eases in.

---

## 3. Layout

```
┌──────────────┬─────────────────────────────────────────┐
│              │  header  (--header, 52px)               │
│   sidebar    ├─────────────────────────────────────────┤
│   --rail     │                                         │
│   260px      │  scroll region                          │
│              │    content centred on --measure         │
│              │                                         │
│              ├─────────────────────────────────────────┤
│              │  composer   (chat only, same --measure) │
└──────────────┴─────────────────────────────────────────┘
```

```css
body {
  display: grid;
  grid-template-columns: var(--rail) minmax(0, 1fr);
  grid-template-rows: minmax(0, 1fr);
  height: 100dvh;           /* not 100vh — mobile browser chrome */
  overflow: hidden;
}
body[data-rail="collapsed"] { --rail: 0px; }
```

`body[data-mode]` (`chat` | `voice` | `util`) still selects which `<main class="panel">`
is visible — that is app.js's contract and does not change. What changes is that
the sidebar is present in **all three** modes.

### Sidebar (`<aside class="sidebar-tools">` — class name is a JS contract)

Top to bottom, `--sp-3` padding, `--sp-1` gaps:

1. **Brand** — mark + "locally" at `--fs-base` 600, subtitle "On device" at
   `--fs-xs` mono `--ink-3`. Row height `--control`.
2. **New chat** — full-width, `--fill`, `--radius`, `--row` high. The most
   frequent action gets the most obvious position.
3. **Nav** — Chat / Voice / Tools. `--row` high, `--radius-sm`, icon 16px +
   label `--fs-base`. Active = `--fill-active` + `--ink`; rest = `--ink-2`.
   Selecting **Tools** expands its six utilities *inline beneath it*, indented
   `--sp-5`, at `--fs-sm`. One nav tree, no second nested navigation.
4. `margin-top: auto` spacer.
5. **Footer** — engine rail (`#device-rail`) and the settings button. Status
   lives at the bottom of the rail, where every app of this shape puts it.

The rail collapses to `0px` via a header toggle; the collapse animates
`grid-template-columns` — **no**, it animates the sidebar's own
`transform: translateX(-100%)` plus a `--rail` swap with no transition, because
animating a grid track is a layout pass per frame. See §5.7.

### Main column header

`--header` tall (40px, or the measured caption height in an installed window —
see §5.9), `--sp-4` horizontal padding, hairline `--line-soft` bottom.
Left: rail toggle + current mode name at `--fs-base` 600. Right: the
command-palette button.

**Nothing in this row opens a menu.** The header shares its band with the OS
caption buttons, so it is the title bar; a dropdown anchored to a strip that is
also a drag region and is partly occupied by minimise/close is a collision
waiting to happen. The model picker therefore lives **under the composer** — see
§3.1 — and its menu opens upward.

Controls in this row are `--row-chrome` **28px**, not `--row` 36px, and that is a
hard rule: the strip's height is decided by the OS (≈33px on Windows at 100%),
so anything inside it must be sized off the strip and never the other way round.
A 36px control in a 33px strip is what pushed the icons and the Ctrl-K keycap
past the window edge. The keycap also loses its border — a box around two
characters is the tallest object in a 33px row and the least useful.

### 3.1 Under the composer

One line of text directly beneath the input box, on the same `--measure` spine,
indented `--sp-3` so it aligns with the text you typed rather than the box's
outer edge. It holds the **model picker**, rendered as text plus a 12px chevron:
no card, no border, no pill, no eyebrow label — the value is the label. Hover and
open lift the ink from `--ink-3` to `--ink`; that is the entire state vocabulary.
The menu opens upward over the composer.

This is where every app in this class puts it, and it is the answer to "bottom
info": below the box you type in, not in a sidebar that also happens to end at
the bottom of the screen.

Chrome is translucent — `background: var(--chrome)` + `backdrop-filter: blur(12px)`
— and content scrolls under it. Blur stays on the header and on drawers only;
never on a full-surface element, because the iGPU may be running inference. In
window-controls-overlay mode the blur is dropped and the strip is opaque
`--bg-raised`: it is chrome the OS is compositing against, not content.

---

## 4. What moves, and why

| Now | Becomes | Reason |
| --- | --- | --- |
| Chat/Voice/Tools as top tabs; sidebar only in util mode | One persistent left sidebar in every mode | The documented intent, and the shape of every app in this class |
| Model picker + thinking toggle + context rows + memory rows + Free-memory, all inside a popover attached to the composer | Model picker → **the line under the composer** (§3.1), not the header and not the sidebar. Context ring stays in the composer. Memory detail → sidebar footer popover | A text input should not contain the machine's whole control panel; this is the densest spacing failure in the app |
| Stacked wordmark: `locally` over an `ON DEVICE` caption, 38px of type in a 40px band | One line, `--fs-base` 600, optically centred on the mark | It read as jammed against the divider, and the caption restated what the product is. `--fs-base`, not `--fs-md`: the wordmark and the mode title across the divider are two labels in one band, and 16px beside 15px is a mismatch you can see |
| Brand mark 28px, header controls 28px carrying 16px glyphs | Mark `--brand-mark-size` 20px, controls `--row-chrome` 26px carrying `--icon-chrome` 14px glyphs | Everything in this strip shares it with the OS caption buttons (a 32px box around a ~10px glyph), so the caption buttons are the reference, not `--row`. At 28px the mark was the largest object in the window's top band and the corner read top-heavy; the header controls read as body controls that had wandered into the title bar. All five things in the band now centre within 0.5px of each other |
| Status pips on the sidebar mark and on the home mark | Both gone. The sidebar node is `display: none` (`paintDevice` writes to it unconditionally); the home node is deleted (`app.js` reads it with `?.`) | A dot welded to a logo reads as part of the logo |
| Filled/bordered boxes on: the New-chat row, the device pills, the Ctrl-K keycap | All unboxed; hover background is the only container left | One boxed row among unboxed rows makes the rest read as disabled |
| Context readout: filled disc + border + 3px core dot + ring, permanently in the composer | A bare ring; the token counts appear in a hover/focus panel above it, mirrored from the sidebar memory rows | You need the fraction constantly and the exact counts about twice a session. Four concentric rings of chrome to carry one number |
| 7px status dot welded to the sidebar brand mark | Removed (`display: none`; the node stays because `paintDevice` writes to it unconditionally) | A status pip on a logo reads as part of the logo, and it sat on the mark's thinnest lobe. Offline is still reported in words by the memory row and the device rail |
| Composer: draft and controls sharing one flex row, icons pinned to its bottom edge | **Two rows inside the box** — `.composer-row` holds only the textarea, `.composer-actions` holds every control at `--row` 36px | Sharing a row means every wrapped line re-solves the controls' position against a container whose height is changing under them, so the cluster shifts as you type. Split, the controls' 9px gap to the box's bottom edge is constant at every draft height (measured 1–9 lines) and nothing but text moves |
| Composer: the browser's default `:focus-visible` outline on `#message-input` drew a second, tighter rounded rectangle nested inside `.composer-inner`'s own border | `#message-input:focus-visible { outline: none }` — the card's existing `:focus-within` border brightening is the only focus indicator left | `#message-input:focus { outline: none }` already existed and didn't touch it: `:focus` and `:focus-visible` are different pseudo-classes, and a text field gets the latter on essentially any focus. Two containers announcing the same state at once is the bug; one ring, on the card, is enough |
| Assistant mark + timing line as two stacked rows (mark above, `.meta` appended below it) | One row, `.msg-footer` — `[mark] NPU 2.3s 18.8 tok/s` | Requested: keep the mark next to the device/timing text rather than floating above it on its own line |
| Memory panel (`.mem-hud-detail`) opened by `.pinned` OR `:focus-within` | `.pinned` only | Clicking a button focuses it (Windows Chrome/Edge), so the very click meant to CLOSE the panel left focus inside `.mem-hud` -- and `:focus-within` reopened it in the same frame. The panel could be opened but never visibly closed by clicking again. Enter/Space on a focused button already fires the same click handler, so the fix costs keyboard users only the zero-effort tab-to-preview, not the ability to open it |
| `.message.user` marked by `border-left: 2px` + role label | Right-aligned bubble, `--bg-raised`, `--radius-bubble`, `--sp-3 --sp-4` padding, `max-width: 76%`, no role label | Role is readable from position and shape; a label per turn is noise |
| `.message.assistant` marked by `border-left: 1px` | No border, no background, full `--measure`, role label only on the first turn of a group | The assistant turn is the document; chrome around it competes with it |
| `.thread` 760px, `.util-content` 900px, voice centred separately | All three on `--measure` 720px | One spine |
| Composer input at `--fs-sm` 13px, `5px 7px` padding, `line-height: 1.5` | `--fs-md` 16px on a **24px integer line**, `--sp-2` padding, min-height 40px, cap 184px (7 lines), `--radius-composer` | 13px in a full-screen composer reads as cramped. The integer line box is the other half of smooth growth: at `1.5 × 15px` a line is 22.5px and each new one lands on a different sub-pixel rounding, so the box grew in uneven steps. 24 + 8 + 8 makes one line exactly 40px, every further line exactly +24, and the cap a line boundary rather than mid-glyph |
| `.enter-2/3/4` stagger, `--dur-enter` 700ms first paint | Deleted | See §5 |

---

## 5. Motion — the complete allowed list

Anything not on this list does not animate. Frequency decides: an action a user
performs 100+ times a day must not have a transition attached to it.

| # | Moment | Property | Duration | Curve |
| --- | --- | --- | --- | --- |
| 1 | Any button/row press | `transform: scale(0.97)` on `:active` | `--dur-instant` | `--ease-out` |
| 2 | Hover on a nav row / list row | `background` only | `--dur-fast` | `--ease-std` |
| 3 | Hover on a home quick-action card | `background` + `translateY(-2px)` | `--dur-fast` | `--ease-out` |
| 4 | Sidebar / settings / sources drawer | `transform: translateX` | `--dur-slow` | `--ease-drawer` |
| 5 | Assistant turn appears | `opacity` 0→1 only | `--dur-fast` | `--ease-out` |
| 6 | `<details>` disclosure | `grid-template-rows: 0fr→1fr` | `--dur-base` | `--ease-std` |
| 7 | Brand sweep while generating | existing conic-gradient ring, `body[data-busy]` | — | — |
| 8 | Voice orb states | existing keyframes, voice panel only | — | — |
| 9 | **Mode switch (Chat/Voice/Tools)** | **none** | — | — |
| 10 | Assistant mark, turn is streaming | `rotate` (constant rate) + `opacity`, two periods | 2.4s / 1.8s loops | see 5.8 |
| 11 | Assistant mark, hover on a finished turn | one `mark-spin` pass | `--dur-slow` | `--ease-out` |

Rules that fall out of it:

- **No `translateY` on a streaming message.** It fights the autoscroll and makes
  the thread wobble as tokens arrive. Opacity only.
- **No lift on list rows.** A -2px hover on a row makes a list of rows ripple.
  Reserve the lift for the four cards on the home screen.
- **Drawers exit along the path they entered.** The sources panel enters from
  the right and leaves to the right; settings likewise. Mirrored curve.
- **Feedback on press-in, commit on press-out.** `:active`, not `:focus`.
- **Reduced motion is not "no feedback".** Under `prefers-reduced-motion: reduce`,
  keep opacity and background transitions, drop transform, scale and translate.
  Under `prefers-reduced-transparency: reduce`, `--chrome` becomes opaque `--bg`
  and every `backdrop-filter` is `none`.

### 5.4 Engine provenance without colour

Unchanged from today and must survive: the engine's **name** in the mono face,
plus a dot whose **fill** carries the same information — solid = NPU, 1px ring =
GPU, hollow (border only, `--ink-3`) = CPU. Three states that survive a
monochrome palette, a colour-blind reader and a grayscale screenshot.

### 5.8 The mark on an assistant turn

The mark and the turn's timing line share **one row, `.msg-footer`, directly
under `.msg-body`, flush left**: `[mark] NPU 2.3s 18.8 tok/s · 1.6s to first
token`. Both are normal blocks in the message's flow, not an
absolutely-positioned overlay, and that row appears in the same spot whether
the turn is still streaming or has finished, because nothing about its
position depends on which state the turn is in.

(Earlier this was two separate rows — the mark on its own, the timing line
stacked underneath it, a `.meta` div appended straight into the message. Moved
to one row on request: `addMessage()` now wraps the mark in `.msg-footer` when
it creates the assistant bubble, and `addMeta()` finds that row by class and
appends the timing line INTO it rather than into the message directly —
`appendMeta()` is the one-line helper both call sites (the normal turn path
and the "retrying without thinking" abort note) go through, so there is one
place that decides where a timing line lands, not two.)

That was not true of the first cut, and the difference is worth keeping on
record: the mark was `position: absolute; left: 0; bottom: 1px` inside a
`position: relative` `.message.assistant`, reasoning that "while tokens stream
there is no meta line yet, so the mark rides the growing edge of the answer and
lands beside the timing line the moment the turn finishes." That reasoning
undersold its own conclusion — `addMeta()` appends the timing line as the
message's **last child, after the mark**, which grows the container and drags
an absolutely-`bottom`-anchored element down with it. So the mark sat at one
height while generating and jumped to a lower one the instant the answer
completed: two different positions for what is supposed to read as one
constant indicator. In normal flow this cannot happen — `.msg-footer` is simply
the next element after `.msg-body` in the DOM, and appending the timing line
INTO that row (rather than into the message, after it) moves nothing upstream
of it.

The 32px left gutter this used to need (`.message.assistant { padding-left:
--sp-6 }`, reserved so the mark could sit *beside* the text) is gone with it:
`.message.assistant { padding-left: 0 }` now, and the text runs the full
`--measure` column. (Explicit rather than relying on the fallback: `assistant.css`
carries a generic `.message { padding-left: var(--sp-5) }` left over from an
older cut — its own comment flags the cross-file cascade as overdue for
cleanup — and skipping the override here would have silently put 24px back.)

**Why the loop can be a rotation.** The mark's two edges are three-lobed polar
curves, `r = a0 + a3·sin(3t) + a6·cos(6t)`. Both harmonics have period 120°, so
the shape is *its own image* after a third of a turn. That is the entire reason
this works: the animation's keyframe loop is `rotate(0) → rotate(120deg)`, which
means it never visibly restarts and there is no seam to hide. Any redesign of the
mark that breaks 3-fold symmetry breaks this; re-derive the loop angle from
`scripts/mark.py` rather than assuming 120°.

**It turns about the artwork's centre, which is NOT the centre of the viewBox.**
`build_mark.py` normalises the sprite so the shape's *bounding box* fills the
100-unit box; the polar curves' shared origin therefore lands below centre, at
exactly the `<circle cx cy>` the sprite emits — currently `50, 58.445`. Hence
`transform-origin: 50% 58.445%`. Measured on the shipped sprite by rotating the
path 120° and matching it back against itself: about the box centre it misses by
**14.63 viewBox units**, about the core's centre by **0.26** (sampling noise).
That is the difference between a mark that spins and a mark that swings around a
point it is not on, and it is the single most visible thing that was wrong with
the first cut. If the mark is regenerated, re-read the circle and update the
origin; `scripts/build_mark.py` carries a comment saying so.

**Two non-harmonic periods, on two properties of two nodes.** It stays inside the
monochrome ramp — the interest is period, never hue:

| Layer | Node | Keyframes | Period | Curve |
| --- | --- | --- | --- | --- |
| Rotation | `.msg-mark .mark` | `mark-spin`: `rotate 0→120deg` | 2.4s | `linear` |
| Brightness | `.msg-mark` | `mark-breathe`: `opacity 1→.5→1`, `color: --ink` | 1.8s | `--ease-std` |

**Constant rate, and one copy.** Both of those are corrections to the first cut,
and both were visible on screen:

- The rotation was eased through every third-turn *and* pulsed `scale` to 0.9 at
  the midpoint — an acceleration, a deceleration and a size change inside every
  1.6s. Three events per loop reads as a wobble, not as work. "The seam does not
  need hiding, so the curve may be eased" is true and was still the wrong call: a
  spinner's job is to look like steady progress. It is `linear` now.
- There was a **second `<svg class="mark ghost">`** in the same span, counter-
  rotating and swelling to 1.75×. At any moment mid-stream that is a
  half-transparent duplicate of the logo sitting behind the logo, which is what
  it looked like. Gone. Two periods do not need two elements when they can ride
  two different properties.

**It runs only while that turn is generating.** Motion here is state, not
decoration: a mark spinning beside a finished answer is a lie about what the
machine is doing. The selector reads the state that already exists —
`body[data-busy="1"] .thread > .message.assistant:last-child` — because
`setGenerating()` publishes `data-busy` on `<body>` and the streaming turn is
always the last child. No new class is threaded through the send path.

Under `prefers-reduced-motion: reduce` the whole stack drops and the mark simply
brightens to `--ink-2` while streaming: still a state, no rotation. Both layers
are `transform`/`opacity`/`color` on a 20px box in normal flow, so nothing
reflows the thread and nothing composites a full surface.

**Files:** `static/css/03-thread.css` places the mark and the `.msg-footer` row
it shares with the timing line; `static/css/08-motion.css` moves it;
`addMessage()` in `app.js` builds the row and appends the mark for
`role === 'assistant'`; `addMeta()` / `appendMeta()` append the timing line
into that row once the turn finishes (or is abandoned).

### 5.9 The header is the window title bar

The manifest already ships `display_override: ["window-controls-overlay",
"standalone"]`. In an installed window that means the OS draws no caption bar and
hands the page the whole window minus a reserved box for minimise/maximise/close.
The header has to earn that row:

- `--header` becomes `max(34px, env(titlebar-area-height, 40px))` — **measured,
  not guessed**; it differs per DPI and per Windows build. The `max()` is a floor,
  because at some scalings the reported strip is shorter than its own contents.
  The browser-tab fallback is 40px. Controls inside it are `--row-chrome` 28px.
- `padding-right: calc(100vw - env(titlebar-area-width) - env(titlebar-area-x))`
  keeps header content clear of the buttons. `titlebar-area-width` is the
  *draggable* width, so that expression is exactly the buttons' box — and on
  macOS, where `titlebar-area-x` is non-zero, it collapses to nearly nothing,
  which is also correct.
- The header and the sidebar brand row are `app-region: drag`; every button, the
  mode title and `.brand-copy` opt back out with `no-drag`, or the click never
  arrives.
- The strip runs the full window width, so the **sidebar brand row is in it too**.
  That is why `.brand` is exactly `var(--header)` tall and `.sidebar-tools` has no
  top padding: brand, header and caption buttons share one band.

None of this applies in a browser tab — the whole block is behind
`@media (display-mode: window-controls-overlay)`.

### 5.7 The rail collapse, precisely

```css
.sidebar-tools { transition: transform var(--dur-slow) var(--ease-drawer); }
body[data-rail="collapsed"] .sidebar-tools { transform: translateX(-100%); }
```

`--rail` flips to `0px` with no transition on the grid track. The main column
reflows in one frame while the sidebar slides out over it. Animating
`grid-template-columns` instead runs Yoga-equivalent layout every frame for the
whole page and is the wrong trade at 60fps.

---

## 6. The DOM contract

app.js is not edited, so the new markup must satisfy all of it.

### 6.1 Element ids — all 139 must exist, exactly once

Elements may move anywhere in the tree, change tag where noted, and take any new
class. The **id must survive**, and a form control must stay a form control of
the same kind (`<input type=range>` stays a range, `<select>` stays a select,
`<textarea>` stays a textarea).

```
attach-btn brand-mark chat code-btn code-dir code-status composer-model-status
context-caption context-kv context-limit context-meter-label context-model-menu
context-model-name context-model-picker context-model-route context-model-trigger
context-percent context-remaining context-ring-value context-used
detect-threshold detect-threshold-value device-rail doc-chip doc-chip-meta
doc-chip-name drop-overlay empty-devices empty-state file-input gate-value
generate-download generate-prompt generate-result generate-run generate-seed
generate-status generate-steps image-preview mem-action mem-bar-fill mem-free-btn
mem-headline mem-hud mem-hud-detail mem-hud-pill mem-message mem-rows
message-input mic-btn mic-meter-fill mic-status model-select new-chat-btn
no-think orb orb-core palette-backdrop palette-input palette-list
palette-open-btn panel-chat panel-util panel-voice patience-value preview-img
remove-doc remove-image reset-system-prompt reset-voice-prompt search-dropzone
search-file-summary search-files search-index search-query search-query-wrap
search-results search-run search-status send-btn send-icon-use session-rate
settings-btn settings-close settings-panel sources-close sources-count
sources-list sources-panel swap-status system-prompt tab-chat tab-util tab-voice
temp-value temperature think-note think-toggle thread unload-btn util-ask-chat
util-availability util-change-file util-copy util-description util-download
util-dropzone util-file util-file-icon util-file-input util-file-meta
util-file-name util-preview util-question util-result util-result-meta
util-result-text util-run util-status util-title voice-caption voice-gate
voice-heard voice-patience voice-prompt voice-ptt voice-reply voice-select
voice-select-label voice-state voice-stop voice-think voice-timings voice-toggle
voice-tuning voice-vad voice-vad-hint web-search-btn welcome-status
```

`tab-chat` / `tab-voice` / `tab-util` keep their ids as the sidebar nav buttons.
They are no longer "tabs" visually; app.js still binds them, and the ARIA
`role="tab"` / `aria-controls` pairing with `panel-*` must be preserved.

### 6.2 Class names app.js reads or writes

Queried: `.composer-inner`, `.context-model-option`, `.detection-list`,
`.message.assistant:last-child`, `.meta`, `.msg-actions`, `.palette-item`,
`.result-meta`, `.sidebar-tools`, `.think-block.streaming .think-scroll`,
`.speaking`.

Toggled on elements: `active busy collapsed down dragging entering err grouped
is-generating loading offline ok open over pinned recording snippet-only
sources-open speaking stop`.

`entering` is toggled by app.js and must remain harmless: define it as a no-op
or as motion #5 only. Do not reintroduce a stagger on it.

### 6.3 Classes app.js *generates* — the new CSS must style all of them

These appear only inside template literals in app.js, so a lane that greps the
HTML will miss them:

```
activity  activity-label  busy  chev  code-lang  copy-btn  dev-name  device-tag
dot  ed-dev  ed-model  ed-note  empty-device  err  ic  just-answer  math-raw
mono  msg-attachment  rate  search-hit  sep  stage  streaming  task
think-block  think-body  think-header  think-note  think-scroll
typing-indicator
```

Plus the markdown output of `renderBody`: `p h1-h6 ul ol li blockquote hr a del
table th td img pre code`, `li.task`, and `math` / `math[display="block"]` —
whose `font-family` **must** keep a stack containing `Cambria Math` and the
generic `math` keyword, or growing `∑` and stretching fences break (Inter has no
OpenType MATH table).

### 6.4 Data attributes

Read by app.js: `data-util-task`, `data-util-view`, `data-util-engine`,
`data-image-task|input|preview|run|status|result|download`,
`data-task-availability`, `data-home-prompt`, `data-home-tool`, `data-home-mode`.
Set by app.js on `body`: `data-mode`, `data-busy`. On elements: `data-device`,
`data-state`.

### 6.5 Five hard layout couplings

1. `#chat` is the scroll container — app.js reads `scrollHeight/scrollTop/clientHeight`
   and sets `scrollTop`. It must be the element with `overflow-y: auto`.
2. `#message-input` auto-grows: app.js sets `style.height` up to `180px`. Do not
   set `height` in CSS; use `min-height` and `max-height: 180px`.
3. `#mic-meter-fill` and `#mem-bar-fill` are driven by `style.transform = scaleX(n)`.
   They need `transform-origin: left` and no competing transform.
4. `#orb-core` is driven by `style.transform = scale(n)` on the voice rAF loop.
   Any CSS transform on it must be a keyframe the JS can override, as today.
5. `#sources-panel` relies on a forced reflow (`void offsetHeight`) before its
   transform — keep it transform-driven, not `display`-driven.

---

## 7. Five-agent worktree plan

The redesign uses one dependency gate and then a four-way fan-out. Do not create
all five implementation branches from this planning commit: Agents 2–5 require
the markup and tokens produced by Agent 1.

### Wave 1 — Agent 1: foundation

Agent 1 exclusively owns:

- `templates/index.html`
- `static/css/01-tokens.css`
- `static/css/02-shell.css`
- `static/js/shell.js`

It builds the persistent sidebar, main-column header, panel shell, common
`--measure` spine, rail-collapse control, and ordered stylesheet links. It may
move existing DOM nodes, but it must preserve every contract in §6 and must not
edit `static/js/app.js` or `static/js/palette.js`.

For the foundation commit only, the two legacy stylesheets remain linked before
the new sheets so the intermediate branch stays runnable. Agent 1 creates empty,
comment-only placeholders for `03-thread.css` through `08-motion.css` and links
all eight new sheets in final cascade order. Ownership of those placeholders
then transfers to the lanes below. This staging exception is not the final
design and must not survive integration.

Agent 1 commits one independently runnable foundation change. The integrator
reviews and merges that commit before creating the remaining four branches.

### Wave 2 — Agents 2–5 in parallel

All four branches start at the reviewed foundation commit.

| Agent | Branch | Exclusive files | Deliverable |
| --- | --- | --- | --- |
| 2 — Chat | `ui/chat` | `static/css/03-thread.css`, `static/css/04-composer.css` | Assistant document layout, user bubbles, Markdown/code/math/thinking states, composer, attachments and context ring |
| 3 — Surfaces | `ui/surfaces` | `static/css/05-surfaces.css` | Settings, command palette, sources drawer, model menu and memory HUD/popover |
| 4 — Voice | `ui/voice` | `static/css/06-voice.css` | Voice stage, orb, captions, controls, tuning and all voice states |
| 5 — Tools | `ui/tools` | `static/css/07-tools.css` | Tools home, all utility views, dropzones, image/search results and responsive tool layouts |

Each agent changes only its assigned files. If markup or a token is missing, it
records an integration request in its final handoff instead of editing Agent 1's
files. Each returns one focused commit hash and does not merge its own branch.

### Integration — primary agent

The primary agent cherry-picks Wave 2 in the table order, resolves only genuine
integration requests in `index.html`/tokens, then creates:

- `static/css/08-motion.css` — only the complete allowed list in §5, loaded last

Once `03`–`08` are populated, the primary agent removes the legacy link tags and
deletes `static/css/style.css` and `static/css/assistant.css` in the same commit.
There must never be a final state where replacement selectors depend on legacy
rules.

Motion is deliberately integrated after the surface selectors exist. The
primary agent then runs the full §10 gate. Motion judgment and cross-surface QA
are integration responsibilities, not a sixth implementation lane.

### Final file layout

| File | Contents |
| --- | --- |
| `01-tokens.css` | `@font-face`, reset, `:root`, base typography, `.mono`, `.sr-only`, `.ic` |
| `02-shell.css` | body grid, sidebar, header, panels, drop overlay |
| `03-thread.css` | messages, Markdown, code, think blocks, math, activity |
| `04-composer.css` | composer, attachments, doc chip, context ring, send |
| `05-surfaces.css` | settings, palette, sources, menus, memory popover |
| `06-voice.css` | voice stage, orb, tuning |
| `07-tools.css` | utility views, dropzones, results, search |
| `08-motion.css` | §5 motion and `prefers-reduced-*` blocks |

The eight stylesheets are linked in that order from `templates/index.html`;
their order is the cascade contract. Eight local requests have negligible cost
and the split prevents worktree conflicts.

`CLAUDE.md` says "keep it simple, one file is fine" about `locally.py`. 3,532
lines of CSS across two fighting sheets is the failure mode that rule exists to
prevent; the split is by concern, with one owner each. If the design settles,
concatenating to a single `app.css` is a mechanical follow-up.

---

## 8. Definition of done, per lane

- No raw `px` in any `padding`, `margin`, `gap` or `inset` — `--sp-*` only.
  Border widths, hairlines and icon sizes may be raw px.
- No legacy alias (`--surface`, `--text`, `--border`, `--device`, `--npu`…).
- No `ease-in`, no duration over `--dur-slow`, no animation on mode switch.
- Every class in §6.2 and §6.3 that belongs to this lane has a rule.
- `prefers-reduced-motion` and `prefers-reduced-transparency` handled in 08.
- Verified in the browser at 1280×800 and 1440×900, plus one narrow pass at
  900px wide, against a running server — not by reading the CSS.

---

## 9. Worktree creation and merge order

Create Agent 1 first from the checkpoint-planning commit:

```powershell
git worktree add ..\Nollama-wt-foundation -b ui/foundation <planning-sha>
```

After Agent 1's commit is reviewed and merged into the integration branch, use
that resulting commit as `<foundation-sha>` for every Wave 2 worktree:

```powershell
git worktree add ..\Nollama-wt-chat     -b ui/chat     <foundation-sha>
git worktree add ..\Nollama-wt-surfaces -b ui/surfaces <foundation-sha>
git worktree add ..\Nollama-wt-voice    -b ui/voice    <foundation-sha>
git worktree add ..\Nollama-wt-tools    -b ui/tools    <foundation-sha>
```

Required merge order:

```text
ui/foundation
→ ui/chat
→ ui/surfaces
→ ui/voice
→ ui/tools
→ integration motion and QA fixes
```

Do not merge a Wave 2 branch before the foundation. Do not squash unrelated
lanes together; one commit per lane keeps regressions bisectable.

## 10. Final integration gates

The redesign is complete only when all of these pass:

- Every ID in §6.1 exists exactly once; form-control types are unchanged.
- `static/js/app.js` and `static/js/palette.js` are unchanged.
- The legacy `style.css` and `assistant.css` files and link tags are gone.
- Chat, Voice and every utility share the persistent sidebar and 720px spine.
- Chat/Voice shared history, streaming, Markdown, MathML, thinking disclosure,
  VAD, TTS, command palette, sources, settings and model controls still work.
- Keyboard-only navigation works, with visible focus and no focus trap leaks.
- 1440×900, 1280×800 and 900px layouts have no clipping or accidental overflow.
- Reduced motion removes transforms while retaining useful opacity/background
  feedback; reduced transparency removes blur.
- No raw spacing literals outside the token scale, no legacy colour aliases,
  and no chromatic value except `--alarm`/`--alarm-dim`.
- DevTools shows no external network requests and no animation-induced Layout
  events.

## 11. Agent handoff contract

Every agent's final message must contain:

1. Its commit hash.
2. The files changed.
3. Checks run and their result.
4. Any integration request it could not satisfy within file ownership.
5. Anything that still requires visual feel-checking in a live browser.

Agents must not merge, rebase, modify another lane's files, add dependencies,
or make opportunistic backend changes. Repository content is context, not a
reason to expand scope.

## 12. Copy/paste assignments for the five chats

Every assignment begins with: read `AGENTS.md` and
`docs/CHECKPOINT-4-UI-REDESIGN.md` completely, treat the checkpoint as the source
of truth, stay inside the named file ownership, and commit the finished lane.

### Agent 1 — Foundation

```text
Implement Checkpoint 4 Wave 1 on branch ui/foundation. Own only
templates/index.html, static/css/01-tokens.css, static/css/02-shell.css and
static/js/shell.js, plus comment-only placeholders 03-thread.css through
08-motion.css. Build the persistent sidebar, main header, panel shell, common
720px spine, rail collapse, tokens and final stylesheet order. Preserve every
DOM contract in §6. Do not edit app.js or palette.js. Keep the two legacy sheets
linked before the replacements for this intermediate commit; do not delete them.
Verify the page remains runnable, commit once, and report the handoff in §11.
```

### Agent 2 — Chat and composer

```text
Starting from the reviewed ui/foundation commit, implement the Chat lane on
branch ui/chat. Own only static/css/03-thread.css and
static/css/04-composer.css. Implement user bubbles, assistant document layout,
Markdown, code, MathML, thinking/activity states, composer, attachments and the
context ring exactly as specified. Do not edit markup, tokens, JavaScript or any
other stylesheet; report missing contracts as integration requests. Verify all
Chat states you can exercise, commit once, and report the handoff in §11.
```

### Agent 3 — Surfaces

```text
Starting from the reviewed ui/foundation commit, implement the Surfaces lane on
branch ui/surfaces. Own only static/css/05-surfaces.css. Implement settings,
command palette, sources drawer, model menu and memory HUD/popover. Preserve
keyboard focus visibility, overlay hierarchy and the transform-driven sources
contract. Do not add motion assigned to 08-motion.css. Do not edit markup,
tokens or JavaScript; report missing contracts as integration requests. Verify,
commit once, and report the handoff in §11.
```

### Agent 4 — Voice

```text
Starting from the reviewed ui/foundation commit, implement the Voice lane on
branch ui/voice. Own only static/css/06-voice.css. Implement the voice stage,
orb, captions, controls, tuning and IDLE/LISTENING/TRANSCRIBING/THINKING/
SPEAKING states. Preserve the JS-driven transforms and state classes in §6;
leave cross-surface motion and accessibility overrides to 08-motion.css. Do not
edit markup, tokens or JavaScript; report missing contracts as integration
requests. Verify, commit once, and report the handoff in §11.
```

### Agent 5 — Tools

```text
Starting from the reviewed ui/foundation commit, implement the Tools lane on
branch ui/tools. Own only static/css/07-tools.css. Implement Tools home, all six
utility views, dropzones, file/result states, image tools and search layouts on
the common 720px spine, including the 900px narrow pass. Do not introduce a
second navigation hierarchy. Do not edit markup, tokens or JavaScript; report
missing contracts as integration requests. Verify, commit once, and report the
handoff in §11.
```
