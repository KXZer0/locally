# locally — design lock

The decisions the UI is built from. One source. If a value is not here, it is
not a token; if it is here, do not redefine it in a stylesheet. Change this
file first, then the CSS. Drafted 2026-09-02 from the resolved values in
`01-tokens.css`, checked against the ui-ux-pro-max dataset (style
"Minimalism & Swiss"), and filtered through TODONT.md.

## Direction

Minimal, monochrome, spacious, high contrast. Every affordance is carried by
light (opacity, a hairline, an inversion), never by hue. The only chromatic
value in the app is `--alarm`, reserved for the live microphone and for
things that actually broke. This is the whole brand; do not soften it with
an accent (TODONT "An accent colour").

**Rejected from the generator, on purpose:** AI purple `#7C3AED` + cyan
`#0891B2`, a Google Fonts `@import`, and GSAP. Those are the statistically
safe defaults every AI-built UI converges on, and they are the tell.

## Colour

Pure neutral: R = G = B everywhere. Ink is one white at four opacities;
surfaces are white alphas over `--bg`.

| Token | Value | Use |
|---|---|---|
| `--bg` | `#0A0A0A` | canvas |
| `--bg-raised` | `#121212` | sidebar, composer |
| `--bg-sunken` | `#070707` | code blocks, wells |
| `--bg-overlay` | `#161616` | menus, palette, panels |
| `--ink` | `#FFFFFF` | primary text |
| `--ink-2` | `rgb(255 255 255 / .62)` | secondary text |
| `--ink-3` | `rgb(255 255 255 / .40)` | labels, eyebrows |
| `--ink-4` | `rgb(255 255 255 / .26)` | placeholders, disabled |
| `--line` | `rgb(255 255 255 / .10)` | hairlines |
| `--line-soft` / `--line-strong` | `.06` / `.20` | dividers / focused borders |
| `--fill` / `--fill-hover` / `--fill-active` | `.04` / `.08` / `.12` | control states |
| `--solid` / `--on-solid` | `#FFFFFF` / `#0A0A0A` | the primary button |
| `--alarm` / `--alarm-dim` | `#C4544A` / `rgb(196 84 74 / .16)` | mic live, real errors |

Contrast floor: `--ink-2` on `--bg` is the lowest text pairing allowed for
body copy (≈ 9:1). `--ink-3` is for labels ≥ 13 px only; `--ink-4` never
carries information. Verify with the chromatic audit: 589 elements, exactly
one hit.

**Provenance is a shape, not a colour.** Engine name in the mono face plus a
dot: solid = NPU, ring = GPU, hollow = CPU, and a fourth state for a remote
model (see plan §5.3) that must read as "left the machine" in grayscale.

## Type

Self-hosted, `static/fonts/`, no CDN. Inter 400/500/600 for UI, JetBrains
Mono 400/500 for machine truth (device names, timings, token counts, paths).
`font-display: swap`; preload the two 400 weights only.

| Token | Value |
|---|---|
| `--ui` | `"Inter", "Segoe UI Variable Text", system-ui, sans-serif` |
| `--mono` | `"JetBrains Mono", "Cascadia Code", ui-monospace, monospace` |
| `--fs-xs` … `--fs-2xl` | 11 / 13 / **15** / 16 / 20 / 28 / 36 px |
| body | `clamp(14px, 0.9rem + 0.2vw, 16px)`, line-height 1.5 |
| measure | `72ch` for message bodies; `--measure: 720px` for the shell |

Numbers in tables, timings and the HUD get `font-variant-numeric:
tabular-nums`. Headings get `text-wrap: balance`. Long tokens (URLs, ids,
paths) get `overflow-wrap: anywhere` on a `min-inline-size: 0` child —
never `word-break: break-all` on prose.

The dataset's pairing for this product class is Mono + Sans (it proposes IBM
Plex Sans with JetBrains Mono). Inter is kept as a deliberate choice, not a
default; if it is ever swapped, the replacement is a single decision made
here, and it stays self-hosted.

## Space

4/8 scale, no other values in layout:

`--sp-1..8` = 4 / 8 / 12 / 16 / 24 / 32 / 48 / 72 px

Shell constants: `--header 40px`, `--row 36px`, `--control 40px`,
`--rail-open 260px` (sidebar, draggable 200–360, persisted).

## Radius — one value

`--radius: 12px` and `--radius-pill: 999px`. That is all. The six radii in
the current sheet (6/8/12/16/18/24) are the layered-generations problem in
miniature; bubbles, cards and the composer all take `--radius`, and anything
smaller than a control is square.

## Elevation — one shadow

`--shadow-overlay: 0 12px 32px rgb(0 0 0 / .45)` on menus, the palette and
panels. Nothing else casts a shadow. No `backdrop-filter` on any full-surface
element (the iGPU may be running inference); at most two small overlays may
blur, and the default is none.

## Motion

Three moves and no more: `enter`, hover lift, `sweep` (CLAUDE.md). Only
`transform` and `opacity` animate; `filter: blur()` leaves the `enter` keyframe.

| Token | Value | Use |
|---|---|---|
| `--dur-instant` | 100 ms | press feedback |
| `--dur-fast` | 150 ms | hover, exits |
| `--dur-base` | 200 ms | state changes, cross-fades |
| `--dur-slow` | 280 ms | drawers, first paint |
| `--ease-out` | `cubic-bezier(.16, 1, .3, 1)` | arriving |
| `--ease-std` | `cubic-bezier(.4, 0, .2, 1)` | symmetric |
| `--ease-drawer` | `cubic-bezier(.32, .72, 0, 1)` | sidebar, panels |

Exits run at ~70 % of their enter. Springs only on gesture release (the
sidebar drag). `prefers-reduced-motion` collapses everything; view
transitions are additionally gated on `body[data-busy]` (plan §2.4a). Never
`transition: all`.

## Focus

`:focus-visible` only, never bare `:focus`, never `outline: none` without
the ring below on the same element:

```css
:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
```

The composer is the one exception: its card carries a `:focus-within` ring
and the textarea inside it suppresses its own. Sticky chrome must not cover a
focused control: `scroll-padding-top: var(--header)` on the scroll root, and
overlays close before focus can land behind them.

## Controls

- Pointer target ≥ 24 × 24 CSS px, touch ≥ 44; expand the hit area, not the glyph.
- Icon-only buttons carry `aria-label`; decorative `<svg class="ic">` beside
  text carry `aria-hidden="true"` (42 of 43 currently do not).
- `touch-action: manipulation` on all controls; `-webkit-tap-highlight-color: transparent`.
- One primary action per surface: the white plate. Everything else is ghost.
- Destructive actions confirm or offer undo; they do not use `--alarm` unless
  something is *wrong* (stop is not an error; remove-attachment is not an error).

## Status and live text

Async status (`swap-status`, `mic-status`, util statuses, `speaker-status`)
is **one** `role="status" aria-atomic="true"` region per surface, announcing a
complete phrase ("Model loaded: Qwen3-8B"), never a bare number and never
several competing live regions. Loading copy ends with `…`.

## Copy

Second person, active voice, specific verbs on buttons ("Read locally", not
"Continue"). Errors state the cause and the next step. Use `…` not `...`,
curly apostrophes, `&nbsp;` inside `Ctrl K`, `512×512`, `50 pages`. No
marketing slogans anywhere in the shell.

## Anti-patterns (the audit's list, kept here so it does not drift)

heavy chrome · box inside box inside overlay · glass + glow · a second hue ·
Inter on a CDN · six radii · decorative shadows · `!important` · `outline:
none` alone · `transition: all` · a status region per widget · images without
`width`/`height` · placeholder as the only label · slides between tabs.
