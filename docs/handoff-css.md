# CSS collapse: 39 sheets → 9, measured (2026-09-03)

Implements `docs/REBUILD-PLAN.md` §2.1 and the two accessibility defects §2.1a says
to fix "before any restyling". Everything below is measured on the running app, not
estimated.

## What shipped

| | before | after |
|---|---|---|
| stylesheets | 39 | **9** |
| rules | 1,108 | 1,212 sites, 1,194 in the CSSOM |
| declarations | 4,307 | **3,811** (496 overridden ones dropped) |
| bytes | 229,323 | **222,717** |
| `!important` declarations | 95 | **62** |
| chromatic values | 1 hue | **1 hue** (`--alarm`, plus its `--alarm-dim`) |
| computed-style diff, 872 elements × 3 widths | — | **0** |

`static/css/{tokens,base,layout,components,chat,voice,tools,motion,responsive}.css`.
Regenerate, never hand-edit:

```bash
python scripts/css_collapse.py --order templates/index.html.bak39 --split static/css
```

`scripts/css-load-order.txt` records the 39 filenames in the exact order that defined
the cascade; the files themselves are recoverable from git at `c6a8c9d`.

## Two of the plan's targets were wrong, and here is the measurement

**"≤60 KB CSS" is not reachable and should not be the goal.** The 39 sheets are
229 KB, of which **85.6 KB (38%) is comments** — and in this repo the comments are
the reasoning, which TODONT.md exists to preserve. Of the remaining 137 KB of code
there is almost nothing dead: a coverage sweep over every rule found 328 that match
no element in the rendered page, and all 78 of the class names involved
(`msg-body`, `think-block`, `source-item`, `stream-tail`, `setup-progress`, …) are
created by JavaScript at runtime. **Nothing is dead; the app simply has 1,194 rules.**
Deduplication is worth 496 declarations (11.5%), and that is the honest ceiling for a
port. 60 KB would require deleting every comment and then rewriting the UI.

**`@layer` cannot be adopted by re-bucketing existing rules.** This was tried and
measured: the same nine files, wrapped in `@layer tokens, base, layout, components,
views, utilities, overrides`, moved **42 of 872 elements**. The reason is structural —
a later layer beats an earlier one *regardless of specificity*, so every
cross-file pair changes meaning, not just the equal-specificity ones. A constraint
analysis over the rendered DOM found 167 ordering constraints binding 213 rules
(82% of rules are order-free), but layers do not respect specificity, so that
analysis understates the blast radius. **Layers are a redesign of the cascade, not a
port of it.** The sheets are shipped unlayered and each carries its intended layer in
its header comment, so the redesign can be done deliberately, one layer at a time,
with the oracle watching.

## The method: port by measurement

`scripts/css-oracle.js` records, for every element in the page, a hash of every
computed property that can change what is drawn (~110 of the 543 the CSSOM exposes —
the full set is 1.5M reads and takes over a minute). Two stylesheet sets are
equivalent iff every hash matches. Four things it took to make that trustworthy:

1. **A real viewport.** Resizing `documentElement` does not move a media query, and
   in a hidden pane `innerWidth` is **0**, which makes every `max-width` query match.
   The first 91-element "regression" was measured that way and was fiction.
   `capture()` now refuses to run at zero width.
2. **`await document.fonts.ready`.** With `font-display: swap`, a capture taken
   before the woff2 lands measures fallback metrics. That produced 94 diffs whose
   only property was `width`.
3. **Two independent page loads, not a live stylesheet swap.** Swapping `<link>`
   elements restarts the `enter` animations, so a third of the "diffs" were
   `opacity: 0 → 1` mid-animation.
4. **A control.** Two loads of the *identical* page differ by **1 element** — the
   sidebar spinner, which animates. That is the noise floor, and the final result
   sits exactly on it.

## Three real defects the oracle caught, and what each taught

**1. Merging a selector to one place moves it forward in the cascade.** The first
resolver merged each selector into a single rule at its last occurrence. That carried
`.mono { font-size: .85em }` from `10-tokens.css` past
`.util-availability { font-size: 10px }` in `16-utilities.css` — equal specificity, so
order decides — and 13 elements went from 10px to 12.75px. **Dropping losers is safe;
moving winners is not.** Every surviving declaration now stays at its own source
position, and only adjacent occurrences of one key are folded.

**2. `@font-face` is not a selector.** Eleven `@font-face` blocks are eleven faces,
not one face declared eleven times. Merging them by name left a single face; every
string in the app was then measured in a fallback font, which showed up as 29 elements
whose only difference was `width`. At-rules are keyed by position and never folded.

**3. Hoisting media queries into `responsive.css` grants them precedence they never
had.** At 375px this moved 29 elements: `.util-nav-item` lost its 8px padding,
`.quick-action` shrank 62px → 46px, `.empty-state` lost 60px of margin — each a media
rule written *before* a base rule that used to beat it. A media rule is now filed with
the component it modifies. **`responsive.css` is empty by design**, and that is the
honest shape of this app's CSS: it has no separable responsive layer, it has
components that change at width. (Under `@layer overrides` the hoist would be correct
by construction — see above.)

Plus two genuine cross-concern conflicts, both pinned rather than "fixed":

- `.engine-overview .ed-dev` vs `.empty-device .ed-dev` — same component filed under
  two concerns. Both now live in `chat.css`. `.ed-*` moved from components to chat.
- `input.palette-input` deliberately sets `border-radius: 0` with the comment *"the
  shared control rule rounds it; flush here"* — and deliberately **loses**, because
  `input[type="text"]` is written later. The comment records an intent that load order
  defeated, and today's palette input is rounded. Granting the intent is a design
  decision, not a port, so the rule is pinned back into `base.css` where the outcome is
  unchanged, and listed in `PIN` in the tool with this note.

## Accessibility (§2.1a), done first as the plan requires

Verified in the running app, not by reading:

- **The setup dialog's `aria-modal="true"` is now true.** `focusablesOutsideDialog
  StillReachable: 0`. Tab from the last control wraps to the first and Shift+Tab wraps
  back. Focus opens on the dialog, not on Continue — so a stray Enter cannot advance
  past unread content. Two bugs found while testing my own fix: inerting
  `document.body.children` inerted the dialog too (it lives inside `.app-column`, not
  under `<body>`), so `dialog.focus()` silently did nothing; and inerting the button
  that opened setup *before* reading `document.activeElement` destroyed the
  return-focus target. Inert is now an ancestor-path walk, applied before focus moves,
  with a `MutationObserver` on that path so the update card and toasts cannot appear
  live behind the dialog.
- `#setup-stage` no longer carries `aria-live`, which re-announced every card, option
  and hint on each step. A visually-hidden `role="status"` announces the transition:
  *"Step 2 of 5: Choose its mind."*
- The dots were `role="tablist"` with no tabpanel anywhere. They are a labelled group
  of buttons carrying `aria-current="step"` — they really do navigate, so they stay
  buttons.
- Errors were announced twice (once by `role="status"`, once as an appended
  `<p>` with an inline `style.color`). One announcement; the visual twin is
  `.setup-note.is-error`. The scroll lock moved from an inline style to
  `body[data-setup-open="1"] { overflow: hidden }`, so one place owns it and a throwing
  close path cannot leave it stuck.
- **`#message-input` has an accessible name** (`<label class="sr-only">Message`) and
  `aria-describedby` pointing at *"Press Enter to send, Shift plus Enter for a new
  line"* — the send shortcut was undiscoverable. `#attach-btn` and `#mic-btn` had
  `title` only; both now have `aria-label`, and their glyphs are `aria-hidden`.
- `autosizeInput()` coalesces its write/read/write to **one measurement per frame**
  instead of one per keystroke. Same arithmetic, same correctness.

## What is next

1. **The `!important` count is 62, not 0.** The collapse dropped the 33 that were
   losing anyway. The rest need the cascade redesign, since each exists to defeat a
   specificity conflict that layers are the proper answer to.
2. **Layers, deliberately.** Start with `tokens` and `base` — the two whose contents
   genuinely should lose to everything — and verify each with the oracle before adding
   the next.
3. **§2.2/§2.3 are untouched**: draggable sidebar, container queries, the 40-node
   thread cap, listener teardown.
4. `templates/index.html.bak39` is kept only so the tool can regenerate. Delete it
   once the 9 sheets are the source of truth, and drop `--order`.
