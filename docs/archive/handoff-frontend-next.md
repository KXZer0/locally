# Frontend, what is actually next (2026-09-03)

Reads `docs/handoff-css.md` (39 sheets to 9) and `docs/handoff-layout.md`
(§2.2/§2.3) and re-measures them on today's tree, which is not the tree either
was written against: the backend split (`55f73a9`), the GPU reserve work and
the Podman sandbox all landed on top of the UI commit. Everything below was
measured with the rig those documents introduced. Four claims in them turned
out to be wrong, one of them a blocker that does not exist.

## The baseline reproduces, so the two handoffs can be trusted on their own terms

`88949ef` is an ancestor of `HEAD`, and the numbers it recorded are the numbers
the page produces today, after three merges:

| | handoff | today |
|---|---|---|
| elements at boot | 777 | **777** |
| renderer DOM nodes | 2,408 | **2,408** |
| live listeners | 164 | **164** |
| stylesheets | 9 | **9** |
| grid at 1280 | `260px 1020px` | **`260px 1020px`** |

Nothing in the backend split disturbed the shell.

Those figures are the **fallback** boot path's, for the reason Defect 1 gives.
With `/v1/ui/bootstrap` stubbed the real path builds **779 elements and 2,428
nodes**, listeners unchanged at 164, stable across three runs. Two elements and
twenty nodes is not a finding on its own; it is the correct baseline to compare
the next change against.

## Defects 1 and 2 are fixed. What follows is what they turned out to be.

## Defect 1 - every measurement so far was taken on the fallback boot path

`static/js/system/bootstrap.js` fetches `/v1/ui/bootstrap`, which answers
health, models, available models and memory in one round trip, and falls back
to the four public endpoints when that fails. `scripts/uiserve.py` stubs the
four and **not** the one, so under the rig the first fetch 404s every time and
the page boots through the `catch`. Measured:

    document.documentElement.dataset.bootstrapSource === "fallback"

The endpoint has existed since `389bf30`, so this was true for the CSS collapse
and for §2.2/§2.3 as well. It does not invalidate the geometry results - the
same DOM is built either way - but it does mean no boot-timing figure taken on
this rig has ever described the path the app actually takes, and one failed
request sits inside every capture.

**Fixed.** `uiserve.py` now serves `/v1/ui/bootstrap` composed from the same
four payload constants it already defined, so the fallback and the fast path
cannot disagree here in a way they would not disagree for real. Verified:
`bootstrapSource` reads `combined`, three runs, 779 elements / 2,428 nodes /
164 listeners each time.

## Defect 2 - the thread's pinned heights go stale the moment the rail moves

`thread-window.js` pins a dehydrated body to the height it had *at dehydration
time*, and that is what makes the cap free: `handoff-layout.md` measures
`scrollHeight` unchanged to the pixel between capped and hydrated. That result
holds only at a fixed width, and §2.2 shipped a control whose whole purpose is
to change the width.

Measured at a 900px viewport, a 60-turn thread, dragging the rail 200 to 360:

| | |
|---|---|
| body width | 668 to **508 px** |
| one sample body, pinned height | 358 px |
| the same body, actually rendered at 508 px wide | **382 px** (error +24) |
| `scrollHeight` with stale pins | 30,296 |
| `scrollHeight` once every pin is corrected | **31,682** |
| **drift** | **+1,386 px** |

So after a drag the document under-reports its own height by roughly a
screenful and a half, and pays it back in jumps as the reader scrolls up
through the capped region - which is the exact failure the height pin was
written to prevent. Nothing is wrong with the pin; what is missing is that
anything which changes the thread's width invalidates it. The same applies to a
plain window resize, which the module also does not watch.

At a 1280px viewport the same drag produced a drift of **0 px**, because the
thread does not reflow there and 100px of rail does not reach the text. That is
why this was not caught: it only appears once the column is narrow enough to
reflow, which is precisely where the draggable rail can now put it.

### The pin was the smaller half, and by a long way

The obvious repair — a `ResizeObserver` on the thread that rehydrates every
capped body and lets `capThread()` re-pin at the new width — was written first
and **moved the drift from 1,386 px to 1,313**. The pins themselves came out
exactly right: measured across all 40, `real - pinned` summed to **0**. So
something else was holding the missing 1,313 px, and re-pinning could never
have found it.

It is `content-visibility: auto`. `contain-intrinsic-size: auto 240px` makes an
off-screen assistant turn contribute *the height it last had* to `scrollHeight`,
and the height it last had is the height it had **at the width it last had**.
Nothing invalidates a remembered size when the column changes width, and the
turns holding stale ones are not only the 40 capped bodies — they are every
assistant message off screen, capped or mounted. Isolated by forcing a real
layout of the thread with the cap untouched:

| | |
|---|---|
| `scrollHeight` after the drag, as reported | 27,146 |
| the same document with `content-visibility: visible` | **28,542** |
| attributable to remembered sizes | **1,396 px** |

and it stays corrected once the flag comes off again.

**The fix, therefore, is both halves.** `thread-window.js` grows a
`ResizeObserver` (§2.2 budgets one "only where JS must know size"; this is such
a place) which, 150 ms after the last width change, rehydrates every capped
body, sets `#thread[data-remeasuring]` so `chat.css` turns `content-visibility`
off for one painted frame, re-caps under it, and restores the reader's scroll
position from an anchor taken beforehand.

**A remembered size is recorded when the box is rendered, not when it is laid
out.** Setting and clearing the attribute inside one task, with a forced
`scrollHeight` read between them, corrects nothing — that was the 73 px version.
The flag has to survive a painted frame, so it is dropped in a double
`requestAnimationFrame`. This is the single non-obvious line in the change and
the reason it is written down here.

Measured, same 900px viewport and 60-turn thread, rail 200 to 360:

| | before | after |
|---|---|---|
| drift after the drag settles | 1,386 px | **9 px** |
| the anchored message moves by | — | **−1 px** |
| worst frame during the correction | — | **45.6 ms**, one frame, median 8.3 |

Nothing else moved. The §2.3 result reproduces unchanged at 1280 (1,380 vs
2,500 elements, 44.8 %, `scrollHeightMoved` 0, `emptyOnScreen` 0), the rail
probe passes, and boot is 779/2,428/164 as before. Two edge cases were
measured rather than reasoned about: switching to the Voice tab takes the
thread to width 0, which the observer ignores (20 capped bodies before, during
and after; `scrollHeight` identical; **0** bodies pinned to a zero-width
layout), and after `resetThreadWindow()` a fresh thread is watched again — the
next drag grows the document by 995 px at the settle and drifts **−5 px**.

The 9 px that remain are per-body rounding, an order of magnitude below the
height of one line.

## Correction 1 - the stated blocker for container queries does not exist

`handoff-layout.md` warns that `container-type: inline-size` implies
`contain: layout`, making the element a containing block for fixed descendants,
and that `.sources-panel` is `position: fixed` **inside the chat panel**, so
the change is a behaviour change rather than a rename.

`.sources-panel` is at `templates/index.html:983`. `#panel-chat` opens at 493.
It is not inside the chat panel, and a sweep of the rendered page finds
**zero** `position: fixed` descendants inside any of the four panels: all of
the app's fixed elements (`settings-panel`, `setup-shell`, `drop-overlay`,
`sources-panel`, `palette-backdrop`, and `mem-hud` in the sidebar) live outside
them. Applying `container-type: inline-size` to `.app-column > .panel` in the
live page moves `.sources-panel` by 0 px on every axis.

Container queries are unblocked. Run the geometry probe anyway - containment
has other effects - but the specific trap named is not there.

## Correction 2 - the number that motivates container queries is bigger than the one recorded

The handoff says the app column "can now be 440 px wide at a 1440 px viewport".
Measured, at 1440 with the rail at its 360px maximum and the sources panel
open, the chat content is **792 px**. 452 px is what happens at a **1100 px**
viewport under the same conditions. The useful figure is neither: it is the
constant error between what the viewport queries read and what the chat column
actually is, and that is **648 px** at both widths - 360 px of rail plus 288 px
of sources padding.

The consequence is concrete. At a 1100px viewport with both open, the composer
and the message bodies are 452 px wide, `(max-width: 780px)` does not match,
`(max-width: 720px)` does not match, and `(min-width: 1100px)` *does* - so a
452px column is being laid out by the widest rule set the app owns. That is the
argument for moving the nine `.app-column`-internal blocks onto the container,
and it is stronger than a 440px anecdote.

## Correction 3 - `!important` is 38, and 26 of them are not the kind you remove

`handoff-css.md` closes on "the `!important` count is 62, not 0", and makes the
`@layer` cascade redesign the route to the rest. Today the count is **38**, and
reading them:

| where | count | what |
|---|---|---|
| `base.css` | 26 | 12 in `prefers-reduced-motion` blocks, 9 in `.sr-only`, 4 `transform: none`, 1 `[hidden]` |
| `components.css` | 6 | 5 reduced-motion `transform: none`, 1 real conflict (`font-size: var(--fs-xs)`) |
| `chat.css`, `layout.css`, `tools.css` | 6 | all reduced-motion `transform: none` |

`prefers-reduced-motion` overrides and the `.sr-only` clip block are the two
canonical, correct uses of `!important` - a reduced-motion rule that can be
beaten by a more specific animation rule is a broken accessibility feature, and
layers do not help, because the whole point is that it must win regardless of
where it is filed. **One declaration in the whole app is a specificity conflict
a cascade redesign would fix.**

`handoff-css.md` already measured that re-bucketing the nine sheets into layers
moves 42 of 872 elements. Spending that blast radius on one `font-size` is not
a trade worth making. The item should be struck, or narrowed to "delete the one
`!important` at `components.css:2220` by giving its selector the specificity it
needs" - a two-line change with no cascade redesign attached.

## What is worth doing, in order

**1. ~~Stub `/v1/ui/bootstrap` in `uiserve.py`.~~ Done.** See Defect 1. Every
figure in this document below the baseline table was taken after it.

**2. ~~Re-measure the thread on width change.~~ Done.** See Defect 2. The
probe is `scripts/probes/thread-resize.js`; it must be run at `--width 900`,
because at 1280 the thread does not reflow and it reports a clean zero for the
wrong reason.

**3. Container queries, §2.2's remaining structural item.** Nine blocks:
`base.css` 378/766, `chat.css` 243/1069/1086, `voice.css` 251/917,
`tools.css` 374/787. Unblocked (Correction 1), and justified by a 648 px error
rather than an anecdote (Correction 2). The other fourteen width blocks are
chrome, the rail, or fixed overlays and stay on the viewport. Note the count is
23 today, not the 22 recorded - `layout.css:1135` is a second 780px block.

**4. §2.5, the template split, and it is the largest number on the table.**
**83.8 % of the boot DOM is hidden**: 651 of 777 elements, and `templates: 0`.
`panel-util` alone is 165 elements, `settings-panel` 95, `panel-voice` 93,
`speaker-panel` 36, `setup-shell` 30, and five util views are 20-25 each. A
hidden element is cheap to lay out and not free to parse, style or keep, and
`?strip=` cannot answer this one - `_strip_element` is documented as handling
only flat, non-nested elements, and every one of these is nested. Converting
the five util views, the speaker panel and the setup shell to `<template>`
cloned on first open is the §2.5 bullet; the measurement to take before and
after is `Memory.getDOMCounters` plus the boot mark the app already emits
(`locally-bootstrap`, in `bootstrap.js`) - which item 1 has to be fixed for
first.

**5. §2.4 motion.** `motion.css:47` still animates `filter: blur()` on every
message enter, which §2.4 says to remove because it forces a paint per frame on
three elements per message, on a machine whose GPU may be running inference.
View transitions (§2.4a) have a complete recipe written and nothing
implemented - `startViewTransition` appears nowhere in `static/js/`.

**6. Fluid type and the 72ch measure**, untouched. Today's measures are 46ch
and 52ch in four places and one `clamp()` on a display heading; the body type
is fixed. This is the item to do *after* container queries, not before: a
measure expressed against a container it cannot see is the same bug twice.

**7. Struck: the `@layer` redesign** (Correction 3), until something other than
the `!important` count motivates it.

## Two things measured and dismissed, so they are not re-opened

**A dehydrated body does not blank while you are reading it.** `capThread()`
walks back from the cap edge and stops at the first already-capped body, so a
message rehydrated deep in the thread is never revisited by a later append.
Verified: scroll to an old assistant turn, let it rehydrate, append a new
answer - it stays hydrated and the scroll does not move.

**Rehydration keeps up with a real scroll.** Scrolling a 60-turn thread upward
300 px at a time, 76 samples, counting bodies with more than 50 px on screen:
**0 visibly blank at every sample**, and 0 left dehydrated at the end. A single
instantaneous jump can leave one body blank with a 9 px sliver at the viewport
edge, which is the observer's `150%` margin behaving correctly and not a
defect.

## Stale outside the frontend, found while checking the tree

`CLAUDE.md`'s first architecture bullet still reads "`locally.py` - Flask
server, DeviceSlot class per device", and the development-preferences section
says "`locally.py` stays the entry point and the Flask app". Since `55f73a9`,
`locally.py` is 32 lines that import `create_app` from `core/app.py` and `main`
from `core/cli.py`; the app is 14,750 lines across `core/`. The entry-point
half is still true, the Flask-app half is not.
