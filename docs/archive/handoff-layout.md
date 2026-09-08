# Layout and paint: §2.2 / §2.3, measured (2026-09-03)

Follows `docs/handoff-css.md`, which collapsed 39 stylesheets to 9 and left
§2.2 (layout and resizing) and §2.3 (RAM and paint) untouched. Everything
below was measured in a real browser at a real viewport; nothing here is
estimated, and two of the plan's items turned out not to exist.

## The rig, first, because the previous method could not be repeated

`scripts/css-oracle.js` says "paste this into the console on the running app".
That needs a human, a visible pane and a steady hand at three widths, and it
cannot be re-run. Worse, in a hidden pane `document.hidden` is **true**, so
`requestAnimationFrame` never fires and any measurement that waits for a frame
hangs rather than returning — which is not a wrong answer, it is no answer,
and it is easy to mistake for a broken feature.

Two scripts replace it, both dependency-free:

- **`scripts/uiserve.py`** serves `templates/index.html` and `static/` with
  stubs for the handful of endpoints the shell calls at boot. The real server
  spends 10–40 s compiling a model onto a device before it emits a byte of
  HTML, and none of the CSS or DOM work has any reason to pay that — on a box
  that is busy exporting a model it may not be payable at all. `?strip=<id>`
  serves the page with one element removed, which is the only way to ask "did
  adding this element move anything else" and get an answer instead of an
  `nth-child` renumbering.
- **`scripts/uidrive.mjs`** drives the Chromium that ships with WebView2/Edge
  over the DevTools protocol — Node 21+ has a global `WebSocket` and CDP is
  JSON over one socket, so there is no npm dependency and nothing to install.
  A headless page is a **visible** page as far as the page can tell: rAF runs,
  fonts load, `IntersectionObserver` fires. It sets the viewport with
  `Emulation.setDeviceMetricsOverride` (a real layout viewport, which is what
  the oracle's "refuses to run at zero width" guard is about), seeds
  `localStorage` before the app's first line runs, and appends the renderer's
  own `Memory.getDOMCounters` and any console errors to every result.

```bash
python scripts/uiserve.py --port 8779
node scripts/uidrive.mjs --url http://127.0.0.1:8779/ --width 1280 --height 800 \
     --script scripts/probes/smoke.js --set locally-onboarding-complete=1
```

Seeding `locally-onboarding-complete` **before** load matters: set afterwards,
the setup dialog is already open and every measurement is of a shell behind a
modal.

## §2.2 — the sidebar is draggable

`static/js/ui/rail-resize.js`, plus a `role="separator"` in the template and
one block in `layout.css`. Clamp 200–360 px, double-click resets, persisted to
`locally-rail-width`.

The width is **one custom property**. `tokens.css` declares `--rail-open`,
`--rail` derives from it, and body's grid column reads `--rail`, so writing
that one property on the root moves the grid, the sidebar and every rule
measured off them. `body[data-rail="collapsed"]` still sets `--rail: 0px` on
the body, which is closer to the grid than the root is, so collapse keeps
winning over whatever width has been dragged — the two controls answer
different questions ("is the rail there" / "how wide is it") and must not
fight.

Measured at 1280×800:

| | |
|---|---|
| drag 259 → 330 | `--rail-open` 260 → **331 px**, grid `331px 949px`, same frame |
| clamp | 900 → **360**, 10 → **200** |
| keyboard | ArrowRight +16, ArrowLeft −16, Home 200, End 360, Enter → default |
| double-click | back to 260, persisted |
| reload with stored 348 | rail restored to **348 px** |
| collapsed | handle `display: none`, drag refused, re-expand restores |
| announced as | `separator "Resize sidebar" 260 of 200-360` |
| tab order | between `mem-hud-pill` and `rail-toggle` — the boundary it is |

**It moves nothing else.** The page was captured with and without the handle
(`?strip=rail-resizer`) at 1440/1024/375, keyed by element signature rather
than `nth-child` so an inserted sibling cannot masquerade as a diff:
**0 of 776 elements differ at 1440 and 1024**. At 375 two differ by 1 px of
height — and two loads of the *identical* page differ in exactly those two
elements, so that is the noise floor, not the change. Separately, every
selector added by this work matches **nothing** at boot except the handle
itself.

Three things found by measuring rather than by reading:

1. **A persisted width must not follow the window to a narrow one.** A 348 px
   rail carried over from a desktop left a 375 px viewport with **27 px** of
   app column and no control on screen to undo it. The width now applies
   exactly where the handle does — `layout.css` hides the handle below 780 px
   and on coarse pointers, and the module gates on the same two media queries
   and listens for them changing. Below the breakpoint the stylesheet's own
   default (260 px) rules, i.e. the behaviour that was there before.
2. **`transition: none` while dragging was both unnecessary and losing.** The
   only transition the sidebar carries is on `transform` (the collapse
   drawer); the drag moves `width` and the grid column, neither transitioned,
   so the rail lands on the pointer in the same frame. The rule would also
   have lost — the reduced-motion block declaring the transition sits later in
   `layout.css` at equal specificity. Deleted, with the reason left in place.
3. **`setPointerCapture` throws on a synthesised pointerdown.** Capture is
   what keeps a drag alive once the pointer outruns an 11 px strip, which at
   any real speed happens in the first frame — but it is optional, so it is
   wrapped and the drag works without it. Without that, the feature could not
   be tested at all.

The hit area is 11 px and the ink is 2 px, deliberately: the target has to be
catchable, and a permanent 2 px line beside a 1 px border reads as a
scrollbar. Ink appears on hover, focus and drag only.

### One §2.2 defect fixed while auditing tab order

`modeTabs` in `ui/tabs.js` was `[tabChat, tabVoice, tabUtil]` — Code was
missing. `setMode()` gave Code `tabIndex 0` when it was selected, so a
keyboard user could land on it, but no arrow key ever reached it and its own
arrow keys did nothing: a roving tabindex with one tab outside the roll, which
is the case the ARIA tablist pattern exists to prevent. Now all four, in DOM
order; verified that ArrowRight walks chat → voice → code → util → chat, that
Home/End land on the ends, and that exactly one tab holds `tabIndex 0`.

## §2.3 — the thread stops growing

`static/js/chat/thread-window.js`, plus two rules in `chat.css`.

**What is actually expensive is the element tree, not the node in the
thread.** One 2,000-token answer parses into several hundred elements, each
with a computed style, while the markup that produced it is one string. So a
capped message keeps its wrapper — role label, footer, meta line, action row
with its live listeners, about ten nodes — and gives up only `.msg-body`,
which is all the rest of it. Rehydration is one `innerHTML` write of a string
that was never let go of: nothing is re-parsed from markdown and no listener
is rebound. **Chat is still the complete record**; nothing here touches
`chatHistory`.

Measured over a 100-turn session (200 messages) at 1280×800:

| | capped | fully hydrated |
|---|---|---|
| elements under `#thread` | **1,380** | 2,500 |
| renderer DOM nodes (whole document) | **8,621** | 11,683 |
| live event listeners | **261** | 341 |
| `scrollHeight` | 45,357 | 45,357 |

44.8 % fewer elements in the thread, 3,062 fewer nodes in the document, and
**the scroll height does not move by a pixel** between the two states — the
emptied body is pinned to the height it had, measured before it is emptied
(reading it afterwards reads zero, and the thread would then shorten by the
height of every capped message at once, which is the scroll jumping to
somewhere you were not). Scrolling the thread rehydrates every one of them and
leaves **0** empty bodies on screen; the next message re-caps.

The 80-listener difference is exactly the 80 dehydrated bodies: `render.js`
emits `<button class="copy-btn" onclick="copyCode(this)">` per fenced block,
so the count cross-checks the element count.

`performance.memory` moves by 0.06 MB across the same change, which is not a
disappointing result — **DOM nodes do not live on the JS heap**. That is why
the table above reports the renderer's own counters. Do not measure this with
`usedJSHeapSize`.

### `content-visibility`, and the one measurement that decides it

`.message.assistant` now carries `content-visibility: auto` and
`contain-intrinsic-size: auto 240px`. Assistant turns only, deliberately:
`.message.user` is `width: fit-content`, and size containment resolves
fit-content against the intrinsic size rather than the text, so an off-screen
bubble would be laid out at the placeholder width. User turns are a line or
two — nothing to win, and a real way to be wrong.

The trap, and it looked damning: appending 400 messages in one task and then
scrolling through them grew `scrollHeight` by **17,945 px** — *with the thread
cap removed entirely*, so `content-visibility` alone. Every one of those boxes
had been given the 240 px estimate because none had ever been rendered.

The same 400 messages, arriving one at a time and each painted before the next
— which is exactly what the app does, since `addMessage()` auto-scrolls to the
new message — grew `scrollHeight` by **0 px**. The intrinsic-size estimate is
only ever used for a box that has never been rendered, and in this app there
is no such box. A test that appends a thread in a loop is measuring an
artifact of the loop.

## Two of §2.3's items do not exist, and here is the measurement

**"127 `addEventListener` / 0 `removeEventListener`" is a count, not a leak.**
A listener bound once at boot to an element that lives as long as the page has
nothing to tear down, and 12 of the 13 registrations on `document`/`window`
are exactly that. The thirteenth is the setup dialog's focus trap, which is
bound when the dialog opens and is the one `removeEventListener` in the
codebase. The question a teardown helper answers is whether the count *grows*.
Measured with the renderer's own counter:

| | elements | DOM nodes | listeners |
|---|---|---|---|
| boot | 777 | 2,408 | 164 |
| after 80 tab switches | 777 | 2,373 | 162 |
| after 20 settings open/close | 777 | 2,377 | 162 |

Flat, and slightly down. **An `AbortController` per view would be machinery
for a leak that is not there**, and the plan's bullet should be struck rather
than implemented.

**Object URLs are already revoked and stream painting is already batched.**
Every `createObjectURL` in the app has a matching `revokeObjectURL`
(`voice/speech-queue.js`, `util/images.js`, `util/read.js`), and
`markdown/stream-painter.js` has coalesced its writes into one
`requestAnimationFrame` since it was written. Both §2.3 bullets were already
satisfied when the plan was drafted.

## What is next

1. **Container queries (§2.2) are the remaining structural item**, and the
   draggable rail is what makes them necessary rather than tidy: the app
   column can now be 440 px wide at a 1440 px viewport, and every rule that
   asks the viewport is asking the wrong question. Nine of the 22
   width-conditional blocks have subjects inside `.app-column` and should
   move: `base.css` 378/766, `chat.css` 243/1069/1086, `voice.css` 251/917,
   `tools.css` 374/787. The other 13 are chrome, the rail itself, or fixed
   overlays and must stay on the viewport. Note before starting:
   `container-type: inline-size` implies `contain: layout`, which makes the
   element a containing block for absolutely and fixed-positioned descendants
   — `.sources-panel` is `position: fixed` inside the chat panel, so this is a
   behaviour change, not a rename. Run the geometry probe at 375/1024/1440
   plus a dragged rail before and after.
2. Fluid type and the 72ch measure (§2.2) are untouched.
3. `@layer` (`docs/handoff-css.md`) is still a cascade redesign and still the
   only route to the remaining 62 `!important`.
4. The 40-node cap is a constant in `thread-window.js`. It has not been tuned;
   40 is the plan's number and the measurement above is what it buys.
