# Probes

Scripts for `scripts/uidrive.mjs`. Each is evaluated as the body of an async
function in the page and returns a JSON-able object; `uidrive` appends the
renderer's `Memory.getDOMCounters` and any console errors to it.

```bash
python scripts/uiserve.py --port 8779 &
node scripts/uidrive.mjs --url http://127.0.0.1:8779/ --width 1280 --height 800 \
     --script scripts/probes/<name>.js --set locally-onboarding-complete=1
```

| probe | question it answers |
|---|---|
| `smoke.js` | does the shell boot, at a real viewport, with no errors |
| `geom.js` | every element's box and key paint properties, keyed independently of `nth-child` so an inserted sibling is not a diff. Run against `?strip=<id>` for the "did adding this move anything" comparison |
| `newrules.js` | which of the recently added selectors match anything at boot |
| `rail.js` | the sidebar drag: clamp, keyboard, double-click, persistence, collapsed |
| `narrow.js` | what a persisted rail width does to the app column at each width |
| `ink.js` | the handle's ink, the drag cursor, and the narrow-viewport hide |
| `a11y.js` | the mode tablist's roving tabindex, and the separator's place in the tab order |
| `thread.js` | the 40-message cap over a 100-turn session: nodes, scroll height, rehydration |
| `thread-phase.js` | the same session stopped at one phase (`--arg capped\|hydrated`) so the renderer's node and listener counters can be read per phase |
| `thread-resize.js` | whether a width change corrects the heights the thread is holding, and whether it moves the reader. **Run at `--width 900`** — at 1280 the thread does not reflow and the probe reports a clean zero for the wrong reason |
| `listeners.js` | whether the listener count grows across 80 tab switches and 20 panel opens (`--arg boot\|tabs\|panels`) |

The noise floor is real and small: two loads of the identical page at 375px
differ in exactly two elements (`.quick-action` and its grid, by 1px of
height). Anything at or below that is not a change.
