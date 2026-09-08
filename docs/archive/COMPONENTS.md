# Three components: onboarding, update card, chatbox

Specs for `docs/REBUILD-PLAN.md` §2.1. Every value traces to `DESIGN.md`.
Findings below were read out of the source on 2026-09-02, not inferred.

## Why not shadcn/ui + Tailwind

The `ui-styling` skill is React + Radix + Tailwind + a bundler. locally's
frontend is vanilla ES modules, hand-rolled CSS, self-hosted fonts, no build
step and no npm tree. Adopting it would add React, ReactDOM, Radix primitives,
Tailwind and a build pipeline to a project whose stated premise is "works on a
plane, single origin, self-hosted" — the same trade TODONT already refused for
marked.js (escape-first rendering) and KaTeX (a math webfont). What transfers
is the *thinking*: composition over boolean props, explicit interaction states,
Radix's accessibility contracts, and design tokens. Those are applied below.

Radix is worth reading as a **specification** even so — its Dialog contract is
exactly what §1 is missing.

---

## 1. Onboarding (`setup-shell`)

`static/js/onboarding.js`, markup `templates/index.html:479-504`.

### What is wrong

**1. `aria-modal="true"` with no focus trap.** The dialog declares itself modal
(index.html:480) and nothing keeps Tab inside it. Two consequences, one severe:
assistive tech is told the rest of the page is inert when it is not, and a
keyboard user can Tab straight into the app behind a dialog that on first run
**cannot be dismissed** (`close` is hidden while `setup.needs_assistant`, and
both Escape and the close button check the same flag). The pattern already
exists in this codebase — `palette.js:267` and `swap.js:232` both cycle Tab —
so this is an omission, not a missing capability.

**2. Focus lands on the primary action.** `setOpen()` calls `next.focus()`.
Opening a dialog with Continue focused means a stray Enter advances a step the
user has not read. Focus belongs on the dialog container.

**3. The whole step is a live region.** `#setup-stage` carries
`aria-live="polite"`, so every step change re-announces the entire rendered
step — every card, every option, every hint. Announce the transition, let the
content be read on demand.

**4. Dots claim to be tabs.** `#setup-dots` is `role="tablist"`
(index.html:497) with no `tabpanel` anywhere. They are progress, and a screen
reader is told to expect tab semantics that never arrive.

**5. Errors announce twice and style inline.** `installAndFinish()` writes the
message to `status` (which is `role="status"`) *and* appends a `<p>` carrying
`errorLine.style.color = 'var(--alarm)'`. One announcement, one class.

### The fix

Keep the visual design — CLAUDE.md records why the containers went away and how
the three alignment bugs were found by measuring. This is behaviour only.

```js
// --- focus trap: the Radix Dialog contract, ~20 lines ---
const FOCUSABLE = 'a[href],button:not([disabled]),input:not([disabled]),' +
                  'select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';

function focusablesIn(root) {
    return [...root.querySelectorAll(FOCUSABLE)].filter(el => el.offsetParent !== null);
}

// One handler, registered only while open. `capture` so it beats the app's
// own Tab handlers, matching how the Escape handler above already works.
function onTrapKey(event) {
    if (event.key !== 'Tab') return;
    const items = focusablesIn(dialog);
    if (!items.length) { event.preventDefault(); return; }
    const first = items[0], last = items[items.length - 1];
    const active = document.activeElement;
    if (event.shiftKey && (active === first || !dialog.contains(active))) {
        event.preventDefault(); last.focus();
    } else if (!event.shiftKey && active === last) {
        event.preventDefault(); first.focus();
    }
}

function setOpen(open) {
    shell.hidden = !open;
    shell.setAttribute('aria-hidden', String(!open));
    document.body.dataset.setupOpen = open ? '1' : '0';   // CSS owns overflow
    // Everything outside the dialog really is inert, so the aria-modal
    // claim becomes true rather than aspirational.
    for (const el of document.body.children) {
        if (el !== shell) el.inert = open;
    }
    if (open) {
        returnFocusTo = document.activeElement;
        document.addEventListener('keydown', onTrapKey, true);
        // The DIALOG, not the primary action: opening a step must not put
        // Enter one keystroke away from advancing past unread content.
        dialog.setAttribute('tabindex', '-1');
        requestAnimationFrame(() => dialog.focus());
    } else {
        document.removeEventListener('keydown', onTrapKey, true);
        returnFocusTo?.focus?.();
    }
}
```

`el.inert` does the heavy lifting and is supported everywhere the app runs;
`shell.js:22` already uses it on the sidebar, so the idiom is established.

Step announcement, replacing `aria-live` on the stage:

```html
<div class="setup-stage" id="setup-stage"></div>          <!-- no aria-live -->
<span class="sr-only" id="setup-step-status" role="status" aria-atomic="true"></span>
```
```js
// in render(), after the stage is painted
stepStatus.textContent = `Step ${step + 1} of ${steps.length}: ${steps[step].title}`;
```

Dots become progress, not tabs:

```html
<ol class="setup-dots" id="setup-dots" aria-label="Setup progress"></ol>
```
```js
li.setAttribute('aria-current', i === step ? 'step' : 'false');
li.querySelector('.sr-only').textContent = `Step ${i + 1}: ${steps[i].title}`;
```

Error styling by class, announced once:

```js
status.textContent = message;                 // role="status" announces it
stage.append(el('p', 'setup-note is-error', message));   // silent duplicate
```
```css
.setup-note.is-error { color: var(--alarm); }
body[data-setup-open="1"] { overflow: hidden; }
.setup-shell { overscroll-behavior: contain; }
```

---

## 2. Update card (`update-notice`)

`static/js/updates.js`, styles `static/css/05-surfaces.css:6`.

### What is wrong

**1. The live region is created with its content already inside.** `showNotice`
builds an `<aside role="status">`, fills it, *then* appends it
(updates.js:27-66). A live region has to be in the accessibility tree **before**
its content changes; injecting a populated one is the classic reason a status
never announces. This is the same lesson as the badge rule: one region, mounted
early, updated later.

**2. `×` as button content.** `dismiss.textContent = '×'` (updates.js:59).
`aria-label` saves the accessible name, but DESIGN.md says icons are SVG from
the sprite — `#i-win-close` already exists and is the right glyph.

**3. Two tokens that DESIGN.md collapses.** `--radius-card` and a bespoke
`box-shadow: 0 18px 48px rgb(0 0 0 / .44)`. There is one radius and one
overlay shadow now.

**4. Fixed bottom-left, and it can cover focus.** WCAG 2.2 "focus not
obscured": a persistent overlay must not hide the focused control. It sits over
the composer's left edge at narrow widths.

**5. It appears with no motion,** which reads as a glitch rather than an
arrival.

### The fix

Mount the region empty at load; fill it when the check returns.

```html
<!-- index.html, once, near the end of body -->
<aside class="update-notice" id="update-notice" role="status" aria-atomic="true" hidden></aside>
```
```js
const notice = document.getElementById('update-notice');

function showNotice(sources) {
    const available = Object.entries(sources).filter(([, v]) => v?.update_available);
    if (!available.length) return;
    const fingerprint = available.map(([n, v]) => `${n}:${v.latest}`).sort().join('|');
    if (localStorage.getItem('locally-dismissed-updates') === fingerprint) return;

    // One atomic phrase, not a bare count (the "3 items in cart" rule).
    const summary = available.length === 1
        ? 'An update is available'
        : `${available.length} updates are available`;
    // ... build copy/links exactly as today ...

    const dismiss = document.createElement('button');
    dismiss.className = 'update-dismiss';
    dismiss.type = 'button';
    dismiss.setAttribute('aria-label', 'Dismiss update notice');
    dismiss.innerHTML = '<svg class="ic" aria-hidden="true"><use href="#i-win-close"/></svg>';
    dismiss.addEventListener('click', () => {
        localStorage.setItem('locally-dismissed-updates', fingerprint);
        notice.hidden = true;
        notice.replaceChildren();
    });

    for (const link of links.querySelectorAll('a')) {
        link.append(Object.assign(document.createElement('span'),
            { className: 'sr-only', textContent: ' (opens in a new tab)' }));
    }

    notice.replaceChildren(copy, links, dismiss);
    notice.hidden = false;                       // content change inside a live region
}
```

```css
.update-notice {
    position: fixed;
    z-index: 70;
    left: var(--sp-4);
    bottom: var(--sp-4);
    width: min(360px, calc(100vw - var(--sp-6)));
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: var(--sp-3);
    padding: var(--sp-4);
    color: var(--ink);
    background: var(--bg-overlay);
    border: 1px solid var(--line-strong);
    border-radius: var(--radius);            /* one radius */
    box-shadow: var(--shadow-overlay);       /* one shadow */
    overscroll-behavior: contain;
}
/* Arrives rather than appears. transform + opacity only. */
@media (prefers-reduced-motion: no-preference) {
    .update-notice:not([hidden]) {
        animation: notice-in var(--dur-base) var(--ease-out) both;
    }
}
@keyframes notice-in { from { opacity: 0; transform: translateY(8px); } }

/* Never sit on top of a focused control. */
@media (max-width: 900px) {
    .update-notice { left: var(--sp-3); right: var(--sp-3); width: auto; bottom: calc(var(--sp-4) + var(--control)); }
}
```

---

## 3. Chatbox (`composer`)

`templates/index.html:546-638`, `static/js/ui/composer.js`,
`static/css/04-composer.css` + `14-empty-composer.css` + `34-composer-settings.css`.

### What is wrong

**1. The textarea has no accessible name.** `#message-input`
(index.html:568) carries a placeholder and nothing else. A placeholder is not a
label; it disappears on first keystroke and is not reliably announced. This is
the single clearest WCAG failure in the app's most-used control.

**2. Enter-to-send is undiscoverable.** `composer.js:45` sends on Enter and
newlines on Shift+Enter. Nothing on screen or in the accessibility tree says so.

**3. A forced layout read on every keystroke.** `autosizeInput()` writes
`height`, reads `scrollHeight`, writes `height` again — a deliberate
read/write interleave, once per input event. The comment defending it is right
that the length-based heuristic it replaced was wrong; the answer is not to go
back to a heuristic but to **coalesce to one measurement per frame**.

**4. Three conflicting focus rules.** `.composer-inner:focus-within` is defined
in `04-composer.css:34` (gradient fill), `14-empty-composer.css:85`
(`border-color: --text-faint`) and `34-composer-settings.css:39`
(`border-color: --line-strong`). Load order decides. §2.1 keeps the 04 rule.

**5. Icon buttons without names.** `#attach-btn` (572) and `#mic-btn` (577)
have `title` only.

### The fix

```html
<div class="composer-row">
    <label class="sr-only" for="message-input">Message</label>
    <textarea id="message-input" rows="1"
              placeholder="Ask anything…"
              aria-describedby="composer-hint"
              autocomplete="off" spellcheck="true"></textarea>
</div>
<span class="sr-only" id="composer-hint">Press Enter to send, Shift plus Enter for a new line.</span>
```
```html
<button class="iconbtn" id="attach-btn" type="button"
        aria-label="Attach a file or image" title="Attach a file or image">
    <svg class="ic" aria-hidden="true"><use href="#i-clip"/></svg>
</button>
<button class="iconbtn" id="mic-btn" type="button"
        aria-label="Hold to talk" title="Hold to talk (push-to-talk)" hidden>
    <svg class="ic" aria-hidden="true"><use href="#i-mic"/></svg>
</button>
```

The hint is also worth showing sighted users once, in the footer row that
already exists, at `--fs-xs` / `--ink-3`, hidden after the first successful
send (`localStorage`), so it teaches without becoming furniture.

Coalesced autosize — same arithmetic, same correctness, one measurement per
frame instead of one per keystroke:

```js
let sizePending = false;

export function autosizeInput() {
    if (sizePending) return;
    sizePending = true;
    requestAnimationFrame(() => {
        sizePending = false;
        // Measure from a collapsed box with the gutter hidden — unchanged, and
        // still the only way the height cannot disagree with what is on screen.
        input.style.overflowY = 'hidden';
        input.style.height = `${INPUT_MIN_H}px`;
        const content = input.scrollHeight;
        input.style.height = `${Math.min(Math.max(content, INPUT_MIN_H), INPUT_MAX_H)}px`;
        if (content > INPUT_MAX_H) input.style.overflowY = 'auto';
    });
}
```

Focus, one definition, in `components.css` under `@layer components`:

```css
.composer-inner { padding: var(--sp-2); display: flex; flex-direction: column;
                  border-radius: var(--radius); border: 0; }
/* The surface answers, not the stroke: with no border there is nothing to
   brighten, and a ring would restore the outline this deliberately removes. */
.composer-inner:focus-within {
    background: linear-gradient(var(--fill-hover), var(--fill-hover)), var(--bg-raised);
}
#message-input:focus-visible { outline: none; }   /* the card carries it */
```

### Composition note

The composer's trailing cluster (mic, context meter, send) is the one place a
Radix-style split would pay off in vanilla: it is assembled inline in the
template and its disabled/busy states are set from three different modules
(`send.js`, `generation.js`, `swap.js`). During §2.1, give it one
`setComposerState(state)` with `'idle' | 'busy' | 'blocked'` rather than three
callers each toggling attributes — that is the boolean-prop problem the skill
warns about, expressed in DOM.
