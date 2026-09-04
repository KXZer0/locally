// One helper for every view transition, so the reasons to skip are checked once.
//
// §2.4a. The React `<ViewTransition>` component does not apply -- there is no
// React -- but the browser API it wraps does. The shell is WebView2 (Chromium);
// Firefox 144+ and Safari 18.2+ cover the LAN case, and anywhere else this is
// a plain function call with no animation, which is the correct fallback rather
// than a degraded one.

// Three reasons to skip, and the third is the one that matters here.
//
// `prefers-reduced-motion` is gated in JS rather than CSS on purpose: the CSS
// route needs `!important` on `::view-transition-*` to beat the UA sheet, and
// the frontend is trying to drive that count down, not up.
//
// `data-busy` is set by setGenerating() while tokens stream. A view transition
// snapshots the old and new document, and taking that snapshot over a long
// thread while the iGPU is mid-inference is exactly the compositing cost the
// project's motion rules exist to avoid. A tab switch during a turn therefore
// happens instantly, which is also what a user pressing it mid-answer wants.
//
// Read the VALUE, not the attribute's presence. setGenerating writes
// `dataset.busy = on ? '1' : '0'` -- it never removes it -- so the
// hasAttribute() form the §2.4a recipe suggests is true forever after the
// first turn, and silently disabled every transition for the rest of the
// session. The CSS already keys on `body[data-busy="1"]`; this now agrees
// with it. Caught by running a real turn: the probe set and removed the
// attribute itself, so it tested my assumption rather than the app.
export function transition(mutate) {
    const skip = !document.startViewTransition
        || matchMedia('(prefers-reduced-motion: reduce)').matches
        || document.body.dataset.busy === '1';
    if (skip) {
        mutate();
        return null;
    }
    try {
        return document.startViewTransition(mutate);
    } catch {
        // A duplicate view-transition-name among rendered elements throws.
        // The mutation still has to happen -- losing the tab switch would be a
        // far worse failure than losing its cross-fade.
        mutate();
        return null;
    }
}
