// Dragging the sidebar's width.
//
// In here: the separator between the rail and the app column, its pointer
// drag, its keyboard equivalent, and the one number both write.
// Not in here: collapsing the rail entirely -- that is the toggle in
// shell.js, and it is a different question ("is the rail there") from this
// one ("how wide is it"). The two must not fight: while the rail is
// collapsed this control is not reachable at all.
//
// The width is a single custom property, `--rail-open`. tokens.css defines
// it, `--rail` derives from it, and body's grid column reads `--rail` -- so
// writing one property on the root moves the grid, the sidebar and every
// rule that measures off them, with no second source of truth to keep in
// step. `body[data-rail="collapsed"]` still sets `--rail: 0px` on the body,
// which is closer to the grid than the root, so collapse keeps winning
// whatever width has been dragged.

const KEY = 'locally-rail-width';
// The clamp is REBUILD-PLAN §2.2's. The floor is where the nav labels stop
// fitting beside their icons; the ceiling is where the rail starts taking
// measure from the thread, which is the column that matters.
const MIN = 200;
const MAX = 360;

(() => {
    const root = document.documentElement;
    const body = document.body;
    const sidebar = document.getElementById('primary-sidebar');
    const handle = document.getElementById('rail-resizer');
    if (!sidebar || !handle) return;

    // The default is whatever tokens.css says, read once. Hardcoding 260 here
    // would mean a token change silently stops being the default.
    const declared = parseFloat(getComputedStyle(root).getPropertyValue('--rail-open'));
    const DEFAULT = Number.isFinite(declared) ? declared : 260;

    const clamp = px => Math.min(MAX, Math.max(MIN, Math.round(px)));

    // The width applies exactly where the handle does. layout.css hides the
    // handle below 780px and on coarse pointers; if the stored width kept
    // applying there, a 348px rail carried over from a desktop window left a
    // 375px viewport with **27px** of app column and no control on screen to
    // fix it. Measured, and it is the reason this gate exists rather than a
    // clamp: below the breakpoint the right width is the one the stylesheet
    // already chose.
    const narrow = matchMedia('(max-width: 780px)');
    const coarse = matchMedia('(pointer: coarse)');
    const applies = () => !narrow.matches && !coarse.matches;

    let width = DEFAULT;
    let raf = 0;
    let pending = DEFAULT;

    function paint() {
        raf = 0;
        if (applies()) root.style.setProperty('--rail-open', `${pending}px`);
        else root.style.removeProperty('--rail-open');
        handle.setAttribute('aria-valuenow', String(pending));
    }

    // Crossing the breakpoint has to repaint: the window can be resized, and
    // the desktop shell starts at whatever size it was last left.
    for (const query of [narrow, coarse]) {
        query.addEventListener('change', () => { if (!raf) raf = requestAnimationFrame(paint); });
    }

    // Every write goes through one rAF. A pointermove can fire several times
    // per frame, and each write invalidates a grid that spans the whole
    // window; coalescing means one layout per painted frame instead of one
    // per event.
    function setWidth(px, persist) {
        width = clamp(px);
        pending = width;
        if (!raf) raf = requestAnimationFrame(paint);
        if (!persist) return;
        try { localStorage.setItem(KEY, String(width)); }
        catch { /* Storage may be unavailable in private browsing. */ }
    }

    let saved = null;
    try { saved = localStorage.getItem(KEY); }
    catch { /* Use the token's default. */ }
    const restored = parseFloat(saved);
    if (Number.isFinite(restored)) setWidth(restored, false);

    handle.setAttribute('aria-valuemin', String(MIN));
    handle.setAttribute('aria-valuemax', String(MAX));
    handle.setAttribute('aria-valuenow', String(width));

    // --- Pointer -------------------------------------------------------------

    let startX = 0;
    let startWidth = 0;
    let dragging = false;

    handle.addEventListener('pointerdown', event => {
        if (event.button !== 0 || body.dataset.rail === 'collapsed') return;
        dragging = true;
        startX = event.clientX;
        startWidth = width;
        // data-rail-resizing kills the drawer transition for the duration.
        // Without it every frame of the drag animates towards the previous
        // frame's target and the rail lags the pointer by --dur-slow.
        body.dataset.railResizing = '1';
        // Capture keeps the drag alive when the pointer outruns an 11px strip,
        // which at any real drag speed it does within the first frame. It is
        // optional, though: a synthesised pointerdown has no active pointer to
        // capture and throws, and the drag itself works without it.
        try { handle.setPointerCapture(event.pointerId); }
        catch { /* No active pointer to capture (synthetic event). */ }
        event.preventDefault();
    });

    handle.addEventListener('pointermove', event => {
        if (!dragging) return;
        setWidth(startWidth + (event.clientX - startX), false);
    });

    function endDrag(event) {
        if (!dragging) return;
        dragging = false;
        delete body.dataset.railResizing;
        try { handle.releasePointerCapture(event.pointerId); } catch { /* already released */ }
        setWidth(width, true);
    }
    handle.addEventListener('pointerup', endDrag);
    handle.addEventListener('pointercancel', endDrag);

    // Double-click resets, which is the convention every resizable pane uses
    // and the only way back to the default once it has been dragged.
    handle.addEventListener('dblclick', () => setWidth(DEFAULT, true));

    // --- Keyboard ------------------------------------------------------------

    // A separator that only responds to a pointer is a control a keyboard user
    // cannot operate at all. The step is one --sp-4; Home/End go straight to
    // the ends, so reaching either does not take ten presses.
    const STEP = 16;
    handle.addEventListener('keydown', event => {
        if (body.dataset.rail === 'collapsed') return;
        let next = null;
        if (event.key === 'ArrowLeft') next = width - STEP;
        if (event.key === 'ArrowRight') next = width + STEP;
        if (event.key === 'Home') next = MIN;
        if (event.key === 'End') next = MAX;
        if (event.key === 'Enter') next = DEFAULT;
        if (next === null) return;
        event.preventDefault();
        setWidth(next, true);
    });
})();
