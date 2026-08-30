(() => {
    const body = document.body;
    const toggle = document.getElementById('rail-toggle');
    const modeTitle = document.getElementById('mode-title');
    const sidebar = document.getElementById('primary-sidebar');
    const contextMeter = document.getElementById('context-meter');
    const contextTip = document.getElementById('context-meter-tip');
    const contextModelMenu = document.getElementById('context-model-menu');
    const memHud = document.getElementById('mem-hud');
    const memHudPill = document.getElementById('mem-hud-pill');
    if (!toggle || !modeTitle) return;

    const STORAGE_KEY = 'locally-rail';
    const MODE_TITLES = { chat: 'Chat', voice: 'Voice', util: 'Tools' };

    function setCollapsed(collapsed, persist = true) {
        if (collapsed) body.dataset.rail = 'collapsed';
        else delete body.dataset.rail;
        toggle.setAttribute('aria-expanded', String(!collapsed));
        toggle.setAttribute('aria-label', collapsed ? 'Expand sidebar' : 'Collapse sidebar');
        if (sidebar) {
            sidebar.inert = collapsed;
            sidebar.setAttribute('aria-hidden', String(collapsed));
        }
        if (!persist) return;
        try { localStorage.setItem(STORAGE_KEY, collapsed ? 'collapsed' : 'expanded'); }
        catch { /* Storage may be unavailable in private browsing. */ }
    }

    function syncModeTitle() {
        modeTitle.textContent = MODE_TITLES[body.dataset.mode] || 'Chat';
    }

    function syncContextMeter() {
        if (!contextMeter || !memHud) return;
        const state = memHud.dataset.context;
        if (state) contextMeter.dataset.context = state;
        else delete contextMeter.dataset.context;
        contextMeter.classList.toggle('busy', memHud.classList.contains('busy'));
        contextMeter.classList.toggle('down', memHud.classList.contains('down'));
    }

    // The composer ring shows a fraction; its numbers are the ones app.js
    // already writes into the sidebar memory panel. Mirroring them on hover
    // keeps a single writer -- a second copy of updateContextDisplay() would be
    // two things to keep in step, and the one that drifts is always the copy.
    const TIP_SOURCES = {
        percent: 'context-percent',
        used: 'context-used',
        limit: 'context-limit',
        remaining: 'context-remaining',
        rate: 'session-rate',
        caption: 'context-caption',
    };

    function syncContextTip() {
        if (!contextTip) return;
        for (const [slot, sourceId] of Object.entries(TIP_SOURCES)) {
            const target = contextTip.querySelector(`[data-tip="${slot}"]`);
            if (!target) continue;
            const source = document.getElementById(sourceId);
            target.textContent = source ? source.textContent : '—';
        }
    }

    function detachModelMenuFromMemoryPopover() {
        if (!contextModelMenu || contextModelMenu.hidden || !memHud || !memHudPill) return;
        memHud.classList.remove('pinned');
        memHudPill.setAttribute('aria-expanded', 'false');
    }

    let saved = null;
    try { saved = localStorage.getItem(STORAGE_KEY); }
    catch { /* Use the expanded default. */ }
    setCollapsed(saved === 'collapsed', false);
    syncModeTitle();

    toggle.addEventListener('click', () => {
        setCollapsed(body.dataset.rail !== 'collapsed');
    });

    new MutationObserver(syncModeTitle).observe(body, {
        attributes: true,
        attributeFilter: ['data-mode'],
    });

    if (contextMeter && contextTip) {
        // Fill on the way in rather than on every context update: the panel is
        // invisible the rest of the time, and app.js repaints these numbers on
        // every keystroke.
        for (const event of ['pointerenter', 'focus']) {
            contextMeter.addEventListener(event, syncContextTip);
        }
        // Clicking the ring opens the full memory panel in the sidebar, which is
        // where the controls (free memory, thinking) live. The ring is a
        // readout; it does not grow its own copy of them.
        contextMeter.addEventListener('click', event => {
            if (!memHudPill) return;
            // app.js closes the memory panel on any document click landing
            // outside it. This click is outside it, so without this the panel
            // would open and shut in the same tick.
            event.stopPropagation();
            memHudPill.click();
            memHudPill.focus();
        });
        syncContextTip();
    }

    if (memHud) {
        syncContextMeter();
        new MutationObserver(syncContextMeter).observe(memHud, {
            attributes: true,
            attributeFilter: ['class', 'data-context'],
        });
    }

    if (contextModelMenu) {
        new MutationObserver(detachModelMenuFromMemoryPopover).observe(contextModelMenu, {
            attributes: true,
            attributeFilter: ['hidden'],
        });
    }

    // --- Desktop shell (pywebview) window controls --------------------------
    // Hidden by default (02-shell.css); revealed only when this page is
    // running inside the frameless pywebview window. pywebview injects
    // window.pywebview asynchronously and fires 'pywebviewready' on window
    // when it lands -- but that can happen before this script even attaches
    // the listener, so both paths are checked. ?shell=1 fakes detection so
    // appearance can be verified in an ordinary browser tab, where
    // window.pywebview (and so window.pywebview.api) never exists -- every
    // control below is guarded so a missing API can never throw.
    const topbar = document.getElementById('topbar');
    const winControls = document.getElementById('window-controls');
    const minimizeBtn = document.getElementById('win-minimize-btn');
    const maximizeBtn = document.getElementById('win-maximize-btn');
    const closeBtn = document.getElementById('win-close-btn');

    if (topbar && winControls && minimizeBtn && maximizeBtn && closeBtn) {
        const maximizeIcon = maximizeBtn.querySelector('use');
        let maximized = false;

        function activateShellChrome() {
            body.dataset.shell = 'native';
        }

        if (window.pywebview) activateShellChrome();
        window.addEventListener('pywebviewready', activateShellChrome);
        try {
            if (new URLSearchParams(location.search).get('shell') === '1') activateShellChrome();
        } catch { /* Malformed query string; leave shell chrome hidden. */ }

        function pywebviewApi() {
            return window.pywebview && window.pywebview.api;
        }

        // The Python side has no maximized/restored callback yet, so this is
        // optimistic local state flipped on every toggle rather than a true
        // readout of the native window. Good enough to drive the icon swap;
        // revisit once toggle_maximize() can report back.
        function setMaximized(next) {
            maximized = next;
            if (maximizeIcon) {
                maximizeIcon.setAttribute('href', maximized ? '#i-win-restore' : '#i-win-maximize');
            }
            const label = maximized ? 'Restore' : 'Maximize';
            maximizeBtn.setAttribute('aria-label', label);
            maximizeBtn.setAttribute('title', label);
        }

        minimizeBtn.addEventListener('click', () => {
            try { pywebviewApi()?.minimize?.(); }
            catch { /* pywebview API not wired up yet. */ }
        });

        maximizeBtn.addEventListener('click', () => {
            try { pywebviewApi()?.toggle_maximize?.(); }
            catch { /* pywebview API not wired up yet. */ }
            setMaximized(!maximized);
        });

        closeBtn.addEventListener('click', () => {
            try { pywebviewApi()?.close?.(); }
            catch { /* pywebview API not wired up yet. */ }
        });

        // Double-clicking the empty title bar toggles maximize, like a native
        // caption. Excluded whenever the double-click landed on a control --
        // dblclick bubbles from the button, and without this guard a
        // double-click on e.g. Close would both close the window and flip
        // the maximize state.
        topbar.addEventListener('dblclick', event => {
            if (event.target.closest('button, a, input, select, textarea, kbd')) return;
            try { pywebviewApi()?.toggle_maximize?.(); }
            catch { /* pywebview API not wired up yet. */ }
            setMaximized(!maximized);
        });
    }
})();
