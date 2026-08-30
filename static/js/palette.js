// Command palette (Ctrl+K).
//
// It does not animate, in either direction. A palette toggle is a 100+/day
// keyboard action, and the review-animations frequency table's rule for that
// band is "no animation, ever" — motion there reads as lag. Raycast makes the
// same call. This opened with a 150ms fade until the standard was checked.
//
// Lives in its own module deliberately: it is self-contained, and its only
// coupling to the app is the handful of imports below.
//
// Everything is rendered from COMMANDS — never from markup. A later feature
// adds itself with registerCommand() and needs no change here, which is the
// whole point of a registry. `available()` returns true, or a string that is
// the reason it cannot run; an unavailable command is shown greyed with the
// reason rather than hidden, matching the honesty rule the util buttons
// already follow via /health.

import { brandMark, input, noThinkCheckbox, settingsBtn, settingsPanel } from './core/dom.js';
import { loadedModelId } from './core/state.js';
import { openContextModelMenu } from './system/swap.js';
import { newChat } from './chat/send.js';
import { setMode } from './ui/tabs.js';
import { UTIL_COPY, selectUtilTask } from './util/engine.js';

(function () {
    'use strict';

    const backdrop = document.getElementById('palette-backdrop');
    const inputEl = document.getElementById('palette-input');
    const listEl = document.getElementById('palette-list');
    const openBtn = document.getElementById('palette-open-btn');
    if (!backdrop || !inputEl || !listEl) return;

    const COMMANDS = [];
    let matches = [];
    let cursor = 0;
    let open = false;
    let lastFocus = null;

    function registerCommand(cmd) {
        COMMANDS.push(cmd);
    }
    window.registerCommand = registerCommand;

    // --- Built-ins ---------------------------------------------------------
    // `setMode`, `newChat`, `selectUtilTask` and the DOM handles come from the
    // imports at the top of this file. `loadedModelId` is a live binding, so
    // `available()` below sees the model that is resident right now.

    registerCommand({ id: 'go-chat',  group: 'Go', label: 'Open Chat', hint: 'Chat', keywords: 'conversation ask assistant message', run: () => setMode('chat') });
    registerCommand({ id: 'go-voice', group: 'Go', label: 'Open Voice', hint: 'Voice', keywords: 'talk speech microphone audio', run: () => setMode('voice') });
    registerCommand({ id: 'go-util',  group: 'Go', label: 'Open Tools', hint: 'Tools', keywords: 'utilities local npu gpu', run: () => setMode('util') });

    // Built from UTIL_COPY rather than a second hardcoded list, so a new
    // utility appears here automatically.
    if (typeof UTIL_COPY === 'object') {
        for (const [task, copy] of Object.entries(UTIL_COPY)) {
            registerCommand({
                id: 'util-' + task,
                group: 'Utilities',
                label: copy[0],
                hint: task,
                keywords: `${task} ${copy[1]}`,
                run: () => { setMode('util'); selectUtilTask(task); },
            });
        }
    }

    registerCommand({
        id: 'new-chat', group: 'Chat', label: 'New chat', hint: 'Ctrl+N', keywords: 'clear reset conversation',
        run: () => newChat(),
    });
    registerCommand({
        id: 'focus-input', group: 'Chat', label: 'Focus message input', hint: 'Enter', keywords: 'ask type prompt',
        run: () => { setMode('chat'); input.focus(); },
    });
    registerCommand({
        id: 'toggle-nothink', group: 'Chat',
        label: () => (noThinkCheckbox.checked ? 'Allow the model to think' : 'Skip thinking (/no_think)'),
        run: () => {
            noThinkCheckbox.checked = !noThinkCheckbox.checked;
            noThinkCheckbox.dispatchEvent(new Event('change'));
        },
    });

    registerCommand({
        id: 'settings', group: 'App', label: 'Open Settings', hint: 'Settings', keywords: 'preferences model prompt voice temperature',
        // Reuse the button's own handler rather than restating what it does —
        // it also moves focus correctly per mode.
        run: () => { if (settingsPanel.hidden) settingsBtn.click(); },
    });
    registerCommand({
        id: 'load-model', group: 'Model', label: 'Load a model…',
        run: () => { setMode('chat'); openContextModelMenu(); },
    });
    registerCommand({
        id: 'unload-model', group: 'Model', label: 'Unload the current model', hint: 'frees memory',
        available: () => (loadedModelId ? true : 'no model is loaded'),
        run: () => fetch('/v1/models/unload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}',
        }),
    });

    // --- Matching ----------------------------------------------------------
    // Substring, prefix-first. No fuzzy library: on a list this size fuzzy
    // matching mostly buys false positives.

    function labelOf(cmd) {
        return typeof cmd.label === 'function' ? cmd.label() : cmd.label;
    }

    function isSubsequence(needle, haystack) {
        let at = 0;
        for (const ch of haystack) {
            if (ch === needle[at]) at++;
            if (at === needle.length) return true;
        }
        return false;
    }

    function rank(cmd, q) {
        if (!q) return 0;
        const label = labelOf(cmd).toLowerCase();
        const group = (cmd.group || '').toLowerCase();
        const text = `${label} ${group} ${cmd.hint || ''} ${cmd.keywords || ''}`.toLowerCase();
        if (label.startsWith(q)) return 0;
        if (label.includes(q) || group.startsWith(q)) return 1;
        const words = q.split(/\s+/).filter(Boolean);
        if (words.every(word => text.includes(word))) return 2;
        if (q.length > 1 && isSubsequence(q, label)) return 3;
        return -1;
    }

    function recompute() {
        const q = inputEl.value.trim().toLowerCase().replace(/^[>/]\s*/, '');
        matches = COMMANDS
            .map(cmd => ({ cmd, r: rank(cmd, q) }))
            .filter(x => x.r >= 0)
            .sort((a, b) => a.r - b.r)
            .map(x => x.cmd);
        cursor = 0;
        render();
    }

    function render() {
        listEl.textContent = '';
        if (!matches.length) {
            const li = document.createElement('li');
            li.className = 'palette-empty';
            li.textContent = 'No matching command';
            listEl.appendChild(li);
            inputEl.removeAttribute('aria-activedescendant');
            return;
        }

        let group = null;
        matches.forEach((cmd, i) => {
            if (cmd.group && cmd.group !== group) {
                group = cmd.group;
                const head = document.createElement('li');
                head.className = 'palette-group';
                head.setAttribute('role', 'presentation');
                head.textContent = group;
                listEl.appendChild(head);
            }

            const why = cmd.available ? cmd.available() : true;
            const usable = why === true;

            const li = document.createElement('li');
            li.className = 'palette-item';
            li.id = 'pal-' + cmd.id;
            li.setAttribute('role', 'option');
            li.setAttribute('aria-selected', String(i === cursor));
            if (!usable) li.setAttribute('aria-disabled', 'true');

            const label = document.createElement('span');
            label.className = 'pal-label';
            label.textContent = labelOf(cmd);
            li.appendChild(label);

            const hint = usable ? cmd.hint : why;
            if (hint) {
                const h = document.createElement('span');
                h.className = 'pal-hint';
                h.textContent = hint;
                li.appendChild(h);
            }

            li.addEventListener('mousemove', () => {
                if (cursor !== i) { cursor = i; paintSelection(); }
            });
            li.addEventListener('click', () => run(i));
            listEl.appendChild(li);
        });
        paintSelection();
    }

    // Selection moves far more often than the list changes, so it gets its own
    // pass — re-rendering every row on each arrow key would be the same
    // rebuild-everything mistake the streaming path just had removed.
    function paintSelection() {
        const rows = listEl.querySelectorAll('.palette-item');
        rows.forEach((row, i) => row.setAttribute('aria-selected', String(i === cursor)));
        const active = rows[cursor];
        if (active) {
            inputEl.setAttribute('aria-activedescendant', active.id);
            active.scrollIntoView({ block: 'nearest' });
        }
    }

    function run(i) {
        const cmd = matches[i];
        if (!cmd) return;
        if (cmd.available && cmd.available() !== true) return;
        close();
        try { cmd.run(); } catch (err) { console.error('command failed:', cmd.id, err); }
    }

    // --- Open / close ------------------------------------------------------

    function openPalette() {
        if (open) return;
        open = true;
        lastFocus = document.activeElement;

        // Take the colour of whichever engine last answered, so the palette
        // agrees with the rest of the device-provenance system.
        const dev = getComputedStyle(brandMark).getPropertyValue('--device').trim();
        if (dev) backdrop.style.setProperty('--device', dev);

        backdrop.hidden = false;
        inputEl.value = '';
        recompute();
        backdrop.classList.add('open');   // state flag only; nothing animates
        inputEl.focus();
    }

    function close() {
        if (!open) return;
        open = false;
        backdrop.classList.remove('open');
        backdrop.hidden = true;
        if (lastFocus && document.contains(lastFocus)) lastFocus.focus();
        lastFocus = null;
    }

    // --- Events ------------------------------------------------------------

    // Capture phase: app.js already binds Escape to cancel generation, and it
    // registered first. Closing the palette must not also abort a running
    // answer, so this has to see the key first and stop it there.
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && ['k', 'p'].includes(e.key.toLowerCase())) {
            e.preventDefault();
            open ? close() : openPalette();
            return;
        }
        if (!open) return;

        if (e.key === 'Escape') {
            e.preventDefault();
            e.stopPropagation();
            close();
        } else if (e.key === 'Tab') {
            // Arrow keys navigate the command options, so the search field is
            // intentionally the dialog's only tab stop. Keep both Tab and
            // Shift+Tab inside the modal instead of leaking focus into the
            // dimmed application behind it.
            e.preventDefault();
            inputEl.focus();
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (matches.length) { cursor = (cursor + 1) % matches.length; paintSelection(); }
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (matches.length) { cursor = (cursor - 1 + matches.length) % matches.length; paintSelection(); }
        } else if (e.key === 'Enter') {
            e.preventDefault();
            run(cursor);
        }
    }, true);

    // `input`, not `keydown`: filtering on keydown reads the value before the
    // key lands, and breaks IME composition outright.
    inputEl.addEventListener('input', recompute);
    openBtn?.addEventListener('click', openPalette);

    backdrop.addEventListener('mousedown', (e) => {
        if (e.target === backdrop) close();
    });
})();
