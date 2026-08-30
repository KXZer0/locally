// Shared visual vocabulary: device provenance, activity lines, scrolling.
//
// In here: small DOM writes several features perform identically, where a
// second copy would be a second thing to keep in step.
// Not in here: anything that renders a message, a panel or a whole surface --
// those belong to the feature that owns the markup.

import { chat } from './dom.js';
import { escapeHtml } from './format.js';

// --- Device provenance ---

// The engine that answered used to be a colour (NPU amber, GPU cyan, CPU
// slate). It is now its own name in the mono face, plus a dot whose FILL says
// the same thing: solid NPU, ring GPU, hollow CPU. Three states that survive a
// monochrome palette, a colour-blind reader, and a screenshot in grayscale.
export function deviceKind(device) {
    const d = (device || '').toUpperCase();
    if (d.startsWith('NPU')) return 'NPU';
    if (d.startsWith('GPU')) return 'GPU';
    return 'CPU';
}

// Still sets --device because ~40 CSS rules read it; it now resolves to ink,
// so those rules stay correct without each one having to be found.
export function paintDevice(el, device) {
    if (!el) return;
    el.dataset.device = deviceKind(device);
    el.style.setProperty('--device', 'var(--ink)');
}
// One activity line for every state a turn can be in — thinking, searching,
// reading, writing. Shared so a turn never changes vocabulary mid-flight, and
// so the search path and the plain chat path look like the same product.
// Each state carries its own icon, so the shape says what is happening before
// the word is read. Anything unmapped falls back to the thinking mark rather
// than rendering an empty slot.
export const ACTIVITY_ICONS = {
    thinking:  'i-brain',
    writing:   'i-pen',
    starting:  'i-globe',
    searching: 'i-globe',
    reading:   'i-doc',
    ranking:   'i-scan',
    answering: 'i-pen',
    calculating: 'i-scan',
};

export function activityHtml(label, kind) {
    const key = kind || String(label).split(' ')[0].toLowerCase();
    const icon = ACTIVITY_ICONS[key] || 'i-brain';
    return '<span class="activity busy">'
         + `<svg class="ic" aria-hidden="true"><use href="#${icon}"/></svg>`
         + `<span class="activity-label">${escapeHtml(label)}</span></span>`;
}
export function shouldAutoScroll() {
    return chat.scrollHeight - chat.scrollTop - chat.clientHeight < 80;
}

export let scrollRaf = 0;
export function scrollToBottom(wantScroll) {
    if (!wantScroll || scrollRaf) return;
    scrollRaf = requestAnimationFrame(() => {
        scrollRaf = 0;
        chat.scrollTop = chat.scrollHeight;
    });
}

// `hidden` makes a surface non-interactive immediately. On reveal, hold the
// compositor-only starting state for one painted frame so the transition has
// a real value to animate from; focusing a child can otherwise flush styles
// and skip an @starting-style entrance in Chromium.
export function revealSurface(el) {
    if (!el || !el.hidden) return;
    el.classList.add('entering');
    el.hidden = false;
    requestAnimationFrame(() => requestAnimationFrame(() => {
        el.classList.remove('entering');
    }));
}
