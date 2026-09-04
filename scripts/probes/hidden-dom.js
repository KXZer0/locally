// How much of the boot DOM is built for something nobody has opened yet.
//
// §2.5. `handoff-frontend-next.md` measured 83.8 % hidden and named the biggest
// subtrees; this reports the same shape but keyed by the element that would
// become a <template>, so the before/after of converting one of them is a
// direct comparison rather than a total that moved.
//
// "Hidden" is the `hidden` attribute or display:none, resolved on the rendered
// page rather than read off the markup -- a panel is hidden by JS at boot, not
// by an attribute in the template.

const all = [...document.querySelectorAll('body *')];
// checkVisibility, not getComputedStyle(el).display: `display: none` does NOT
// cascade to a child's computed style, so testing the element alone finds the
// seven hidden containers and none of their 600-odd descendants -- which is the
// whole quantity in question. A first cut of this probe read 8.8% that way,
// against the 83.8% the handoff measured, and the handoff was right.
const isHidden = el => !el.checkVisibility({ checkVisibilityCSS: true });

// The candidates §2.5 names: surfaces cloned on first open rather than parsed
// at boot. Keyed by id so the list survives markup moving around.
const CANDIDATES = [
    'panel-util', 'settings-panel', 'panel-voice', 'speaker-panel',
    'setup-shell', 'panel-code', 'palette', 'sources-panel',
];

const subtree = id => {
    const el = document.getElementById(id);
    if (!el) return null;
    return {
        elements: el.querySelectorAll('*').length + 1,
        hidden: isHidden(el),
        // A <template> is parsed but inert: its contents are NOT in the main
        // document tree, so they do not count here once converted.
        inTemplate: !!el.closest('template'),
    };
};

const hidden = all.filter(isHidden);
const perCandidate = {};
for (const id of CANDIDATES) {
    const s = subtree(id);
    if (s) perCandidate[id] = s;
}

// Util views are five near-identical blocks; worth their own line because
// converting them is one change, not five.
const utilViews = [...document.querySelectorAll('.util-view')]
    .map(v => ({ id: v.id || '(unnamed)', elements: v.querySelectorAll('*').length + 1 }));

return {
    totalElements: all.length,
    hiddenElements: hidden.length,
    hiddenPct: +(100 * hidden.length / all.length).toFixed(1),
    templates: document.querySelectorAll('template').length,
    templateElements: [...document.querySelectorAll('template')]
        .reduce((n, t) => n + t.content.querySelectorAll('*').length, 0),
    perCandidate,
    utilViews,
    utilViewTotal: utilViews.reduce((n, v) => n + v.elements, 0),
};
