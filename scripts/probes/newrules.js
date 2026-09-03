// Which elements can the rules added for §2.2/§2.3 reach at boot?
// A rule that matches nothing cannot have changed the page; a rule that
// matches something has to be looked at. This is the check that replaces
// "it should be additive".
const ADDED = [
    '.rail-resizer',
    '.rail-resizer:hover',
    '.rail-resizer:focus-visible',
    'body[data-rail-resizing="1"] .rail-resizer',
    'body[data-rail="collapsed"] .rail-resizer',
    'body[data-rail-resizing="1"]',
    'body[data-rail-resizing="1"] > .sidebar-tools',
    '.message.assistant',
    '.msg-body[data-dehydrated]',
];
const matches = {};
for (const sel of ADDED) {
    matches[sel] = [...document.querySelectorAll(sel)]
        .map(el => el.id || el.className || el.tagName.toLowerCase());
}
// And confirm the sheets really do carry them, so a typo cannot pass as
// "matches nothing".
const present = new Set();
for (const sheet of document.styleSheets) {
    const walk = rules => { for (const r of rules) {
        if (r.selectorText) present.add(r.selectorText.replace(/\s+/g, ' ').trim());
        if (r.cssRules) walk(r.cssRules);
    } };
    try { walk(sheet.cssRules); } catch { /* cross-origin */ }
}
return {
    matchesAtBoot: matches,
    inSheets: ADDED.filter(s => present.has(s)),
    missingFromSheets: ADDED.filter(s => !present.has(s)),
};
