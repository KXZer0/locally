// Geometry of every element, keyed independently of sibling position.
// nth-child paths renumber the moment an element is inserted, which is
// exactly the change being measured; this keys on the element's own identity
// plus its index among elements with the SAME signature.
const seen = new Map();
const rows = {};
for (const el of document.querySelectorAll('*')) {
    if (el.id === 'rail-resizer') continue;
    const sig = el.tagName.toLowerCase()
        + (el.id ? `#${el.id}` : '')
        + (el.className && typeof el.className === 'string'
            ? '.' + el.className.trim().split(/\s+/).join('.') : '');
    const n = (seen.get(sig) || 0);
    seen.set(sig, n + 1);
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    rows[`${sig}[${n}]`] = [
        Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height),
        cs.display, cs.position, cs.zIndex, cs.color, cs.backgroundColor,
        cs.fontSize, cs.overflowX, cs.overflowY,
    ].join('|');
}
return rows;
