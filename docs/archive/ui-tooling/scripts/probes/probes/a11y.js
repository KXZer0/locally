// Two things: the mode tablist's roving tabindex, and where the new
// separator lands in the tab order.
const ids = ['tab-chat', 'tab-voice', 'tab-code', 'tab-util'].map(id => document.getElementById(id));
const key = (el, k) => el.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: k }));
const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const state = () => ids.map(t => `${t.id}:${t.tabIndex}${t.getAttribute('aria-selected') === 'true' ? '*' : ''}`).join(' ');
const out = { initial: state(), walk: [] };
let cur = ids[0];
cur.focus();
for (let i = 0; i < 5; i++) {
    key(cur, 'ArrowRight'); await frame();
    cur = document.activeElement;
    out.walk.push(`${cur.id} | ${state()}`);
}
key(cur, 'Home'); await frame(); out.home = document.activeElement.id;
key(document.activeElement, 'End'); await frame(); out.end = document.activeElement.id;
document.getElementById('tab-chat').click(); await frame();

// Tab order: every natively focusable element, in document order, with the
// separator's position called out.
const focusables = [...document.querySelectorAll(
    'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])')]
    .filter(el => !el.disabled && el.offsetParent !== null
        && !el.closest('[inert]') && !el.closest('[hidden]'));
out.tabStops = focusables.length;
out.separatorIndex = focusables.findIndex(el => el.id === 'rail-resizer');
out.separatorNeighbours = [
    focusables[out.separatorIndex - 1]?.id || focusables[out.separatorIndex - 1]?.className,
    'rail-resizer',
    focusables[out.separatorIndex + 1]?.id || focusables[out.separatorIndex + 1]?.className,
];
// Accessible name and value, as a screen reader would read them.
const h = document.getElementById('rail-resizer');
out.separatorAnnounced = `${h.getAttribute('role')} "${h.getAttribute('aria-label')}" `
    + `${h.getAttribute('aria-valuenow')} of ${h.getAttribute('aria-valuemin')}`
    + `-${h.getAttribute('aria-valuemax')}`;
return out;
