const h = document.getElementById('rail-resizer');
const body = document.body;
const rail = () => getComputedStyle(document.documentElement).getPropertyValue('--rail-open').trim();
const cols = () => getComputedStyle(body).gridTemplateColumns;
const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const ev = (type, x) => h.dispatchEvent(new PointerEvent(type, {
    bubbles: true, clientX: x, clientY: 400, button: 0, pointerId: 1,
}));
async function drag(from, to) { ev('pointerdown', from); ev('pointermove', to); await frame();
    const mid = { rail: rail(), flag: body.dataset.railResizing || null };
    ev('pointerup', to); await frame();
    return { mid, after: rail(), flag: body.dataset.railResizing || null,
             stored: localStorage.getItem('locally-rail-width') }; }
const out = { start: { rail: rail(), cols: cols() } };
out.dragTo330 = await drag(259, 330);
out.colsAfter = cols();
out.clampMax = await drag(331, 900);
out.clampMin = await drag(361, 10);
h.focus();
const key = async k => { h.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: k })); await frame(); return rail(); };
out.arrowRight = await key('ArrowRight');
out.arrowLeft = await key('ArrowLeft');
out.end = await key('End');
out.home = await key('Home');
h.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })); await frame();
out.dblclick = { rail: rail(), stored: localStorage.getItem('locally-rail-width') };
out.aria = { role: h.getAttribute('role'), orientation: h.getAttribute('aria-orientation'),
    min: h.getAttribute('aria-valuemin'), max: h.getAttribute('aria-valuemax'),
    now: h.getAttribute('aria-valuenow'), label: h.getAttribute('aria-label'),
    focusable: h.tabIndex };
// Collapsed: the handle must leave the tab order AND refuse to drag.
document.getElementById('rail-toggle').click(); await frame();
out.collapsed = { display: getComputedStyle(h).display, cols: cols() };
out.collapsedDrag = await drag(259, 340);
document.getElementById('rail-toggle').click(); await frame();
out.reexpanded = { rail: rail(), cols: cols() };
return out;
