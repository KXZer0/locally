const h = document.getElementById('rail-resizer');
const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const ev = (t, x) => h.dispatchEvent(new PointerEvent(t, { bubbles: true, clientX: x, clientY: 400, button: 0, pointerId: 1 }));
const ink = () => getComputedStyle(h).getPropertyValue('--handle-ink').trim() || '(unset)';
const out = { display: getComputedStyle(h).display, idle: ink(), idleBg: getComputedStyle(h).backgroundImage.slice(0, 90) };
ev('pointerdown', 259); ev('pointermove', 300); await frame();
out.dragging = { ink: ink(), bodyFlag: document.body.dataset.railResizing || null,
                 bodyCursor: getComputedStyle(document.body).cursor,
                 sidebarTransition: getComputedStyle(document.querySelector('.sidebar-tools')).transition };
ev('pointerup', 300); await frame();
out.afterDrag = { ink: ink(), bodyCursor: getComputedStyle(document.body).cursor };
return out;
