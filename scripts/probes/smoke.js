const h = document.getElementById('rail-resizer');
const r = h && h.getBoundingClientRect();
return {
    hidden: document.hidden,
    innerWidth,
    setupOpen: document.body.dataset.setupOpen || null,
    elements: document.querySelectorAll('*').length,
    sheets: document.styleSheets.length,
    handle: r && { x: Math.round(r.x), w: Math.round(r.width), h: Math.round(r.height) },
    cols: getComputedStyle(document.body).gridTemplateColumns,
    rail: getComputedStyle(document.documentElement).getPropertyValue('--rail-open').trim(),
};
