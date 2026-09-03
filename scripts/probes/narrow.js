const body = document.body;
return {
    innerWidth,
    cols: getComputedStyle(body).gridTemplateColumns,
    rail: getComputedStyle(document.documentElement).getPropertyValue('--rail-open').trim(),
    railAttr: body.dataset.rail || null,
    sidebar: (() => { const r = document.querySelector('.sidebar-tools').getBoundingClientRect();
        return { x: Math.round(r.x), w: Math.round(r.width) }; })(),
    appColumn: (() => { const r = document.querySelector('.app-column').getBoundingClientRect();
        return { x: Math.round(r.x), w: Math.round(r.width) }; })(),
    resizer: getComputedStyle(document.getElementById('rail-resizer')).display,
};
