// Same question, one phase per run so the renderer's own listener counter can
// be read for each: --arg boot | tabs | panels
const phase = window.__arg || 'boot';
const { setMode } = await import('/static/js/ui/tabs.js');
const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const settle = (ms = 150) => new Promise(r => setTimeout(r, ms));
if (phase !== 'boot') {
    for (let round = 0; round < 20; round++) {
        for (const m of ['chat', 'voice', 'util', 'code']) { setMode(m); await frame(); }
    }
    setMode('chat'); await frame();
}
if (phase === 'panels') {
    for (let i = 0; i < 20; i++) {
        document.getElementById('settings-btn')?.click(); await frame();
        document.getElementById('settings-btn')?.click(); await frame();
    }
}
await settle();
return { phase, elements: document.querySelectorAll('*').length };
