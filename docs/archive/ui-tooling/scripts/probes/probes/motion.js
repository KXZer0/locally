// What the enter animation costs, and which properties it animates.
//
// §2.4 says only transform and opacity may animate, and names `filter: blur()`
// in the `enter` keyframes as the thing to remove: blur forces a paint every
// frame, on three elements per message, on a machine whose GPU may be busy
// running inference. That is an argument, not a measurement, so this probe
// takes the measurement.
//
// Two halves. First, read the keyframes out of the CSSOM and report every
// property any of them animates — an audit that stays true as rules change,
// rather than a grep for one word. Second, insert a burst of messages and
// record the frame intervals while they animate in, because "forces a paint"
// only matters if it shows up as a long frame.
//
// Run at a real width; the animation is width-independent but the message
// bodies are not.

const { addMessage } = await import('/static/js/chat/thread.js');

// --- 1. audit: what do the keyframes actually animate? --------------------
const animated = {};
for (const sheet of document.styleSheets) {
    let rules;
    try { rules = sheet.cssRules; } catch { continue; }
    for (const rule of rules) {
        if (!(rule instanceof CSSKeyframesRule)) continue;
        for (const frame of rule.cssRules) {
            for (const prop of frame.style) {
                (animated[rule.name] ||= new Set()).add(prop);
            }
        }
    }
}
// Anything outside this set costs layout or paint per frame.
const CHEAP = new Set(['transform', 'opacity']);
const offenders = [];
for (const [name, props] of Object.entries(animated)) {
    const bad = [...props].filter(p => !CHEAP.has(p) && !p.startsWith('--'));
    if (bad.length) offenders.push(`${name}: ${bad.join(', ')}`);
}

// --- 2. cost: frame intervals while a burst of messages enters ------------
// Frame *intervals*, not durations: a long frame is one that arrives late, and
// that is what a dropped frame looks like to a reader.
const frames = [];
let recording = true;
function tick(prev) {
    return function step(now) {
        if (!recording) return;
        if (prev) frames.push(now - prev);
        requestAnimationFrame(tick(now));
    };
}
requestAnimationFrame(tick(0));

const BURST = 12;
const t0 = performance.now();
for (let i = 0; i < BURST; i++) {
    addMessage('assistant',
        `Message ${i}. ` + 'The runtime compiles the model graph for the device '
        + 'before any token is produced, and the compiled graph is cached so the '
        + 'second load is cheap. '.repeat(3), { html: false });
    // Let each one actually start animating rather than inserting all twelve
    // into one frame, which would measure a single relayout instead.
    await new Promise(r => setTimeout(r, 40));
}
// The enter animation is under 300 ms; give the tail room to finish.
await new Promise(r => setTimeout(r, 500));
recording = false;
const elapsed = performance.now() - t0;

const sorted = frames.slice().sort((a, b) => a - b);
const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
const worst = sorted.length ? sorted[sorted.length - 1] : 0;
// 20 ms is a frame and a bit at 60 Hz: late enough that a reader can see it.
const late = frames.filter(f => f > 20).length;

// What the animating elements are actually being asked to do, right now.
const sample = document.querySelector('.message .msg-body');
const cs = sample ? getComputedStyle(sample) : null;

return {
    keyframesAnimatingExpensiveProps: offenders,
    burst: BURST,
    elapsedMs: Math.round(elapsed),
    frames: frames.length,
    medianFrameMs: +median.toFixed(1),
    worstFrameMs: +worst.toFixed(1),
    lateFrames: late,
    enterBlurToken: cs ? cs.getPropertyValue('--enter-blur').trim() || '(unset)' : null,
    reducedMotionHonoured: !!matchMedia('(prefers-reduced-motion: reduce)').media,
};
