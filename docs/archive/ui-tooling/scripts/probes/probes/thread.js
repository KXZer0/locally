const { addMessage } = await import('/static/js/chat/thread.js');
const { threadWindowStats } = await import('/static/js/chat/thread-window.js');
const thread = document.getElementById('thread');
const chat = document.getElementById('chat');
const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const settle = (ms = 120) => new Promise(r => setTimeout(r, ms));

// A realistic answer, not a one-liner: a heading, prose, a list and a fenced
// block is what the renderer actually has to build for a coding turn.
const answer = n => `## Answer ${n}\n\nThe engine chose the **GPU** for this turn `
    + `because the model is group-quantized and the NPU compiler rejects that.\n\n`
    + `- prefill ${1000 + n} tok/s\n- decode ${20 + (n % 9)} tok/s\n- KV ${96} KB/token\n\n`
    + '```python\ndef fit(weights, kv, budget):\n    return weights + kv <= budget\n```\n\n'
    + `Measured on the B390 at ratio ${n % 60}, which is the smallest that fits.`;

const nodes = () => thread.querySelectorAll('*').length;
const heap = () => performance.memory && performance.memory.usedJSHeapSize;

const out = { start: { nodes: nodes(), heap: heap() } };

// Turns arrive one at a time and each is painted before the next, which is
// how the app behaves -- addMessage() auto-scrolls to the new message. A loop
// that appends 400 nodes in one task never paints any of them, and then every
// height in the thread is content-visibility's 240px estimate rather than a
// measurement.
for (let i = 0; i < 100; i++) {
    addMessage('user', `Question ${i}: why did turn ${i} land on the GPU?`);
    addMessage('assistant', answer(i), { device: 'GPU' });
    await frame();
}
await frame();
out.capped = { ...threadWindowStats(), nodes: nodes(), heap: heap(),
               scrollHeight: chat.scrollHeight };

// Scroll the whole thread so every observer fires, which both proves
// rehydration works and produces the uncapped cost for comparison.
for (let y = 0; y <= chat.scrollHeight; y += Math.max(200, chat.clientHeight - 100)) {
    chat.scrollTop = y;
    await settle(40);
}
chat.scrollTop = 0; await settle(200);
chat.scrollTop = chat.scrollHeight; await settle(200);
out.hydrated = { ...threadWindowStats(), nodes: nodes(), heap: heap(),
                 scrollHeight: chat.scrollHeight };

// One more message re-runs the cap over a fully hydrated thread.
addMessage('assistant', answer(999), { device: 'NPU' });
await frame();
out.recapped = { ...threadWindowStats(), nodes: nodes(), heap: heap() };

out.saving = {
    nodesDropped: out.hydrated.nodes - out.capped.nodes,
    nodesPct: +(100 * (1 - out.capped.nodes / out.hydrated.nodes)).toFixed(1),
    heapDeltaMB: out.hydrated.heap && +(((out.hydrated.heap - out.capped.heap) / 1048576).toFixed(2)),
    scrollHeightMoved: out.hydrated.scrollHeight - out.capped.scrollHeight,
};
// Nothing may be left empty on screen at the end.
out.emptyOnScreen = [...thread.querySelectorAll('.msg-body[data-dehydrated]')]
    .filter(b => { const r = b.getBoundingClientRect();
                   return r.bottom > 0 && r.top < innerHeight; }).length;
return out;
