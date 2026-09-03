// The same 100-turn session as probe-thread.js, stopped at one phase so the
// renderer's own DOM counter (which includes text nodes, where most of a
// rendered markdown answer lives) can be read for that phase.
//   --arg capped    stop with the cap in force
//   --arg hydrated  scroll the whole thread first, so nothing is capped
const phase = window.__arg || 'capped';
const { addMessage } = await import('/static/js/chat/thread.js');
const { threadWindowStats } = await import('/static/js/chat/thread-window.js');
const thread = document.getElementById('thread');
const chat = document.getElementById('chat');
const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const settle = (ms = 40) => new Promise(r => setTimeout(r, ms));
const answer = n => `## Answer ${n}\n\nThe engine chose the **GPU** for this turn `
    + `because the model is group-quantized and the NPU compiler rejects that.\n\n`
    + `- prefill ${1000 + n} tok/s\n- decode ${20 + (n % 9)} tok/s\n- KV 96 KB/token\n\n`
    + '```python\ndef fit(weights, kv, budget):\n    return weights + kv <= budget\n```\n\n'
    + `Measured on the B390 at ratio ${n % 60}, which is the smallest that fits.`;
for (let i = 0; i < 100; i++) {
    addMessage('user', `Question ${i}: why did turn ${i} land on the GPU?`);
    addMessage('assistant', answer(i), { device: 'GPU' });
    await frame();
}
if (phase === 'hydrated') {
    for (let y = 0; y <= chat.scrollHeight; y += Math.max(200, chat.clientHeight - 100)) {
        chat.scrollTop = y; await settle(30);
    }
    chat.scrollTop = 0; await settle(300);
}
await settle(200);
return { phase, ...threadWindowStats(), elements: thread.querySelectorAll('*').length,
         scrollHeight: chat.scrollHeight };
