// Does a width change correct the thread's pinned heights, and does it move
// the reader while doing it?
//
// The cap in chat/thread-window.js pins a dehydrated body to the height it had
// AT DEHYDRATION TIME, which is what makes the cap free: scrollHeight is
// unchanged to the pixel between capped and hydrated. That holds at a fixed
// width only, and §2.2 shipped a draggable rail. Run at --width 900: the
// thread reflows there, which at 1280 it does not, so a 1280 run reports a
// clean zero for the wrong reason.
//
//   node scripts/uidrive.mjs --url http://127.0.0.1:8779/ --width 900 \
//        --height 800 --script scripts/probes/thread-resize.js \
//        --set locally-onboarding-complete=1

const { addMessage } = await import('/static/js/chat/thread.js');
const { threadWindowStats } = await import('/static/js/chat/thread-window.js');
const thread = document.getElementById('thread');
const chat = document.getElementById('chat');
const root = document.documentElement;
const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const settle = (ms = 150) => new Promise(r => setTimeout(r, ms));

// Long enough to wrap differently at 668px and at 508px, which is the whole
// point: a one-line answer is the same height at every width.
const answer = n => `## Answer ${n}\n\nThe engine chose the **GPU** for this turn `
    + `because the model is group-quantized and the NPU compiler rejects that outright. `
    + `A longer paragraph so the body actually reflows when the column narrows, which is `
    + `what this measures and needs a good deal more than one line of text to show.\n\n`
    + `- prefill ${1000 + n} tok/s\n- decode ${20 + (n % 9)} tok/s\n- KV 96 KB/token\n\n`
    + '```python\ndef fit(w, kv, b):\n    return w + kv <= b\n```\n\n'
    + `Measured on the B390 at ratio ${n % 60}.`;

root.style.setProperty('--rail-open', '200px');
await frame();
for (let i = 0; i < 60; i++) {
    addMessage('user', `Question ${i}: why did turn ${i} land on the GPU?`);
    addMessage('assistant', answer(i), { device: 'GPU' });
    await frame();
}
await frame();

const out = { cap: threadWindowStats() };
const anyBody = thread.querySelector('.message.assistant .msg-body');
out.narrow = { bodyWidth: Math.round(anyBody.getBoundingClientRect().width),
               cols: getComputedStyle(document.body).gridTemplateColumns,
               scrollHeight: chat.scrollHeight };

// Park the reader mid-thread, so anchor drift is visible rather than absorbed
// by being pinned to the bottom.
chat.scrollTop = Math.round(chat.scrollHeight * 0.6);
await settle(250);
const watched = [...thread.children].find(m => m.getBoundingClientRect().bottom > 0);
const watchedTop = watched.getBoundingClientRect().top;

const pinned = [...thread.querySelectorAll('.msg-body[data-dehydrated]')];
const sample = pinned[Math.floor(pinned.length / 2)];
out.pinnedHeight = parseFloat(sample.style.height);

// --- The drag. Everything below is what the reader sees afterwards. ---
root.style.setProperty('--rail-open', '360px');
await frame();
await settle(500);                       // past SETTLE_MS, with room to spare

out.wide = { bodyWidth: Math.round(anyBody.getBoundingClientRect().width),
             cols: getComputedStyle(document.body).gridTemplateColumns,
             scrollHeight: chat.scrollHeight };
out.capAfter = threadWindowStats();
out.anchorMoved = Math.round(watched.getBoundingClientRect().top - watchedTop);

// The truth: hydrate every remaining pin and see whether the document was
// already the height it now is.
chat.scrollTop = 0;
await settle(150);
for (let y = 0; y <= chat.scrollHeight; y += Math.max(200, chat.clientHeight - 100)) {
    chat.scrollTop = y;
    await settle(35);
}
await settle(250);
out.fullyHydratedScrollHeight = chat.scrollHeight;
out.drift = out.fullyHydratedScrollHeight - out.wide.scrollHeight;
out.leftDehydrated = thread.querySelectorAll('.msg-body[data-dehydrated]').length;
return out;
