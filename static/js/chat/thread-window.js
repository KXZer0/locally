// Keeping a long thread from growing without bound (REBUILD-PLAN §2.3).
//
// In here: the cap on how many message bodies stay mounted, and the observer
// that puts one back when you scroll to it.
// Not in here: the conversation. Chat is the complete record and nothing in
// this file touches chatHistory -- what is dropped is DOM, never data, and
// the text is still in the message's own copy button and in the history the
// next turn is built from.
//
// WHAT IS ACTUALLY EXPENSIVE, measured rather than assumed. A message's cost
// is its element tree, not its node in the thread: one 2,000-token answer
// parses into several hundred elements, each with a computed style, while the
// markup that produced it is one string. So a capped message keeps its
// wrapper -- the role label, the footer, the meta line, the action row with
// its live listeners, roughly ten nodes -- and gives up only `.msg-body`,
// which is all of the rest of it. Rehydration is one innerHTML write of a
// string we never let go of, so nothing is re-parsed from markdown and no
// listener has to be rebound.
//
// Two things this must not do, both of which a naive version does:
//   - move the scroll. The emptied body is pinned to the height it had, so
//     the document is exactly as tall afterwards as before.
//   - hide the streaming answer. The newest CAP messages are never touched,
//     and the observer's margin rehydrates a screenful ahead of the viewport.

import { thread } from '../core/dom.js';

// 40 is the plan's number. It is a cap on mounted BODIES; the thread's own
// child count is not capped, because a placeholder has to stay in the flow to
// hold its place in the scroll.
const CAP = 40;

// Rehydrate a screenful early. An observer that fires when the message is
// already on screen shows an empty box for a frame, which is worse than
// keeping it mounted.
const AHEAD = '150% 0px';

const stored = new WeakMap();   // .msg-body -> its innerHTML while dehydrated
let observer = null;

function ensureObserver() {
    if (observer) return observer;
    if (typeof IntersectionObserver !== 'function') return null;
    observer = new IntersectionObserver(entries => {
        for (const entry of entries) {
            if (entry.isIntersecting) rehydrate(entry.target);
        }
    }, { root: null, rootMargin: AHEAD, threshold: 0 });
    return observer;
}

function dehydrate(msgDiv) {
    const body = msgDiv.body || msgDiv.querySelector('.msg-body');
    if (!body || stored.has(body)) return;
    const height = body.getBoundingClientRect().height;
    // Nothing to gain on a one-line answer, and a placeholder for it costs an
    // observer entry and a rehydrate.
    if (height < 60) return;
    stored.set(body, body.innerHTML);
    // Pin the height BEFORE emptying: reading it afterwards reads zero, and
    // the thread would then shorten by the height of every capped message at
    // once, which is the scroll jumping to somewhere you were not.
    body.style.height = `${Math.round(height)}px`;
    body.replaceChildren();
    body.dataset.dehydrated = '1';
    ensureObserver()?.observe(msgDiv);
}

function rehydrate(msgDiv) {
    const body = msgDiv.body || msgDiv.querySelector('.msg-body');
    if (!body || !stored.has(body)) return;
    body.innerHTML = stored.get(body);
    stored.delete(body);
    body.style.height = '';
    delete body.dataset.dehydrated;
    observer?.unobserve(msgDiv);
}

// Called after every message is appended. Walks back from the newest, keeps
// CAP of them mounted, and caps the rest -- from the far end, so the common
// case (one new message, everything already capped) stops at the first
// already-capped body instead of walking the whole thread.
export function capThread() {
    const messages = thread.children;
    const limit = messages.length - CAP;
    for (let i = limit - 1; i >= 0; i--) {
        const msg = messages[i];
        const body = msg.body || msg.querySelector('.msg-body');
        if (body && stored.has(body)) break;
        dehydrate(msg);
    }
}

// New Chat empties the thread outright. The WeakMap lets the strings go with
// the nodes, but the observer holds hard references to elements that are no
// longer in the document, so it has to be dropped.
export function resetThreadWindow() {
    observer?.disconnect();
    observer = null;
}

// For the measurement in docs/, and for anything that needs to know whether a
// body it is about to read is currently mounted.
export function threadWindowStats() {
    let dehydrated = 0;
    for (const msg of thread.children) {
        if (msg.querySelector('.msg-body[data-dehydrated]')) dehydrated++;
    }
    return { messages: thread.children.length, dehydrated, cap: CAP };
}
