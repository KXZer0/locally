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

import { chat, thread } from '../core/dom.js';

// 40 is the plan's number. It is a cap on mounted BODIES; the thread's own
// child count is not capped, because a placeholder has to stay in the flow to
// hold its place in the scroll.
const CAP = 40;

// Rehydrate a screenful early. An observer that fires when the message is
// already on screen shows an empty box for a frame, which is worse than
// keeping it mounted.
const AHEAD = '150% 0px';

// A pin is only true at the width it was measured at, and nothing noticed
// that until §2.2 shipped a control whose whole purpose is changing the
// width. Measured at a 900px viewport, dragging the rail 200 -> 360 narrows a
// body from 668px to 508px, which makes it 382px tall where it was pinned at
// 358; across 40 capped bodies the document then under-reports its own height
// by 1,386px, and pays that back as a jump the moment the reader scrolls up
// into the capped region -- the exact failure the pin exists to prevent. A
// window resize does the same thing; the rail is only what made it easy to
// hit. At 1280px the same drag drifts 0px, because the thread does not reflow
// there, which is why this survived the first measurement.
//
// A pin cannot be corrected without laying the body out, and laying it out
// means rehydrating it, so half the repair is to rehydrate everything and let
// capThread() re-cap at the new width. That is 40 innerHTML writes of strings
// already in hand, which is what lets it be this blunt -- but not once per
// frame of a drag, hence the settle.
//
// Only half, though: re-pinning alone moved the drift from 1,386px to 1,313,
// and the pins it produced were exactly right (real minus pinned summed to 0
// over all 40). The rest is content-visibility, and remeasure() says what.
const SETTLE_MS = 150;

const stored = new WeakMap();   // .msg-body -> its innerHTML while dehydrated
let observer = null;
let sizeObserver = null;
let lastWidth = 0;
let settleTimer = 0;

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

function ensureSizeObserver() {
    if (sizeObserver || typeof ResizeObserver !== 'function') return;
    sizeObserver = new ResizeObserver(entries => {
        const width = Math.round(entries[entries.length - 1].contentRect.width);
        // Zero is the Chat tab being hidden, not a resize. Acting on it would
        // rehydrate the whole thread and re-pin every body to whatever a
        // zero-width layout reports, which is worse than the stale number.
        if (!width || width === lastWidth) return;
        const first = lastWidth === 0;
        lastWidth = width;
        // observe() reports the current size immediately; that first report
        // is the width every pin was already taken at.
        if (first) return;
        clearTimeout(settleTimer);
        settleTimer = setTimeout(remeasure, SETTLE_MS);
    });
    sizeObserver.observe(thread);
}

// The first message with any part of itself on screen: what the reader is
// looking at, and the thing whose position must not move. Correcting the
// heights above them is the point of this; moving the text they are reading
// while doing it would be a different bug of the same size.
function topVisibleMessage() {
    for (const msg of thread.children) {
        const rect = msg.getBoundingClientRect();
        if (rect.bottom > 0) return { msg, top: rect.top };
    }
    return null;
}

function remeasure() {
    settleTimer = 0;
    const stale = [...thread.querySelectorAll('.msg-body[data-dehydrated]')];
    if (!stale.length) return;
    const anchor = topVisibleMessage();
    // Rehydrate first, cap second: two passes rather than one, so the writes
    // are not interleaved with the reads capThread() takes to re-pin.
    for (const body of stale) rehydrate(body.closest('.message'));
    // The pins are only half of it, and the smaller half. Every off-screen
    // assistant turn is also carrying a content-visibility size remembered at
    // the old width -- 1,396px of the 900px-viewport case, against 0 once this
    // runs. See the note beside #thread[data-remeasuring] in chat.css: one
    // real layout at the new width is the only way to replace them.
    thread.dataset.remeasuring = '1';
    void thread.scrollHeight;                 // flush style and layout now
    capThread();                              // reads true heights under the flag
    // The flag has to survive a painted frame. A remembered size is recorded
    // when the box is rendered, not when it is laid out, so setting and
    // clearing the attribute inside one task corrects nothing -- measured, and
    // it was the reason the first version of this moved the drift by 73px out
    // of 1,386.
    requestAnimationFrame(() => requestAnimationFrame(() => {
        delete thread.dataset.remeasuring;
        if (anchor) chat.scrollTop += anchor.msg.getBoundingClientRect().top - anchor.top;
    }));
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
    // Nothing is pinned until now, so nothing can go stale until now.
    ensureSizeObserver();
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
    // Same reason, and one more: lastWidth describes pins that no longer
    // exist, so keeping it would make the first resize of the next
    // conversation look like no resize at all.
    sizeObserver?.disconnect();
    sizeObserver = null;
    lastWidth = 0;
    clearTimeout(settleTimer);
    settleTimer = 0;
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
