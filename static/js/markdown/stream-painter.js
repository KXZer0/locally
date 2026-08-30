// Painting a message while it is still being written.
//
// In here: the three-part streaming DOM (think / stable / tail) and the
// boundary search that decides what has stopped changing. This exists because
// re-rendering the whole answer per token is O(n^2) over a message.
// Not in here: the canonical render. Both callers run renderMarkdown() once
// the stream ends, so the committed DOM is always markdown/render.js's.

import { chat } from '../core/dom.js';
import { shouldAutoScroll } from '../core/paint.js';
import { renderBody, renderMarkdown } from './render.js';
import { splitThink } from './think-split.js';

// --- Streaming ---

// Index of the end of the last COMPLETE markdown block: the last blank line
// that is not inside a fenced code block. Everything before it is finished
// and cannot change; everything after is still being written.
//
// Fences are tracked rather than ignored because a blank line inside ``` is
// not a block boundary, and treating it as one splits a code block in half
// mid-stream. While a fence is open the loop stops advancing `boundary`, so
// an unterminated fence yields the last boundary from before it opened.
export function lastStableBoundary(text) {
    let inFence = false, boundary = 0, pos = 0;
    const lines = text.split('\n');
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (/^\s{0,3}(```|~~~)/.test(line)) inFence = !inFence;
        // `i < lines.length - 1` is load-bearing: the final element of a
        // split has no newline after it, so `pos + len + 1` would point one
        // past the end of the string. Caught by the round-trip test on a
        // lone newline character, which returned boundary 2 for a string of
        // length 1.
        else if (!inFence && line.trim() === '' && i > 0 && i < lines.length - 1)
            boundary = pos + line.length + 1;
        pos += line.length + 1;
    }
    return boundary;
}

// Paints a streaming message without re-parsing it from scratch every token.
//
// The old path was `target.innerHTML = renderMarkdown(fullText)` per SSE
// delta: ~26 regex passes plus a TeX parse over the ENTIRE answer so far,
// then a full subtree teardown, then a forced layout from scrollToBottom
// reading scrollHeight straight after invalidating that tree. O(n) per token
// is O(n^2) over a message -- a 2000-token answer re-parsed itself 2000
// times. That is why text strobed instead of flowing, and why a text
// selection inside a streaming message could never survive.
//
// Split the message in three:
//   think   re-rendered only when its markup actually changes
//   stable  completed blocks; re-rendered only when a new block completes
//   tail    the block being written; short, so parsing it per token is cheap
//
// Per-token cost stops growing with the answer and becomes proportional to
// the current paragraph. Writes coalesce into one rAF, so several tokens
// landing inside a frame paint once and the scroll happens in that same
// frame instead of forcing a layout per token.
//
// Known and deliberate: a list with blank lines between items can render as
// two adjacent lists mid-stream, since a blank line reads as a boundary. It
// self-corrects -- both callers run the full renderMarkdown once the stream
// ends, so the committed message is always the canonical render.
export function makeStreamPainter(target) {
    let raf = 0, pending = '', stableEnd = 0, lastThink = null, wantScroll = true;
    let thinkEl = null, stableEl = null, tailEl = null;

    function ensureNodes() {
        if (stableEl) return;
        target.innerHTML = '';
        thinkEl  = document.createElement('div');
        stableEl = document.createElement('div');
        tailEl   = document.createElement('div');
        thinkEl.className  = 'stream-think';
        stableEl.className = 'stream-stable';
        tailEl.className   = 'stream-tail';
        // display:contents in the stylesheet, so these wrappers generate no
        // boxes and the message keeps the spacing it has always had.
        target.appendChild(thinkEl);
        target.appendChild(stableEl);
        target.appendChild(tailEl);
    }

    function flush() {
        raf = 0;
        ensureNodes();
        const split = splitThink(pending, true);

        if (split.thinkHtml !== lastThink) {
            thinkEl.innerHTML = split.thinkHtml;
            lastThink = split.thinkHtml;
            // Pin the reasoning to its newest line. The box is height-capped
            // while streaming, so without this it shows the FIRST few lines
            // and grows silently past them -- a chain that is very much moving
            // looks frozen. Set inside the same rAF as the write above.
            const ts = thinkEl.querySelector('.think-block.streaming .think-scroll');
            if (ts) ts.scrollTop = ts.scrollHeight;
        }

        // mainText is '' while a think block is open and jumps to the full
        // answer when </think> lands. If it ever shrank, anything already
        // promoted to stable is no longer trustworthy.
        const mainText = split.mainText;
        if (mainText.length < stableEnd) { stableEnd = 0; stableEl.innerHTML = ''; }

        const b = lastStableBoundary(mainText);
        if (b > stableEnd) {
            stableEl.innerHTML = renderBody(mainText.slice(0, b));
            stableEnd = b;
        }
        tailEl.innerHTML = renderBody(mainText.slice(stableEnd));

        // Same frame as the writes above, so this rides a layout the browser
        // was going to compute anyway instead of forcing a fresh one.
        if (wantScroll) chat.scrollTop = chat.scrollHeight;
    }

    return {
        paint(text) {
            pending = text;
            // Sampled BEFORE the write: shouldAutoScroll() measures the
            // pre-write scroll position, and asking after the DOM grew would
            // report the user as scrolled up on every token.
            wantScroll = shouldAutoScroll();
            if (!raf) raf = requestAnimationFrame(flush);
        },
        stop() { if (raf) cancelAnimationFrame(raf); raf = 0; }
    };
}
