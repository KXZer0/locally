// Writing messages into the chat thread.
//
// In here: creating a message bubble, stamping the engine that answered onto
// it, the timing/rate line, the per-answer action row, and the running
// throughput average those numbers feed.
// Not in here: markdown (markdown/render.js), the request (chat/send.js), and
// the streaming paint (markdown/stream-painter.js). This module owns the
// message's STRUCTURE; something else owns its content.

import { brandMark, emptyState, thread } from '../core/dom.js';
import { escapeHtml } from '../core/format.js';
import { paintDevice, scrollToBottom, shouldAutoScroll } from '../core/paint.js';
import { renderMarkdown } from '../markdown/render.js';
import { capThread } from './thread-window.js';

// --- Message actions ---------------------------------------------------------

// Copy the whole answer, not just a code block. Copies the SOURCE text rather
// than the rendered DOM, so what lands on the clipboard is the markdown the
// model wrote — pasting a rendered table into an editor is not what anyone
// wants.
export function addMessageActions(msgDiv, text, extra) {
    if (!msgDiv || msgDiv.querySelector('.msg-actions')) return null;
    const row = document.createElement('div');
    row.className = 'msg-actions';

    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'msg-action';
    copy.title = 'Copy this answer';
    copy.innerHTML = '<svg class="ic" aria-hidden="true"><use href="#i-copy"/></svg><span>Copy</span>';
    copy.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(text);
            copy.classList.add('ok');
            copy.querySelector('span').textContent = 'Copied';
            setTimeout(() => {
                copy.classList.remove('ok');
                copy.querySelector('span').textContent = 'Copy';
            }, 1400);
        } catch { copy.querySelector('span').textContent = 'Press Ctrl+C'; }
    });
    row.appendChild(copy);

    for (const btn of (extra || [])) row.appendChild(btn);
    msgDiv.appendChild(row);
    return row;
}

export function actionButton(label, icon, onClick) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'msg-action';
    b.title = label;
    b.innerHTML = `<svg class="ic" aria-hidden="true"><use href="#${icon}"/></svg><span>${escapeHtml(label)}</span>`;
    b.addEventListener('click', onClick);
    return b;
}
// --- Session throughput ------------------------------------------------------

// A per-turn rate answers "was that one fast"; the running average answers "is
// this model worth keeping loaded", which is the question actually being asked
// on a machine that swaps models. Weighted by tokens, not a mean of rates, or
// one short turn would skew it.
export let sessionTokens = 0;
export let sessionSeconds = 0;
export function recordThroughput(tokens, seconds) {
    if (!(tokens > 0) || !(seconds > 0)) return;
    sessionTokens += tokens;
    sessionSeconds += seconds;
    renderSessionRate();
}

export function renderSessionRate() {
    const el = document.getElementById('session-rate');
    if (!el) return;
    if (!sessionSeconds) { el.textContent = ''; return; }
    // Decode-only, same as the per-turn figure, so the two are comparable and
    // neither is dragged down by how long the prompt was.
    el.textContent = `${(sessionTokens / sessionSeconds).toFixed(1)} tok/s avg`;
}
export let lastRenderedRole = null; // for message grouping

// New Chat empties the thread, so the next message must not be grouped with a
// bubble that is no longer on screen.
export function resetMessageGrouping() { lastRenderedRole = null; }
// --- Message rendering ---

export function addMessage(role, content, options) {
    // Sample before any DOM write; reading scrollHeight after append would
    // synchronously lay out the thread we just invalidated.
    const wantScroll = shouldAutoScroll();
    const opts = options || {};
    const div = document.createElement('div');
    div.className = `message ${role} entering`;
    if (lastRenderedRole === role) div.classList.add('grouped');
    lastRenderedRole = role;

    const label = document.createElement('div');
    label.className = 'msg-role';
    label.textContent = role === 'user' ? 'You' : (opts.device || 'Assistant');
    div.appendChild(label);

    const bodyDiv = document.createElement('div');
    bodyDiv.className = 'msg-body';
    if (opts.html) bodyDiv.innerHTML = content;
    else bodyDiv.innerHTML = renderMarkdown(content);
    div.appendChild(bodyDiv);

    // The mark sits bottom-left below the answer and turns while that turn is
    // streaming (03-thread.css places it, 08-motion.css moves it). One copy:
    // the streaming state used to stack a second, counter-rotating one to carry a
    // second period, and a half-faded duplicate of the logo behind the logo is
    // what that looks like on screen. The two periods now ride two properties --
    // rotation on the svg, brightness on the wrapper. Static markup, so innerHTML
    // here is the sprite reference and nothing else.
    //
    // It sits inside a `.msg-footer` row rather than directly in the message,
    // because the timing line (addMeta(), appended later once the turn ends)
    // belongs beside it, not stacked under it -- one line reading "[mark] NPU
    // 2.3s 18.8 tok/s" rather than the mark on its own row with the numbers
    // underneath. addMeta() finds this row by class and appends into it.
    if (role === 'assistant') {
        const footer = document.createElement('div');
        footer.className = 'msg-footer';
        const mark = document.createElement('span');
        mark.className = 'msg-mark';
        mark.setAttribute('aria-hidden', 'true');
        mark.innerHTML =
            '<svg class="mark" viewBox="0 0 100 100"><use href="#i-mark"/></svg>';
        footer.appendChild(mark);
        div.appendChild(footer);
    }

    if (opts.device) paintDevice(div, opts.device);

    emptyState.hidden = true;
    thread.appendChild(div);
    requestAnimationFrame(() => requestAnimationFrame(() => {
        div.classList.remove('entering');
    }));
    scrollToBottom(wantScroll);
    // Callers stream into the body, not the wrapper, so the role label and
    // meta line survive every re-render.
    div.body = bodyDiv;
    div.label = label;
    // After the append, never before: capThread() measures the height it is
    // about to pin, and an element that is not in the document measures zero.
    capThread();
    return div;
}

// Stamp the device onto a message once the response headers reveal it, and
// echo it to the brand mark so the last-used engine is always on screen.
export function attachDevice(msgDiv, device) {
    if (!device) return;
    paintDevice(msgDiv, device);
    msgDiv.label.textContent = device;
    paintDevice(brandMark, device);
}

// `rate` is tokens/second for THIS turn. It sits next to the elapsed time
// because the two are only meaningful together: 4.7s says nothing without
// knowing whether that produced ten tokens or four hundred, which is exactly
// what a benchmark needs.
export function addMeta(msgDiv, device, elapsed, extra, rate) {
    const meta = document.createElement('div');
    meta.className = 'meta';
    const bits = [];
    if (device) bits.push(`<span class="device-tag">${escapeHtml(device)}</span>`);
    bits.push(`${elapsed}s`);
    if (rate > 0) bits.push(`<span class="rate">${rate.toFixed(1)} tok/s</span>`);
    if (extra) bits.push(`<span class="sep">·</span> ${escapeHtml(extra)}`);
    meta.innerHTML = bits.join(' ');
    // Lands beside the mark, not under it: addMessage() built a `.msg-footer`
    // row around the mark for exactly this. Falls back to the message itself
    // for any caller that didn't go through addMessage('assistant', ...).
    appendMeta(msgDiv, meta);
    return meta;
}

export function appendMeta(msgDiv, node) {
    (msgDiv.querySelector('.msg-footer') || msgDiv).appendChild(node);}

export function renderPythonRuns(runs) {
    if (!Array.isArray(runs) || !runs.length) return '';
    return runs.map((run, index) => {
        const status = run.timed_out ? `timed out · ${run.elapsed_ms}ms`
            : `exit ${run.exit_status} · ${run.elapsed_ms}ms`;
        const output = `stdout:\n${run.stdout || '(empty)'}\n\n` +
            `stderr:\n${run.stderr || '(empty)'}`;
        return `<details class="python-run" open>` +
            `<summary>Python calculation ${index + 1} · ${escapeHtml(status)}` +
            `${run.truncated ? ' · output truncated' : ''}</summary>` +
            `<div class="python-run-body">` +
            `<pre><code>${escapeHtml(run.code || '')}</code></pre>` +
            `<pre><code>${escapeHtml(output)}</code></pre>` +
            `</div></details>`;
    }).join('');
}
