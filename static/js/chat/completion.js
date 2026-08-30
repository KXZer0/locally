// One request/response cycle against /v1/chat/completions.
//
// In here: the request body, the SSE consumer shared by every send path, the
// turn that runs on it, and "Just answer me, dammit!".
// Not in here: composing the messages (chat/prompt.js), what a send DOES with
// the answer (chat/send.js), and the voice turn, which needs its own loop
// because it feeds a speech queue while the tokens are still arriving.

import { invalidateContextCount, scheduleExactContextCount, updateContextDisplay } from './context.js';
import { abortController, cancelGeneration, isGenerating, setAbortController, setGenerating } from './generation.js';
import { buildMessages } from './prompt.js';
import { addMessage, addMessageActions, addMeta, appendMeta, attachDevice, recordThroughput, renderPythonRuns } from './thread.js';
import { input, temperatureSlider, thread } from '../core/dom.js';
import { escapeHtml, estimateTokensIn } from '../core/format.js';
import { chatHistory, loadedModelId } from '../core/state.js';
import { renderMarkdown } from '../markdown/render.js';
import { makeStreamPainter } from '../markdown/stream-painter.js';

// --- Request builder ---

export function buildRequestBody(overrides) {
    const temp = temperatureSlider.value / 100;

    const body = {
        // The resident model's id, not the dropdown's value — the dropdown now
        // lists what is on disk, which is a different question.
        model: loadedModelId,
        messages: buildMessages(false),
        stream: true,
        max_tokens: 16384,
        // The web UI is the "does it work" surface, and a thinking-loop on a
        // slow iGPU reads as "it doesn't". Ollama's 1.1 breaks loops faster
        // than the server default (1.05, kept mild for coding agents).
        repetition_penalty: 1.1,
    };

    if (temp > 0) body.temperature = temp;

    return { ...body, ...overrides };
}
// Seconds to the first token of the most recent turn. Prefill, not generation.
export let lastTurnTtft = 0;
// Shared by every send path: consumes the SSE body, paints tokens into
// `target` as they arrive, and returns the full text. `onFinish` reports the
// terminal finish_reason, which is the only way to tell a model that stopped
// because it was done from one that stopped because it ran out of tokens.
export async function consumeStream(resp, target, onDelta, onFinish) {
    const painter = target ? makeStreamPainter(target) : null;
    // Prefill and decode are different machines' problems and must not be
    // averaged together. On the NPU prefill is a fixed ~3.3s (no prefix cache
    // there), so dividing output by TOTAL time reports 3 tok/s for a short
    // answer and 21 for a long one on a model whose real rate is 17 both
    // times. lastTurnTtft is written here and read by the meta line.
    lastTurnTtft = 0;
    const streamStart = performance.now();
    let fullText = '';
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete line

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            try {
                const chunk = JSON.parse(data);
                const reason = chunk.choices?.[0]?.finish_reason;
                if (reason && onFinish) onFinish(reason);
                const delta = chunk.choices?.[0]?.delta?.content;
                if (delta) {
                    if (!lastTurnTtft) {
                        lastTurnTtft = (performance.now() - streamStart) / 1000;
                    }
                    fullText += delta;
                    if (painter) painter.paint(fullText);
                    if (onDelta) onDelta(fullText, delta);
                }
            } catch {}
        }
    }
    // Drop any frame still queued: both callers overwrite `target` with the
    // full render on the very next line, and a late flush would paint the
    // three-part streaming DOM back over it.
    if (painter) painter.stop();
    return fullText;
}
// One request/response cycle. `messages` overrides history composition so
// "Just answer me" can force no-think without mutating settings.
export async function runCompletion(assistantDiv, overrides, metaNote) {
    const t0 = performance.now();
    setGenerating(true);

    // After 3s with no response, check if a model is reloading and show that
    const reloadCheckTimer = setTimeout(async () => {
        try {
            const r = await fetch('/health');
            const data = await r.json();
            const reloading = Object.values(data.devices || {}).some(
                d => d.status === 'loading' || d.status === 'warming_up'
            );
            if (reloading && isGenerating) {
                assistantDiv.body.innerHTML =
                    '<span class="typing-indicator"></span> ' +
                    '<span style="color:var(--text-dim)">Reloading model…</span>';
            }
        } catch {}
    }, 3000);

    try {
        setAbortController(new AbortController());
        const resp = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: abortController.signal,
            body: JSON.stringify(buildRequestBody(overrides)),
        });
        clearTimeout(reloadCheckTimer);

        const device = resp.headers.get('X-Device') || '';
        attachDevice(assistantDiv, device);

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            assistantDiv.body.innerHTML =
                `<span style="color:var(--error)">${escapeHtml(err.error?.message || 'Error')}</span>`;
            return null;
        }

        const contentType = resp.headers.get('content-type') || '';
        let text;
        let pythonRuns = [];
        if (contentType.includes('text/event-stream')) {
            text = await consumeStream(resp, assistantDiv.body);
        } else {
            const data = await resp.json();
            text = data.choices?.[0]?.message?.content || '';
            pythonRuns = data.python_runs || [];
            lastTurnTtft = 0;
        }

        // Re-render with streaming=false to collapse the think block
        assistantDiv.body.innerHTML = renderPythonRuns(pythonRuns) +
            renderMarkdown(text, false);
        const elapsed = (performance.now() - t0) / 1000;
        const turnTokens = estimateTokensIn(text);
        // Decode only. Including prefill measures how long the question was,
        // not how fast the model writes.
        const decode = Math.max(0.001, elapsed - lastTurnTtft);
        const pythonNote = pythonRuns.length
            ? `${pythonRuns.length} calculation${pythonRuns.length === 1 ? '' : 's'}`
            : '';
        const note = lastTurnTtft
            ? `${lastTurnTtft.toFixed(1)}s to first token`
                + (metaNote ? ` · ${metaNote}` : '')
                + (pythonNote ? ` · ${pythonNote}` : '')
            : [metaNote, pythonNote].filter(Boolean).join(' · ');
        addMeta(assistantDiv, device, elapsed.toFixed(1), note, turnTokens / decode);
        addMessageActions(assistantDiv, text);
        recordThroughput(turnTokens, decode);
        chatHistory.push({ role: 'assistant', content: text });
        invalidateContextCount(true);
        updateContextDisplay();
        scheduleExactContextCount(0);
        return text;
    } catch (err) {
        clearTimeout(reloadCheckTimer);
        if (err.name === 'AbortError') {
            assistantDiv.body.innerHTML +=
                '<br><span style="color:var(--text-dim)">[cancelled]</span>';
        } else {
            assistantDiv.body.innerHTML =
                `<span style="color:var(--error)">${escapeHtml(err.message)}</span>`;
        }
        return null;
    } finally {
        setGenerating(false);
        setAbortController(null);
    }
}
// --- Just answer me, dammit! ---

export async function justAnswerMe(event) {
    event.stopPropagation();
    cancelGeneration();

    const hasUserMsg = chatHistory.some(m => m.role === 'user');
    if (!hasUserMsg) return;

    await new Promise(r => setTimeout(r, 100));   // let the abort settle

    // Mark the abandoned bubble so the retry isn't mistaken for a duplicate
    const lastBubble = thread.querySelector('.message.assistant:last-child');
    if (lastBubble && !lastBubble.querySelector('.meta')) {
        const note = document.createElement('div');
        note.className = 'meta';
        note.textContent = '[retrying without thinking]';
        appendMeta(lastBubble, note);
    }

    const assistantDiv = addMessage('assistant', '<span class="typing-indicator"></span>',
                                    { html: true });
    await runCompletion(assistantDiv, { messages: buildMessages(true) }, 'no-think');
    input.focus();
}
window.justAnswerMe = justAnswerMe;
