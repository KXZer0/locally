// The web-search tool: whether it is offered, and the grounded turn itself.
//
// In here: the composer toggle -- which says whether the model is OFFERED the
// tool, not whether it must use it -- and the single /v1/search request that
// does search, fetch, chunk, embed, rerank and answer server-side.
// Not in here: the sources list (chat/sources-panel.js) and the decision to
// search, which belongs to the model.

import { invalidateContextCount } from './context.js';
import { setGenerating } from './generation.js';
import { buildMessages } from './prompt.js';
import { openSources, renderSources } from './sources-panel.js';
import { actionButton, addMessageActions, addMeta, recordThroughput } from './thread.js';
import { input } from '../core/dom.js';
import { escapeHtml, estimateTokensIn } from '../core/format.js';
import { activityHtml, scrollToBottom } from '../core/paint.js';
import { chatHistory, lastHealthData } from '../core/state.js';
import { renderMarkdown } from '../markdown/render.js';
import { mode } from '../ui/tabs.js';

// Web search and local Python are no longer composer buttons. They are tools
// the server offers the model on every turn it can support them, so deciding
// whether a question needs a search or an exact calculation is the model's
// job -- which is what it is for. The toggles asked the user to answer that
// in advance, and got it wrong in both directions: a forced search on
// "hello", none on a question about last week.
export let webSearchContext = null;

// The one control that survived, and it is not a mode: it says whether the
// model is OFFERED the search tool at all. It still decides when to use it --
// but on a train with no signal, or when you simply do not want the machine
// reaching the network, the tool should not be on the table. Off means the
// spec is never rendered into the prompt, so it costs nothing either.
export const webSearchBtn = document.getElementById('web-search-btn');
export let webSearchAllowed = localStorage.getItem('locally-web-search') !== 'off';
export let webSearchReady = false;

export function applyWebSearchHealth(health) {
    if (!webSearchBtn) return;
    const ws = (health && health.web_search) || {};
    // on_demand counts as ready: the backend is startable and the first search
    // wakes it. Greying out a backend that is merely asleep is the same
    // mistake as greying out a model that has idle-unloaded.
    webSearchReady = !!(ws.enabled && (ws.reachable || ws.on_demand));
    webSearchBtn.disabled = !webSearchReady;
    const on = webSearchReady && webSearchAllowed;
    webSearchBtn.classList.toggle('active', on);
    webSearchBtn.setAttribute('aria-pressed', String(on));
    webSearchBtn.setAttribute('aria-label',
        on ? 'Web search allowed' : 'Web search off');
    webSearchBtn.title = !webSearchReady
        ? (ws.reason || 'No search backend is configured')
        : on ? 'The model may search the web when it needs to'
             : 'Web search off — the model will answer from what it knows';
}

if (webSearchBtn) {
    webSearchBtn.addEventListener('click', () => {
        if (!webSearchReady) return;
        webSearchAllowed = !webSearchAllowed;
        localStorage.setItem('locally-web-search', webSearchAllowed ? 'on' : 'off');
        applyWebSearchHealth(lastHealthData);
        input.focus();
    });
}
// Web-search turn. /v1/search does the whole pipeline server-side -- search,
// fetch, chunk, embed and rerank on the NPU, then a grounded answer -- so this
// is one request, not an orchestration. It does not stream: the answer only
// exists once the passages have been ranked, and pretending otherwise would
// mean showing a cursor blinking at nothing for ten seconds.
export async function runWebSearch(assistantDiv, question, mode = 'brief') {
    setGenerating(true);
    const t0 = performance.now();
    // Say what is happening. A search turn is ~20-33s against ~12s for a plain
    // answer, and for most of that the model has not been called yet — a bare
    // typing dot for that long is indistinguishable from a hung request.
    // These are real server stages, not a timer: each arrives when that phase
    // actually starts.
    const showStage = (text) => {
        assistantDiv.body.innerHTML = activityHtml(text);
        scrollToBottom(true);
    };
    showStage(mode === 'analyze' && webSearchContext
        ? 'analysing the saved sources' : 'searching the web');

    let data = null;
    try {
        const resp = await fetch('/v1/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: question, top_k: 5, answer: true, stream: true, mode,
                // The server owns the /no_think decision for grounded turns;
                // analyse must be able to reason even when the chat switch is off.
                messages: buildMessages(false, false, true),
                ...(mode === 'analyze' && webSearchContext
                    ? { passages: webSearchContext.results,
                        sources: webSearchContext.sources }
                    : {}),
            }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => null);
            throw new Error(err?.error?.message || `HTTP ${resp.status}`);
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const raw = line.slice(6);
                if (raw === '[DONE]') continue;
                let ev; try { ev = JSON.parse(raw); } catch { continue; }
                if (ev.event === 'stage') showStage(ev.detail);
                else if (ev.event === 'error') throw new Error(ev.message);
                else if (ev.event === 'done') data = ev;
            }
        }
        if (!data) throw new Error('the search ended without a result');

        // Say WHY there is no answer. The generic line was shown even when the
        // server had reported a concrete failure in answer_error, which made a
        // broken generate() call look like the web simply had nothing on the
        // subject -- with a meta line underneath claiming 3 pages were read.
        if (data.answer_error) throw new Error(data.answer_error);
        const body = (data.answer || '').trim()
            || (data.note
                || 'The sources were read but the model returned nothing.');
        assistantDiv.body.innerHTML = renderMarkdown(body);
        chatHistory.push({ role: 'assistant', content: body });
        invalidateContextCount(true);
        if (data.results?.length && mode === 'brief') {
            // Keep the ranked passages, not just the prose answer: Analyse is
            // a synthesis pass over this cache and must not search SearXNG again.
            webSearchContext = { results: data.results, sources: data.sources || [] };
        }

        const t = data.timings_ms || {};
        const answerSecs = (t.answer || 0) / 1000;
        addMeta(assistantDiv, 'WEB', ((performance.now() - t0) / 1000).toFixed(1),
                `${data.pages_read}/${data.pages_found} pages`
                + (data.reranked && t.rerank != null ? ` · ranked ${t.rerank}ms` : ''),
                answerSecs > 0 ? estimateTokensIn(body) / answerSecs : 0);

        // Sources move to the panel. The answer keeps a button rather than a
        // trailing list, so the prose ends where the prose ends.
        const extras = [];
        if (data.sources?.length) {
            renderSources(data.sources, data.passages_dropped);
            const readCount = data.sources.filter(x => x.read !== false).length;
            const label = readCount === data.sources.length
                ? `${data.sources.length} sources`
                : `${readCount}/${data.sources.length} sources`;
            // Deliberately NOT opened automatically. The answer is what was
            // asked for; a panel sliding over it the moment it arrives
            // interrupts the thing you were waiting to read.
            extras.push(actionButton(label, 'i-panel',
                () => openSources(data.sources, data.passages_dropped)));
        }
        addMessageActions(assistantDiv, body, extras);
        if (t.answer) recordThroughput(estimateTokensIn(body), t.answer / 1000);
        return body;
    } catch (err) {
        assistantDiv.body.innerHTML =
            `<span class="err">Web search failed: ${escapeHtml(err.message)}</span>`;
        return '';
    } finally {
        setGenerating(false);
        scrollToBottom(true);
    }
}
