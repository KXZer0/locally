// How much of the context window this turn will use, and what to drop.
//
// In here: the token count behind the composer ring -- an approximation while
// typing, reconciled against the server's real tokenizer after a pause -- plus
// compaction, which trims history BEFORE a send rather than after a failure.
// Not in here: the memory HUD's RAM figures (system/memory-hud.js). The ring
// and the HUD share a surface, not a subject.

import { buildMessages } from './prompt.js';
import { addMessage } from './thread.js';
import { contextCaption, contextKv, contextLimit, contextMeterLabel, contextModelName, contextModelRoute, contextPercent, contextRemaining, contextRingValue, contextUsed, input, memHud, memHudPill } from '../core/dom.js';
import { escapeHtml } from '../core/format.js';
import { paintDevice } from '../core/paint.js';
import { attachedDoc, chatHistory, lastHealthData, loadedModelId, setChatHistory } from '../core/state.js';
import { _slotUp } from '../system/devices.js';
import { modelSwapBusy } from '../system/swap.js';

// --- Composer model + context ---------------------------------------------
// The model chip is progressive disclosure (idle only); the context ring is
// persistent because running out of prompt space is a state worth seeing
// before Send. Counts update approximately while typing and are reconciled
// with the server's real tokenizer after a short idle pause.
export let currentChatSlot = null;
export let contextCountTimer = 0;
export let contextCountSerial = 0;
export let exactContextTokens = null;
export let exactContextRevision = -1;
export let contextRevision = 0;
export let cachedBaseContextTokens = null;

export function activeChatSlot(health) {
    const slots = Object.values(health?.devices || {});
    if (!slots.length) return null;
    const at = loadedModelId.lastIndexOf('@');
    const model = at >= 0 ? loadedModelId.slice(0, at) : loadedModelId;
    const device = at >= 0 ? loadedModelId.slice(at + 1).toUpperCase() : '';
    return slots.find(s => (s.device_name || '').toUpperCase() === device)
        || slots.find(s => s.model === model)
        || slots[0];
}

export function contextTokenLimit(slot) {
    if (!slot) return null;
    // The server now resolves this for every device — the smallest of the
    // model's max_position_embeddings, the NPU prompt cap and the KV pool.
    // The NPU fallback below is only for talking to an older server; it used
    // to be the sole reason the NPU was the only device that tracked context.
    if (slot.context_tokens) return Number(slot.context_tokens);
    if ((slot.device_name || '').toUpperCase() === 'NPU') return 4096;
    return null;
}

// "36k · limited by kv pool" — the reason belongs next to the number, because
// a small window on a large model is only alarming without it.
export function contextLimitReason(slot) {
    return slot && slot.context_limit_by ? slot.context_limit_by : null;
}

export function countableContent(content) {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';
    return content.map(block => block?.type === 'text' ? (block.text || '') : '').join('\n');
}

export function contextCountPayload(includeDraft = true) {
    const built = buildMessages(false).map(m => ({ ...m }));
    if (includeDraft) {
        const draft = input.value.trim();
        const attachmentText = attachedDoc?.text || '';
        if (draft || attachmentText) {
            built.push({ role: 'user', content: [attachmentText, draft].filter(Boolean).join('\n\n') });
        }
    }

    const system = built.filter(m => m.role === 'system')
        .map(m => countableContent(m.content)).filter(Boolean).join('\n\n');
    const messages = built.filter(m => m.role !== 'system').map(m => ({
        role: m.role === 'assistant' ? 'assistant' : 'user',
        content: countableContent(m.content),
    }));
    return { model: loadedModelId, max_tokens: 1, system, messages };
}

export function estimateContextTokens(payload) {
    const chars = payload.system.length
        + payload.messages.reduce((n, m) => n + m.content.length, 0);
    return Math.max(0, Math.ceil(chars / 4) + payload.messages.length * 4);
}

export function invalidateContextCount(baseChanged = false) {
    contextRevision++;
    if (baseChanged) cachedBaseContextTokens = null;
}

export function liveContextEstimate() {
    if (cachedBaseContextTokens == null) {
        cachedBaseContextTokens = estimateContextTokens(contextCountPayload(false));
    }
    const draftChars = input.value.trim().length + (attachedDoc?.text?.length || 0);
    return cachedBaseContextTokens + (draftChars ? Math.ceil(draftChars / 4) + 4 : 0);
}

export function tokenText(value) {
    if (value == null) return '—';
    if (value < 1000) return value.toLocaleString();
    const digits = value < 10000 ? 1 : 0;
    return `${(value / 1000).toFixed(digits)}k`;
}

export function updateContextDisplay() {
    const exact = exactContextRevision === contextRevision ? exactContextTokens : null;
    const used = exact ?? liveContextEstimate();
    const limit = contextTokenLimit(currentChatSlot);
    const remaining = limit == null ? null : Math.max(0, limit - used);
    const ratio = limit ? Math.max(0, Math.min(1, used / limit)) : 0;
    const percent = Math.round(ratio * 100);

    contextRingValue.style.strokeDashoffset = String(100 - percent);
    memHud.dataset.context = ratio >= 0.95 ? 'critical' : ratio >= 0.8 ? 'warning' : 'normal';
    contextPercent.textContent = limit == null ? '—' : `${percent}%`;
    contextUsed.textContent = `${tokenText(used)}${exact == null ? ' est.' : ''}`;
    contextLimit.textContent = tokenText(limit);
    contextRemaining.textContent = tokenText(remaining);
    contextKv.textContent = currentChatSlot?.kv_pool_gb
        ? `${currentChatSlot.kv_pool_gb} GB`
        : ((currentChatSlot?.device_name || '').toUpperCase() === 'NPU'
            ? 'fixed NPU cap' : 'model managed');

    const model = currentChatSlot?.model || 'No model';
    contextCaption.textContent = `${exact == null ? 'Live estimate' : 'Tokenizer count'} · ${model}`;
    contextMeterLabel.textContent = limit == null
        ? `${tokenText(used)} tokens used. Context limit unavailable.`
        : `${tokenText(used)} of ${tokenText(limit)} context tokens used, ${percent} percent.`;
}

export async function refreshExactContextCount() {
    const payload = contextCountPayload();
    const serial = ++contextCountSerial;
    const revision = contextRevision;
    memHud.dataset.count = 'loading';
    try {
        const resp = await fetch('/v1/messages/count_tokens', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            memHud.dataset.count = 'estimate';
            contextCaption.title = `Tokenizer count unavailable (HTTP ${resp.status})`;
            return;
        }
        const data = await resp.json();
        if (serial !== contextCountSerial || revision !== contextRevision) return;
        exactContextTokens = Number(data.input_tokens) || 0;
        exactContextRevision = revision;
        memHud.dataset.count = 'exact';
        contextCaption.title = 'Counted with the loaded model tokenizer';
        updateContextDisplay();
    } catch (err) {
        memHud.dataset.count = 'estimate';
        contextCaption.title = `Tokenizer count unavailable: ${err.message}`;
    }
}

export function scheduleExactContextCount(delay = 550) {
    clearTimeout(contextCountTimer);
    contextCountTimer = setTimeout(refreshExactContextCount, delay);
}

export function syncComposerDraftState() {
    invalidateContextCount(false);
    updateContextDisplay();
    scheduleExactContextCount();
}

export function updateComposerHealth(health) {
    currentChatSlot = activeChatSlot(health);
    const device = currentChatSlot?.device_name || '';
    const status = currentChatSlot?.status || 'unavailable';
    const model = currentChatSlot?.model || 'No chat model';
    paintDevice(memHud, device);
    memHud.classList.toggle('busy', status === 'loading' || status === 'warming_up');
    memHud.classList.toggle('down', !_slotUp(currentChatSlot || {}));
    memHudPill.title = currentChatSlot
        ? `${device} · ${model} · ${status}` : 'No chat model ready';
    if (!modelSwapBusy) {
        contextModelName.textContent = model;
        contextModelRoute.textContent = device || '—';
    }
    updateContextDisplay();
}
// --- Auto-compaction ---------------------------------------------------------
// The NPU window is 8192 tokens and hard: the pipeline REFUSES an over-long
// prompt, it does not truncate (see TODONT.md). Nothing used to trim history,
// so a long chat walked straight into a failure and the model degraded into
// repeating itself on the way there.
//
// Compaction runs BEFORE the send, at 75% of the window rather than at 100%:
// the reply needs room too, and a guard that only fires on failure has already
// failed. Turns are dropped oldest-first in whole user+assistant pairs -- a
// lone user message with its answer removed is worse than no message at all.

export const COMPACT_AT = 0.75;    // start compacting once the window is this full
export const COMPACT_TO = 0.55;    // and free down to here, so it is not every turn

export function compactHistory(forVoice) {
    const limit = contextTokenLimit(activeChatSlot(lastHealthData));
    if (!limit || chatHistory.length < 4) return null;

    // Measures the real composed payload, so the system prompt and the voice
    // directive are counted too -- they are re-sent every turn and on an 8k
    // window they are not a rounding error.
    const measure = () => estimateContextTokens(contextCountPayload(false));
    if (measure() <= limit * COMPACT_AT) return null;

    const target = limit * COMPACT_TO;
    let dropped = 0;
    // Oldest first, in whole pairs, never touching the most recent exchange.
    while (chatHistory.length > 2 && measure() > target) {
        setChatHistory(chatHistory.slice(2));
        invalidateContextCount(true);
        dropped += 2;
    }
    if (!dropped) return null;

    // Say it happened. Silent memory loss is indistinguishable from the model
    // being stupid, which is exactly how this read before.
    const note = `${dropped} earlier message${dropped === 1 ? '' : 's'} dropped `
               + `to stay within the ${tokenText(limit)}-token window`;
    if (!forVoice) {
        addMessage('assistant', `<em>Compacted</em> — ${escapeHtml(note)}.`,
                   { html: true });
    }
    console.info('[compact]', note);
    return note;
}
