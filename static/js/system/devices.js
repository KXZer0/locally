// What each engine is holding, painted into the rail and the empty state.
//
// In here: the device pills, the empty-state rows, and the flags that say
// which audio models exist -- both surfaces read the same /health payload, so
// "which model is on which engine" is answered identically in both.
// Not in here: loading or swapping a model (system/swap.js) and the utility
// engines' availability (util/engine.js), which answers a different question.

import { invalidateContextCount, scheduleExactContextCount, updateComposerHealth } from '../chat/context.js';
import { deviceRail, emptyDevices, micBtn, vadHint, voicePtt, voiceToggle, voiceTuning, voiceVadCheckbox } from '../core/dom.js';
import { escapeHtml, shortModel } from '../core/format.js';
import { paintDevice } from '../core/paint.js';
import { asrReady, lastHealthData, setAsrReady, setLoadedModelId, setTtsReady, setVadReady, vadReady } from '../core/state.js';
import { renderUtilAvailability } from '../util/engine.js';
import { populateVoices } from '../voice/state.js';

export let lastDevicesRenderSignature = '';

// Cleared when a poll fails, so the next successful one repaints from scratch
// instead of matching a signature for a rail that is no longer on screen.
export function resetDevicesRenderSignature() { lastDevicesRenderSignature = ''; }
// Both the top rail and the empty state read from the same /health payload,
// so "which model is on which engine" is answered identically in both places.
export function renderDevices(health) {
    const renderSignature = JSON.stringify({
        devices: health.devices || {},
        util: health.util || {},
        whisper: health.whisper || null,
        tts: health.tts || null,
        vad: health.vad || null,
    });
    if (renderSignature === lastDevicesRenderSignature) return;
    lastDevicesRenderSignature = renderSignature;

    const devices = health.devices || {};
    const entries = Object.values(devices);
    const ocrReady = Object.values(health.util || {}).some(u =>
        u.tasks?.read && _slotUp(u));

    deviceRail.innerHTML = '';
    emptyDevices.innerHTML = '';

    for (const d of entries) {
        const dev = d.device_name || '';

        const pill = document.createElement('div');
        pill.className = 'device-pill' + (d.status === 'ready' ? '' : ' down');
        paintDevice(pill, dev);
        pill.title = `${d.model} on ${d.device || dev} — ${d.status}`;
        pill.innerHTML =
            `<span class="dot"></span><span class="dev-name">${escapeHtml(dev)}</span>` +
            `<span>${escapeHtml(shortModel(d.model))}</span>`;
        deviceRail.appendChild(pill);

        const row = document.createElement('div');
        row.className = 'empty-device';
        paintDevice(row, dev);
        const notes = [d.type === 'vlm' ? 'vision' : 'text', 'files'];
        // A text-only NPU model still receives readable image content through
        // the utility OCR bridge. Name that path explicitly: it can read text
        // in screenshots/scans, but it does not gain full visual understanding.
        if (d.type !== 'vlm' && ocrReady) notes.push('images via OCR');
        if (d.tools) notes.push('tools');
        if (d.status !== 'ready') notes.push(d.status);
        row.innerHTML =
            `<span class="ed-dev">${escapeHtml(dev)}</span>` +
            `<span class="ed-model">${escapeHtml(d.model || '—')}</span>` +
            `<span class="ed-note">${escapeHtml(notes.join(' · '))}</span>`;
        emptyDevices.appendChild(row);
    }

    // The mic only appears when the server actually has an ASR model loaded —
    // an inert button that fails on click is worse than no button.
    setAsrReady(!!(health.whisper && _slotUp(health.whisper)));
    setTtsReady(!!(health.tts && _slotUp(health.tts)));
    // Auto turn-taking is a capability, not a preference: without the VAD model
    // there is nothing that can tell speech from a noisy room, so the control
    // is disabled and says why rather than silently doing nothing useful.
    setVadReady(!!(health.vad && _slotUp(health.vad)));
    micBtn.hidden = !asrReady;
    voiceToggle.disabled = !asrReady;
    voicePtt.disabled = !asrReady;
    voiceVadCheckbox.disabled = !vadReady;
    if (vadHint) vadHint.hidden = vadReady;
    if (voiceTuning) voiceTuning.hidden = !vadReady;

    if (health.tts) populateVoices(health.tts);

    if (health.tts) {
        const t = health.tts;
        const row = document.createElement('div');
        row.className = 'empty-device';
        paintDevice(row, t.device_name || 'CPU');
        row.innerHTML =
            `<span class="ed-dev">${escapeHtml(t.device_name || 'CPU')}</span>` +
            `<span class="ed-model">${escapeHtml(t.model || 'tts')}</span>` +
            `<span class="ed-note">text-to-speech</span>`;
        emptyDevices.appendChild(row);
    }

    if (health.whisper) {
        const w = health.whisper;
        const row = document.createElement('div');
        row.className = 'empty-device';
        paintDevice(row, w.device_name || 'CPU');
        row.innerHTML =
            `<span class="ed-dev">${escapeHtml(w.device_name || 'CPU')}</span>` +
            `<span class="ed-model">${escapeHtml(w.model || 'whisper')}</span>` +
            `<span class="ed-note">speech-to-text</span>`;
        emptyDevices.appendChild(row);
    }

    if (!entries.length) {
        emptyDevices.innerHTML =
            '<div class="empty-device"><span class="ed-model">No models loaded</span></div>';
    }

    renderUtilAvailability(health.util || {});
}

// An idle-unloaded slot still serves — it reloads on the next request — so
// treat it as available rather than hiding the control that would wake it.
export function _slotUp(slot) {
    return slot.status === 'ready' || slot.status === 'idle_unloaded';
}

export async function loadModels() {
    try {
        const resp = await fetch('/v1/models');
        applyModels(await resp.json());
    } catch {}
}

export function applyModels(data) {
    const chat = (data.data || []).filter(m => !m.id.startsWith('whisper@'));
    setLoadedModelId(chat.length ? chat[0].id : '');
    invalidateContextCount(false);
    updateComposerHealth(lastHealthData);
    scheduleExactContextCount(0);
}
