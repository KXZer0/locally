// Startup and the /health poll.
//
// In here: the one-request bootstrap, the fallback to the public endpoints
// when it fails, and the poll that follows -- fast while something is loading,
// slow once everything is ready.
// Not in here: what any of the payloads MEAN. This module fans them out to
// system/devices.js, chat/context.js, chat/websearch.js and voice/speaker.js
// and holds no opinion of its own.

import { syncComposerDraftState, updateComposerHealth } from '../chat/context.js';
import { applyWebSearchHealth } from '../chat/websearch.js';
import { brandMark, deviceRail, input, modelSelect, welcomeStatus } from '../core/dom.js';
import { setLastHealthData } from '../core/state.js';
import { applyModels, loadModels, renderDevices, resetDevicesRenderSignature } from './devices.js';
import { refreshMemory, renderMemory } from './memory-hud.js';
import { applyAvailableModels, loadAvailableModels } from './swap.js';
import { applySpeakerHealth } from '../voice/speaker.js';

// --- Init ---

export async function init() {
    // Focus is local state; never hold it behind network work, even localhost.
    input.focus();
    syncComposerDraftState();
    const bootstrapStarted = performance.now();
    let bootstrapSource = 'combined';
    performance.mark('locally-bootstrap-start');
    try {
        const resp = await fetch('/v1/ui/bootstrap');
        if (!resp.ok) throw new Error(`bootstrap returned ${resp.status}`);
        const data = await resp.json();
        applyHealth(data.health || {});
        applyModels(data.models || {});
        applyAvailableModels(data.available_models || {});
        renderMemory(data.memory || null);
    } catch {
        bootstrapSource = 'fallback';
        // Old/stale server or a transient failure: the public polling path is
        // still complete. This matters during development because Flask can
        // serve a freshly edited app.js from a process running old Python.
        await checkHealth();
    }
    performance.mark('locally-bootstrap-ready');
    performance.measure('locally-bootstrap',
                        'locally-bootstrap-start', 'locally-bootstrap-ready');
    document.documentElement.dataset.bootstrap = 'ready';
    document.documentElement.dataset.bootstrapSource = bootstrapSource;
    document.documentElement.dataset.bootstrapMs =
        (performance.now() - bootstrapStarted).toFixed(1);
    // Poll fast while something is still loading (the browser opens seconds
    // after launch now), then settle down once everything is ready.
    let tick = 0;
    const poll = async () => {
        await checkHealth();
        const busy = modelSelect.disabled;
        tick++;
        setTimeout(poll, busy || tick < 4 ? 2000 : 15000);
    };
    setTimeout(poll, 2000);
}
// Which slots are actually serveable, as a string — when this changes, the
// dropdowns are stale.
export let lastSlotSignature = '';
export async function checkHealth() {
    try {
        const resp = await fetch('/health');
        const data = await resp.json();
        const changed = applyHealth(data);
        // After the one-request initial snapshot, polling keeps the existing
        // public endpoints. The independent refreshes run together: no reason
        // for a directory scan to block a RAM reading, or vice versa.
        const refreshes = [refreshMemory()];
        if (changed) refreshes.push(loadModels(), loadAvailableModels());
        await Promise.all(refreshes);
    } catch {
        deviceRail.innerHTML = '';
        resetDevicesRenderSignature();
        brandMark.classList.add('offline');
        welcomeStatus?.classList.add('offline');
        brandMark.title = 'disconnected';
    }
}

export function applyHealth(data) {
    setLastHealthData(data || {});
    renderDevices(data);
    updateComposerHealth(data);
    applyWebSearchHealth(data);
    applySpeakerHealth(data);
    brandMark.classList.remove('offline');
    welcomeStatus?.classList.remove('offline');
    brandMark.title = `status: ${data.status}`;

    // The browser can open while models are loading, so /v1/models may be
    // empty at first. The signature tells the poll when those controls need a
    // refresh without touching them on every tick.
    const sig = Object.values(data.devices || {})
        .map(d => `${d.device_name}:${d.model}:${d.status}`).sort().join('|');
    const changed = sig !== lastSlotSignature;
    lastSlotSignature = sig;
    return changed;
}
