// Choosing a model, which IS loading it -- there is no separate Load button.
//
// In here: the list of models on disk, the composer's model menu, and the
// swap request. The swap deliberately sends NO device: the server places the
// model on the engine that can actually run it and moves the slot there.
// Not in here: which model is RESIDENT. That is loadedModelId in core/state.js
// and it answers a different question from the menu's selection; conflating
// the two is how requests came to name a model that was not loaded yet.

import { currentChatSlot, scheduleExactContextCount } from '../chat/context.js';
import { isGenerating } from '../chat/generation.js';
import { composerModelStatus, contextModelMenu, contextModelName, contextModelPicker, contextModelRoute, contextModelTrigger, memHud, memHudPill, modelSelect, swapStatus } from '../core/dom.js';
import { checkHealth } from './bootstrap.js';
import { loadModels } from './devices.js';


// The models on disk, and whether a swap is in flight. chat/context.js reads
// `modelSwapBusy` so the composer does not overwrite "Loading X" with the model
// that is still resident.
export let swapDevices = [];
export let modelSwapBusy = false;
export let availableChatModels = [];
// --- Model swap (Ollama-style, one at a time) ---

export async function loadAvailableModels() {
    try {
        const resp = await fetch('/v1/models/available');
        applyAvailableModels(await resp.json());
    } catch {}
}

export function applyAvailableModels(data) {
    // Settings keeps its compact native fallback; the everyday model switcher
    // is the custom menu inside the context surface. Never replace the native
    // select while its OS menu is open.
    if (document.activeElement === modelSelect) return;
    const models = data.data || [];
    const live = models.find(m => m.loaded_on);
    availableChatModels = models;

    modelSelect.innerHTML = '';
    for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.dataset.device = m.loaded_on || '';
        opt.textContent = m.loaded_on
            ? `${m.name} · ${m.loaded_on}`
            : `${m.name} (${m.type})`;
        modelSelect.appendChild(opt);
    }
    if (live) modelSelect.value = live.name;
    if (!modelSelect.options.length) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'no models found';
        modelSelect.appendChild(opt);
    }

    renderContextModelMenu();
    swapDevices = data.devices || [];
    syncModelControlState();
}

export function renderContextModelMenu() {
    const live = availableChatModels.find(m => m.loaded_on);
    const activeName = live?.name || currentChatSlot?.model || '';
    const activeDevice = live?.loaded_on || currentChatSlot?.device_name || '';

    if (!modelSwapBusy) {
        contextModelName.textContent = activeName || 'No model found';
        contextModelRoute.textContent = activeDevice || '—';
    }

    contextModelMenu.innerHTML = '';
    if (!availableChatModels.length) {
        const empty = document.createElement('span');
        empty.className = 'context-model-empty';
        empty.textContent = 'No local chat models found';
        contextModelMenu.appendChild(empty);
        return;
    }

    for (const model of availableChatModels) {
        const selected = model.name === activeName;
        const option = document.createElement('button');
        option.type = 'button';
        option.className = 'context-model-option';
        option.dataset.model = model.name;
        option.dataset.device = model.loaded_on || '';
        option.setAttribute('role', 'menuitemradio');
        option.setAttribute('aria-checked', String(selected));
        option.tabIndex = -1;

        const copy = document.createElement('span');
        const name = document.createElement('strong');
        const meta = document.createElement('small');
        name.textContent = model.name;
        meta.textContent = model.loaded_on
            ? `${model.loaded_on} · loaded`
            : `${(model.type || 'model').toUpperCase()} · available`;
        copy.append(name, meta);

        const check = document.createElement('span');
        check.className = 'context-model-check';
        check.textContent = selected ? '✓' : '';
        check.setAttribute('aria-hidden', 'true');
        option.append(copy, check);
        contextModelMenu.appendChild(option);
    }
}

export function syncModelControlState() {
    const disabled = modelSwapBusy || isGenerating;
    modelSelect.disabled = disabled || !modelSelect.options.length || !modelSelect.value;
    contextModelTrigger.disabled = disabled || !availableChatModels.length;
    contextModelPicker.classList.toggle('loading', modelSwapBusy);
    contextModelPicker.setAttribute('aria-busy', String(modelSwapBusy));
    for (const option of contextModelMenu.querySelectorAll('.context-model-option')) {
        option.disabled = disabled;
    }
}

export async function swapModel(name, loadedOn = '') {
    if (!name) return;
    // Already resident: selecting what is loaded should be a no-op, not a
    // pointless unload-and-reload of the same weights.
    if (loadedOn) return;
    // Deliberately send NO device: the server places the model on the engine
    // that can actually run it (NPU for channel-wise text models, GPU for
    // vision or group-quantized weights) and moves the slot there. Passing the
    // current model's device — which this used to do — pinned every swap to
    // whatever was already loaded and made the NPU unreachable from the UI.

    modelSwapBusy = true;
    contextModelName.textContent = `Loading ${name}`;
    contextModelRoute.textContent = 'placing…';
    syncModelControlState();
    swapStatus.textContent = `Unloading current model, then loading ${name} `
        + 'on whichever engine can run it. A large model can take a minute.';
    composerModelStatus.textContent = `Loading ${name}.`;
    contextModelPicker.title = `Loading ${name}…`;
    try {
        const resp = await fetch('/v1/models/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: name }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error?.message || 'Load failed');
        // Say where it landed and why — the placement can legitimately differ
        // from what you'd guess (a VLM can't go on the NPU, for instance).
        swapStatus.textContent =
            `${data.model} ready on ${data.device} (${data.type}) in ${data.load_seconds}s.`
            + (data.placement ? ` — ${data.placement}` : '');
        composerModelStatus.textContent = `${data.model} ready on ${data.device}.`;
        contextModelPicker.title = `${data.model} ready on ${data.device}`;
        await checkHealth();
        await loadModels();
        await loadAvailableModels();
        // A model switch changes the engine for the next turn; it does not
        // erase the conversation. New Chat remains an explicit user action.
        scheduleExactContextCount(0);
    } catch (err) {
        swapStatus.textContent = err.message;
        composerModelStatus.textContent = `Model load failed: ${err.message}`;
        contextModelPicker.title = err.message;
    } finally {
        modelSwapBusy = false;
        renderContextModelMenu();
        syncModelControlState();
    }
}

// Choosing a model *is* loading it — there is no separate Load button to press,
// which is the whole point of collapsing the two controls into one.
modelSelect.addEventListener('change', () => {
    const option = modelSelect.selectedOptions[0];
    swapModel(modelSelect.value, option?.dataset.device || '');
});

export function contextModelOptions() {
    return [...contextModelMenu.querySelectorAll('.context-model-option:not(:disabled)')];
}

export function openContextModelMenu(direction = 1) {
    if (contextModelTrigger.disabled) return;
    memHud.classList.add('pinned');
    memHudPill.setAttribute('aria-expanded', 'true');
    contextModelMenu.hidden = false;
    contextModelTrigger.setAttribute('aria-expanded', 'true');
    const options = contextModelOptions();
    const selected = options.find(option => option.getAttribute('aria-checked') === 'true');
    (selected || options[direction < 0 ? options.length - 1 : 0])?.focus();
}

export function closeContextModelMenu({ focusTrigger = false } = {}) {
    if (contextModelMenu.hidden) return;
    contextModelMenu.hidden = true;
    contextModelTrigger.setAttribute('aria-expanded', 'false');
    if (focusTrigger) contextModelTrigger.focus();
}

contextModelTrigger.addEventListener('click', () => {
    if (contextModelMenu.hidden) openContextModelMenu();
    else closeContextModelMenu({ focusTrigger: true });
});

contextModelTrigger.addEventListener('keydown', event => {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    event.preventDefault();
    openContextModelMenu(event.key === 'ArrowUp' ? -1 : 1);
});

contextModelMenu.addEventListener('click', event => {
    const option = event.target.closest('.context-model-option');
    if (!option || option.disabled) return;
    const name = option.dataset.model;
    const loadedOn = option.dataset.device || '';
    closeContextModelMenu({ focusTrigger: true });
    swapModel(name, loadedOn);
});

contextModelMenu.addEventListener('keydown', event => {
    const options = contextModelOptions();
    const index = options.indexOf(document.activeElement);
    if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        closeContextModelMenu({ focusTrigger: true });
        return;
    }
    if (event.key === 'Tab') {
        closeContextModelMenu();
        contextModelTrigger.focus();
        return;
    }
    let next = null;
    if (event.key === 'ArrowDown') next = options[(index + 1) % options.length];
    if (event.key === 'ArrowUp') next = options[(index - 1 + options.length) % options.length];
    if (event.key === 'Home') next = options[0];
    if (event.key === 'End') next = options[options.length - 1];
    if (!next) return;
    event.preventDefault();
    next.focus();
});

document.addEventListener('pointerdown', event => {
    if (!contextModelPicker.contains(event.target)) closeContextModelMenu();
});
