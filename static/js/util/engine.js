// Which engine serves the Tools tab, and which tasks it can actually do.
//
// In here: NPU/GPU selection, the per-task availability read out of /health,
// and the workspace switcher. Availability has THREE states -- ready, not
// installed, installed but not served here -- because each needs a different
// action from the user, and a task can be unavailable on this engine and fine
// on the other one.
// Not in here: running any task. Each workspace owns its own request.

import { generateRun, panelUtil, searchFilesInput, searchIndexBtn, sidebarTools, utilAvailability, utilDescription, utilEngineBtns, utilNavBtns, utilRunBtn, utilStatus, utilTitle, utilViews } from '../core/dom.js';
import { paintDevice, revealSurface } from '../core/paint.js';
import { _slotUp } from '../system/devices.js';
import { imageTaskFiles } from './images.js';
import { utilBusy, utilFile } from './read.js';

export function renderUtilAvailability(slots) {
    utilEngines = {
        npu: !!(slots.npu && _slotUp(slots.npu)),
        gpu: !!(slots.gpu && _slotUp(slots.gpu)),
    };
    utilTasks = {
        npu: slots.npu?.tasks || {},
        gpu: slots.gpu?.tasks || {},
    };
    utilDisabled = {
        npu: slots.npu?.disabled || {},
        gpu: slots.gpu?.disabled || {},
    };

    for (const btn of utilEngineBtns) {
        const engine = btn.dataset.utilEngine;
        const slot = slots[engine];
        const available = utilEngines[engine];
        btn.disabled = !available;
        btn.title = available
            ? `${engine.toUpperCase()} utilities are ready`
            : (slot ? `${engine.toUpperCase()} utilities: ${slot.status}`
                    : `${engine.toUpperCase()} utility models are not loaded`);
    }

    // NPU is the default every fresh session. Fall over only when it is not a
    // serveable option; never silently move a user's explicit GPU choice back.
    if (!utilEngines[utilEngine]) {
        if (utilEngines.npu) setUtilEngine('npu');
        else if (utilEngines.gpu) setUtilEngine('gpu');
    } else {
        setUtilEngine(utilEngine);
    }

    const ready = Object.entries(utilEngines).filter(([, up]) => up).map(([e]) => e.toUpperCase());
    if (ready.length) {
        utilAvailability.textContent = `${ready.join(' + ')} ready`;
        utilAvailability.classList.remove('err');
    } else {
        utilAvailability.textContent = 'native docs ready · OCR models not loaded';
        utilAvailability.classList.add('err');
    }
    syncUtilRunState();
    syncUtilityTaskAvailability();
}

export let utilEngine = 'npu';
export let utilEngines = { npu: false, gpu: false };
export let utilTasks = { npu: {}, gpu: {} };
// task -> why it is off, when the reason is not "the model is missing".
export let utilDisabled = { npu: {}, gpu: {} };
export let utilTask = 'read';
export function utilFileUsesNativeParser(file) {
    if (!file) return false;
    const ext = (file.name.match(/\.[^.]+$/)?.[0] || '').toLowerCase();
    return [
        '.pdf', '.docx', '.pptx', '.xlsx', '.xls', '.html', '.htm', '.epub',
        '.csv', '.tsv', '.txt', '.md', '.json', '.xml', '.yaml', '.yml',
    ].includes(ext);
}

export function setUtilEngine(engine) {
    if (!['npu', 'gpu'].includes(engine)) return;
    if (utilEngines[engine] === false && (utilEngines.npu || utilEngines.gpu)) return;
    utilEngine = engine;
    for (const btn of utilEngineBtns) {
        const active = btn.dataset.utilEngine === engine;
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-checked', String(active));
    }
    paintDevice(panelUtil, engine);
    paintDevice(sidebarTools, engine);
    syncUtilRunState();
    syncUtilityTaskAvailability();
}
export const UTIL_COPY = {
    read: ['Read document', 'Extract text from documents and images.'],
    background: ['Remove background', 'Remove a portrait background and save a transparent PNG.'],
    upscale: ['Upscale', 'Enlarge an image with a local super-resolution model.'],
    detect: ['Detect objects', 'Find common objects and people in an image.'],
    generate: ['Generate image', 'Create a 512×512 image with a local diffusion model.'],
    search: ['Search files', 'Build a temporary index and search local files by meaning.'],
};

export function taskIsAvailable(task) {
    return !!(utilEngines[utilEngine] && utilTasks[utilEngine]?.[task]);
}

export function syncUtilityTaskAvailability() {
    for (const el of document.querySelectorAll('[data-task-availability]')) {
        const task = el.dataset.taskAvailability;
        const ready = taskIsAvailable(task);
        // "Not installed" and "installed but switched off" need different
        // words: only the first is fixed by downloading a model, and telling
        // someone to fetch one for the second sends them in circles.
        const why = utilDisabled[utilEngine]?.[task];
        // With per-device tiers a task can be unavailable *here* and fine on
        // the other engine — saying "unavailable on this hardware" in that case
        // is simply false, and hides the one-click fix.
        const other = utilEngine === 'npu' ? 'gpu' : 'npu';
        const elsewhere = utilEngines[other] && utilTasks[other]?.[task];
        el.textContent = ready
            ? `${utilEngine.toUpperCase()} ready`
            : elsewhere ? `switch to ${other.toUpperCase()}`
            : why ? 'unavailable on this hardware'
            : 'model not installed';
        el.title = ready ? ''
            : elsewhere ? `${why || 'Not served on ' + utilEngine.toUpperCase()}`
                          + ` — the ${other.toUpperCase()} engine can run this.`
            : (why || '');
        el.classList.toggle('err', !ready);
    }
    for (const btn of document.querySelectorAll('[data-image-run]')) {
        const task = btn.dataset.imageRun;
        btn.disabled = !imageTaskFiles[task] || !taskIsAvailable(task);
    }
    generateRun.disabled = !taskIsAvailable('generate');
    searchIndexBtn.disabled = !searchFilesInput.files.length || !taskIsAvailable('search');
}

export function selectUtilTask(task) {
    if (!UTIL_COPY[task]) return;
    utilTask = task;
    [utilTitle.textContent, utilDescription.textContent] = UTIL_COPY[task];
    for (const view of utilViews) {
        if (view.dataset.utilView === task) revealSurface(view);
        else view.hidden = true;
    }
    // Tool shortcuts live in the persistent sidebar. Their active state is
    // independent of the parent Tools tab, so the last workspace is preserved
    // when the user visits Chat or Voice and returns.
    for (const btn of utilNavBtns) {
        const on = btn.dataset.utilTask === task;
        btn.classList.toggle('active', on);
        if (on) btn.setAttribute('aria-current', 'page');
        else btn.removeAttribute('aria-current');
    }
    syncUtilityTaskAvailability();
}

export function syncUtilRunState() {
    if (!utilRunBtn) return;
    const canRun = utilFile && (utilFileUsesNativeParser(utilFile) || utilEngines[utilEngine]);
    utilRunBtn.disabled = utilBusy || !canRun;
}
export function utilSetStatus(text, isError = false) {
    utilStatus.textContent = text || '';
    utilStatus.classList.toggle('err', isError);
}

for (const btn of utilEngineBtns) {
    btn.addEventListener('click', () => setUtilEngine(btn.dataset.utilEngine));
}
