// Which engine serves the Tools tab, and which tasks it can actually do.
//
// In here: NPU/GPU/CPU selection, the per-task availability read out of /health,
// and the workspace switcher. Availability has THREE states -- ready, not
// installed, installed but not served here -- because each needs a different
// action from the user, and a task can be unavailable on this engine and fine
// on the other one.
// Not in here: running any task. Each workspace owns its own request.
//
// CPU is a first-class engine here, not an afterthought: `--util-engines cpu`
// is the entire configuration on a machine with no Intel accelerator, and while
// it was missing from this list the models loaded, served requests over HTTP,
// and were unreachable from the UI. The engine list is ONE constant for that
// reason — a fourth engine must not need four literals found by grep.

import { generateRun, panelUtil, searchFilesInput, searchIndexBtn, sidebarTools, utilAvailability, utilDescription, utilEngineBtns, utilNavBtns, utilRunBtn, utilStatus, utilTitle, utilViews } from '../core/dom.js';
import { paintDevice, revealSurface } from '../core/paint.js';
import { _slotUp } from '../system/devices.js';
import { imageTaskFiles } from './images.js';
import { utilBusy, utilFile } from './read.js';

// Preference order for the default engine, and the only list of engine names
// in this module. It matches the server's `_default_util_engine()`: NPU first
// (it is the low-power engine and the point of the project), CPU last.
export const UTIL_ENGINE_ORDER = ['npu', 'gpu', 'cpu'];

export function renderUtilAvailability(slots) {
    utilEngines = {};
    utilTasks = {};
    utilDisabled = {};
    for (const engine of UTIL_ENGINE_ORDER) {
        const slot = slots[engine];
        utilEngines[engine] = !!(slot && _slotUp(slot));
        utilTasks[engine] = slot?.tasks || {};
        utilDisabled[engine] = slot?.disabled || {};
    }

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

    // NPU is the default every fresh session, then GPU, then CPU. Fall through
    // only when the current engine is not a serveable option; never silently
    // move a user's explicit choice.
    if (!utilEngines[utilEngine]) {
        const fallback = UTIL_ENGINE_ORDER.find(e => utilEngines[e]);
        if (fallback) setUtilEngine(fallback);
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
export let utilEngines = Object.fromEntries(UTIL_ENGINE_ORDER.map(e => [e, false]));
export let utilTasks = Object.fromEntries(UTIL_ENGINE_ORDER.map(e => [e, {}]));
// task -> why it is off, when the reason is not "the model is missing".
export let utilDisabled = Object.fromEntries(UTIL_ENGINE_ORDER.map(e => [e, {}]));
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
    if (!UTIL_ENGINE_ORDER.includes(engine)) return;
    // Refuse a dead engine only when some other one could serve instead —
    // with nothing loaded there is no better choice to move to.
    if (utilEngines[engine] === false
        && UTIL_ENGINE_ORDER.some(e => utilEngines[e])) return;
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
        // another engine — saying "unavailable on this hardware" in that case
        // is simply false, and hides the one-click fix. Search the whole engine
        // list in preference order: with three engines "the other one" is not a
        // single answer, and assuming it was is what hid CPU entirely.
        const other = UTIL_ENGINE_ORDER.find(
            e => e !== utilEngine && utilEngines[e] && utilTasks[e]?.[task]);
        const elsewhere = !!other;
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
