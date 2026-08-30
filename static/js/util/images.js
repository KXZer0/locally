// The image workspaces: background removal, upscale, detect, generate.
//
// In here: the shared per-task file slot, the four requests, and their
// results. They are together because they are the same shape -- one image in,
// one image out, one meta line -- not because they share a model.
// Not in here: document reading (util/read.js) and semantic file search
// (util/search.js), which take different inputs and produce different results.

import { detectThreshold, detectThresholdValue, generateDownload, generatePrompt, generateResult, generateRun, generateSeed, generateStatus, generateSteps, imageTaskInputs, panelUtil } from '../core/dom.js';
import { escapeHtml } from '../core/format.js';
import { paintDevice } from '../core/paint.js';
import { syncUtilityTaskAvailability, taskIsAvailable, utilEngine } from './engine.js';

export const imageTaskFiles = {};
export const imageTaskPreviewUrls = {};
export let generatedImageUrl = '';
export function setImageTaskFile(task, file) {
    if (!file || !file.type.startsWith('image/')) return;
    if (file.size > 50 * 1024 * 1024) {
        document.querySelector(`[data-image-status="${task}"]`).textContent = 'File exceeds 50 MB.';
        return;
    }
    if (imageTaskPreviewUrls[task]) URL.revokeObjectURL(imageTaskPreviewUrls[task]);
    imageTaskFiles[task] = file;
    imageTaskPreviewUrls[task] = URL.createObjectURL(file);
    const preview = document.querySelector(`[data-image-preview="${task}"]`);
    preview.src = imageTaskPreviewUrls[task];
    preview.hidden = false;
    document.querySelector(`[data-image-task="${task}"] strong`).textContent = file.name;
    document.querySelector(`[data-image-result="${task}"]`).hidden = true;
    document.querySelector(`[data-image-status="${task}"]`).textContent = '';
    syncUtilityTaskAvailability();
}

export function downloadDataUrl(url, filename) {
    if (!url) return;
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
}

export async function runImageUtility(task) {
    const file = imageTaskFiles[task];
    if (!file || !taskIsAvailable(task)) return;
    const run = document.querySelector(`[data-image-run="${task}"]`);
    const status = document.querySelector(`[data-image-status="${task}"]`);
    const result = document.querySelector(`[data-image-result="${task}"]`);
    const label = run.textContent;
    run.disabled = true;
    run.textContent = 'Working…';
    status.textContent = `running on ${utilEngine.toUpperCase()}`;
    const form = new FormData();
    form.append('file', file, file.name);
    form.append('engine', utilEngine);
    if (task === 'detect') form.append('threshold', (Number(detectThreshold.value) / 1000).toString());
    try {
        const response = await fetch(`/v1/util/${task}`, { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error?.message || `${label} failed`);
        result.querySelector('img').src = data.image;
        result.dataset.image = data.image;
        const ms = Math.round(data.timings_ms?.infer || 0);
        if (task === 'upscale') {
            result.querySelector('.result-meta').textContent =
                `${data.engine.toUpperCase()} · ${data.output_width}×${data.output_height} · ${data.scale}× · ${ms} ms`;
        } else if (task === 'detect') {
            result.querySelector('.result-meta').textContent =
                `${data.engine.toUpperCase()} · ${data.detections.length} found · ${ms} ms`;
            const list = result.querySelector('.detection-list');
            list.innerHTML = data.detections.map(item =>
                `<li>${escapeHtml(item.label.replaceAll('_', ' '))} · ${(item.score * 100).toFixed(1)}%</li>`
            ).join('');
        } else {
            result.querySelector('.result-meta').textContent =
                `${data.engine.toUpperCase()} · ${data.width}×${data.height} · ${ms} ms`;
        }
        result.hidden = false;
        status.textContent = 'done';
        paintDevice(panelUtil, data.engine);
    } catch (error) {
        result.hidden = true;
        status.textContent = error.message;
        status.classList.add('err');
    } finally {
        run.textContent = label;
        syncUtilityTaskAvailability();
    }
}

export async function runImageGeneration() {
    const prompt = generatePrompt.value.trim();
    if (!prompt || !taskIsAvailable('generate')) return;
    generateRun.disabled = true;
    generateStatus.textContent = `running on ${utilEngine.toUpperCase()}`;
    try {
        const response = await fetch('/v1/util/generate', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt, engine: utilEngine, steps: Number(generateSteps.value),
                seed: Number(generateSeed.value),
            }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error?.message || 'Image generation failed');
        generatedImageUrl = data.image;
        generateResult.querySelector('img').src = data.image;
        generateResult.querySelector('.result-meta').textContent =
            `${data.engine.toUpperCase()} · ${data.steps} steps · ${Math.round(data.timings_ms?.infer || 0)} ms`;
        generateResult.hidden = false;
        generateStatus.textContent = 'done';
    } catch (error) {
        generateResult.hidden = true;
        generateStatus.textContent = error.message;
        generateStatus.classList.add('err');
    } finally {
        syncUtilityTaskAvailability();
    }
}
for (const inputEl of imageTaskInputs) {
    const task = inputEl.dataset.imageInput;
    const drop = document.querySelector(`[data-image-task="${task}"]`);
    drop.addEventListener('click', () => inputEl.click());
    inputEl.addEventListener('change', () => {
        if (inputEl.files[0]) setImageTaskFile(task, inputEl.files[0]);
    });
    for (const eventName of ['dragenter', 'dragover']) {
        drop.addEventListener(eventName, (event) => {
            event.preventDefault();
            drop.classList.add('dragging');
        });
    }
    for (const eventName of ['dragleave', 'drop']) {
        drop.addEventListener(eventName, (event) => {
            event.preventDefault();
            drop.classList.remove('dragging');
            if (eventName === 'drop' && event.dataTransfer.files[0]) {
                setImageTaskFile(task, event.dataTransfer.files[0]);
            }
        });
    }
}
for (const btn of document.querySelectorAll('[data-image-run]')) {
    btn.addEventListener('click', () => runImageUtility(btn.dataset.imageRun));
}
for (const btn of document.querySelectorAll('[data-image-download]')) {
    btn.addEventListener('click', () => {
        const task = btn.dataset.imageDownload;
        const result = document.querySelector(`[data-image-result="${task}"]`);
        const base = (imageTaskFiles[task]?.name || task).replace(/\.[^.]+$/, '');
        downloadDataUrl(result.dataset.image, `${base}-${task}.png`);
    });
}
detectThreshold.addEventListener('input', () => {
    detectThresholdValue.textContent = `${(Number(detectThreshold.value) / 10).toFixed(1)}%`;
});
generateRun.addEventListener('click', runImageGeneration);
generateDownload.addEventListener('click', () => downloadDataUrl(generatedImageUrl, 'generated.png'));
