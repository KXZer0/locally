// The Read-document workspace.
//
// In here: picking a file, running /v1/util/read on it, and what can be done
// with the text afterwards -- copy, download, or hand it to Chat.
// Not in here: the same endpoint's other caller. chat/attachments.js reads a
// document to feed a conversation; this is the tool itself, and it keeps its
// file and result between visits to other tabs.

import { clearImage } from '../chat/attachments.js';
import { sendMessage } from '../chat/send.js';
import { input, panelUtil, utilAskChat, utilChangeFile, utilCopyBtn, utilDownloadBtn, utilDropzone, utilFileEl, utilFileIcon, utilFileInput, utilFileMeta, utilFileName, utilPreview, utilQuestion, utilResult, utilResultMeta, utilResultText, utilRunBtn } from '../core/dom.js';
import { formatBytes } from '../core/format.js';
import { paintDevice } from '../core/paint.js';
import { setMode } from '../ui/tabs.js';
import { syncUtilRunState, utilEngine, utilEngines, utilFileUsesNativeParser, utilSetStatus } from './engine.js';

export let utilFile = null;
export let utilPreviewUrl = '';
export let utilBusy = false;
export function selectUtilFile(file) {
    if (!file) return;
    const ext = (file.name.match(/\.[^.]+$/)?.[0] || '').toLowerCase();
    const supported = new Set([
        '.pdf', '.docx', '.pptx', '.xlsx', '.xls', '.html', '.htm', '.epub',
        '.csv', '.tsv', '.txt', '.md', '.json', '.xml', '.yaml', '.yml',
        '.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tif', '.tiff',
    ]);
    const isImage = file.type.startsWith('image/') ||
        ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tif', '.tiff'].includes(ext);
    if (!supported.has(ext) && !isImage) {
        utilSetStatus(`Unsupported file type ${ext || '(unknown)'}.`, true);
        return;
    }
    if (file.size > 50 * 1024 * 1024) {
        utilSetStatus('The file is over the server’s 50 MB request limit.', true);
        return;
    }
    if (utilPreviewUrl) URL.revokeObjectURL(utilPreviewUrl);
    utilFile = file;
    utilPreviewUrl = isImage ? URL.createObjectURL(file) : '';
    utilPreview.src = utilPreviewUrl;
    utilPreview.hidden = !isImage;
    utilFileIcon.hidden = isImage;
    utilFileIcon.textContent = (ext.slice(1) || 'DOC').toUpperCase().slice(0, 5);
    utilFileName.textContent = file.name || 'pasted image';
    utilFileMeta.textContent = formatBytes(file.size);
    utilFileEl.hidden = false;
    utilDropzone.hidden = true;
    utilResult.hidden = true;
    utilResultText.textContent = '';
    utilSetStatus('');
    syncUtilRunState();
}

export function clearUtilFile() {
    if (utilPreviewUrl) URL.revokeObjectURL(utilPreviewUrl);
    utilPreviewUrl = '';
    utilFile = null;
    utilPreview.src = '';
    utilPreview.hidden = false;
    utilFileIcon.hidden = true;
    utilFileEl.hidden = true;
    utilDropzone.hidden = false;
    utilResult.hidden = true;
    utilResultText.textContent = '';
    utilSetStatus('');
    syncUtilRunState();
}

export async function runUtilRead() {
    if (!utilFile || utilBusy ||
            (!utilFileUsesNativeParser(utilFile) && !utilEngines[utilEngine])) return;
    utilBusy = true;
    syncUtilRunState();
    const oldLabel = utilRunBtn.innerHTML;
    utilRunBtn.textContent = `Reading on ${utilEngine.toUpperCase()}…`;
    utilSetStatus('working locally');

    const form = new FormData();
    form.append('file', utilFile, utilFile.name);
    form.append('engine', utilEngine);
    const started = performance.now();
    try {
        const resp = await fetch('/v1/util/read', { method: 'POST', body: form });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error?.message || 'Document read failed');

        const elapsed = performance.now() - started;
        const text = data.markdown || data.text || '';
        utilResultText.textContent = text || '(No readable text found.)';
        const modelMs = Object.values(data.timings_ms || {})
            .reduce((sum, value) => sum + (Number(value) || 0), 0);
        const dropped = data.dropped_low_conf
            ? ` · ${data.dropped_low_conf} uncertain region${data.dropped_low_conf === 1 ? '' : 's'} omitted`
            : '';
        const engineLabel = data.engine ? data.engine.toUpperCase() : 'LOCAL';
        const workLabel = data.source === 'native'
            ? 'native parser'
            : `${data.regions} region${data.regions === 1 ? '' : 's'}`;
        const pages = data.pages_total
            ? ` · ${data.pages_processed}/${data.pages_total} page${data.pages_total === 1 ? '' : 's'}`
            : '';
        const truncated = data.truncated ? ' · TRUNCATED' : '';
        utilResultMeta.textContent = `${engineLabel} · ${workLabel}${pages}`
            + ` · ${Math.round(modelMs || elapsed)} ms${dropped}${truncated}`;
        utilResult.hidden = false;
        utilSetStatus(data.engine ? `done on ${data.engine.toUpperCase()}` : 'done locally');
        if (data.engine) paintDevice(panelUtil, data.engine);
    } catch (err) {
        utilResult.hidden = true;
        utilSetStatus(err.message, true);
    } finally {
        utilBusy = false;
        utilRunBtn.innerHTML = oldLabel;
        syncUtilRunState();
    }
}

export async function copyUtilText() {
    const text = utilResultText.textContent;
    if (!text) return;
    await navigator.clipboard.writeText(text);
    const oldTitle = utilCopyBtn.title;
    utilCopyBtn.title = 'Copied';
    setTimeout(() => { utilCopyBtn.title = oldTitle; }, 1500);
}

export function downloadUtilText() {
    const text = utilResultText.textContent;
    if (!text) return;
    const base = (utilFile?.name || 'document').replace(/\.[^.]+$/, '');
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
    link.download = `${base}.txt`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 0);
}

export async function askUtilTextInChat() {
    const extracted = utilResultText.textContent.trim();
    const question = utilQuestion.value.trim() || 'Summarize this document.';
    if (!extracted || extracted === '(No readable text found.)') return;
    const name = utilFile?.name || 'image';
    // A stale Chat attachment must not hitch a ride with OCR text from Util.
    clearImage();
    input.value = `Text extracted locally from ${name}:\n\n${extracted}\n\nQuestion: ${question}`;
    input.dispatchEvent(new Event('input'));
    setMode('chat');
    await sendMessage();
}

utilDropzone.addEventListener('click', () => utilFileInput.click());
utilChangeFile.addEventListener('click', () => utilFileInput.click());
utilFileInput.addEventListener('change', () => {
    if (utilFileInput.files[0]) selectUtilFile(utilFileInput.files[0]);
    utilFileInput.value = '';
});
utilRunBtn.addEventListener('click', runUtilRead);
utilCopyBtn.addEventListener('click', () => copyUtilText().catch(err => utilSetStatus(err.message, true)));
utilDownloadBtn.addEventListener('click', downloadUtilText);
utilAskChat.addEventListener('click', askUtilTextInChat);
utilQuestion.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); askUtilTextInChat(); }
});
