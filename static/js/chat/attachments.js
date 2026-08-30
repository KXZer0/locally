// Files on their way into a chat turn.
//
// In here: images (attached raw, so the SERVER decides whether a VLM sees them
// or the OCR utilities read them) and documents (extracted to text here,
// because the model needs words).
// Not in here: the Tools tab's own file handling (util/read.js). They look
// alike and are not the same feature: one feeds a conversation, the other is
// the product.

import { invalidateContextCount, scheduleExactContextCount, updateContextDisplay } from './context.js';
import { docChip, docChipMeta, docChipName, imagePreview, input, previewImg } from '../core/dom.js';
import { attachedImage, setAttachedDoc, setAttachedImage } from '../core/state.js';
import { utilEngine } from '../util/engine.js';

// --- Attachments (images and documents) ---

// An image is attached raw so the server can route it (VLM sees it, text model
// has it read by OCR). A document is extracted to text here, because the model
// needs words and the extraction endpoint already exists.
export function attachFile(file) {
    if (!file) return;
    if (file.type.startsWith('image/')) return attachImage(file);
    return attachDocument(file);
}

export function attachImage(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = () => {
        setAttachedImage(reader.result);
        previewImg.src = attachedImage;
        imagePreview.style.display = 'block';
        // Attach-image-first leaves focus on the attach button (or wherever
        // the paste happened) — keystrokes then go nowhere and the input
        // feels dead. Hand focus to the box the user will type in next.
        input.focus();
    };
    reader.readAsDataURL(file);
}

export async function attachDocument(file) {
    setAttachedDoc(null);
    docChip.hidden = false;
    docChipName.textContent = file.name;
    docChipMeta.textContent = 'reading…';
    try {
        const form = new FormData();
        form.append('file', file);
        form.append('engine', utilEngine);
        const resp = await fetch('/v1/util/read', { method: 'POST', body: form });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data?.error?.message || `HTTP ${resp.status}`);
        const textOut = (data.markdown || data.text || '').trim();
        if (!textOut) throw new Error('no text found in this file');
        setAttachedDoc({ name: file.name, text: textOut, chars: textOut.length });
        invalidateContextCount(false);
        // "native" means the file carried its own text and no model was used —
        // worth showing, because it is the fast, lossless path.
        const how = data.source === 'ocr' ? `${(data.engine || '').toUpperCase()} OCR` : 'native text';
        docChipMeta.textContent = `${textOut.length.toLocaleString()} chars · ${how}`;
        updateContextDisplay();
        scheduleExactContextCount();
        input.focus();
    } catch (err) {
        docChipMeta.textContent = err.message;
        docChip.classList.add('err');
        setTimeout(clearDoc, 4000);
    }
}

export function clearImage() {
    setAttachedImage(null);
    previewImg.src = '';
    imagePreview.style.display = 'none';
    updateContextDisplay();
}

export function clearDoc() {
    setAttachedDoc(null);
    invalidateContextCount(false);
    docChip.hidden = true;
    docChip.classList.remove('err');
    docChipMeta.textContent = '';
    updateContextDisplay();
    scheduleExactContextCount();
}

export function clearAttachments() {
    clearImage();
    clearDoc();
}
