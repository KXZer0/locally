// The message box and everything you can drop into it.
//
// In here: autosizing, Enter-to-send, the send/stop button, paste, drag and
// drop, and the global keyboard shortcuts. The three autosize numbers are the
// same arithmetic 04-composer.css uses and must stay in step with it.
// Not in here: what a send DOES (chat/send.js) and how a file is read
// (chat/attachments.js).

import { attachFile, attachImage, clearDoc, clearImage } from '../chat/attachments.js';
import { justAnswerMe } from '../chat/completion.js';
import { syncComposerDraftState } from '../chat/context.js';
import { abortController, cancelGeneration, isGenerating } from '../chat/generation.js';
import { newChat, sendMessage } from '../chat/send.js';
import { attachBtn, chat, dropOverlay, fileInput, imagePreview, input, newChatBtn, previewImg, removeDocBtn, removeImageBtn, sendBtn, tempValue, temperatureSlider } from '../core/dom.js';
import { attachedImage, setAttachedImage, setThinkExpanded, thinkExpanded } from '../core/state.js';
import { mode } from './tabs.js';
import { utilSetStatus } from '../util/engine.js';
import { selectUtilFile } from '../util/read.js';

// --- Events ---

temperatureSlider.addEventListener('input', () => {
    tempValue.textContent = (temperatureSlider.value / 100).toFixed(1);
});

// Event delegation for think blocks (survives DOM re-renders)
chat.addEventListener('click', (e) => {
    if (e.target.closest('[data-just-answer]')) {
        e.stopPropagation();
        justAnswerMe(e);
        return;
    }
    const thinkBlock = e.target.closest('[data-think-toggle]');
    if (thinkBlock) {
        setThinkExpanded(!thinkExpanded);
        thinkBlock.classList.toggle('collapsed');
    }
});

sendBtn.addEventListener('click', () => {
    if (isGenerating) cancelGeneration();
    else sendMessage();
});
input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea. The three numbers below are the same arithmetic
// 04-composer.css uses, and they must stay in step with it: an integer line box
// is what makes each added line an exact 24px step instead of whatever a 1.5x
// line-height rounds to on this DPI, which is half of why growth looked jumpy.
export const INPUT_LINE = 24;   // line-height
export const INPUT_PAD = 16;    // block padding, 8 top + 8 bottom
export const INPUT_ROWS_MAX = 7;
export const INPUT_MIN_H = INPUT_LINE + INPUT_PAD;                       // 40
export const INPUT_MAX_H = INPUT_LINE * INPUT_ROWS_MAX + INPUT_PAD;      // 184

// Measure from a collapsed box every time. The old version tried to skip the
// collapse by inferring growth from the draft's length, which is wrong the
// moment an edit changes the wrap without changing the character count -- select
// a word, type a longer one, and the box kept a height that no longer matched
// its content. One textarea's worth of forced layout per keystroke is not a cost
// worth a heuristic that can disagree with what is on screen.
// The measurement is a deliberate write/read/write, which forces layout. That
// is unavoidable -- the length-based heuristic it replaced was simply wrong
// about wrapped text -- but it does not have to happen once per keystroke. One
// per frame is enough: nothing between two keystrokes in the same frame can be
// seen. Same arithmetic, same correctness, a fraction of the layout work.
let sizePending = false;

export function autosizeInput() {
    if (sizePending) return;
    sizePending = true;
    requestAnimationFrame(() => {
        sizePending = false;
        // Measure with the gutter hidden: a visible scrollbar narrows the content
        // box, so leaving it on would over-report the height the text needs and the
        // box would never come back down off the cap.
        input.style.overflowY = 'hidden';
        input.style.height = `${INPUT_MIN_H}px`;
        const content = input.scrollHeight;
        const next = Math.min(Math.max(content, INPUT_MIN_H), INPUT_MAX_H);
        input.style.height = `${next}px`;
        // Only past the cap is there anything to scroll. Below it the gutter would
        // be a scrollbar over content that already fits.
        input.style.overflowY = content > INPUT_MAX_H ? 'auto' : 'hidden';
    });
}

input.addEventListener('input', () => {
    autosizeInput();
    syncComposerDraftState();
});
autosizeInput();

newChatBtn.addEventListener('click', newChat);

attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) attachFile(fileInput.files[0]);
    fileInput.value = '';
});
removeImageBtn.addEventListener('click', clearImage);
removeDocBtn.addEventListener('click', clearDoc);
// Paste image
document.addEventListener('paste', (e) => {
    for (const item of e.clipboardData.items) {
        if (item.type.startsWith('image/')) {
            e.preventDefault();
            if (mode === 'util') selectUtilFile(item.getAsFile());
            else attachImage(item.getAsFile());
            return;
        }
    }
    // Images copied from web pages often arrive as a text/html flavor with a
    // data: URL and no image/ item — without this, the whole base64 string
    // lands in the input box as text.
    const dataUrl = (e.clipboardData.getData('text/html') + ' ' +
                     e.clipboardData.getData('text/plain'))
        .match(/data:image\/(?:png|jpe?g|gif|webp);base64,[A-Za-z0-9+/=]+/);
    if (dataUrl) {
        e.preventDefault();
        if (mode === 'util') {
            fetch(dataUrl[0]).then(r => r.blob()).then(blob => {
                const ext = blob.type.split('/')[1]?.replace('jpeg', 'jpg') || 'png';
                selectUtilFile(new File([blob], `pasted-image.${ext}`, { type: blob.type }));
            }).catch(err => utilSetStatus(err.message, true));
        } else {
            setAttachedImage(dataUrl[0]);
            previewImg.src = attachedImage;
            imagePreview.style.display = 'block';
            input.focus();
        }
    }
});

// Drag and drop
export let dragCounter = 0;
document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    if (e.dataTransfer.types.includes('Files')) {
        dropOverlay.textContent = mode === 'util' ? 'Drop file to read locally' : 'Drop a file or image here';
        dropOverlay.classList.add('active');
    }
});
document.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) {
        dropOverlay.classList.remove('active');
        dragCounter = 0;
    }
});
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => {
    e.preventDefault();
    dropOverlay.classList.remove('active');
    dragCounter = 0;
    const file = e.dataTransfer.files[0];
    if (file) {
        if (mode === 'util') selectUtilFile(file);
        else attachFile(file);          // documents too, not only images
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'n') {
        e.preventDefault();
        newChat();
    }
    if (e.key === 'Escape' && isGenerating && abortController) {
        cancelGeneration();
    }
});
