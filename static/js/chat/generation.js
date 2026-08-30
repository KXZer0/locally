// Is a turn in flight, and how to stop it.
//
// In here: the one flag every surface reads to know the app is busy, the
// abort controller behind it, and the cancel path.
// Not in here: the request itself (chat/completion.js) and the voice turn's
// own abort, which is separate because a spoken turn must be interruptible
// even when OpenVINO is blocked in native code and ignores the cancel.

import { composerInner, sendBtn, sendIconUse } from '../core/dom.js';
import { syncModelControlState } from '../system/swap.js';

export let isGenerating = false;
export let abortController = null;
export function setGenerating(on) {
    isGenerating = on;
    // Drives the light that travels round the brand mark while tokens stream.
    // One flag on <body> rather than a class on the mark itself, so the voice
    // path and any later surface can show the same state without knowing where
    // the mark lives.
    document.body.dataset.busy = on ? '1' : '0';
    sendBtn.classList.toggle('stop', on);
    sendIconUse?.setAttribute('href', on ? '#i-stop' : '#i-send');
    sendBtn.title = on ? 'Stop generating' : 'Send message';
    sendBtn.setAttribute('aria-label', sendBtn.title);
    composerInner?.classList.toggle('is-generating', on);
    syncModelControlState();
}

export function cancelGeneration() {
    if (abortController) abortController.abort();
    fetch('/v1/cancel', { method: 'POST' }).catch(() => {});
}

export function setAbortController(next) { abortController = next; }
