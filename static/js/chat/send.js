// Sending what is in the composer, and clearing the thread.
//
// In here: the composed user turn (text, image, extracted document), the
// pasted-link path that reads a page before answering about it, and New Chat.
// Not in here: the HTTP turn (chat/completion.js), attachment capture
// (chat/attachments.js) and the composer's own DOM (ui/composer.js).

import { clearAttachments } from './attachments.js';
import { runCompletion } from './completion.js';
import { compactHistory, invalidateContextCount, scheduleExactContextCount, syncComposerDraftState, updateContextDisplay } from './context.js';
import { isGenerating } from './generation.js';
import { addMessage, resetMessageGrouping } from './thread.js';
import { webSearchAllowed, webSearchReady } from './websearch.js';
import { emptyState, input, noThinkCheckbox, thread } from '../core/dom.js';
import { escapeHtml } from '../core/format.js';
import { activityHtml, scrollToBottom } from '../core/paint.js';
import { attachedDoc, attachedImage, chatHistory, setChatHistory, setThinkExpanded } from '../core/state.js';
import { autosizeInput } from '../ui/composer.js';
import { setMode } from '../ui/tabs.js';

// --- Reading a pasted link ---------------------------------------------------

// The honest version of "read the site I am on". This window cannot see your
// tabs -- that needs a browser extension -- but pasting the link is the same
// intent, and the server already has a hardened fetcher for it.
//
// It only fires when the message is essentially just a URL. A link mentioned
// inside a real question is context for that question, not a request to go
// read it, and fetching every URL anyone types would be both slow and nosy.

export const URL_ONLY_RE = /^\s*(https?:\/\/\S+)\s*$/i;
export const URL_ANYWHERE_RE = /https?:\/\/\S+/i;

export function soleUrlIn(text) {
    const m = URL_ONLY_RE.exec(text || '');
    return m ? m[1].replace(/[)\].,]+$/, '') : null;
}

export async function readUrlInto(assistantDiv, url) {
    assistantDiv.body.innerHTML = activityHtml('reading the page', 'reading');
    scrollToBottom(true);
    const resp = await fetch('/v1/read-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data?.error?.message || `HTTP ${resp.status}`);
    return data;
}
// --- Chat ---

export async function sendMessage(overrideText) {
    const text = (overrideText !== undefined ? overrideText : input.value).trim();
    if (!text && !attachedImage && !attachedDoc) return;
    if (isGenerating) return;
    setThinkExpanded(false);

    let userContent;
    let displayHtml;

    // A document was already turned into text when it was attached, so it goes
    // in as text. An image goes in as an image and the SERVER decides how to
    // handle it — a VLM sees it, a text model gets it read by the local OCR
    // utilities. Doing that server-side means every client benefits, not just
    // this page.
    const docPrefix = attachedDoc
        ? `Text extracted locally from ${attachedDoc.name}:\n\n${attachedDoc.text}\n\n`
        : '';
    const sentText = docPrefix + text;

    if (attachedImage) {
        userContent = [];
        if (sentText) userContent.push({ type: 'text', text: sentText });
        userContent.push({ type: 'image_url', image_url: { url: attachedImage } });
        displayHtml = escapeHtml(text) + `<img src="${attachedImage}" alt="attached">`;
    } else {
        userContent = sentText;
        displayHtml = escapeHtml(text);
    }
    // The transcript shows the file as a chip, not the whole extracted text —
    // a 40-page PDF would bury the conversation.
    if (attachedDoc) {
        displayHtml = `<span class="msg-attachment">${escapeHtml(attachedDoc.name)}` +
            ` <span class="mono">${attachedDoc.chars.toLocaleString()} chars</span></span>` +
            (displayHtml ? '<br>' + displayHtml : '');
    }

    // Bypass renderMarkdown: displayHtml is already escaped where needed, and
    // with an attached image it contains a real <img> — markdown rendering
    // would escape it and dump the base64 src as visible text.
    addMessage('user', displayHtml.replace(/\n/g, '<br>'), { html: true });
    // Compact before the turn is composed, not after it fails: the NPU refuses
    // an over-long prompt outright.
    compactHistory(false);
    chatHistory.push({ role: 'user', content: userContent });
    invalidateContextCount(true);

    if (overrideText === undefined) {
        input.value = '';
        // Back to one line, and through the same path the keystrokes use, so the
        // cached height cannot go stale against what is on screen.
        autosizeInput();
    }
    clearAttachments();
    syncComposerDraftState();

    const assistantDiv = addMessage('assistant',
                                    activityHtml(noThinkCheckbox.checked ? 'writing' : 'thinking'),
                                    { html: true });
    // A message that is just a link means "read this". Fetch it first and hand
    // the text to the model as context, so the answer is about the page rather
    // than about the model's memory of the domain name.
    const soleUrl = soleUrlIn(text);
    if (soleUrl) {
        try {
            const page = await readUrlInto(assistantDiv, soleUrl);
            const note = page.truncated
                ? ` (first ${page.text.length.toLocaleString()} of ${page.chars.toLocaleString()} characters)`
                : '';
            // Replace the user's turn in history with the page text. The thread
            // still shows the link the user typed; the model gets the content.
            chatHistory[chatHistory.length - 1] = {
                role: 'user',
                content: [
                    `Content of ${soleUrl}${note}:`, '',
                    page.text, '',
                    'Summarise this page and say what it is for.',
                ].join(String.fromCharCode(10)),
            };
            invalidateContextCount(true);
            compactHistory(false);
        } catch (err) {
            assistantDiv.body.innerHTML =
                `<span class="err">${escapeHtml(err.message)}</span>`;
            input.focus();
            return '';
        }
    }
    // Always streamed. A turn that calls a tool still streams: the server
    // holds back only the first few characters of visible text, long enough to
    // tell a tool call from prose, then either goes live or runs the tool and
    // streams the answer it produces.
    const answer = await runCompletion(assistantDiv, {
        stream: true,
        // Not "search now" -- "you may search". The model still decides.
        web_search: webSearchAllowed && webSearchReady,
    });
    input.focus();
    return answer;
}
export function newChat() {
    setChatHistory([]);
    invalidateContextCount(true);
    thread.innerHTML = '';
    resetMessageGrouping();
    emptyState.hidden = false;
    setMode('chat');
    updateContextDisplay();
    scheduleExactContextCount(0);
}
