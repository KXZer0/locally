// Separating a model's <think> block from its answer, for DISPLAY.
//
// In here: the whole-document view of think blocks -- complete, still open,
// or arriving character by character -- and the markup each of those renders.
// Not in here: the voice path's version. voice/think-stream.js decides from
// stream STATE rather than from a regex over a half-written document, because
// mid-stream there is no closing tag to anchor on and speaking reasoning aloud
// is a much worse failure than showing it.

import { escapeHtml } from '../core/format.js';
import { thinkExpanded } from '../core/state.js';

export function splitThink(text, isStreaming) {
    // Handle <think>...</think> blocks BEFORE escaping HTML
    // These are raw model output tags, not user HTML
    let thinkHtml = '';
    // An orphan closing tag with no opener. Qwen3 under /no_think sometimes
    // emits "</think>" alone, and every pattern below anchors on "^<think>",
    // so it fell through and rendered as the first characters of the answer.
    let mainText = String(text).replace(/^\s*<\/think>\s*/, '');
    text = mainText;

    // Complete: <think>...</think> followed by the actual answer
    let thinkMatch = text.match(/^<think>([\s\S]*?)<\/think>\s*([\s\S]*)$/);
    // Partial: <think> started but no closing tag yet (streaming)
    let thinkOpen = !thinkMatch && text.match(/^<think>([\s\S]*)$/);
    // Very early: just the opening tag arriving character by character
    let thinkStarting = !thinkMatch && !thinkOpen && /^<(?:t(?:h(?:i(?:n(?:k)?)?)?)?)?$/.test(text.trim());

    if (thinkMatch) {
        const thinkContent = thinkMatch[1].trim();
        mainText = thinkMatch[2].trim();
        // Skip empty think blocks (no-think mode sometimes emits <think></think>)
        if (thinkContent) {
            const lines = thinkContent.split('\n');
            const preview = lines.slice(-3).join('\n');
            const cls = thinkExpanded ? '' : 'collapsed';
            thinkHtml = `<div class="think-block ${cls}" data-think-toggle>
                <div class="think-header"><svg class="ic" aria-hidden="true"><use href="#i-brain"/></svg>Thought<svg class="ic chev" aria-hidden="true"><use href="#i-chevron"/></svg></div>
                <div class="think-body"><div class="think-scroll">${escapeHtml(thinkContent).replace(/\n/g, '<br>')}</div></div>
            </div>`;
        }
    } else if (thinkOpen && !isStreaming) {
        // The turn ENDED inside an unclosed <think>. Qwen3 does this when it
        // spends its whole budget reasoning, and under /no_think it sometimes
        // opens the tag, writes the answer inside it and stops without closing.
        //
        // The streaming branch below sets mainText = '' and renders a
        // "Thinking…" header, so hitting it after the turn finished left a
        // message frozen mid-thought with a completed meta line under it and
        // the content discarded. That is the bug in the screenshot.
        //
        // Returning the reasoning beats returning nothing — the same call the
        // server's Anthropic path already makes for the same situation.
        const thinkContent = thinkOpen[1].trim();
        mainText = thinkContent;
        if (thinkContent) {
            thinkHtml = '<div class="think-note">'
                      + 'The model stopped while still reasoning, so this is its '
                      + 'working rather than a finished answer.</div>';
        }
    } else if (thinkOpen) {
        // Still thinking — show content live
        const thinkContent = thinkOpen[1].trim();
        if (thinkContent) {
            const lines = thinkContent.split('\n');
            if (lines.length > 4) {
                // Enough lines — expandable + just-answer button
                const preview = lines.slice(-4).join('\n');
                const justAnswerBtn = lines.length > 8
                    ? `<button class="just-answer" data-just-answer>Just answer me, dammit!</button>`
                    : '';
                const cls = thinkExpanded ? '' : 'collapsed';
                thinkHtml = `<div class="think-block streaming ${cls}" data-think-toggle>
                    <div class="think-header"><svg class="ic" aria-hidden="true"><use href="#i-brain"/></svg>Thinking…<svg class="ic chev" aria-hidden="true"><use href="#i-chevron"/></svg></div>
                    <div class="think-body"><div class="think-scroll">${escapeHtml(thinkContent).replace(/\n/g, '<br>')}</div></div>
                    ${justAnswerBtn}
                </div>`;
            } else {
                // Few lines — show all, no collapse needed
                thinkHtml = `<div class="think-block streaming ${thinkExpanded ? '' : 'collapsed'}" data-think-toggle>
                    <div class="think-header"><svg class="ic" aria-hidden="true"><use href="#i-brain"/></svg>Thinking…<svg class="ic chev" aria-hidden="true"><use href="#i-chevron"/></svg></div>
                    <div class="think-body"><div class="think-scroll">${escapeHtml(thinkContent).replace(/\n/g, '<br>')}</div></div>
                </div>`;
            }
        } else {
            thinkHtml = `<div class="think-block streaming">
                <div class="think-header"><svg class="ic" aria-hidden="true"><use href="#i-brain"/></svg>Thinking…</div>
            </div>`;
        }
        mainText = '';
    } else if (thinkStarting && isStreaming) {
        // Partial <think> tag still arriving
        thinkHtml = `<div class="think-block streaming">
            <div class="think-header"><svg class="ic" aria-hidden="true"><use href="#i-brain"/></svg>Thinking…</div>
        </div>`;
        mainText = '';
    }

    return { thinkHtml: thinkHtml, mainText: mainText };
}
