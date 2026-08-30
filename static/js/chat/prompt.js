// What the model is told, and the switch that decides how hard it thinks.
//
// In here: the default system and voice prompts, the /no_think control token
// and which families implement it, and buildMessages() -- the ONE place that
// composes a turn. Every send path goes through it, so the system prompt
// cannot be live on one path and missing on another.
// Not in here: the user's stored text (localStorage, read here on demand) and
// the settings panel that edits it (ui/settings.js).

import { activeChatSlot } from './context.js';
import { noThinkCheckbox } from '../core/dom.js';
import { chatHistory, lastHealthData, loadedModelId } from '../core/state.js';

// --- Prompt composition ---

// Qwen models drift into Chinese mid-answer without an explicit language
// instruction — a documented family-wide quirk, not a bug in this stack.
// Solving it once here beats every user rediscovering it.
//
// Everything else here is aimed at the failure modes of 4-30B local models
// specifically, which are not the failure modes of a frontier model: the
// "Certainly! Here's..." preamble, padding an answer to look thorough,
// inventing a plausible API or citation rather than admitting a gap, and
// bolting a safety disclaimer onto a question about pasta. Each line below
// is one of those, stated concretely — small models follow "read text
// exactly as printed" and ignore "be helpful and accurate".
//
// It is deliberately ~220 tokens (measured on Qwen3-8B and gemma-4-E4B).
// That is not a style choice: on the NPU the whole prompt has to fit
// MAX_PROMPT_LEN=4096, and a system prompt is re-sent on every single turn.
// The 21k-token prompts that agent tools ship would eat half the NPU's
// budget before the user typed anything. Add to it sparingly.
export const DEFAULT_SYSTEM_PROMPT =
    'You are a local assistant running on this machine. Reply in English '
    + 'unless the user writes in another language, then use theirs.\n\n'
    + 'Lead with the answer. No preamble, no restating the question, no '
    + '"Certainly" or "Great question". Stop when the answer is complete; '
    + 'length is not quality.\n\n'
    + 'Be concrete: real numbers, names, commands, code. If you don\'t know '
    + 'or aren\'t sure, say so in one line instead of inventing. Never invent '
    + 'quotes, citations, URLs, APIs, file contents or figures.\n\n'
    + 'Prose by default. Use markdown only when it earns its place: fenced '
    + 'blocks for code, a short list for genuinely parallel items. No '
    + 'headings in short answers.\n\n'
    + 'For images: describe only what is visible. Read any text exactly as '
    + 'printed, and say when it is blurred, cut off or ambiguous rather than '
    + 'guessing.\n\n'
    + 'If a request is ambiguous, make the most reasonable assumption, say '
    + 'which one in a sentence, and answer anyway. Ask only when you '
    + 'genuinely cannot proceed.\n\n'
    + 'No lecturing, no safety caveats on ordinary requests, no apologies '
    + 'unless you got something wrong.';
export const NO_THINK_INSTRUCTION =
    'Respond directly and concisely. Do not use <think> blocks or internal reasoning.';
// Qwen3 has a native soft switch for this, and the English sentence above does
// not trigger it: asked "what is an NPU?" with only the sentence, Qwen3-8B
// opened a <think> block and spent all 220 voice tokens inside it without ever
// closing the tag or answering. `/no_think` is a real control token for that
// family and inert text for every other, so it costs nothing to send.
export const NO_THINK_TOKEN = '/no_think';

// Which families actually implement /no_think as a control token. The original
// comment here claimed it was "inert text for every other" model — it is not.
// A model that does not recognise it reads it as content and answers ABOUT it,
// which is the same failure Qwen3 showed when it was appended inline. Sending
// it only where it means something costs those models nothing: the English
// NO_THINK_INSTRUCTION is already in their system prompt.
export const NO_THINK_MODELS = /qwen\s*3|deepseek\S*r1|\br1\b/i;

export function modelUsesNoThinkToken() {
    const slot = activeChatSlot(lastHealthData);
    const name = (slot && slot.model) || loadedModelId || '';
    return NO_THINK_MODELS.test(name);
}

// Voice and text are different media, so they get different instructions.
// Markdown is meaningless out loud — a table read aloud is noise, and a
// bulleted list becomes a stream of "dash". Spoken answers also want to be
// short: you can skim a screen, but you have to sit through audio.
export const DEFAULT_VOICE_PROMPT =
    'You are speaking out loud. Reply in at most 2-3 short sentences of plain '
    + 'conversational English. Never use markdown: no tables, bullet points, '
    + 'numbered lists, headings, code blocks or asterisks. Write numbers and '
    + 'symbols as words where it reads more naturally. If the full answer needs '
    + 'a list, table or code, give the short spoken summary and say the details '
    + 'are in the chat tab.';

// Voice replies are capped harder than text ones: without a ceiling a chatty
// model will happily talk for two minutes.
export const VOICE_MAX_TOKENS = 220;

export function getVoicePrompt() {
    const stored = localStorage.getItem('locally-voice-prompt');
    return stored === null ? DEFAULT_VOICE_PROMPT : stored;
}

// null (never set) falls back to the default; "" is a deliberate opt-out and
// must survive a reload, so don't collapse the two with `|| DEFAULT`.
export function getSystemPrompt() {
    const stored = localStorage.getItem('locally-system-prompt');
    return stored === null ? DEFAULT_SYSTEM_PROMPT : stored;
}

// The one place that decides what the model is told. Every send path goes
// through it, so the system prompt can't be live on one and missing on another.
export function buildMessages(forceNoThink, forVoice, suppressNoThink = false) {
    const directives = [];
    const noThink = !suppressNoThink && (forceNoThink || noThinkCheckbox.checked);
    const sys = getSystemPrompt().trim();
    if (sys) directives.push(sys);
    if (noThink) directives.push(NO_THINK_INSTRUCTION);
    // Last, so the spoken-style rules override anything the general prompt
    // said about formatting.
    if (forVoice) {
        const vp = getVoicePrompt().trim();
        if (vp) directives.push(vp);
    }

    let history = chatHistory.filter(m => m.role !== 'system');
    // Qwen3 reads the switch off the most recent turn, so it goes on the last
    // user message rather than the system block. On a COPY — chatHistory is
    // what the user sees in the thread, and it must not grow control tokens.
    //
    // On its OWN LINE, never appended inline. Measured 4/4 deterministic on
    // Qwen3-8B with the real system prompt: "sup /no_think" made the model
    // explain what /no_think does, every single time, because on a one-word
    // message the token is the most substantive thing in the turn. The same
    // message with the token on its own line answered the greeting, 4/4.
    // Putting it in the system prompt was also tried and was worse still — it
    // misread "sup" as a report of system trouble.
    if (noThink && history.length && modelUsesNoThinkToken()) {
        const i = history.map(m => m.role).lastIndexOf('user');
        if (i !== -1 && typeof history[i].content === 'string') {
            history = history.slice();
            history[i] = { ...history[i],
                           content: `${history[i].content}\n\n${NO_THINK_TOKEN}` };
        }
    }
    if (!directives.length) return history;
    return [{ role: 'system', content: directives.join('\n\n') }, ...history];
}

// --- Thinking switch + web-search toggle -------------------------------------

export const thinkToggle = document.getElementById('think-toggle');
// The switch reads positively ("Thinking on") while every send path reads
// no-think. Inverting the sense here is safer than rewriting all of them, and
// it keeps the persisted key meaning what it always meant.
export const thinkNote = document.getElementById('think-note');

export function syncThinkSwitch() {
    if (!thinkToggle) return;
    thinkToggle.checked = !noThinkCheckbox.checked;
    // The switch reads positively; the note says what that costs, because
    // "thinking on" without "slower" is only half the trade.
    if (thinkNote) thinkNote.textContent = thinkToggle.checked ? 'slower' : 'off';
}

if (thinkToggle) {
    thinkToggle.addEventListener('change', () => {
        noThinkCheckbox.checked = !thinkToggle.checked;
        noThinkCheckbox.dispatchEvent(new Event('change'));
    });
}
