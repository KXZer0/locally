// One spoken turn, end to end.
//
// In here: transcript -> chat history -> streamed answer -> speech queue, with
// the bookkeeping that keeps the thread and the history telling the same story
// when a turn is interrupted halfway.
// Not in here: playback (voice/speech-queue.js), the think filter
// (voice/think-stream.js) and the state machine (voice/state.js). This module
// is the sequence; those are its parts.

import { transcribe } from '../audio/asr.js';
import { buildRequestBody, consumeStream } from '../chat/completion.js';
import { compactHistory } from '../chat/context.js';
import { VOICE_MAX_TOKENS, buildMessages } from '../chat/prompt.js';
import { addMessage, addMeta, attachDevice } from '../chat/thread.js';
import { orb, voiceHeard, voiceReply } from '../core/dom.js';
import { escapeHtml } from '../core/format.js';
import { paintDevice } from '../core/paint.js';
import { chatHistory, ttsReady } from '../core/state.js';
import { renderMarkdown } from '../markdown/render.js';
import { SPEAK_FIRST_MIN_CHARS, SPEAK_MIN_CHARS, bumpSpeechEpoch, finishReveal, makeSpeechQueue, revealSpoken, setSpeakStartedAt, speechEpoch, splitSpeakable } from './speech-queue.js';
import { resetTimings, setVoiceState, showTimings, timings, voiceSession, voiceThinkAllowed } from './state.js';
import { ThinkStream, speakableText } from './think-stream.js';

export let voiceAbort = null;
// `preset` is a transcript the server produced during the end-of-turn pause;
// when it is there the upload-and-wait stage of the turn simply does not exist.
export async function voiceTurn(blob, preset) {
    // Declared out here so an interrupted turn can still settle the bubble it
    // created: aborting used to leave an assistant message in the DOM with no
    // matching chatHistory entry, and the two drifted apart from then on.
    let assistantDiv = null;
    let partial = '';
    try {
        resetTimings();
        setVoiceState('transcribing', '');
        voiceHeard.hidden = true;
        voiceReply.hidden = true;
        voiceReply.textContent = '';

        const asr = preset ? { text: (preset.text || '').trim(), ms: preset.ms || 0 }
                           : await transcribe(blob);
        timings.asr = asr.ms;
        showTimings();
        if (!asr.text) {
            setVoiceState('idle', "Didn't catch that — try again.");
            return;
        }
        voiceHeard.textContent = asr.text;
        voiceHeard.hidden = false;

        setVoiceState('thinking', '');
        // Mirror into the chat thread so the two tabs really are one
        // conversation, not two transcripts that happen to share a model.
        addMessage('user', escapeHtml(asr.text), { html: true });
        // Voice forgets silently: the chat thread still holds the full
        // record, and interrupting a spoken turn to report bookkeeping is
        // worse than quietly dropping the oldest exchange.
        compactHistory(true);
        chatHistory.push({ role: 'user', content: asr.text });

        assistantDiv = addMessage('assistant', '<span class="typing-indicator"></span>',
                                  { html: true });
        const t0 = performance.now();
        voiceAbort = new AbortController();
        const resp = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: voiceAbort.signal,
            body: JSON.stringify(buildRequestBody({
                // No-think unless the user opted in; voice styling either way.
                messages: buildMessages(!voiceThinkAllowed(), true),
                max_tokens: VOICE_MAX_TOKENS,
            })),
        });

        const device = resp.headers.get('X-Device') || '';
        attachDevice(assistantDiv, device);
        paintDevice(orb, device);          // the orb takes the answering engine's colour
        paintDevice(voiceReply, device);

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            const msg = err.error?.message || 'Chat failed';
            assistantDiv.body.innerHTML =
                `<span style="color:var(--error)">${escapeHtml(msg)}</span>`;
            assistantDiv = null;          // settled — don't also treat it as partial
            setVoiceState('idle', msg);
            return;
        }

        // Stays hidden until there is something to put in it — under karaoke
        // the first text arrives with the first clip, and an empty bordered box
        // in the meantime just looks like a failure.

        // Speak as it streams: each finished sentence is synthesized while the
        // model is still writing the next one, so the first audio lands
        // seconds earlier than waiting for the whole answer.
        const epoch = bumpSpeechEpoch();
        setSpeakStartedAt(performance.now());
        const think = new ThinkStream(voiceThinkAllowed());
        // Karaoke: with TTS present the queue owns #voice-reply and reveals
        // each chunk as it STARTS PLAYING, so what you read is what you hear.
        // Painting it from the token stream instead put the text seconds ahead
        // of the voice, which is what made the two feel unrelated.
        const queue = ttsReady ? makeSpeechQueue(epoch, revealSpoken) : null;
        let emitted = '';        // answer text already handed to the splitter
        let pendingTail = '';    // split remainder, not yet a whole chunk
        let sawFirstChunk = false;

        const feed = (full) => {
            partial = full;
            const clean = speakableText(think.push(full));
            if (!queue) {                       // nothing to sync to — show it live
                voiceReply.hidden = !clean;
                voiceReply.textContent = clean;
                voiceReply.scrollTop = voiceReply.scrollHeight;
                return;
            }
            if (think.state === 'thinking') {
                setVoiceState('thinking', 'Reasoning…');
                return;
            }
            // The visible text can retract when </think> lands: everything
            // before it was reasoning and is not part of the answer at all.
            // Nothing was queued while thinking, so resetting is safe.
            if (!clean.startsWith(emitted)) { emitted = ''; pendingTail = ''; }
            const fresh = clean.slice(emitted.length);
            emitted = clean;
            const [chunks, remainder] = splitSpeakable(
                pendingTail + fresh,
                sawFirstChunk ? SPEAK_MIN_CHARS : SPEAK_FIRST_MIN_CHARS);
            for (const c of chunks) {
                if (!sawFirstChunk) { sawFirstChunk = true; setVoiceState('speaking', ''); }
                queue.enqueue(c);
            }
            pendingTail = remainder;
        };

        let text;
        let finishReason = '';
        if ((resp.headers.get('content-type') || '').includes('text/event-stream')) {
            text = await consumeStream(resp, assistantDiv.body, feed,
                                       (r) => { finishReason = r; });
        } else {
            const data = await resp.json();
            text = data.choices?.[0]?.message?.content || '';
            finishReason = data.choices?.[0]?.finish_reason || '';
        }
        timings.llm = performance.now() - t0;
        assistantDiv.body.innerHTML = renderMarkdown(text, false);
        addMeta(assistantDiv, device, (timings.llm / 1000).toFixed(1), 'voice');
        chatHistory.push({ role: 'assistant', content: text });
        assistantDiv = null;              // recorded — the catch must not re-add it

        let spoken = speakableText(think.flush(text));
        showTimings();
        // An unclosed <think> alone does NOT mean the model was still
        // reasoning. Qwen3 under /no_think opens the tag, writes the answer
        // inside it and stops without ever closing — measured here. What
        // separates the two is why generation ended: hitting the token cap
        // means it never got to the answer, stopping naturally means what it
        // wrote IS the answer. Only the first case is worth apologising for;
        // reading a page of internal monologue out loud is worse than silence,
        // and the voice prompt already teaches "the details are in the chat
        // tab" as the shape of that answer.
        if (think.discarded && finishReason === 'length') {
            spoken = 'I got stuck thinking about that one and ran out of room '
                   + 'before answering. The reasoning is in the chat tab.';
            emitted = '';
            pendingTail = '';
        }
        if (!spoken) { setVoiceState('idle', ''); return; }

        if (!queue) {
            voiceReply.hidden = false;
            voiceReply.textContent = spoken;
            setVoiceState('idle', 'No TTS model loaded (--tts-dir) — answer shown, not spoken.');
            return;
        }

        // The closing sentence has no trailing whitespace, so splitSpeakable
        // never closed it — it is sitting in pendingTail. This used to be
        // recomputed as a slice that always evaluated to '', which is why the
        // last sentence of every answer was shown but never spoken.
        const tail = spoken.startsWith(emitted)
            ? (pendingTail + spoken.slice(emitted.length))
            : spoken;      // retracted or unclosed-think fallback: nothing queued
        const [tailChunks, tailRest] = splitSpeakable(
            tail, sawFirstChunk ? SPEAK_MIN_CHARS : SPEAK_FIRST_MIN_CHARS);
        for (const c of [...tailChunks, tailRest]) {
            if (!c.trim()) continue;
            if (!sawFirstChunk) { sawFirstChunk = true; setVoiceState('speaking', ''); }
            queue.enqueue(c.trim());
        }
        if (sawFirstChunk) setVoiceState('speaking', '');
        await queue.done();
        // A barge-in during that await already moved us on. Forcing idle here
        // would strand the capture that the interrupt just started.
        if (epoch !== speechEpoch) return;
        finishReveal(spoken);
        setVoiceState('idle', voiceSession ? 'Listening for your next turn.' : '');
    } catch (err) {
        if (err.name === 'AbortError') setVoiceState('idle', 'Interrupted.');
        else setVoiceState('idle', err.message);
    } finally {
        // An interrupted turn still said something — keep the thread and the
        // history telling the same story.
        if (assistantDiv) {
            if (partial.trim()) {
                assistantDiv.body.innerHTML = renderMarkdown(partial, false) +
                    '<br><span style="color:var(--text-dim)">[interrupted]</span>';
                chatHistory.push({ role: 'assistant', content: partial });
            } else {
                assistantDiv.remove();
            }
        }
        voiceAbort = null;
    }
}
