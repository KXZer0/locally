// Speaking an answer while it is still being written.
//
// In here: the sentence splitter, the synthesise-ahead/play-in-order queue,
// the karaoke reveal (text appears when the clip carrying it STARTS PLAYING,
// not when the token arrives), and every interrupt path.
// Not in here: what to speak. voice/think-stream.js decides that, and
// voice/turn.js feeds it. Two measured details live in the comments below --
// why the first clip has a lower character floor, and why pause() has to
// settle its promise explicitly.

import { input, voiceReply } from '../core/dom.js';
import { setVoiceState, showTimings, timings, voiceName } from './state.js';
import { voiceAbort } from './turn.js';

export let currentAudio = null;      // the <audio> currently playing TTS
// TTS has no streaming, so there is a dead window — often seconds — between
// asking for audio and having any. A barge-in during that window used to be
// ignored, and the assistant would start talking *after* being interrupted.
// The epoch makes any speech request older than the last interrupt inert.
export let speechEpoch = 0;
// --- Sentence-chunked speech ---
//
// The pipeline can't stream audio, but it doesn't have to wait for the whole
// answer either: synthesize sentence by sentence as the text streams in, and
// play the clips back to back. First audio then arrives after the FIRST
// sentence instead of the last, and synthesis of sentence N+1 overlaps
// playback of N — so after the first clip the speech is usually gapless.
export const SPEAK_MIN_CHARS = 40;      // don't ship a 3-word fragment to TTS
export const SPEAK_MAX_CHARS = 320;     // flush long run-ons so audio keeps up
// The FIRST clip of a turn is the one the user is waiting on, and every
// character in it is dead air. Ship any complete sentence, however short:
// "Sure." starts the audio a full clip earlier, and the pause while the next
// sentence synthesizes is one a person would leave there anyway. Later clips
// keep the 40-char floor so speech doesn't fragment once it's running. The
// floor of 4 exists only to reject a bare "." left by markdown stripping,
// which would synthesize to a click.
export const SPEAK_FIRST_MIN_CHARS = 4;

export function splitSpeakable(text, minChars = SPEAK_MIN_CHARS) {
    // Returns [completeChunks, remainder]. A chunk ends at sentence
    // punctuation followed by whitespace, so decimals and "e.g." survive.
    const chunks = [];
    let rest = text;
    const boundary = /([.!?…]+)(\s+)/g;
    let last = 0, m;
    while ((m = boundary.exec(rest)) !== null) {
        const end = m.index + m[1].length;
        if (end - last >= minChars) {
            chunks.push(rest.slice(last, end).trim());
            last = end + m[2].length;
        }
    }
    let remainder = rest.slice(last);
    // A very long clause with no punctuation would otherwise never flush.
    while (remainder.length > SPEAK_MAX_CHARS) {
        const cut = remainder.lastIndexOf(' ', SPEAK_MAX_CHARS);
        const at = cut > SPEAK_MIN_CHARS ? cut : SPEAK_MAX_CHARS;
        chunks.push(remainder.slice(0, at).trim());
        remainder = remainder.slice(at).trimStart();
    }
    return [chunks.filter(Boolean), remainder];
}

// A pipeline: synthesis runs ahead of playback, both in submission order.
// Karaoke reveal. The text appears when the clip carrying it actually starts
// playing, so the reply reads at speaking pace instead of racing ahead of it.
export function revealSpoken(text) {
    voiceReply.hidden = false;
    const prev = voiceReply.querySelector('.speaking');
    if (prev) prev.classList.remove('speaking');
    const span = document.createElement('span');
    span.className = 'speaking';
    span.textContent = (voiceReply.childNodes.length ? ' ' : '') + text;
    voiceReply.appendChild(span);
    voiceReply.scrollTop = voiceReply.scrollHeight;
}

// End of turn: nothing is being spoken any more, so drop the highlight. If
// every clip failed to synthesize the reveal never ran, and showing nothing
// would lose an answer we already have.
export function finishReveal(spoken) {
    const prev = voiceReply.querySelector('.speaking');
    if (prev) prev.classList.remove('speaking');
    if (!voiceReply.textContent.trim() && spoken) {
        voiceReply.hidden = false;
        voiceReply.textContent = spoken;
    }
}

export function makeSpeechQueue(epoch, onSpeakStart) {
    const pending = [];          // promises for audio blobs, in order
    let playing = Promise.resolve();
    let firstAudioAt = null;

    function enqueue(text) {
        if (!text || epoch !== speechEpoch) return;
        const job = synthesize(text, epoch);
        pending.push(job);
        playing = playing.then(async () => {
            if (epoch !== speechEpoch) return;
            const blob = await job;
            if (epoch !== speechEpoch) return;
            // Synthesis failed: still reveal the text, or the answer would be
            // neither heard nor read.
            if (!blob) { if (onSpeakStart) onSpeakStart(text); return; }
            if (firstAudioAt === null) {
                firstAudioAt = performance.now();
                timings.tts_first = firstAudioAt - speakStartedAt;
                showTimings();
            }
            await playBlob(blob, () => { if (onSpeakStart) onSpeakStart(text); });
        }).catch(() => {});
    }

    return {
        enqueue,
        // Resolves when everything queued has finished playing.
        done: () => playing,
    };
}

export let speakStartedAt = 0;

export async function synthesize(text, epoch) {
    const controller = new AbortController();
    speechAborts.add(controller);
    try {
        const resp = await fetch('/v1/audio/speech', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({ input: text, voice: voiceName() }),
        });
        if (!resp.ok || epoch !== speechEpoch) return null;
        return await resp.blob();
    } catch {
        return null;             // aborted or failed — the turn carries on
    } finally {
        speechAborts.delete(controller);
    }
}

export const speechAborts = new Set();  // in-flight synthesis requests, cancellable

export let stopCurrentPlayback = null;   // resolves the in-flight playBlob promise

export function playBlob(blob, onStart) {
    return new Promise((resolve) => {
        const url = URL.createObjectURL(blob);
        const el = new Audio(url);
        currentAudio = el;
        let settled = false;
        const done = () => {
            if (settled) return;
            settled = true;
            URL.revokeObjectURL(url);
            if (currentAudio === el) currentAudio = null;
            stopCurrentPlayback = null;
            resolve();
        };
        // pause() fires neither 'ended' nor 'error', so an interrupt has to
        // settle this promise explicitly or the turn hangs here forever.
        stopCurrentPlayback = done;
        el.onended = done;
        el.onerror = done;
        // play() resolves once audio is genuinely running, which is the only
        // honest moment to reveal the text this clip is saying.
        el.play().then(() => { if (onStart) onStart(); }).catch(done);
    });
}
// Cut audio first, then try to stop generation. The repo's own note applies:
// if OpenVINO is blocked in a native call the cancel may not land, so the
// user-visible interrupt must not depend on it.
export function interruptSpeech() {
    speechEpoch++;                       // invalidates every queued/in-flight clip
    for (const c of speechAborts) { try { c.abort(); } catch {} }
    speechAborts.clear();
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.src = '';
        currentAudio = null;
    }
    if (stopCurrentPlayback) stopCurrentPlayback();
    if (voiceAbort) voiceAbort.abort();
    fetch('/v1/cancel', { method: 'POST' }).catch(() => {});
    setVoiceState('idle', 'Interrupted — go ahead.');
}
export function interruptAudioOnly() {
    speechEpoch++;
    for (const c of speechAborts) { try { c.abort(); } catch {} }
    speechAborts.clear();
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.src = '';
        currentAudio = null;
    }
    if (stopCurrentPlayback) stopCurrentPlayback();
}

export function bumpSpeechEpoch() { return ++speechEpoch; }
export function setSpeakStartedAt(at) { speakStartedAt = at; }
