// The chat tab's microphone button.
//
// In here: its OWN stream and MediaRecorder, push-to-talk, and the transcription
// request. The transcript lands in the composer rather than being sent, so it
// can be read and fixed before it commits.
// Not in here: the voice tab's capture. They shared one stream global once,
// which meant ending a chat recording tore down the voice graph's tracks.

import { toWav16kMono } from './wav.js';
import { input, micBtn, micStatusEl } from '../core/dom.js';
import { asrReady, pttActive, setPttActive } from '../core/state.js';

export let mediaRecorder = null;
export let recChunks = [];
export function micStatus(text, isError) {
    micStatusEl.hidden = !text;
    micStatusEl.textContent = text || '';
    micStatusEl.classList.toggle('err', !!isError);
}
// The chat mic keeps its OWN stream. It used to share `mediaStream` with voice
// mode, which means stopping a chat recording tore down the tracks the voice
// capture graph was running on. The tabs happen not to overlap today, so it
// never fired — but one shared global between two independent features is a
// trap waiting for the first person who makes them overlap.
export let chatStream = null;

export async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Microphone unavailable (needs localhost or HTTPS)');
    }
    chatStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    recChunks = [];
    mediaRecorder = new MediaRecorder(chatStream);
    mediaRecorder.ondataavailable = (e) => { if (e.data.size) recChunks.push(e.data); };
    mediaRecorder.start();
}

export function stopRecording() {
    return new Promise((resolve) => {
        if (!mediaRecorder) return resolve(null);
        const rec = mediaRecorder;
        rec.onstop = () => {
            const blob = new Blob(recChunks, { type: rec.mimeType || 'audio/webm' });
            if (chatStream) chatStream.getTracks().forEach(t => t.stop());
            chatStream = null;
            mediaRecorder = null;
            resolve(blob);
        };
        rec.stop();
    });
}

// Returns { text, ms } — the timing is reported separately from the LLM's so
// each stage of the voice pipeline can be measured on its own.
export async function transcribe(blob) {
    // The voice path already hands us 16 kHz mono PCM from the capture worklet;
    // only the chat tab's MediaRecorder still needs decoding and resampling.
    const wav = blob.type === 'audio/wav' ? blob : await toWav16kMono(blob);
    const fd = new FormData();
    fd.append('file', wav, 'speech.wav');
    // Naming the language skips Whisper's language-detection pass, which is a
    // measurable chunk of a short utterance's latency. "auto" leaves it out.
    const lang = localStorage.getItem('locally-asr-lang');
    if (lang && lang !== 'auto') fd.append('language', lang);
    const t0 = performance.now();
    const resp = await fetch('/v1/audio/transcriptions', { method: 'POST', body: fd });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error?.message || `Transcription failed (${resp.status})`);
    }
    const data = await resp.json();
    return { text: (data.text || '').trim(), ms: performance.now() - t0 };
}

// Push-to-talk in the Chat tab drops the transcript into the composer rather
// than sending it: you can see what it heard and fix it before committing.
export async function pttStart(e) {
    if (e) e.preventDefault();
    if (pttActive || !asrReady) return;
    try {
        await startRecording();
        setPttActive(true);
        micBtn.classList.add('recording');
        micStatus('listening — release to transcribe');
    } catch (err) {
        micStatus(err.message, true);
    }
}

export async function pttEnd() {
    if (!pttActive) return;
    setPttActive(false);
    micBtn.classList.remove('recording');
    micStatus('transcribing…');
    try {
        const blob = await stopRecording();
        if (!blob || !blob.size) { micStatus(''); return; }
        const { text, ms } = await transcribe(blob);
        if (!text) {
            micStatus(`nothing heard (${(ms / 1000).toFixed(1)}s)`);
            return;
        }
        input.value = input.value ? `${input.value} ${text}` : text;
        input.dispatchEvent(new Event('input'));
        input.focus();
        micStatus(`transcribed in ${(ms / 1000).toFixed(1)}s`);
        setTimeout(() => micStatus(''), 4000);
    } catch (err) {
        micStatus(err.message, true);
    }
}
// Push-to-talk: pointer events cover mouse, pen and touch in one path, and
// pointercancel matters — a drag off the button must not leave the mic open.
micBtn.addEventListener('pointerdown', pttStart);
micBtn.addEventListener('pointerup', pttEnd);
micBtn.addEventListener('pointercancel', pttEnd);
micBtn.addEventListener('pointerleave', () => { if (pttActive) pttEnd(); });
// Keyboard equivalent, so push-to-talk isn't mouse-only.
micBtn.addEventListener('keydown', (e) => {
    if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) { e.preventDefault(); pttStart(e); }
});
micBtn.addEventListener('keyup', (e) => {
    if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); pttEnd(); }
});
