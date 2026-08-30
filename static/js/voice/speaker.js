// Speaker enrolment: Silero says SOMEONE is talking, this says WHO.
//
// In here: the enrol / test / forget panel and the voice profile it builds.
// Its own stream and its own MediaRecorder, deliberately -- sharing either
// with the capture graph is the exact bug the chat mic already had once.
// Not in here: the gating decision, which is the server's. The threshold
// depends on model, microphone and room, so the panel's job is to SHOW you the
// cosine you need to set --speaker-threshold from, not to pick one.

import { toWav16kMono } from '../audio/wav.js';
import { speakerBadge, speakerBadgeText, speakerEnrollBtn, speakerEnrollLabel, speakerFigures, speakerForgetBtn, speakerForgetCancel, speakerModel, speakerPanel, speakerProgress, speakerProgressFill, speakerScore, speakerSeconds, speakerStale, speakerStatusEl, speakerThreshold, speakerVerdict, speakerVerdictHint, speakerVerdictName, speakerVerifyBtn } from '../core/dom.js';

// --- Speaker enrolment (Voice tab) ---
//
// Its own stream and its own MediaRecorder, deliberately. The voice tab's
// capture is an AudioWorklet graph feeding the VAD socket; this is a fixed-
// length clip that has to be uploaded as a WAV. Sharing either the stream or
// the recorder between them is the exact bug the chat mic already had once —
// ending one recording stopped the tracks the other was still reading — so
// these two globals belong to this panel and nothing else touches them.

export const SPEAKER_ENROLL_MS = 5000;   // one utterance; the server folds several
export const SPEAKER_VERIFY_MS = 3000;   // well over the server's 0.55s abstain floor

export let speakerStream = null;
export let speakerRecorder = null;
export let speakerBusy = false;
export let speakerForgetArmed = false;
export let speakerCountdownTimer = null;
export let speakerLastProfile = null;
export let speakerAbort = null;          // ends the current clip early
export let speakerCancelled = false;

export const SPEAKER_CANCELLED = 'speaker-cancelled';

// Leaving the Voice tab must not leave a microphone open, and the clip in
// flight is worthless once the panel is off screen. Ends the recording through
// the recorder's own stop path so the promise chain still settles.
export function cancelSpeakerRecording() {
    if (!speakerAbort) return;
    speakerCancelled = true;
    speakerAbort();
}

export function speakerStatus(text, isError) {
    if (!speakerStatusEl) return;
    speakerStatusEl.hidden = !text;
    speakerStatusEl.textContent = text || '';
    speakerStatusEl.classList.toggle('err', !!isError);
}

// The whole panel is gated on /health carrying a `speaker` key: without the
// model the endpoints 503, and a control wired to a 503 is worse than no
// control. Read from the poll's payload — no second polling loop.
export function applySpeakerHealth(health) {
    if (!speakerPanel) return;
    const slot = health && health.speaker;
    speakerPanel.hidden = !slot;
    if (!slot) return;
    renderSpeakerProfile(slot.profile);
    const line = [slot.model, slot.device_name].filter(Boolean).join(' · ');
    speakerModel.textContent = line;
    speakerModel.hidden = !line;
}

export function renderSpeakerProfile(profile) {
    if (!speakerPanel) return;
    const p = profile || speakerLastProfile || {};
    speakerLastProfile = p;
    const needed = Number(p.needed_seconds) || 6;
    const secs = Number(p.seconds) || 0;
    const utterances = Number(p.utterances) || 0;
    const frac = needed > 0 ? Math.min(1, secs / needed) : 0;

    if (speakerProgressFill) speakerProgressFill.style.transform = `scaleX(${frac})`;
    speakerProgress.classList.toggle('full', !!p.ready);
    speakerProgress.setAttribute('aria-valuemax', needed.toFixed(1));
    speakerProgress.setAttribute('aria-valuenow', secs.toFixed(1));
    speakerFigures.textContent =
        `${secs.toFixed(1)} / ${needed.toFixed(1)}s · ` +
        `${utterances} recording${utterances === 1 ? '' : 's'}`;

    // Four states, told apart by dot fill rather than by colour: solid ready,
    // ring part-way, hairline none. Stale is the one that is genuinely broken.
    let level = 'none';
    let text = 'not taught yet';
    if (p.stale) { level = 'stale'; text = 'needs redoing'; }
    else if (p.ready) { level = 'ready'; text = 'ready'; }
    else if (p.enrolled) { level = 'learning'; text = 'still learning'; }
    speakerBadge.dataset.level = level;
    speakerBadgeText.textContent = text;
    speakerStale.hidden = !p.stale;

    if (!speakerEnrollLabel.dataset.busy) {
        speakerEnrollLabel.textContent = p.enrolled && !p.stale
            ? 'Record another' : 'Teach it your voice';
    }
    syncSpeakerButtons();
}

export function syncSpeakerButtons() {
    const p = speakerLastProfile || {};
    speakerEnrollBtn.disabled = speakerBusy;
    speakerVerifyBtn.disabled = speakerBusy || !p.ready;
    speakerVerifyBtn.title = p.ready
        ? 'Record a few seconds and see how closely it matches'
        : 'Needs a finished voice profile first';
    speakerForgetBtn.disabled = speakerBusy || !p.enrolled;
}

// Records a fixed-length clip on a private stream and returns 16 kHz mono WAV.
// MediaRecorder gives WebM/Opus, which the server's reader cannot open, so it
// goes through the same decode+resample the chat mic uses.
export async function recordSpeakerClip(ms, label) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Microphone unavailable (needs localhost or HTTPS)');
    }
    speakerCancelled = false;
    speakerStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    const chunks = [];
    const rec = new MediaRecorder(speakerStream);
    speakerRecorder = rec;
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    // `pause()`/`stop()` failures must still settle this, or a failed stop
    // would leave the panel stuck in its busy state forever.
    let settle;
    const done = new Promise((resolve) => { settle = resolve; });
    rec.onstop = () => {
        const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' });
        releaseSpeakerMic();
        settle(blob);
    };
    rec.start();
    speakerEnrollBtn.classList.add('recording');

    const started = performance.now();
    const tick = () => {
        const left = Math.max(0, ms - (performance.now() - started));
        speakerStatus(`${label} — ${(left / 1000).toFixed(1)}s left`);
    };
    tick();
    speakerCountdownTimer = setInterval(tick, 100);
    await new Promise((resolve) => {
        const timer = setTimeout(finish, ms);
        speakerAbort = () => { clearTimeout(timer); finish(); };
        function finish() { speakerAbort = null; resolve(); }
    });
    clearInterval(speakerCountdownTimer);
    speakerCountdownTimer = null;
    speakerEnrollBtn.classList.remove('recording');
    try { rec.stop(); } catch { releaseSpeakerMic(); settle(null); }

    const blob = await done;
    if (speakerCancelled) throw new Error(SPEAKER_CANCELLED);
    if (!blob || !blob.size) throw new Error('Nothing was recorded');
    return toWav16kMono(blob);
}

export function releaseSpeakerMic() {
    if (speakerStream) speakerStream.getTracks().forEach(t => t.stop());
    speakerStream = null;
    speakerRecorder = null;
}

export async function speakerRequest(method, wav) {
    const opts = { method };
    if (wav) {
        const fd = new FormData();
        fd.append('file', wav, 'voice.wav');
        opts.body = fd;
    }
    const resp = await fetch('/v1/audio/enroll', opts);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
        throw new Error(data.error?.message || `Request failed (${resp.status})`);
    }
    return data;
}

export async function speakerEnroll() {
    if (speakerBusy) return;
    speakerBusy = true;
    disarmSpeakerForget();
    syncSpeakerButtons();
    speakerEnrollLabel.dataset.busy = '1';
    speakerEnrollLabel.textContent = 'Listening…';
    try {
        const wav = await recordSpeakerClip(SPEAKER_ENROLL_MS, 'Keep talking');
        speakerStatus('Learning your voice…');
        const state = await speakerRequest('POST', wav);
        renderSpeakerProfile(state);
        const left = Math.max(0, (state.needed_seconds || 6) - (state.seconds || 0));
        speakerStatus(state.ready
            ? 'Done — it knows your voice.'
            : `Good. About ${left.toFixed(1)}s more to go.`);
    } catch (err) {
        if (err.message === SPEAKER_CANCELLED) speakerStatus('');
        else speakerStatus(err.message, true);
    } finally {
        releaseSpeakerMic();
        speakerEnrollBtn.classList.remove('recording');
        delete speakerEnrollLabel.dataset.busy;
        speakerBusy = false;
        renderSpeakerProfile(speakerLastProfile);
    }
}

// The point of this is the raw cosine, not the pass/fail: it is what
// --speaker-threshold gets set from. Record yourself and record someone else,
// and the threshold belongs in the gap between the two numbers.
export async function speakerVerify() {
    if (speakerBusy) return;
    speakerBusy = true;
    disarmSpeakerForget();
    syncSpeakerButtons();
    speakerVerdict.hidden = true;
    try {
        const wav = await recordSpeakerClip(SPEAKER_VERIFY_MS, 'Say something');
        speakerStatus('Comparing…');
        const fd = new FormData();
        fd.append('file', wav, 'voice.wav');
        const resp = await fetch('/v1/audio/verify', { method: 'POST', body: fd });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
            throw new Error(data.error?.message || `Test failed (${resp.status})`);
        }
        showSpeakerVerdict(data);
        speakerStatus('');
    } catch (err) {
        if (err.message === SPEAKER_CANCELLED) speakerStatus('');
        else speakerStatus(err.message, true);
    } finally {
        releaseSpeakerMic();
        speakerEnrollBtn.classList.remove('recording');
        speakerBusy = false;
        syncSpeakerButtons();
    }
}

export function showSpeakerVerdict(data) {
    const verdict = data.verdict || 'abstain';
    speakerVerdict.hidden = false;
    speakerVerdict.dataset.verdict = verdict;
    speakerVerdictName.textContent = {
        match: 'that was you',
        reject: 'not you',
        abstain: "couldn't tell",
    }[verdict] || verdict;
    speakerScore.textContent =
        typeof data.score === 'number' ? data.score.toFixed(4) : '—';
    speakerThreshold.textContent =
        typeof data.threshold === 'number' ? data.threshold.toFixed(4) : '—';
    speakerSeconds.textContent =
        typeof data.seconds === 'number' ? `${data.seconds.toFixed(2)}s` : '—';
    speakerVerdictHint.textContent = data.reason
        ? data.reason
        : 'Set --speaker-threshold between your own score and a stranger’s.';
}

export function disarmSpeakerForget() {
    if (!speakerPanel) return;
    speakerForgetArmed = false;
    speakerForgetBtn.textContent = 'Forget my voice';
    speakerForgetCancel.hidden = true;
}

// Destructive and not undoable, so it takes two clicks. The second click is
// the confirmation; "Keep it" backs out.
export async function speakerForget() {
    if (speakerBusy) return;
    if (!speakerForgetArmed) {
        speakerForgetArmed = true;
        speakerForgetBtn.textContent = 'Yes, forget it';
        speakerForgetCancel.hidden = false;
        speakerStatus('This deletes the voice profile. It cannot be undone.');
        return;
    }
    disarmSpeakerForget();
    speakerBusy = true;
    syncSpeakerButtons();
    try {
        const state = await speakerRequest('DELETE');
        renderSpeakerProfile(state);
        speakerVerdict.hidden = true;
        speakerStatus('Forgotten.');
    } catch (err) {
        speakerStatus(err.message, true);
    } finally {
        speakerBusy = false;
        renderSpeakerProfile(speakerLastProfile);
    }
}

if (speakerPanel) {
    speakerEnrollBtn.addEventListener('click', speakerEnroll);
    speakerVerifyBtn.addEventListener('click', speakerVerify);
    speakerForgetBtn.addEventListener('click', speakerForget);
    speakerForgetCancel.addEventListener('click', () => {
        disarmSpeakerForget();
        speakerStatus('');
    });
}
