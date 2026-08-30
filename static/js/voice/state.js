// The voice tab's state machine, its settings and its session.
//
// In here: IDLE -> LISTENING -> TRANSCRIBING -> THINKING -> SPEAKING -> IDLE,
// the per-turn timings line, the tuning sliders, voice selection, and the
// tab's controls. TTS cannot stream, so honest state reporting matters more
// here than anywhere else in the app.
// Not in here: the microphone (voice/capture.js), endpointing
// (voice/vad-socket.js), a turn (voice/turn.js) and playback
// (voice/speech-queue.js). This module says WHERE the tab is, not what it does.

import { gateValue, orb, patienceValue, voiceCaption, voiceGate, voicePatience, voicePtt, voiceSelect, voiceSelectLabel, voiceStateLabel, voiceStopBtn, voiceThinkCheckbox, voiceTimings, voiceToggle, voiceVadCheckbox } from '../core/dom.js';
import { asrReady, pttActive, setPttActive, ttsReady, vadReady } from '../core/state.js';
import { beginCapture, closeMic, endCapture, mediaStream, openMic } from './capture.js';
import { interruptAudioOnly, interruptSpeech } from './speech-queue.js';
import { sendVadConfig, sendVadState } from './vad-socket.js';

// --- Voice mode ---
//
//   IDLE ──mic──▶ LISTENING ──silence──▶ TRANSCRIBING ──▶ THINKING ──▶ SPEAKING ──▶ IDLE
//                     ▲                                                     │
//                     └──────────────── barge-in / interrupt ───────────────┘
//
// TTS does not stream, so honest state reporting matters more than usual: the
// user must never be left wondering whether it heard them. The transcript is
// shown the moment it lands, and the answer streams in while audio is still
// being synthesized.

export let voiceSession = false;     // is the mic open in voice mode
export let voiceState = 'idle';
export const timings = {};

// Turn-taking used to be a loudness gate: RMS against
// max(absolute floor, noise floor x multiple). It cannot work, and the reason
// is not tuning. Loudness does not distinguish speech from a fan, a road, or
// the next table — so in any room with steady background noise the level sits
// above the gate permanently: turns open by themselves and never close,
// because the silence counter never gets to run. See TODONT.md.
//
// Endpointing now belongs to the server, which has a model that can tell
// speech from noise. What is left here is a level for the orb and the meter,
// so the picture still reacts to the microphone with no network in the path.
export const VAD = {
    maxUtteranceMs: 60000,
    // A natural pause mid-sentence is ~0.5-1s. Silero is confident enough to
    // wait less than the old gate had to.
    endMsDefault: 900,
    meterFullScale: 0.25,      // RMS that fills the meter
};

export const VAD_SENSITIVITY = {      // slider 1..5 -> Silero speech probability
    threshold: [0.25, 0.35, 0.50, 0.65, 0.80],
    words: ['very open', 'open', 'medium', 'strict', 'very strict'],
};

export function vadEndMs() {
    return Number(localStorage.getItem('locally-vad-patience') || VAD.endMsDefault);
}

export function vadSensitivity() {
    return Math.min(5, Math.max(1, Number(localStorage.getItem('locally-vad-gate') || 3)));
}

export function vadThreshold() {
    return VAD_SENSITIVITY.threshold[vadSensitivity() - 1];
}

export function setVoiceState(state, caption) {
    const changed = voiceState !== state;
    voiceState = state;
    if (changed) sendVadState();
    orb.dataset.state = state;
    voiceStateLabel.textContent = state;
    if (caption !== undefined) voiceCaption.textContent = caption;
    voiceStopBtn.hidden = state !== 'speaking';
}

export function showTimings() {
    const parts = [];
    for (const [k, v] of Object.entries(timings)) {
        if (v != null) parts.push(`<span class="stage"><b>${k}</b> ${(v / 1000).toFixed(1)}s</span>`);
    }
    voiceTimings.innerHTML = parts.join('');
}

// Timings are per-turn. Left to accumulate, last turn's tts number sits on
// screen through the whole of the next turn and reads as this turn's.
export function resetTimings() {
    for (const k of Object.keys(timings)) delete timings[k];
    showTimings();
}
// Opening the Voice tab is the earliest honest signal that a spoken turn is
// coming. If idle-unload has evicted Whisper and the TTS model, reloading them
// costs seconds — and without this those seconds are spent inside the user's
// first turn, where they are indistinguishable from a slow model.
export let warmedAt = 0;
export function warmAudio() {
    if (!asrReady && !ttsReady) return;
    const now = Date.now();
    if (now - warmedAt < 30000) return;      // tab-flipping shouldn't spam it
    warmedAt = now;
    fetch('/v1/audio/warm', { method: 'POST' }).catch(() => {});
}

// Off by default: reasoning is dead air on a voice turn — you sit listening to
// silence while the model argues with itself, and VOICE_MAX_TOKENS can be gone
// before it says anything. On, it reasons silently and speaks only the answer.
export function voiceThinkAllowed() { return localStorage.getItem('locally-voice-think') === 'on'; }
export function voiceName() {
    return localStorage.getItem('locally-voice') || '';
}

// Only shown when the model actually ships voices (Kokoro does, SpeechT5
// doesn't) — an empty dropdown would just be a dead control.
export function populateVoices(tts) {
    const names = tts.voices || [];
    const show = names.length > 1;
    voiceSelect.hidden = !show;
    voiceSelectLabel.hidden = !show;
    if (!show || voiceSelect.options.length === names.length) return;

    const saved = voiceName() || tts.voice || names[0];
    voiceSelect.innerHTML = '';
    for (const n of names) {
        const opt = document.createElement('option');
        opt.value = n;
        opt.textContent = n;
        if (n === saved) opt.selected = true;
        voiceSelect.appendChild(opt);
    }
}
export async function startVoiceSession() {
    try {
        await openMic();
        voiceSession = true;
        voiceToggle.textContent = 'Stop listening';
        setVoiceState('idle', voiceReadyCaption());
    } catch (err) {
        setVoiceState('idle', err.message);
    }
}

export function stopVoiceSession() {
    voiceSession = false;
    closeMic();
    interruptAudioOnly();
    voiceToggle.textContent = 'Start listening';
    setVoiceState('idle', 'Microphone off.');
}
// Voice tuning — both persist, and both are pushed to the server on change so
// a mid-session adjustment takes effect on the next frame.
export function syncVoiceTuning() {
    patienceValue.textContent = (voicePatience.value / 1000).toFixed(1) + 's';
    gateValue.textContent = VAD_SENSITIVITY.words[voiceGate.value - 1];
}
voicePatience.value = vadEndMs();
voiceGate.value = vadSensitivity();
syncVoiceTuning();
voicePatience.addEventListener('input', () => {
    localStorage.setItem('locally-vad-patience', voicePatience.value);
    syncVoiceTuning();
    sendVadConfig();
});
voiceGate.addEventListener('input', () => {
    localStorage.setItem('locally-vad-gate', voiceGate.value);
    syncVoiceTuning();
    sendVadConfig();
});
// Voice controls
voiceToggle.addEventListener('click', () => {
    if (voiceSession) stopVoiceSession(); else startVoiceSession();
});
voiceStopBtn.addEventListener('click', interruptSpeech);
voiceSelect.addEventListener('change', () => {
    localStorage.setItem('locally-voice', voiceSelect.value);
});
voiceVadCheckbox.addEventListener('change', () => {
    localStorage.setItem('locally-vad', voiceVadCheckbox.checked ? 'on' : 'off');
    sendVadConfig();
    if (voiceSession) setVoiceState('idle', voiceReadyCaption());
});
voiceVadCheckbox.checked = localStorage.getItem('locally-vad') !== 'off';
// One place decides what the voice tab says it is waiting for, because the
// answer depends on three things (session, model, user setting) and having it
// spelled out at each call site is how they drifted.
export function voiceReadyCaption() {
    if (!vadReady) return 'Hold the mic button to talk (auto turn-taking needs --vad-dir).';
    if (!voiceVadCheckbox.checked) return 'Hold the mic button to talk.';
    return 'Just talk — it takes its turn when you pause.';
}

voiceThinkCheckbox.addEventListener('change', () => {
    localStorage.setItem('locally-voice-think',
                         voiceThinkCheckbox.checked ? 'on' : 'off');
});
voiceThinkCheckbox.checked = voiceThinkAllowed();
// Push-to-talk inside voice mode: same capture path, but the turn ends on
// release instead of on silence.
export async function voicePttStart(e) {
    if (e) e.preventDefault();
    if (pttActive || voiceState === 'transcribing' || voiceState === 'thinking') return;
    interruptAudioOnly();
    if (!mediaStream) {
        try { await openMic(); voiceSession = true; voiceToggle.textContent = 'Stop listening'; }
        catch (err) { setVoiceState('idle', err.message); return; }
    }
    setPttActive(true);
    voicePtt.classList.add('recording');
    sendVadConfig();          // stand the server's endpointing down for now
    beginCapture();
}

export async function voicePttEnd() {
    if (!pttActive) return;
    setPttActive(false);
    voicePtt.classList.remove('recording');
    sendVadConfig();
    await endCapture();
}

voicePtt.addEventListener('pointerdown', voicePttStart);
voicePtt.addEventListener('pointerup', voicePttEnd);
voicePtt.addEventListener('pointercancel', voicePttEnd);
voicePtt.addEventListener('pointerleave', () => { if (pttActive) voicePttEnd(); });
