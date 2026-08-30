// The microphone graph behind the voice tab.
//
// In here: one AudioWorklet at 16 kHz emitting 512-sample frames, the 500 ms
// pre-roll ring so a turn begins with the audio that opened it, the orb's
// level, and where a turn starts and ends.
// Not in here: the decision that a turn has ended. That is a model, and it
// runs server-side (voice/vad-socket.js) -- a loudness gate cannot tell speech
// from a noisy room and never could.

import { encodeWav } from '../audio/wav.js';
import { micMeterFill, orbCore, voiceVadCheckbox } from '../core/dom.js';
import { vadReady } from '../core/state.js';
import { VAD, setVoiceState, vadEndMs, voiceState } from './state.js';
import { voiceTurn } from './turn.js';
import { closeVadSocket, openVadSocket, sendVadFrame } from './vad-socket.js';

export let mediaStream = null;
export let audioCtx = null;
export let captureNode = null;
export let vadRaf = null;
export let micRms = 0;               // last frame's level, for the orb and the meter
// --- Capture ---
//
// One capture path serves both turn-taking modes. The worklet delivers 512
// sample frames of 16 kHz mono float; each frame goes three places: a level for
// the picture, the socket (when the server is doing the endpointing), and a
// local ring so push-to-talk still works with no socket at all.

export const CAPTURE_SR = 16000;
export const PREROLL_FRAMES = 16;         // ~500 ms of lead-in

export let preroll = [];                  // rolling frames captured before a turn opens
export let utterance = null;              // frames of the turn in progress, or null

export async function openMic() {
    if (mediaStream) return;
    mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    // Ask the graph for 16 kHz directly: it resamples from whatever the device
    // runs at, which is both what Whisper wants and what Silero requires, and
    // it retires the decode-and-resample pass the old blob path needed.
    const Ctx = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctx({ sampleRate: CAPTURE_SR });
    await audioCtx.audioWorklet.addModule('/static/js/vad-worklet.js');
    const src = audioCtx.createMediaStreamSource(mediaStream);
    captureNode = new AudioWorkletNode(audioCtx, 'locally-capture');
    captureNode.port.onmessage = (e) => onFrame(e.data);
    src.connect(captureNode);
    // Chrome will not pull from a worklet that reaches no destination. A zero
    // gain node keeps the graph running without playing the mic back.
    const mute = audioCtx.createGain();
    mute.gain.value = 0;
    captureNode.connect(mute).connect(audioCtx.destination);
    preroll = [];
    utterance = null;
    openVadSocket();
    tickOrb();
}

export function closeMic() {
    if (vadRaf) cancelAnimationFrame(vadRaf);
    vadRaf = null;
    closeVadSocket();
    if (captureNode) { try { captureNode.port.onmessage = null; captureNode.disconnect(); } catch {} }
    captureNode = null;
    if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
    if (audioCtx) audioCtx.close().catch(() => {});
    audioCtx = null;
    preroll = [];
    utterance = null;
    micRms = 0;
    orbCore.style.transform = '';
}

export function onFrame(frame) {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    micRms = Math.sqrt(sum / frame.length);

    if (utterance) {
        utterance.push(frame);
        // Safety net: a stuck turn must not grow without bound.
        if (utterance.length * frame.length / CAPTURE_SR * 1000 > VAD.maxUtteranceMs) {
            if (voiceState === 'listening') endCapture();
        }
    } else {
        preroll.push(frame);
        if (preroll.length > PREROLL_FRAMES) preroll.shift();
    }
    sendVadFrame(frame);
}

// Everything a turn recorded, as one 16 kHz mono buffer.
export function utteranceSamples() {
    const frames = utterance || [];
    const total = frames.reduce((n, f) => n + f.length, 0);
    const out = new Float32Array(total);
    let at = 0;
    for (const f of frames) { out.set(f, at); at += f.length; }
    return out;
}

// The picture updates from the local level only, so the orb never stutters
// with the network even though the turn decisions come from the server.
export function tickOrb() {
    vadRaf = requestAnimationFrame(tickOrb);
    const level = micRms;
    if (micMeterFill) {
        const meterScale = Math.min(1, level / VAD.meterFullScale);
        micMeterFill.style.transform = `scaleX(${meterScale.toFixed(3)})`;
        micMeterFill.classList.toggle('over', voiceState === 'listening');
    }
    if (voiceState === 'listening' || voiceState === 'idle') {
        const amp = Math.min(1, level / VAD.meterFullScale);
        orbCore.style.transform = `scale(${(1 + amp * 0.28).toFixed(3)})`;
    }
}

// Auto turn-taking needs a model that can tell speech from noise. Without one
// the honest answer is push-to-talk, not a loudness gate that opens on traffic.
export function vadEnabled() { return vadReady && voiceVadCheckbox.checked; }
export function beginCapture() {
    if (!captureNode || utterance) return;
    // Start from the pre-roll, not from now. The old path created the recorder
    // only after speech had already been detected, so the attack of the first
    // word was cut off before Whisper ever saw it.
    utterance = preroll.slice();
    preroll = [];
    setVoiceState('listening', vadEnabled()
        ? `Listening — pause ${(vadEndMs() / 1000).toFixed(1)}s to send.`
        : 'Listening — release to send.');
}

// Throw away a capture that turned out to be noise, without running a turn.
export function discardCapture() {
    utterance = null;
    setVoiceState('idle', 'That was too short — still listening.');
}

// Push-to-talk only. Under auto turn-taking the server already holds the audio
// and answers with a transcript, so nothing is uploaded on that path at all.
export async function endCapture() {
    if (!utterance) return;
    const samples = utteranceSamples();
    utterance = null;
    // Under half a second of audio is a cough or a bumped desk, not a turn.
    if (samples.length < CAPTURE_SR * 0.5) {
        setVoiceState('idle', 'Say something.');
        return;
    }
    await voiceTurn(encodeWav(samples, CAPTURE_SR));
}

// The server holds the turn's audio once it has endpointed it, so the socket
// stands the local capture down rather than uploading a second copy.
export function clearUtterance() { utterance = null; }
