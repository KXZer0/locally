// Endpointing, over a socket to the Silero VAD slot.
//
// In here: the WebSocket, the frames pushed down it, and the events that come
// back -- speech_start, speech_end, transcript. The model is server-side on
// purpose: the turn's audio is already where Whisper needs it when the turn
// ends, which is most of why transcription overlaps the pause.
// Not in here: any fallback loudness gate. Without --vad-dir the tab disables
// auto turn-taking and says so; it does not fall back to what was broken.

import { pttActive, vadReady } from '../core/state.js';
import { beginCapture, clearUtterance, discardCapture, vadEnabled } from './capture.js';
import { interruptSpeech } from './speech-queue.js';
import { resetTimings, setVoiceState, vadEndMs, vadThreshold, voiceSession, voiceState } from './state.js';
import { voiceTurn } from './turn.js';

// --- Endpointing, over a socket to the VAD model ---
//
// Everything is on localhost, so the round trip costs about as much as a
// function call. Putting the model server-side buys three things a browser
// build could not: the audio for the turn is already where Whisper needs it
// when the turn ends (which is most of why transcription now overlaps the
// pause), the VAD is an OpenVINO slot like every other model here, and no
// megabytes of inference runtime get vendored into static/.

export let vadSock = null;
export let vadSockReady = false;

export function openVadSocket() {
    if (!vadReady || vadSock) return;
    try {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        vadSock = new WebSocket(`${proto}//${location.host}/v1/audio/stream`);
        vadSock.binaryType = 'arraybuffer';
        vadSock.onopen = () => {
            vadSockReady = true;
            sendVadConfig();
        };
        vadSock.onmessage = (e) => {
            let msg;
            try { msg = JSON.parse(e.data); } catch { return; }
            handleVadEvent(msg);
        };
        vadSock.onclose = () => { vadSockReady = false; vadSock = null; };
        // A dead socket must not look like a dead microphone: fall back to
        // push-to-talk and say so, rather than silently never taking a turn.
        vadSock.onerror = () => {
            vadSockReady = false;
            if (voiceSession && voiceState === 'idle') {
                setVoiceState('idle', 'Auto turn-taking lost its connection — hold the mic to talk.');
            }
        };
    } catch { vadSock = null; }
}

export function closeVadSocket() {
    vadSockReady = false;
    if (vadSock) { try { vadSock.close(); } catch {} }
    vadSock = null;
}

export function sendVadConfig() {
    if (!vadSockReady) return;
    vadSock.send(JSON.stringify({
        type: 'config',
        threshold: vadThreshold(),
        patienceMs: vadEndMs(),
        language: localStorage.getItem('locally-asr-lang') || '',
        enabled: vadEnabled() && !pttActive,
    }));
}

// The server needs to know we are playing audio: the microphone hears the
// speakers even through echo cancellation, so barge-in has to be harder to
// trigger than a normal turn start.
export function sendVadState() {
    if (!vadSockReady) return;
    vadSock.send(JSON.stringify({ type: 'state', voice: voiceState }));
}

export function sendVadFrame(frame) {
    if (!vadSockReady || vadSock.readyState !== WebSocket.OPEN) return;
    // Int16 on the wire: half the bytes of float32 for identical quality at
    // this bit depth, and exactly what the server's reader wants.
    const pcm = new Int16Array(frame.length);
    for (let i = 0; i < frame.length; i++) {
        const s = Math.max(-1, Math.min(1, frame[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    try { vadSock.send(pcm.buffer); } catch {}
}

export function handleVadEvent(msg) {
    if (msg.type === 'speech_start') {
        if (voiceState === 'speaking') { interruptSpeech(); return; }
        // Barge-in during transcription or generation: the old build had no
        // way out of those states at all, so a wrong turn had to be waited out.
        if (voiceState === 'transcribing' || voiceState === 'thinking') {
            interruptSpeech();
            return;
        }
        if (voiceState === 'idle' && vadEnabled() && !pttActive) beginCapture();
    } else if (msg.type === 'speech_end') {
        if (voiceState !== 'listening' || pttActive) return;
        if (msg.tooShort) { discardCapture(); return; }
        // Somebody spoke, but not the enrolled voice. Say so briefly rather
        // than silently: an assistant that ignores speech with no explanation
        // is indistinguishable from one that has crashed. The score is shown
        // because the threshold is the user's to tune.
        if (msg.notEnrolled) {
            discardCapture();
            setVoiceState('idle', msg.score != null
                ? `Not your voice (${msg.score.toFixed(2)}) — ignored.`
                : 'Not your voice — ignored.');
            return;
        }
        // The server has the audio and is already transcribing it — started
        // during the pause, not after it. Stop capturing and wait; uploading a
        // second copy of the same audio would only make it slower.
        clearUtterance();
        resetTimings();
        setVoiceState('transcribing', '');
        awaitingTranscript = msg.utteranceId;
    } else if (msg.type === 'transcript') {
        if (msg.utteranceId !== awaitingTranscript) return;
        awaitingTranscript = null;
        if (voiceState !== 'transcribing') return;      // interrupted meanwhile
        if (!msg.text) { setVoiceState('idle', "Didn't catch that — try again."); return; }
        voiceTurn(null, { text: msg.text, ms: msg.ms });
    } else if (msg.type === 'error') {
        setVoiceState('idle', msg.message || 'Voice detection failed.');
    }
}
export let awaitingTranscript = null;
