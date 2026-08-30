// Microphone capture for voice mode.
//
// Replaces an AnalyserNode + MediaRecorder pair. Two reasons it has to be a
// worklet rather than a script processor or a recorder:
//
//   1. Silero wants exactly 512 samples at 16 kHz per inference, and the audio
//      graph hands out 128-sample render quanta. Something has to accumulate
//      them, and doing it here means neither the VAD nor the socket ever sees
//      a partial frame.
//   2. MediaRecorder gives WebM/Opus, which the server's reader cannot open —
//      so the old path decoded and resampled the blob after the fact. Raw
//      float from the graph is already what Whisper wants, and it is available
//      DURING the utterance rather than only at the end of it.
//
// The AudioContext is created at 16 kHz, so the graph resamples the microphone
// for us and this processor never has to know the device's native rate.

const FRAME = 512;

class locallyCapture extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buf = new Float32Array(FRAME);
        this.n = 0;
    }

    process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        // No input yet (or the track ended): stay alive, the node is reused
        // across the whole session.
        if (!ch) return true;

        for (let i = 0; i < ch.length; i++) {
            this.buf[this.n++] = ch[i];
            if (this.n === FRAME) {
                // A copy, not the buffer itself — this one is refilled
                // immediately and the receiver may still be holding it.
                this.port.postMessage(this.buf.slice());
                this.n = 0;
            }
        }
        return true;
    }
}

registerProcessor('locally-capture', locallyCapture);
