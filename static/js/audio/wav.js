// Audio format plumbing.
//
// In here: PCM -> WAV, and decode-and-resample to the 16 kHz mono the server's
// reader wants. Kept apart from the two features that need it so neither the
// chat mic nor speaker enrolment grows its own copy.
// Not in here: capture. The voice tab's worklet already delivers 16 kHz mono
// and needs none of this.

// --- Speech to text ---
//
// The browser records WebM/Opus, which the server's soundfile reader cannot
// open. Rather than teach the server another codec, decode and resample here:
// the wire format becomes exactly what WhisperPipeline wants — 16 kHz mono
// PCM — and the server stays a single `sf.read()`.

export function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeStr = (off, s) => {
        for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);                 // fmt chunk size
    view.setUint16(20, 1, true);                  // PCM
    view.setUint16(22, 1, true);                  // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);     // byte rate
    view.setUint16(32, 2, true);                  // block align
    view.setUint16(34, 16, true);                 // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++, off += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([view], { type: 'audio/wav' });
}

export async function toWav16kMono(blob) {
    const buf = await blob.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    let decoded;
    try {
        decoded = await ctx.decodeAudioData(buf);
    } finally {
        ctx.close();
    }
    // A 1-channel OfflineAudioContext at 16 kHz does the downmix and the
    // resample in one pass, whatever the mic's native rate was.
    const frames = Math.max(1, Math.ceil(decoded.duration * 16000));
    const off = new OfflineAudioContext(1, frames, 16000);
    const src = off.createBufferSource();
    src.buffer = decoded;
    src.connect(off.destination);
    src.start();
    const rendered = await off.startRendering();
    return encodeWav(rendered.getChannelData(0), 16000);
}
