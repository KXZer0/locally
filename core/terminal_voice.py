"""Spoken turns for the terminal client.

The server already owns the hard parts -- Whisper, Kokoro and Silero -- so the
only thing missing from a terminal was the microphone and the speaker. That is
`sounddevice`, a ~2 MB PortAudio wheel, imported HERE and only when /voice is
used: attaching a terminal to a running server must not require an audio stack.

Two decisions carried over from the web UI, both measured there:

* **Speech starts before the answer finishes.** TTS cannot stream -- genai
  returns the whole utterance -- but it does not need the whole message. Clips
  are peeled off the token stream a sentence at a time and clip N+1 is
  synthesized while clip N plays. The first clip is allowed to be much shorter
  than the rest, because every character before it is dead air.
* **The text is revealed by playback, not by the token stream.** A line is
  printed when its audio starts, so what you read is what you are hearing.
  Painting from the deltas instead put the text seconds ahead of the voice.

Push-to-talk only. The server's Silero VAD can take turns on its own over
/v1/audio/stream, but that wants a live socket and a pre-roll ring; this is the
version that works in any terminal, and the socket can come later.
"""

import io
import queue
import re
import threading
import wave

SAMPLE_RATE = 16000

# A spoken reply is not a written one: a table read aloud is gibberish and a
# list becomes a stream of "dash". Appended last so it wins over the general
# prompt's formatting advice.
VOICE_PROMPT = (
    "Answer out loud in 2-3 plain sentences. No Markdown, no lists, no tables, "
    "no code. If the real answer needs any of those, say so briefly and say it "
    "is in the chat instead."
)
# Enough for 2-3 sentences and little enough that a model which starts
# reasoning runs out before it can read an essay at you.
VOICE_MAX_TOKENS = 220

# Every character before the first clip is silence the user is sitting through,
# so "Sure." is worth speaking on its own; later clips are long enough to sound
# like sentences rather than fragments.
FIRST_MIN_CHARS = 4
MIN_CHARS = 40
# A clause this long with no sentence end is someone dictating without pausing.
FLUSH_CHARS = 300

_SENTENCE_END = re.compile(r"[.!?…](?=\s)")


def split_speakable(buffer, first):
    """(clip, rest): the longest complete sentence run worth speaking yet.

    Boundaries need whitespace AFTER the punctuation, which is what keeps
    "3.14159" and "e.g." in one piece without a list of exceptions.
    """
    minimum = FIRST_MIN_CHARS if first else MIN_CHARS
    end = None
    for match in _SENTENCE_END.finditer(buffer):
        if match.end() >= minimum:
            end = match.end()
    if end is None:
        if len(buffer) < FLUSH_CHARS:
            return None, buffer
        # No sentence in sight -- break at the last comma so the audio keeps
        # up instead of waiting for a full stop that never comes. The comma
        # stays with the clip it ends: a clip that OPENS with one reads as a
        # fragment out loud. A space is the fallback when there is no comma.
        comma = buffer.rfind(", ")
        cut = comma + 1 if comma >= minimum else buffer.rfind(" ")
        if cut < minimum:
            return None, buffer
        end = cut
    return buffer[:end].strip(), buffer[end:].lstrip()


def import_audio():
    """sounddevice, or a sentence saying how to get it."""
    try:
        import sounddevice
    except Exception as exc:                       # OSError when PortAudio is missing
        raise RuntimeError(
            "Voice needs the sounddevice package (PortAudio): "
            "python -m pip install sounddevice. " + str(exc)) from exc
    return sounddevice


def wav_bytes(pcm, rate=SAMPLE_RATE):
    """Wrap raw int16 mono PCM as a WAV file, in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return buf.getvalue()


class Recorder:
    """One push-to-talk take. Stop is whatever the caller decides."""

    def __init__(self, audio):
        self.audio = audio
        self.frames = []

    def __enter__(self):
        def callback(indata, frames, time_info, status):
            self.frames.append(bytes(indata))
        self.stream = self.audio.RawInputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback)
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()

    @property
    def pcm(self):
        return b"".join(self.frames)


class SpeechQueue:
    """Synthesis and playback on separate threads, so they overlap.

    One thread would make clip N+1's synthesis wait for clip N's playback,
    which is the whole latency the pipelining exists to remove.
    """

    def __init__(self, audio, client, on_speak):
        self.audio, self.client, self.on_speak = audio, client, on_speak
        self.pending = queue.Queue()
        self.ready = queue.Queue(maxsize=2)
        self.stopped = threading.Event()
        self.error = None
        self.synth = threading.Thread(target=self._synthesize, daemon=True)
        self.player = threading.Thread(target=self._play, daemon=True)
        self.synth.start()
        self.player.start()

    def say(self, text):
        if text.strip():
            self.pending.put(text.strip())

    def _synthesize(self):
        while True:
            text = self.pending.get()
            if text is None or self.stopped.is_set():
                self.ready.put(None)
                return
            try:
                pcm, rate = self.client.speech(text)
            except Exception as exc:                     # surfaced by finish()
                self.error = self.error or exc
                self.ready.put(None)
                return
            self.ready.put((text, pcm, rate))

    def _play(self):
        stream = None
        while True:
            item = self.ready.get()
            if item is None or self.stopped.is_set():
                break
            text, pcm, rate = item
            self.on_speak(text)
            try:
                if stream is None or stream.samplerate != rate:
                    if stream is not None:
                        stream.stop(); stream.close()
                    stream = self.audio.RawOutputStream(
                        samplerate=rate, channels=1, dtype="int16")
                    stream.start()
                # In blocks, so a barge-in takes effect inside a long clip
                # rather than after it.
                block = rate  # one second
                for offset in range(0, len(pcm), block * 2):
                    if self.stopped.is_set():
                        break
                    stream.write(pcm[offset:offset + block * 2])
            except Exception as exc:
                self.error = self.error or exc
                break
        if stream is not None:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass

    def finish(self, timeout=120):
        """Wait for everything queued to be spoken."""
        self.pending.put(None)
        self.synth.join(timeout=timeout)
        self.player.join(timeout=timeout)
        if self.error:
            raise self.error

    def cancel(self):
        """Barge-in: silence first, tidy up after."""
        self.stopped.set()
        try:
            self.audio.stop()
        except Exception:
            pass
