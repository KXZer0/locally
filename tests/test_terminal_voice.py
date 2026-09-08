import io
import json
import unittest
import wave
from unittest import mock

from rich.console import Console

from core import terminal_voice as voice
from core.terminal import NO_THINK, Chat, read_stream
from tests.test_terminal import stream_bytes


class StubStream:
    def __init__(self, **kwargs):
        self.samplerate = kwargs.get("samplerate")
        self.written = bytearray()
        self.closed = False

    def start(self):
        pass

    def write(self, data):
        self.written += data

    def stop(self):
        pass

    def close(self):
        self.closed = True


class StubAudio:
    """Just enough sounddevice to run the queue without a sound card."""

    def __init__(self):
        self.streams = []

    def RawOutputStream(self, **kwargs):
        stream = StubStream(**kwargs)
        self.streams.append(stream)
        return stream

    def stop(self):
        pass


class SplitSpeakableTests(unittest.TestCase):
    def test_the_first_clip_may_be_tiny_and_later_ones_may_not(self):
        # Every character before the first clip is dead air, so "Sure." is
        # worth speaking on its own; mid-answer that would sound chopped.
        clip, rest = voice.split_speakable("Sure. And then we ", first=True)
        self.assertEqual(clip, "Sure.")
        self.assertEqual(rest, "And then we ")
        self.assertEqual(voice.split_speakable("Sure. And then we ", first=False),
                         (None, "Sure. And then we "))

    def test_a_decimal_is_not_a_sentence_end(self):
        text = "Pi is about 3.14159 and it never repeats, which matters here. Next"
        clip, rest = voice.split_speakable(text, first=False)
        self.assertTrue(clip.endswith("matters here."))
        self.assertNotIn("3.14159", rest)

    def test_an_endless_clause_is_flushed_so_audio_keeps_up(self):
        text = "and then, " * 40 + "we keep going"  # no sentence end at all
        clip, rest = voice.split_speakable(text, first=False)
        self.assertIsNotNone(clip)
        # The comma ends the clip rather than opening the next one, which out
        # loud is the difference between a pause and a fragment.
        self.assertTrue(clip.endswith(","))
        self.assertEqual(rest, "we keep going")

    def test_nothing_to_say_yet_returns_the_buffer_untouched(self):
        self.assertEqual(voice.split_speakable("half a sen", first=True),
                         (None, "half a sen"))


class WavTests(unittest.TestCase):
    def test_wav_bytes_round_trips_through_the_wave_module(self):
        pcm = b"\x01\x02" * 800
        data = voice.wav_bytes(pcm, 16000)
        with wave.open(io.BytesIO(data), "rb") as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getsampwidth(), 2)
            self.assertEqual(handle.getframerate(), 16000)
            self.assertEqual(handle.readframes(handle.getnframes()), pcm)


class SpeechQueueTests(unittest.TestCase):
    def test_clips_are_spoken_in_order_and_announced_on_playback(self):
        audio = StubAudio()
        client = mock.Mock()
        client.speech.side_effect = lambda text, **kw: (b"\x00\x00" * 240, 24000)
        spoken = []
        queue = voice.SpeechQueue(audio, client, spoken.append)
        for line in ("One.", "Two.", "Three."):
            queue.say(line)
        queue.finish(timeout=20)
        self.assertEqual(spoken, ["One.", "Two.", "Three."])
        self.assertEqual(len(audio.streams), 1)      # one stream, reused
        self.assertEqual(len(audio.streams[0].written), 3 * 480)

    def test_a_synthesis_failure_surfaces_instead_of_hanging(self):
        audio = StubAudio()
        client = mock.Mock()
        client.speech.side_effect = RuntimeError("no TTS model")
        queue = voice.SpeechQueue(audio, client, lambda line: None)
        queue.say("One.")
        with self.assertRaises(RuntimeError):
            queue.finish(timeout=20)

    def test_cancel_stops_the_queue(self):
        audio = StubAudio()
        client = mock.Mock()
        client.speech.side_effect = lambda text, **kw: (b"\x00\x00" * 240, 24000)
        spoken = []
        queue = voice.SpeechQueue(audio, client, spoken.append)
        queue.cancel()
        queue.say("One.")
        queue.finish(timeout=20)
        self.assertEqual(spoken, [])


class SpokenTurnTests(unittest.TestCase):
    def setUp(self):
        self.api = mock.Mock()
        self.api.speech.side_effect = lambda text, **kw: (b"\x00\x00" * 240, 24000)
        self.console = Console(file=io.StringIO(), width=80, color_system=None)
        self.chat = Chat(self.api, self.console)

    def test_a_spoken_turn_is_shaped_for_speech_and_shares_the_conversation(self):
        self.api.stream.side_effect = lambda *a: read_stream(
            stream_bytes("An NPU runs neural networks. It is low power.\n"))
        self.chat.speak_turn("what is an NPU", StubAudio(), voice)
        sent, model, max_tokens = self.api.stream.call_args[0]
        # The voice directive goes last so it beats the general prompt's
        # formatting advice, and the budget is small on purpose.
        self.assertEqual(sent[-1], {"role": "system", "content": voice.VOICE_PROMPT})
        self.assertEqual(max_tokens, voice.VOICE_MAX_TOKENS)
        # Reasoning must never be spoken, so no-think is the voice default.
        self.assertTrue(sent[-2]["content"].endswith(NO_THINK))
        # ... and neither the token nor the directive is kept: the same
        # conversation continues in text afterwards.
        self.assertEqual(self.chat.messages[-2]["content"], "what is an NPU")
        self.assertEqual(self.chat.messages[-1]["role"], "assistant")
        self.assertNotIn(voice.VOICE_PROMPT,
                         json.dumps(self.chat.messages))

    def test_reasoning_is_never_sent_to_the_speaker(self):
        self.api.stream.side_effect = lambda *a: read_stream(
            stream_bytes("<think>weigh it up</think>It is a chip for neural nets.\n"))
        self.chat.speak_turn("what is an NPU", StubAudio(), voice)
        self.assertNotIn("weigh it up", " ".join(
            call.args[0] for call in self.api.speech.call_args_list))
        self.assertNotIn("weigh it up", self.chat.last_answer)


if __name__ == "__main__":
    unittest.main()
