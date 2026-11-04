"""The voice session's transcription runs, batch and streaming.

Two ways to get a turn's transcript, and the session picks per turn:

  * Batch (Whisper, Parakeet TDT): the whole utterance goes to the ASR slot
    after the pause, with one speculative run allowed to start early.
  * Streaming (parakeet-unified, Nemotron): a StreamingTranscriber is opened
    when the turn opens and fed every frame as it arrives, so its worker has
    decoded all but the last unit by the time the pause is over. speech_end
    then only waits for that tail.

Both are sequence-tagged the same way: every run takes a number, and only
the run the turn is waiting for (`asr_want`) may become `final`. The number
is recorded BEFORE the worker starts, because a streaming tail can finish in
a few milliseconds -- quicker than the caller could store the return value
-- and a result that lands before anyone is waiting for it would be dropped.
"""

from datetime import datetime
import threading
import time

from core import runtime
from core.slots.select import _slot_serviceable


def _log(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


class TranscriptionRuns:
    """Mixed into VoiceStreamSession; uses its utt_id / awaiting / asr_* state."""

    def _open_stream(self):
        """A live transcriber for the turn that just opened, or None.

        Only for a slot that can stream AND is resident: opening one on an
        idle-unloaded slot would reload the model on the socket's own thread,
        stalling every frame behind it. Such a turn takes the batch path,
        which loads on its worker."""
        slot = runtime.whisper_slot
        if (not slot or not hasattr(slot, "open_stream")
                or not _slot_serviceable(slot) or slot.status != "ready"):
            return None
        try:
            return slot.open_stream(language=self.language or None)
        except Exception as e:
            _log(f"!! [voice] could not open a streaming transcription ({e}); "
                 "this turn is transcribed after the pause")
            return None

    def _drop_stream(self):
        if self.stream is not None:
            self.stream.cancel()
            self.stream = None

    def close(self):
        """The socket went away: stop any half-fed turn's worker."""
        self._drop_stream()

    def _finish_stream(self, stream, samples, utt_id):
        """Decode the streamed turn's tail on a worker; returns its sequence id."""
        self.asr_seq += 1
        seq = self.asr_seq
        self.asr_want = seq

        def run():
            t0 = time.perf_counter()
            text = None
            try:
                text = stream.finalize()
                _log(f"-- [voice] streamed {stream.units} unit(s) during the turn, "
                     f"transcript {(time.perf_counter() - t0) * 1000:.0f} ms "
                     "after the pause")
            except Exception as e:
                _log(f"!! [voice] streaming transcription failed ({e}); "
                     "transcribing the whole turn instead")
                try:
                    text = runtime.whisper_slot.transcribe(
                        samples, language=self.language or None)
                except Exception as e2:
                    _log(f"!! [voice] transcription failed: {e2}")
            result = {"id": utt_id, "text": (text or "").strip(),
                      "ms": (time.perf_counter() - t0) * 1000}
            if seq == self.asr_want and utt_id == self.awaiting:
                self.final = result

        threading.Thread(target=run, daemon=True).start()
        return seq

    def _transcribe(self, samples, utt_id, speculative):
        """Run the ASR slot on a background thread; returns the run's sequence id.

        Every run is tagged, and only the run the turn is waiting for can
        become the final answer -- otherwise a speculative run started before
        the user carried on could overwrite the real one. The slot takes the
        device lock it shares with chat, so a wasted run is not free for
        anything else on that device; hence one speculation at a time.
        """
        self.asr_seq += 1
        seq = self.asr_seq
        if not speculative:
            self.asr_want = seq
        if not runtime.whisper_slot or not _slot_serviceable(runtime.whisper_slot):
            self.spec_busy = False
            return seq
        language = self.language
        spoken_at = self.spoken_ms

        def run():
            t0 = time.perf_counter()
            text = None
            try:
                runtime.whisper_slot.ensure_loaded()
                text = runtime.whisper_slot.transcribe(samples, language=language or None)
            except Exception as e:
                _log(f"!! [voice] transcription failed: {e}")
            result = {"id": utt_id, "text": (text or "").strip(),
                      "ms": (time.perf_counter() - t0) * 1000}
            if speculative:
                self.spec_busy = False
                if utt_id == self.utt_id and self.speaking:
                    self.spec = result
                    self.spec_at = spoken_at
                    return
            if seq == self.asr_want and utt_id == self.awaiting:
                self.final = result

        threading.Thread(target=run, daemon=True).start()
        return seq
