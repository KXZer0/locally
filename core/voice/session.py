"""One voice turn, decided server-side.

The client streams 512-sample frames and this decides when a turn opens,
when it ends, and -- once a speaker profile exists -- whether the voice is
the enrolled user's at all. Because the audio is already here, the
transcription happens here too, speculatively during the pause rather than
after it."""

from collections import deque
from datetime import datetime
import numpy as np
import threading
import time
from core import runtime
from core.slots.vad import VadSlot
from core.slots.select import _slot_serviceable


class VoiceStreamSession:
    """Endpointing for one open microphone.

    Holds the turn's audio as it arrives, which is the point: when the user
    stops talking the samples are already here, so transcription can start
    during the pause rather than after an upload that begins once it ends.

    Because the audio is here, the transcription happens here too. Speculation
    alone is not enough to hide the latency — the head start between "they went
    quiet" and "the pause is over" is ~500 ms against a Whisper run of about a
    second, so the guess usually would not have landed in time. What it does
    buy is that same ~500 ms off the front of a run that has to happen anyway,
    with no upload after it. The client sends nothing and waits for `transcript`.
    """

    PREROLL_MS = 500        # kept before a turn opens, so the first word survives
    MIN_SPEECH_MS = 350     # shorter than this is a cough, not a turn
    SPECULATE_MS = 400      # silence before we start transcribing on spec
    SPECULATE_MIN_MS = 1000  # too short to be worth a wasted Whisper run
    BARGE_MS = 400          # sustained speech over playback before interrupting
    MIN_BARGE_VERIFY_MS = 650   # ... and this much when it must also be identified

    def __init__(self):
        self.state = runtime._vad_slot.new_state() if runtime._vad_slot else None
        self.threshold = 0.5
        self.patience_ms = 900
        self.language = ""
        self.enabled = True
        self.voice_state = "idle"
        self.speaking = False        # is a turn currently open
        self.preroll = deque(maxlen=max(1, int(self.PREROLL_MS / 32)))
        self.utterance = []
        self.speech_ms = 0.0         # sustained speech, for turn start / barge-in
        self.barge_buf = []          # candidate interruption audio, for speaker ID
        self.silence_ms = 0.0
        self.spoken_ms = 0.0
        self.utt_id = 0
        self.spec = None             # speculative result: {"id", "text", "ms"}
        self.spec_at = None          # spoken_ms when that guess was started
        self.spec_busy = False       # one speculative run at a time
        self.final = None            # result ready to send
        self.awaiting = None         # utterance id whose transcript is owed
        self.asr_seq = 0             # every run is tagged; only one can win
        self.asr_want = None
        self.spec_seq = None

    @property
    def frame_ms(self):
        return VadSlot.FRAME / VadSlot.SR * 1000

    def samples(self):
        if not self.utterance:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.utterance)

    def drain(self):
        """Transcripts that finished since the last call. Sent from the socket's
        own thread — simple_websocket has one writer and it is this loop."""
        out = []
        if self.final and self.final["id"] == self.awaiting:
            out.append({"type": "transcript", "utteranceId": self.final["id"],
                        "text": self.final["text"], "ms": self.final["ms"]})
            self.awaiting = None
            self.final = None
        return out

    def _gating(self):
        """Is speaker verification actually in force right now?

        Every condition has to hold: a slot, a loaded model, and a profile
        enrolled by THAT model with enough audio behind it. Any of them
        missing means the system cannot tell who is talking, and the whole
        design rule is that a system which cannot tell must let speech
        through rather than swallow it.
        """
        return bool(runtime._speaker_slot) and runtime._speaker_slot.profile_ready             and runtime._speaker_slot.status == "ready"

    def _verify(self, samples, why):
        """Verify a candidate utterance, and say so in the log.

        The score is printed on every decision because the threshold cannot be
        set from the outside: it depends on the model, the microphone and the
        room. Someone tuning --speaker-threshold needs to see their own
        numbers, and a rejection with no number attached is indistinguishable
        from the feature being broken.
        """
        if not self._gating():
            return "abstain", None
        verdict, score = runtime._speaker_slot.verify(samples)
        if verdict != "abstain":
            print(f"{datetime.now():%H:%M:%S} -- [speaker] {why}: {verdict} "
                  f"(cosine {score:.3f} vs threshold "
                  f"{runtime._speaker_slot.threshold:.2f})", flush=True)
        return verdict, score

    def feed(self, frame):
        """One 512-sample frame in, a list of events out."""
        events = []
        prob = runtime._vad_slot.probability(frame, self.state)
        dt = self.frame_ms
        # Hysteresis: it takes more evidence to open a turn than to keep one
        # open, so a dip between words does not close it.
        loud = prob >= (self.threshold if not self.speaking
                        else max(0.15, self.threshold - 0.15))

        if self.speaking:
            self.utterance.append(frame)
        else:
            self.preroll.append(frame)

        # While audio is playing, the microphone hears the speakers. Require a
        # longer run of speech before calling it an interruption.
        if self.voice_state == "speaking":
            self.speech_ms = self.speech_ms + dt if loud else 0.0
            if loud:
                self.barge_buf.append(frame)
            else:
                self.barge_buf.clear()
            # With speaker gating on, an interruption needs enough audio to
            # identify, not just enough to detect: the 500 ms pre-roll ring is
            # under the verifier's minimum, and it is mostly the assistant's
            # own voice bleeding back through the microphone anyway. So the
            # barge candidate is collected separately, and the bar rises from
            # 400 ms to MIN_BARGE_VERIFY_MS. That is ~250 ms of extra latency
            # before the assistant stops talking, and it buys the difference
            # between "someone spoke" and "you spoke".
            need = (max(self.BARGE_MS, self.MIN_BARGE_VERIFY_MS)
                    if self._gating() else self.BARGE_MS)
            if self.speech_ms >= need:
                self.speech_ms = 0.0
                verdict, score = self._verify(np.concatenate(self.barge_buf)
                                              if self.barge_buf else np.zeros(0),
                                              "barge-in")
                self.barge_buf.clear()
                if verdict != "reject":
                    events.append({"type": "speech_start"})
            return events

        if not self.enabled:
            return events

        if not self.speaking:
            if self.voice_state != "idle":
                return events
            self.speech_ms = self.speech_ms + dt if loud else max(0.0, self.speech_ms - dt)
            if self.speech_ms >= self.MIN_SPEECH_MS / 2:
                self.speaking = True
                self.utterance = list(self.preroll)
                self.preroll.clear()
                self.speech_ms = self.silence_ms = 0.0
                self.spoken_ms = self.MIN_SPEECH_MS / 2
                self.utt_id += 1
                self.spec = None
                events.append({"type": "speech_start"})
            return events

        if loud:
            self.silence_ms = 0.0
            self.spoken_ms += dt
        else:
            self.silence_ms += dt
            if (self.silence_ms >= self.SPECULATE_MS and not self.spec_busy
                    and self.spec_at != self.spoken_ms
                    and self.spoken_ms >= self.SPECULATE_MIN_MS):
                self.spec_busy = True
                self.spec_at = self.spoken_ms
                self.spec_seq = self._transcribe(self.samples(), self.utt_id,
                                                 speculative=True)

        if self.silence_ms >= self.patience_ms:
            done = {"type": "speech_end", "utteranceId": self.utt_id}
            samples = self.samples()
            verdict, score = ("abstain", None)
            if self.spoken_ms >= self.MIN_SPEECH_MS:
                verdict, score = self._verify(samples, "turn")
            if self.spoken_ms < self.MIN_SPEECH_MS:
                done["tooShort"] = True
            elif verdict == "reject":
                # Somebody spoke, but not the enrolled user. Drop the turn
                # before Whisper ever sees it: transcribing it would spend the
                # device lock on audio that is going to be discarded, and the
                # old behaviour -- sending it as the user's own message -- is
                # the bug this whole feature exists to fix.
                done["notEnrolled"] = True
                if score is not None:
                    done["score"] = round(score, 3)
            elif (self.spec and self.spec["id"] == self.utt_id
                  and self.spec_at == self.spoken_ms):
                # The guess covered every word this turn contains — trailing
                # silence cannot change a transcript — and it landed in time.
                self.awaiting = self.utt_id
                self.final = self.spec
            elif self.spec_busy and self.spec_at == self.spoken_ms:
                # Still running, and running on the whole turn. Wait for it
                # rather than starting a second copy: they would queue on the
                # same device lock and the duplicate would only add latency.
                self.awaiting = self.utt_id
                self.asr_want = self.spec_seq
            else:
                # Still running, never started, or started before the user said
                # more. Run it for real; the client is owed a transcript and
                # must not go and make one itself.
                self.awaiting = self.utt_id
                self.asr_want = self._transcribe(samples, self.utt_id,
                                                 speculative=False)
            self.speaking = False
            self.utterance = []
            self.silence_ms = self.spoken_ms = self.speech_ms = 0.0
            self.spec = self.spec_at = None
            events.append(done)
        return events

    def _transcribe(self, samples, utt_id, speculative):
        """Run Whisper on a background thread; returns the run's sequence id.

        Every run is tagged, and only the run the turn is waiting for can
        become the final answer — otherwise a speculative run started before
        the user carried on could overwrite the real one. Whisper takes the
        device lock this slot shares with chat, so a wasted run is not free for
        anything else on that device; hence one speculation at a time.
        """
        self.asr_seq += 1
        seq = self.asr_seq
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
                print(f"{datetime.now():%H:%M:%S} !! [voice] transcription "
                      f"failed: {e}", flush=True)
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
