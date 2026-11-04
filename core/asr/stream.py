"""Transcribe while the audio is still arriving.

A streaming model (core/asr/geometry.py) can decode a unit of audio the
moment its last sample -- plus the right context, in window mode -- exists,
and the only thing that crosses a unit boundary is the predictor state, the
last token and (cache mode) the encoder caches. So a voice turn can be
transcribed DURING the turn: feed() the frames as the microphone sends them,
a worker thread decodes each unit as it becomes ready, and finalize() at the
end of the turn only has the last unit or two left to run.

Offline transcription is the same object fed all at once with no worker,
which is why the two cannot drift: there is one loop, and the geometry
decides when a unit is ready in both cases.

The slot provides four things: `geometry`, `new_carry(language)`,
`run_unit(carry, samples, valid)` and `text(carry)`. Each unit takes the slot
lock -- the device lock chat shares -- for one encoder pass and its decode,
not for the whole turn, so a long turn never holds the device against chat.
"""

import threading

import numpy as np


class StreamingTranscriber:
    def __init__(self, slot, language=None, background=True):
        self.slot = slot
        self.geo = slot.geometry
        self.carry = slot.new_carry(language)
        self.background = background
        self.pos = 0                 # first sample of the next undecoded unit
        self.n = 0                   # samples received
        self.units = 0               # units decoded so far
        self.error = None
        self._parts = []
        self._audio = np.zeros(0, dtype=np.float32)
        self._cv = threading.Condition()
        self._final = False
        self._cancelled = False
        self._worker = None

    # --- producer side -----------------------------------------------------

    def feed(self, samples):
        """Append 16 kHz float32 audio. Cheap: never runs inference itself."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        with self._cv:
            if self._final or self._cancelled:
                raise RuntimeError("feed() after finalize() or cancel()")
            self._parts.append(samples)
            self.n += samples.size
            self._cv.notify()
        if self.background and self._worker is None:
            self._worker = threading.Thread(target=self._loop, daemon=True,
                                            name="asr-stream")
            self._worker.start()

    def finalize(self):
        """Decode what is left and return the transcript. Blocks."""
        with self._cv:
            self._final = True
            self._cv.notify()
        if self._worker is not None:
            self._worker.join()
        if self.error is not None:
            raise self.error
        self._drain(final=True)
        return self.slot.text(self.carry)

    def cancel(self):
        """Abandon the turn: the worker stops after the unit it is on."""
        with self._cv:
            self._cancelled = True
            self._cv.notify()

    # --- consumer side -----------------------------------------------------

    def _snapshot(self):
        """(audio, n) with every fed part joined. Caller holds self._cv."""
        if self._parts:
            self._audio = np.concatenate([self._audio, *self._parts])
            self._parts = []
        return self._audio, self.n

    def _drain(self, final):
        while True:
            with self._cv:
                audio, n = self._snapshot()
                if self._cancelled or not self.geo.ready(self.pos, n, final):
                    return
            samples, valid = self.geo.unit(audio, self.pos, n)
            if self.background:
                self.slot.ensure_loaded()     # idle unload may have run mid-turn
            with self.slot.lock:
                self.slot.run_unit(self.carry, samples, valid)
            self.pos += self.geo.step
            self.units += 1

    def _loop(self):
        while True:
            with self._cv:
                while not (self._final or self._cancelled
                           or self.geo.ready(self.pos, self.n, False)):
                    self._cv.wait()
                if self._final or self._cancelled:
                    # finalize() runs the tail on its own thread after join().
                    # A unit that became ready in the same instant is part of it.
                    return
            try:
                self._drain(final=False)
            except Exception as e:      # surfaced by finalize()
                self.error = e
                return
