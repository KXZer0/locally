

_HOLD_CHARS = 24          # enough to tell an opener from prose

"""Stripping <think> from a token stream, one token at a time.

The whole-string form lives in core/tools/text.py. This one cannot use a
regex: mid-stream there is no closing tag to match, and the tags themselves
arrive split across tokens as '<', 'think', '>'. So it holds back eight
characters and decides from stream STATE instead."""



class _ThinkFilter:
    """Strips <think> blocks from a token stream, one token at a time.

    Tags arrive split across tokens ("<", "think", ">"), so a per-token
    `in` test can't see them. Hold back the last few characters — anything
    that could still turn out to be the start of a tag — and only release
    text once it can no longer be part of one.
    """

    _HOLD = len("</think>")

    def __init__(self):
        self.buf = ""
        self.inside = False
        # Kept so a turn that produced *only* reasoning (a small model can
        # burn its whole token budget before closing the tag) can fall back
        # to showing it. Returning an empty message would look like a hang.
        self.discarded = ""

    def feed(self, token):
        if self.inside:
            self.discarded += token
        self.buf += token
        out = []
        while True:
            if self.inside:
                end = self.buf.find("</think>")
                if end < 0:
                    # Keep only what might still complete the closing tag.
                    self.buf = self.buf[-self._HOLD:]
                    break
                self.buf = self.buf[end + self._HOLD:]
                self.inside = False
                continue
            start = self.buf.find("<think>")
            if start < 0:
                break
            out.append(self.buf[:start])
            self.buf = self.buf[start + len("<think>"):]
            self.inside = True
        if self.inside:
            return "".join(out)
        # A partial tag can only be at the very end; release the rest.
        safe = self.buf[:-self._HOLD] if len(self.buf) > self._HOLD else ""
        self.buf = self.buf[len(safe):]
        out.append(safe)
        return "".join(out)

    def flush(self):
        rest = "" if self.inside else self.buf
        self.buf = ""
        return rest
