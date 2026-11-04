"""Which text of a tool turn may be shown NOW, decided token by token.

A tool turn used to be generated whole before a byte of it was sent, because
a structured `tool_calls` delta needs the whole call. That made every reply
on a tool-enabled request -- including the many that call nothing -- arrive
in one lump after the last token. The gate streams the text and holds back
only what could still turn out to be a call:

- **The tail.** The last few characters are held while they are a prefix of
  some opener parse_tool_calls acts on (`<tool_call>`, `<function=`,
  `[TOOL_CALLS]`, `<|python_tag|>`, DeepSeek's `<｜tool▁calls▁begin｜>`,
  Liquid's `<|tool_call_start|>`, Gemma's `<|tool_call>`). `<tool` in prose
  is released the moment the next character rules the opener out.
- **Everything after an opener.** Once a whole opener appears, nothing more
  is shown; the turn is parsed at the end exactly as the buffered path parsed
  it, and the text that parser keeps (anything after the call) is sent then.
- **A reply whose answer begins with `{` or `[`.** That is the bare-JSON
  fallback, which only fires when the WHOLE stripped answer is JSON naming
  declared tools, so there is no way to tell early. The answer is held to the
  end, which costs streaming only for JSON answers on tool turns. A fenced
  ```json block is not held: the fallback never reads one, so it is prose.
  "Begins" means after a leading `<think>...</think>`, which streams live.

Reasoning is not hidden: a `<think>` block stays content on this path (the
OpenAI path has always kept it), and on Gemma's native dialect the thought
channel is lifted into `reasoning` deltas, as split_reasoning does for a
buffered turn, with its end-of-turn tokens dropped.
"""

from core.tools.gemma import split_reasoning
from core.tools.parse import parse_tool_calls

OPENERS = ("<tool_call>", "<function=", "[TOOL_CALLS]", "<|python_tag|>",
           "<｜tool▁calls▁begin｜>", "<|tool_call_start|>", "<|tool_call>")
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
# Gemma 4 native dialect (core/tools/gemma.py).
CHANNEL_OPENS = ("<|channel>thought\n", "<|channel>thought", "<|channel>")
CHANNEL_CLOSE = "<channel|>"
DROPPED = ("<eos>", "<turn|>", "<bos>", "<pad>", "<|tool_response>")


def _names(tools):
    out = set()
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if fn.get("name"):
            out.add(fn["name"])
    return out


class ToolStreamGate:
    """Feed it deltas; it returns [(kind, text)] that are safe to send.

    kind is "content" or "reasoning". `finish()` flushes what is still held
    and returns the parsed tool calls, so the caller never parses twice.
    """

    def __init__(self, tools, native=False):
        self.tools = tools or []
        self.known = bool(_names(self.tools))
        self.native = native
        self.buf = ""
        self.pos = 0
        self.mode = "content"           # content | reasoning | call | json
        self.in_think = False
        self.answered = False           # visible, non-think, non-space text sent
        self.sent = {"content": [], "reasoning": []}

    # -- markers -----------------------------------------------------------

    def _markers(self):
        if self.mode == "reasoning":
            return (CHANNEL_CLOSE,) + DROPPED
        if self.native:
            return OPENERS + CHANNEL_OPENS + (CHANNEL_CLOSE,) + DROPPED
        return OPENERS + (THINK_OPEN, THINK_CLOSE)

    def _find(self, seg, final):
        """(index, marker) of the earliest whole marker, or (None, None).

        A whole marker that is also the start of a longer one still arriving
        (`<|channel>` vs `<|channel>thought\\n`) is not decided yet: it is
        reported as a hold at that index instead.
        """
        markers = self._markers()
        best, found = None, None
        for m in markers:
            j = seg.find(m)
            if j != -1 and (best is None or j < best or (j == best and len(m) > len(found))):
                best, found = j, m
        if best is None:
            return None, None
        if not final:
            rest = seg[best:]
            if any(len(m) > len(rest) and m.startswith(rest) for m in markers):
                return best, ""            # hold here, longer marker may follow
        return best, found

    def _hold_from(self, seg):
        """Earliest index whose suffix could still grow into a marker."""
        markers = self._markers()
        longest = max(len(m) for m in markers)
        for i in range(max(0, len(seg) - longest + 1), len(seg)):
            tail = seg[i:]
            if any(m.startswith(tail) for m in markers):
                return i
        return len(seg)

    # -- feeding -----------------------------------------------------------

    def _emit(self, out, text):
        if not text:
            return
        kind = "reasoning" if self.mode == "reasoning" else "content"
        if kind == "content" and not self.in_think and text.strip():
            self.answered = True
        self.sent[kind].append(text)
        if out and out[-1][0] == kind:
            out[-1] = (kind, out[-1][1] + text)
        else:
            out.append((kind, text))

    def _advance(self, out, final):
        while self.mode in ("content", "reasoning") and self.pos < len(self.buf):
            seg = self.buf[self.pos:]
            j, marker = self._find(seg, final)
            limit = len(seg) if j is None else j
            if (self.mode == "content" and self.known and not self.answered
                    and not self.in_think):
                head = seg[:limit]
                k = len(head) - len(head.lstrip())
                if k == len(head) and j is None and not final:
                    return                 # only whitespace so far: undecided
                if k < len(head) and head[k] in "{[":
                    self.pos += k          # the bare-JSON case: hold it all
                    self.mode = "json"
                    return
            if j is None:
                cut = len(seg) if final else self._hold_from(seg)
                self._emit(out, seg[:cut])
                self.pos += cut
                return
            if marker == "":               # a longer marker may still arrive
                self._emit(out, seg[:j])
                self.pos += j
                return
            self._emit(out, seg[:j])
            self.pos += j
            if marker in OPENERS and self.mode == "content":
                self.mode = "call"         # the rest is a call: hold it all
                return
            self.pos += len(marker)
            if marker in (THINK_OPEN, THINK_CLOSE):
                self.in_think = True       # the tag itself is not the answer
                self._emit(out, marker)
                self.in_think = marker == THINK_OPEN
            elif marker in CHANNEL_OPENS:
                self.mode = "reasoning"
            elif marker == CHANNEL_CLOSE and self.mode == "reasoning":
                self.mode = "content"
            # DROPPED tokens and a stray <channel|> are simply skipped.

    def feed(self, delta):
        self.buf += delta
        out = []
        self._advance(out, final=False)
        return out

    # -- the end -----------------------------------------------------------

    def finish(self):
        """([(kind, text)] still to send, tool_calls, reasoning) for the turn."""
        out = []
        held_mode = self.mode
        self._advance(out, final=True)
        reasoning = ""
        visible = self.buf
        if self.native:
            reasoning, visible = split_reasoning(self.buf)
        content, calls = parse_tool_calls(visible, self.tools)
        if calls or held_mode in ("call", "json"):
            rest = self._remainder(content, calls)
            if rest:
                prev = self.mode
                self.mode = "content"
                self._emit(out, rest)
                self.mode = prev
        return out, calls, reasoning

    def _remainder(self, content, calls):
        """The part of the parsed content the client has not been sent."""
        sent = "".join(self.sent["content"])
        lead = sent.strip()
        if content.startswith(lead):
            rest = content[len(lead):]
            trailing = sent[len(sent.rstrip()):]
            if trailing and rest.startswith(trailing):
                rest = rest[len(trailing):]
            elif trailing:
                rest = rest.lstrip()
            return rest
        # The parser rewrote text already shown (it strips stray tags); what
        # was sent cannot be unsent, so add only what was never shown.
        return "" if calls else self.buf[self.pos:]
