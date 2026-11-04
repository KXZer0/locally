"""A client's tool calls, constrained while they are written: genai structural tags.

The tool prompt (render.py) asks for `<tool_call>\\n<function=NAME>` XML, and a
small model mostly complies -- then names a tool that does not exist, drops a
required argument, or writes "three" where the schema says integer, and the
agent loop fails a step later with nothing pointing back here. genai 2026.4
ships xgrammar's structural tags, which keep sampling free until a TRIGGER
string appears and then mask every token that cannot continue one of the
declared tags. So prose stays prose, and the moment the model writes
`<tool_call>` the rest must be a call to a DECLARED tool whose arguments match
that tool's `parameters` schema. Two tag shapes share the trigger, because
both are what parse_tool_calls reads after it and Qwen3 is trained on the
second: the XML we asked for (`QwenXMLParametersFormat`) and Hermes JSON
(`JSONSchema`).

What it deliberately does not do:

- **Gemma's native dialect** (`<|tool_call>call:NAME{k:<|"|>v<|"|>}`) is left
  unconstrained. Its arguments are not JSON (unquoted keys, `<|"|>` fences), so
  no schema tag describes them, and its opener is a special token, which
  xgrammar matches against decoded text it may never see. Unverifiable here.
- **A compile failure is never a 400.** The client did not ask for a
  constraint; it is a guarantee we add. The fallback walks three levels --
  the client's schemas, the same schemas folded by gemma.simplify_tools
  ($ref inlined, anyOf/oneOf merged), then the tool NAME only with free
  arguments -- and then gives up, logging each step. A level that failed for
  a tool set is remembered for that (device, model, tool set), so an agent
  re-sending the same thirty schemas does not pay for the failure every turn;
  an error that is about the device rather than the schemas (the NPU's static
  pipeline is unverified) turns the constraint off for that (device, model).
- **The output is checked anyway**, as structured_verdict does: a finished
  turn that wrote the trigger but produced no declared call means the
  pipeline accepted the config and ignored it, and the slot stops paying for
  a constraint that does nothing.

`tool_choice`: "none" -> no tags; "required"/"any" -> a call is forced
(`at_least_one`); a named function -> only that tool's tags, forced.
`--no-constrain-tools` turns all of it off.
"""

import hashlib
import json
from datetime import datetime

import openvino_genai as ovg

from core import config
from core.tools.gemma import simplify_tools

_S = ovg.StructuredOutputConfig
TRIGGER = "<tool_call>"
LEVELS = ("schema", "simplified", "name")

_device_off = {}        # (device, model) -> reason
_toolset_level = {}     # (device, model, digest) -> first level index to try


def _functions(tools):
    out = []
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if fn.get("name"):
            out.append((fn["name"], fn.get("parameters") or fn.get("input_schema") or {}))
    return out


def resolve_choice(tool_choice, names):
    """('none'|'auto'|'required', [names]) from an OpenAI or Anthropic tool_choice."""
    if isinstance(tool_choice, str):
        kind, wanted = tool_choice, None
    elif isinstance(tool_choice, dict):
        kind = tool_choice.get("type")
        wanted = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
    else:
        kind, wanted = "auto", None
    if kind == "none":
        return "none", []
    if kind in ("function", "tool") and wanted:
        return ("required", [wanted]) if wanted in names else ("auto", list(names))
    if kind in ("required", "any"):
        return "required", list(names)
    return "auto", list(names)


def _params(schema):
    schema = dict(schema) if isinstance(schema, dict) else {}
    schema.setdefault("type", "object")
    return schema


def _tags(name, params, level):
    xml_begin = f"{TRIGGER}\n<function={name}>\n"
    json_begin = TRIGGER + '\n{"name": ' + json.dumps(name) + ', "arguments": '
    if level == "name":
        xml, js = _S.AnyText(), _S.AnyText()
    else:
        text = json.dumps(params)
        js = _S.JSONSchema(text)
        # QwenXML renders one <parameter=...> per property and cannot express
        # "none at all" (its grammar comes out empty): allow an optional newline.
        xml = (_S.QwenXMLParametersFormat(text) if params.get("properties")
               else _S.Regex(r"\n?"))
    return [_S.Tag(xml_begin, xml, "</function>\n</tool_call>"),
            _S.Tag(json_begin, js, "}\n</tool_call>")]


def _kind(error_text):
    """'toolset' (these schemas), 'device' (this pipeline), or None (not ours)."""
    low = (error_text or "").lower()
    if any(s in low for s in ("structural tag", "grammar_parser", "json_schema_converter",
                              "ebnf parser", "function calling must")):
        return "toolset"
    if any(s in low for s in ("xgrammar", "structured output", "structured_output", "grammar")):
        return "device"
    return None


class ToolConstraint:
    """The constraint for one turn, and the fallback state behind it."""

    def __init__(self, slot, tools, mode, names):
        self.slot = slot
        self.mode = mode
        self.names = set(names)
        self.raw = [(n, _params(p)) for n, p in _functions(tools) if n in self.names]
        simple = {t["function"]["name"]: t["function"]["parameters"]
                  for t in simplify_tools(tools)}
        self.simple = [(n, _params(simple.get(n, p))) for n, p in self.raw]
        self.device_key = (slot.device_name, slot.model_name)
        digest = hashlib.sha256(json.dumps([self.raw, mode], sort_keys=True,
                                           default=str).encode()).hexdigest()[:16]
        self.key = self.device_key + (digest,)
        self.level = _toolset_level.get(self.key, 0)

    @property
    def active(self):
        return self.level < len(LEVELS)

    def config(self):
        level = LEVELS[self.level]
        tags = []
        for name, params in (self.simple if level == "simplified" else self.raw):
            tags += _tags(name, params, level)
        return _S(structural_tags_config=_S.TriggeredTags(
            triggers=[TRIGGER], tags=tags, at_least_one=self.mode == "required",
            stop_after_first=False))

    def apply(self, gen):
        gen.structured_output_config = self.config() if self.active else None
        return gen

    def failed(self, gen, error_text):
        """Step down after a generation error; True when it is worth retrying."""
        if not self.active:
            return False
        kind = _kind(error_text)
        if kind is None:
            return False
        reason = str(error_text or "?").strip()
        if ".cc:" in reason:                 # drop xgrammar's build path
            reason = reason.split(".cc:", 1)[1].split(":", 1)[-1].strip()
        reason = (reason.splitlines() or ["?"])[0][:160]
        if kind == "device":
            _device_off[self.device_key] = reason
            self.level = len(LEVELS)
        else:
            self.level += 1
            _toolset_level[self.key] = self.level
        now = LEVELS[self.level] if self.active else "off"
        print(f"{datetime.now():%H:%M:%S} -- [{self.slot.device_name}] tool-call "
              f"constraint failed ({kind}: {reason}); retrying with {now}", flush=True)
        self.apply(gen)
        return True

    def run(self, gen, fn):
        """fn() under the constraint, stepping down until it compiles."""
        while True:
            try:
                return fn()
            except Exception as e:
                if not self.failed(gen, str(e)):
                    raise

    def judge(self, text, calls, finish):
        """Notice a pipeline that accepted the tags and ignored them."""
        if not self.active or finish in ("length", "cancelled", "error"):
            return
        names = [(c.get("function") or {}).get("name") for c in calls or []]
        if TRIGGER in str(text) and (not names or any(n not in self.names for n in names)):
            _device_off[self.device_key] = "constraint accepted and ignored"
            print(f"{datetime.now():%H:%M:%S} !! [{self.slot.device_name}] tool-call "
                  f"constraint IGNORED on {self.slot.model_name}; turning it off for "
                  "this slot", flush=True)

    def stream_options(self, gen, restart):
        """gated_stream's retry/judge for a stream `restart()` can rebuild."""
        return {"retry": lambda err: restart() if self.failed(gen, err) else None,
                "judge": self.judge}


def build(slot, tools, tool_choice=None, native=False):
    """A ToolConstraint for this turn, or None when there is nothing to add."""
    if not config.CONSTRAIN_TOOLS or native or slot is None or not tools:
        return None
    if type(slot).__name__ == "ProxySlot":
        return None                     # the remote server owns its sampler
    if (slot.device_name, slot.model_name) in _device_off:
        return None
    names = [n for n, _p in _functions(tools)]
    mode, chosen = resolve_choice(tool_choice, names)
    if mode == "none" or not chosen:
        return None
    tc = ToolConstraint(slot, tools, mode, chosen)
    return tc if tc.active else None


def apply(gen, tc):
    return tc.apply(gen) if tc is not None else gen


def run(tc, gen, fn):
    return fn() if tc is None else tc.run(gen, fn)


def judge(tc, text, calls, finish):
    if tc is not None:
        tc.judge(text, calls, finish)


def stream_options(tc, gen, restart):
    return {} if tc is None else tc.stream_options(gen, restart)
