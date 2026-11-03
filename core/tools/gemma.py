"""Gemma 4's own tool-calling dialect, end to end.

Why this exists. The injected Qwen-style prompt (render.py) asks Gemma to write
`<tool_call><function=NAME><parameter=ARG>` XML, which it was never trained on.
Over a multi-step run the model drifts: it writes `<parameter>` with no name
and a JSON object inside, or `<session_id>` child tags, and the Qwen parser
(`<parameter=NAME>` only) threw all of that away, so the call went out with
`{}` -- browser_fetch with no url, again and again, each failure making the
next one likelier because it sits in the history.

Gemma 4 ships a template for exactly this (`<|tool>declaration:...`,
`<|tool_call>call:NAME{k:<|"|>v<|"|>}<tool_call|>`,
`<|tool_response>response:NAME{...}<tool_response|>`, `<|think|>`). genai's
own jinja engine cannot use it: it refuses the template's multi-line
raise_exception, and for tool_calls / tool messages it replaces the template's
rendering with a JSON dump. So the template is rendered here with real jinja2
and handed to the pipeline as a finished string (`apply_chat_template=False`).

Pieces: `GemmaChat` (render), `simplify_tools` (Gemma's template cannot follow
$ref / anyOf / oneOf, which is what MCP servers emit), `split_reasoning`
(`<|channel>thought ... <channel|>` -> reasoning_content) and
`thinking_requested` (the request's own knob; thinking costs latency).
"""

import json
import os
import re

# `<|tool_response>` is how the model hands control back to the runtime after a
# call: it is generated, it ends the turn, and it is not part of the answer.
_SPECIAL_TAIL_RE = re.compile(r"<eos>|<turn\|>|<bos>|<pad>|<\|tool_response>")
_CHANNEL_RE = re.compile(r"<\|channel>(?:thought\n?)?(.*?)(?:<channel\|>|\Z)", re.DOTALL)
THOUGHT_PREFIX = "<|channel>thought\n"


# ---------------------------------------------------------------------------
# Request knobs
# ---------------------------------------------------------------------------

_OFF = {"", "none", "off", "false", "0", "disabled", "disable", "no", "minimal"}


def thinking_requested(body):
    """True when the request asks for reasoning.

    Hermes sends top-level `reasoning_effort`; vLLM/SGLang-style clients send
    `chat_template_kwargs.enable_thinking` or `enable_thinking`; Anthropic-style
    clients send `thinking: {"type": "enabled"}`. Absent means off: thinking
    multiplies a step's decode time, so it is opt-in per request.
    """
    kw = body.get("chat_template_kwargs")
    if isinstance(kw, dict) and "enable_thinking" in kw:
        return bool(kw["enable_thinking"])
    if "enable_thinking" in body:
        return bool(body["enable_thinking"])
    th = body.get("thinking")
    if isinstance(th, dict):
        return str(th.get("type", "")).lower() in ("enabled", "adaptive", "on", "true")
    if isinstance(th, bool):
        return th
    eff = body.get("reasoning_effort")
    if isinstance(eff, dict):
        eff = eff.get("effort") if eff.get("enabled", True) else "none"
    if eff is None:
        r = body.get("reasoning")
        if isinstance(r, dict):
            eff = r.get("effort") if r.get("enabled", True) else "none"
    return eff is not None and str(eff).strip().lower() not in _OFF


# ---------------------------------------------------------------------------
# Tool schemas Gemma's template can read
# ---------------------------------------------------------------------------

def _scalar_default(v):
    return isinstance(v, (bool, int, float)) or (isinstance(v, str) and 0 < len(v) <= 24)


def _simplify(node, defs, seen=()):
    if not isinstance(node, dict):
        return {"type": "string"}
    if "$ref" in node:
        name = str(node["$ref"]).rsplit("/", 1)[-1]
        target = defs.get(name)
        if not isinstance(target, dict) or name in seen:
            return {"type": "string"}
        # Siblings of a $ref (a description, a default) win over the target's.
        return _simplify({**target, **{k: v for k, v in node.items() if k != "$ref"}},
                         defs, seen + (name,))
    node = dict(node)
    if "const" in node:
        node.setdefault("enum", [node["const"]])
    nullable = bool(node.get("nullable"))
    desc = node.get("description") or ""

    for key in ("anyOf", "oneOf"):
        if isinstance(node.get(key), list):
            raw = node[key]
            members = [_simplify(m, defs, seen) for m in raw
                       if not (isinstance(m, dict) and m.get("type") == "null")]
            nullable = nullable or len(members) != len(raw)
            if not members:
                out = {"type": "string"}
            elif len(members) == 1:
                out = members[0]
            elif all(m.get("type") == "object" for m in members):
                out = _merge_variants(members, desc)
                desc = out.get("description", "")
            else:
                out = dict(members[0])
                alts = sorted({m.get("type", "string") for m in members[1:]} - {out.get("type")})
                if alts:
                    out["description"] = ((out.get("description") or desc) +
                                          f" (may also be {'/'.join(alts)})").strip()
            if desc and not out.get("description"):
                out["description"] = desc
            elif desc and out.get("description") != desc and key == "anyOf" and len(members) == 1:
                out["description"] = desc
            if nullable:
                out["nullable"] = True
            return out

    typ = node.get("type")
    if isinstance(typ, list):
        nullable = nullable or "null" in typ
        typ = next((t for t in typ if t != "null"), "string")
    if typ is None:
        typ = ("object" if "properties" in node else
               "array" if "items" in node else "string")
    out = {"type": typ}
    if desc:
        out["description"] = desc
    if "default" in node and _scalar_default(node["default"]) and node["default"] != "":
        d = json.dumps(node["default"]) if not isinstance(node["default"], str) else node["default"]
        out["description"] = (out.get("description", "") + f" [default: {d}]").strip()
    if typ == "string" and isinstance(node.get("enum"), list):
        out["enum"] = [str(e) for e in node["enum"]]
    if typ == "object":
        props = node.get("properties") or {}
        out["properties"] = {k: _simplify(v, defs, seen) for k, v in props.items()}
        req = [r for r in node.get("required") or [] if r in props]
        if req:
            out["required"] = req
    elif typ == "array":
        out["items"] = _simplify(node.get("items") or {}, defs, seen)
    if nullable:
        out["nullable"] = True
    return out


def _merge_variants(members, desc):
    """A tagged union of objects (browser_actions' twelve action kinds) as ONE
    object: the union of every field, `type` enumerating the tags, and a
    one-line catalogue saying which fields each tag takes."""
    props, tags, catalogue = {}, [], []
    for m in members:
        mp = m.get("properties") or {}
        tag = (mp.get("type") or {}).get("enum") or []
        label = tag[0] if tag else "variant"
        tags.extend(tag)
        for k, v in mp.items():
            if k != "type" and k not in props:
                props[k] = v
        req = [r for r in m.get("required") or [] if r != "type"]
        opt = [k for k in mp if k != "type" and k not in req]
        catalogue.append(f"{label}({', '.join(req)}{'[,' + ','.join(opt) + ']' if opt else ''})")
    props["type"] = {"type": "string", "description": "Which kind of item this is.",
                     **({"enum": tags} if tags else {})}
    props = {"type": props.pop("type"), **props}
    out = {"type": "object", "properties": props, "required": ["type"],
           "description": (desc + " " if desc else "") + "One of: " + "; ".join(catalogue) +
                          " -- fields in [] are optional."}
    return out


def simplify_tools(tools):
    """OpenAI `tools` -> [{type, function:{name, description, parameters}}] with
    every $ref inlined and every anyOf/oneOf folded, which is the only kind of
    schema the Gemma template can print."""
    out = []
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if not fn.get("name"):
            continue
        params = fn.get("parameters") or fn.get("input_schema") or {}
        defs = {**(params.get("$defs") or {}), **(params.get("definitions") or {})}
        p = _simplify({**params, "type": params.get("type") or "object"}, defs)
        p.pop("nullable", None)
        out.append({"type": "function", "function": {
            "name": fn["name"], "description": fn.get("description") or "",
            "parameters": p}})
    return out


# ---------------------------------------------------------------------------
# Messages the template can read
# ---------------------------------------------------------------------------

def _text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def has_images(messages):
    for m in messages:
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") != "text" for b in c):
            return True
    return False


def normalize_messages(messages):
    """OpenAI messages -> what the template's loop expects: tool_call
    arguments as mappings (it raises on a JSON string), content always text."""
    out = []
    for i, m in enumerate(messages):
        role = m.get("role", "user")
        if role == "system" and out:
            role = "user"                      # the template honours only the first
        msg = {"role": role, "content": _text_of(m.get("content"))}
        for key in ("reasoning_content", "reasoning"):
            if isinstance(m.get(key), str) and m[key].strip():
                msg[key] = m[key]
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                calls.append({"id": tc.get("id") or f"call_{i}", "type": "function",
                              "function": {"name": fn.get("name", ""), "arguments": args}})
            msg["tool_calls"] = calls
        if role == "tool":
            msg["tool_call_id"] = m.get("tool_call_id")
            if m.get("name"):
                msg["name"] = m["name"]
        out.append(msg)
    return out


# ---------------------------------------------------------------------------
# The template
# ---------------------------------------------------------------------------

class GemmaChat:
    """Renders a conversation with the model's own chat_template.jinja."""

    def __init__(self, template_text):
        import jinja2

        def raise_exception(message):
            raise ValueError(message)

        env = jinja2.Environment()
        env.globals["raise_exception"] = raise_exception
        self._template = env.from_string(template_text)

    @classmethod
    def for_model(cls, model_dir):
        """A renderer for Gemma 4's tool dialect, or None for any other model
        (no template file, no `<|tool_call>` in it, or jinja2 missing)."""
        path = os.path.join(str(model_dir), "chat_template.jinja")
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            return None
        if "<|tool_call>" not in text or "<|think|>" not in text:
            return None
        try:
            return cls(text)
        except Exception:
            return None

    def render(self, messages, tools, thinking):
        """The full prompt string, generation prompt included."""
        prepared = normalize_messages(messages)
        if thinking and tools:
            brief = ("For this tool turn, think briefly about ONE next action. "
                     "Do not restate the task or plan every later step. Use observed tool results, "
                     "then make the next call; revise your approach if a tool failed.")
            if prepared and prepared[0]["role"] == "system":
                prepared[0]["content"] += "\n\n" + brief
            else:
                prepared.insert(0, {"role": "system", "content": brief})
        return self._template.render(
            messages=prepared,
            tools=simplify_tools(tools) or None,
            bos_token="<bos>", eos_token="<eos>",
            add_generation_prompt=True, enable_thinking=bool(thinking))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def split_reasoning(text):
    """(reasoning, rest) from Gemma output that still carries its special
    tokens. `rest` has the thought channel and end-of-turn markers removed and
    keeps any `<|tool_call>` blocks for parse_tool_calls."""
    if "<|channel>" not in text and "<channel|>" not in text:
        return "", _SPECIAL_TAIL_RE.sub("", text).strip()
    thoughts = [m.group(1).strip() for m in _CHANNEL_RE.finditer(text)]
    rest = _CHANNEL_RE.sub("", text)
    rest = rest.replace("<channel|>", "")
    return "\n\n".join(t for t in thoughts if t), _SPECIAL_TAIL_RE.sub("", rest).strip()
