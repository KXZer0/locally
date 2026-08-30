"""Reading a tool call back out of whatever the model actually emitted.

A model often ignores the format it was asked for and falls back to what it
was trained on, so every native dialect is recognised: Qwen3-Coder XML,
Hermes JSON-in-<tool_call>, bare <function=> with no wrapper (Qwen2.5-Coder),
Mistral [TOOL_CALLS], Llama <|python_tag|>, DeepSeek blocks, and a bare-JSON
fallback."""

import json
import re
import uuid

from .text import _strip_tool_markup


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)


_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


# Other model families emit their own native tool-call syntax. A small model
# often ignores our Qwen3-Coder system prompt and falls back to whatever it was
# trained on, so we recognize those too:
#   Mistral   : [TOOL_CALLS][{"name": ..., "arguments": {...}}, ...]
#   Llama 3.x : <|python_tag|>{"name": ..., "parameters": {...}}  (';'-separated)
#   DeepSeek  : <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
#               ```json\n{...}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
_MISTRAL_RE = re.compile(r"\[TOOL_CALLS\]")


_PYTHON_TAG_RE = re.compile(r"<\|python_tag\|>")


_DS_BEGIN = "<｜tool▁calls▁begin｜>"


_DS_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>\s*\w+\s*<｜tool▁sep｜>\s*([^\n`]+?)\s*"
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)


def _tool_param_types(tools):
    """Map {tool_name: {param_name: json_schema_type}} for value coercion."""
    types = {}
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        types[name] = {k: (v or {}).get("type", "string") for k, v in props.items()}
    return types


def _coerce_value(raw, json_type):
    """Best-effort coerce an XML <parameter> string to its schema type."""
    raw = raw.strip()
    try:
        if json_type in ("number", "integer", "object", "array"):
            return json.loads(raw)
        if json_type == "boolean":
            return raw.strip().lower() == "true"
    except (ValueError, TypeError):
        pass
    return raw


def _extract_name_args(obj):
    """Pull (name, args dict) from a tool-call dict across naming conventions.

    Handles name vs function.name vs function (string), and arguments vs
    parameters vs function.arguments, with args possibly JSON-encoded as a
    string. Returns (None, {}) if obj isn't a usable call.
    """
    if not isinstance(obj, dict):
        return None, {}
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else None
    name = obj.get("name") or (fn or {}).get("name")
    if isinstance(obj.get("function"), str):
        name = obj["function"]
    args = obj.get("arguments")
    if args is None and fn:
        args = fn.get("arguments")
    if args is None:
        args = obj.get("parameters")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def _iter_json_objects(s):
    """Yield top-level {...} substrings from s, respecting string escaping.

    Lets us pull several JSON objects out of one blob (e.g. Llama's
    ';'-separated parallel calls) without a brittle balanced-brace regex.
    """
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield s[start:i + 1]
                start = None


def _json_calls(blob):
    """Parse one-or-more tool-call dicts from a blob.

    Accepts a JSON array, a single JSON object, or several objects run together
    / separated by ';' or whitespace (the variants Mistral and Llama emit).
    """
    blob = blob.strip()
    if not blob:
        return []
    try:
        obj = json.loads(blob)
        if isinstance(obj, list):
            return [o for o in obj if isinstance(o, dict)]
        if isinstance(obj, dict):
            return [obj]
    except (ValueError, TypeError):
        pass
    calls = []
    for chunk in _iter_json_objects(blob):
        try:
            o = json.loads(chunk)
        except (ValueError, TypeError):
            continue
        if isinstance(o, dict):
            calls.append(o)
    return calls


def parse_tool_calls(text, tools):
    """Extract tool calls from generated text.

    Returns (content_text, tool_calls) where tool_calls is a list of OpenAI
    tool_call dicts (empty if none found). Handles Qwen3-Coder XML, Hermes-style
    JSON-in-<tool_call>, Mistral [TOOL_CALLS], Llama <|python_tag|>, DeepSeek's
    <｜tool▁calls▁begin｜> blocks, and a bare JSON fallback.
    """
    param_types = _tool_param_types(tools)
    known = set(param_types)
    tool_calls = []

    def add(name, args):
        tool_calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args, ensure_ascii=False)},
        })

    blocks = _TOOL_CALL_RE.findall(text)
    if blocks:
        content = _TOOL_CALL_RE.sub("", text).strip()
        for block in blocks:
            block = block.strip()
            m = _FUNCTION_RE.search(block)
            if m:
                name = m.group(1).strip()
                args = {}
                for k, v in _PARAM_RE.findall(m.group(2)):
                    k = k.strip()
                    args[k] = _coerce_value(v, param_types.get(name, {}).get(k, "string"))
                add(name, args)
            else:
                # JSON inside <tool_call> (Hermes / Qwen2.5 style)
                try:
                    obj = json.loads(block)
                except (ValueError, TypeError):
                    obj = None
                name, args = _extract_name_args(obj)
                if name:
                    add(name, args)
        return content, tool_calls

    # Qwen2.5-Coder native: bare <function=NAME>...</function> with NO
    # surrounding <tool_call> wrapper. The Qwen2.5-Coder models emit this form
    # regardless of the prompt we render, so the wrapped path above never fires
    # for them. _FUNCTION_RE only matches the literal <function=...> syntax, so
    # this won't trip on ordinary prose.
    fn_blocks = list(_FUNCTION_RE.finditer(text))
    if fn_blocks:
        content = _strip_tool_markup(_FUNCTION_RE.sub("", text))
        for m in fn_blocks:
            name = m.group(1).strip()
            args = {}
            for k, v in _PARAM_RE.findall(m.group(2)):
                k = k.strip()
                args[k] = _coerce_value(v, param_types.get(name, {}).get(k, "string"))
            add(name, args)
        if tool_calls:
            return content, tool_calls

    # Mistral: [TOOL_CALLS] followed by a JSON array of {name, arguments}.
    m = _MISTRAL_RE.search(text)
    if m:
        content = text[:m.start()].strip()
        for obj in _json_calls(text[m.end():]):
            name, args = _extract_name_args(obj)
            if name:
                add(name, args)
        if tool_calls:
            return content, tool_calls

    # Llama 3.x: <|python_tag|> then JSON object(s) (';'-separated for parallel).
    m = _PYTHON_TAG_RE.search(text)
    if m:
        content = text[:m.start()].strip()
        for obj in _json_calls(text[m.end():]):
            name, args = _extract_name_args(obj)
            if name:
                add(name, args)
        if tool_calls:
            return content, tool_calls

    # DeepSeek: function name + ```json args``` inside <｜tool▁call▁begin｜> blocks.
    if _DS_BEGIN in text:
        content = text.split(_DS_BEGIN, 1)[0].strip()
        for name, blob in _DS_CALL_RE.findall(text):
            try:
                args = json.loads(blob)
            except (ValueError, TypeError):
                args = {}
            if name.strip():
                add(name.strip(), args if isinstance(args, dict) else {})
        if tool_calls:
            return content, tool_calls

    # Fallback: bare JSON (object or array) where every call names a known tool
    # — the failure mode when the model isn't told the proper format. We only
    # treat it as tool calls if all names match, so normal JSON answers pass
    # through untouched.
    stripped = text.strip()
    if known and stripped[:1] in ("{", "["):
        named = []
        for obj in _json_calls(stripped):
            name, args = _extract_name_args(obj)
            if name is None:
                # Some models emit {"function": "x", "k": v, ...} with no
                # arguments wrapper — treat leftover keys as the arguments.
                continue
            if not args:
                args = {k: v for k, v in obj.items()
                        if k not in ("name", "function", "type",
                                     "arguments", "parameters")}
            named.append((name, args))
        if named and all(n in known for n, _ in named):
            for n, a in named:
                add(n, a)
            return "", tool_calls

    return text, tool_calls
