"""Reading a tool call back out of whatever the model actually emitted.

A model often ignores the format it was asked for and falls back to what it
was trained on, so every native dialect is recognised: Qwen3-Coder XML,
Hermes JSON-in-<tool_call>, bare <function=> with no wrapper (Qwen2.5-Coder),
Mistral [TOOL_CALLS], Liquid <|tool_call_start|>, Llama <|python_tag|>, DeepSeek blocks, and a bare-JSON
fallback."""

import ast
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


# Liquid LFM2/LFM2.5: <|tool_call_start|>[fn(a="x", b=2), g()]<|tool_call_end|>
# -- a Python list of calls, not JSON.
_LFM_RE = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL)


def _lfm_calls(block):
    """[(name, args)] from one Pythonic call list; [] when it is not one."""
    try:
        tree = ast.parse(block.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return []
    nodes = tree.elts if isinstance(tree, ast.List) else [tree]
    out = []
    for node in nodes:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            return []
        try:
            args = {kw.arg: ast.literal_eval(kw.value)
                    for kw in node.keywords if kw.arg}
        except (ValueError, SyntaxError):
            return []
        out.append((node.func.id, args))
    return out


# Google Gemma 4: <|tool_call>call:NAME{key:<|"|>text<|"|>,n:3,ok:true}<tool_call|>
# -- unquoted keys, strings fenced by <|"|>, nested {..} and [..] allowed.
_GEMMA_RE = re.compile(r"<\|tool_call>\s*call:([^\s{]+)\s*\{(.*?)\}\s*<tool_call\|>", re.DOTALL)
_GEMMA_Q = '<|"|>'


def _gemma_value(s, i):
    """Parse one Gemma-encoded value starting at s[i]; return (value, next_i)."""
    n = len(s)
    if s.startswith(_GEMMA_Q, i):
        j = s.index(_GEMMA_Q, i + len(_GEMMA_Q))
        return s[i + len(_GEMMA_Q):j], j + len(_GEMMA_Q)
    if s[i] == "{":
        out, i = {}, i + 1
        while i < n and s[i] != "}":
            k = i
            while s[i] != ":":
                i += 1
            key = s[k:i].strip().replace(_GEMMA_Q, "")
            out[key], i = _gemma_value(s, i + 1)
            if i < n and s[i] == ",":
                i += 1
        return out, i + 1
    if s[i] == "[":
        out, i = [], i + 1
        while i < n and s[i] != "]":
            v, i = _gemma_value(s, i)
            out.append(v)
            if i < n and s[i] == ",":
                i += 1
        return out, i + 1
    j = i
    while j < n and s[j] not in ",}]":
        j += 1
    raw = s[i:j].strip()
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true", j
    if low in ("null", "none"):
        return None, j
    try:
        return int(raw), j
    except ValueError:
        pass
    try:
        return float(raw), j
    except ValueError:
        return raw, j


def _gemma_args(body):
    """Arguments dict from the text between the call's outer braces."""
    try:
        val, _ = _gemma_value("{" + body + "}", 0)
    except (ValueError, IndexError):
        return None
    return val if isinstance(val, dict) else None


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


_NAMED_PARAM_RE = re.compile(
    r"<parameter(?:=([^>\s]+)|\s+name\s*=\s*[\"']?([^\"'>\s]+)[\"']?)\s*>(.*?)</parameter>",
    re.DOTALL)
_BARE_PARAM_RE = re.compile(r"<parameter\s*>(.*?)</parameter>", re.DOTALL)
_CHILD_TAG_RE = re.compile(r"<([A-Za-z_][\w.-]*)\s*>(.*?)</\1\s*>", re.DOTALL)


def _function_params(body, name, param_types):
    """Arguments from the inside of one `<function=NAME>...</function>`.

    Models say this several ways and only the first is the one we asked for:
      <parameter=url>x</parameter>            (the requested form)
      <parameter name="url">x</parameter>
      <parameter>{"url": "x"}</parameter>     (JSON object, parameter unnamed)
      <function=f>{"url": "x"}</function>     (JSON object, no parameter tag)
      <parameter><url>x</url></parameter>     (child tags named for the argument)
    Gemma 4 E4B drifts into the unnamed forms a few steps into a long run; each
    used to parse to {} and go out as a tool call with no arguments at all.
    """
    types = param_types.get(name, {})
    args = {}
    for m in _NAMED_PARAM_RE.finditer(body):
        k = (m.group(1) or m.group(2)).strip()
        args[k] = _coerce_value(m.group(3), types.get(k, "string"))
    if args:
        return args
    chunks = [m.group(1) for m in _BARE_PARAM_RE.finditer(body)] or [body]
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk[:1] == "{":
            for obj in _json_calls(chunk):
                if isinstance(obj, dict):
                    inner = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else obj
                    args.update(inner)
        else:
            for k, v in _CHILD_TAG_RE.findall(chunk):
                if not types or k in types:
                    args[k] = _coerce_value(v, types.get(k, "string"))
    return args


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


_LEADING_THINK_RE = re.compile(r"\s*<think>.*?</think>", re.DOTALL)


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

    lfm = _LFM_RE.findall(text)
    if lfm:
        content = _LFM_RE.sub("", text).strip()
        for block in lfm:
            for name, args in _lfm_calls(block):
                add(name, args)
        if tool_calls:
            return content, tool_calls

    gem = _GEMMA_RE.findall(text)
    if gem:
        content = _GEMMA_RE.sub("", text).strip()
        for name, body in gem:
            args = _gemma_args(body)
            if args is not None:
                add(name.strip(), args)
        if tool_calls:
            return content, tool_calls

    blocks = _TOOL_CALL_RE.findall(text)
    if blocks:
        content = _TOOL_CALL_RE.sub("", text).strip()
        for block in blocks:
            block = block.strip()
            m = _FUNCTION_RE.search(block)
            if m:
                name = m.group(1).strip()
                add(name, _function_params(m.group(2), name, param_types))
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
            add(name, _function_params(m.group(2), name, param_types))
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
    # A leading closed <think> block is reasoning, not the answer: the answer
    # after it is what has to be JSON, and the block stays as content (the
    # OpenAI path keeps it; the streaming gate decides the same way).
    lead = _LEADING_THINK_RE.match(text)
    lead = lead.group(0) if lead else ""
    stripped = text[len(lead):].strip()
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
            return lead.strip(), tool_calls

    return text, tool_calls
