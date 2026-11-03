"""Structured output, after the request: genai's failures and the answer's verdict.

The request half (parsing a format field, building the constraint) is in
core/chat/structured.py. This half decides what to tell the client when the
constraint could not be applied, and checks that it held when it was: a
device that accepted the config and ignored it would otherwise be invisible.
The verdict is remembered per (device, model), so one failure turns every
later constrained request on that slot into an up-front 400.
"""

import json
from datetime import datetime

from core.errors import openai_error

# (device_name, model_name) -> why that pairing cannot constrain output.
_unsupported = {}


def _key(slot):
    return (getattr(slot, "device_name", None), getattr(slot, "model_name", None))


def refusal(slot):
    """Why this slot cannot constrain output, if that is already known."""
    return _unsupported.get(_key(slot))


def classify_error(exc):
    """'schema' (the request's fault), 'unsupported' (the device's), or None.

    xgrammar reports from json_schema_converter.cc for every schema problem
    measured here -- an unknown type, unparsable JSON, a dangling $ref -- so
    that file name is the reliable signal; the message text varies.
    """
    msg = str(exc)
    low = msg.lower()
    if "json_schema_converter" in low or "failed to parse json" in low:
        return "schema"
    if "xgrammar" in low or "structured" in low or "grammar" in low:
        return "unsupported"
    return None


def _schema_message(exc):
    """The useful tail of an xgrammar error, without the build path."""
    msg = str(exc).strip()
    if ".cc:" in msg:
        msg = msg.split(".cc:", 1)[1].split(":", 1)[-1].strip()
    return msg.splitlines()[0][:300] if msg else "schema rejected"


def _refuse_message(slot, reason):
    return (f"{slot.device_name} ({slot.model_name}) cannot constrain output to a "
            f"JSON schema: {reason}. Send the request without response_format/format, "
            "or load the model on GPU or CPU.")


def error_response(turn, exc, ollama=False):
    """A 400 for a structured-output failure, or None if `exc` is not one."""
    spec = turn.get("structured") if isinstance(turn, dict) else None
    if spec is None:
        return None
    kind = classify_error(exc)
    if kind is None:
        return None
    slot = turn["slot"]
    if kind == "schema":
        message = f"Invalid JSON schema in {spec.source}: {_schema_message(exc)}"
    else:
        reason = _schema_message(exc)
        _unsupported[_key(slot)] = reason
        message = _refuse_message(slot, reason)
    print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] structured output "
          f"refused ({kind}): {message}", flush=True)
    if ollama:
        from flask import jsonify
        return jsonify({"error": message}), 400
    return openai_error(message)


def refusal_response(slot, ollama=False):
    """Fail fast on a slot already known not to constrain."""
    reason = refusal(slot)
    if reason is None:
        return None
    message = _refuse_message(slot, reason)
    if ollama:
        from flask import jsonify
        return jsonify({"error": message}), 400
    return openai_error(message)


def verdict(slot, spec, text):
    """'constrained' | 'truncated' | 'violated' for a finished answer.

    'truncated' is not a constraint failure: max_tokens ends a document
    wherever it is. 'violated' is -- a constrained generation that stopped on
    its own must be a complete document -- so the slot is remembered as one
    that ignores the constraint.
    """
    try:
        value = json.loads(str(text).strip())
        if spec.kind == "json_object" and not isinstance(value, dict):
            raise ValueError("not an object")
        return "constrained"
    except ValueError:
        pass
    if getattr(text, "finish_reason", "stop") == "length":
        return "truncated"
    reason = "the device accepted the constraint and produced output that does not parse"
    _unsupported[_key(slot)] = reason
    print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] structured output "
          f"VIOLATED on {slot.model_name}; refusing further constrained requests "
          "on this slot", flush=True)
    return "violated"


def header_value(turn):
    """X-Structured-Output before the answer exists (streams, and the default)."""
    if turn.get("structured") is not None:
        return "constrained"
    if turn.get("structured_skipped"):
        return "skipped-tools"
    return None


def watch_stream(frames, turn):
    """Pass OpenAI SSE frames through, judging the finished document.

    Headers have left before the answer exists, so a stream cannot carry the
    verdict; what it can do is make the next request honest. A structured
    error surfaces as the stream's error frame, and an ignored constraint as a
    finished answer that does not parse -- either way the slot is remembered,
    and the following request gets the 400 instead of a stream.
    """
    spec, slot = turn.get("structured"), turn["slot"]
    pieces, finish = [], None
    for frame in frames:
        yield frame
        if spec is None or not frame.startswith("data: {"):
            continue
        try:
            choice = json.loads(frame[6:])["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        pieces.append((choice.get("delta") or {}).get("content") or "")
        finish = choice.get("finish_reason") or finish
    if spec is None or finish in (None, "cancelled"):
        return
    text = "".join(pieces)
    if finish == "error":
        if classify_error(text) == "unsupported":
            _unsupported[_key(slot)] = _schema_message(text)
        return

    class _Done(str):
        finish_reason = finish
    verdict(slot, spec, _Done(text))
