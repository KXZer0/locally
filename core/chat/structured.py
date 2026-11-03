"""Structured output: a request's JSON format, enforced while decoding.

A 4-8B model asked for JSON produces JSON most of the time, and "most of the
time" is the failure: a missing brace or a chatty preamble breaks a client's
parser on the turn nobody is watching. OpenVINO GenAI ships xgrammar inside
its sampler (`GenerationConfig.structured_output_config`), which masks every
token that cannot continue a document matching the schema -- the model cannot
emit malformed JSON, rather than being asked nicely not to.

This module only translates: OpenAI `response_format`, Ollama `format` and
Anthropic `output_format` into one spec, the spec into a StructuredOutputConfig,
and genai's failures into answers a client can act on.

Three decisions, each deliberate:

- **Tool turns are left alone.** A constrained answer cannot also be a tool
  call -- the grammar would forbid the call's markup -- so when a turn's tools
  are active the format is skipped and `X-Structured-Output: skipped-tools`
  says so. The server's own built-in tools are not offered on a constrained
  turn for the same reason.
- **A device that cannot constrain gets a 400, never a silent retry.** The
  client asked for a guarantee. Retrying unconstrained would hand back exactly
  the possibly-malformed JSON this exists to prevent, under a header most
  clients never read. The verdict is remembered per (device, model) so the
  next request is refused before it generates.
- **The output is checked anyway.** If a pipeline accepted the config and
  ignored it (unverified on the NPU's static pipeline), a finished answer that
  does not parse is the evidence: the reply is marked `violated`, the slot is
  remembered as unable to constrain, and later requests get the 400.
"""

import json

import openvino_genai as ovg

# The failure half lives next door; re-exported so callers import one module.
from core.chat.structured_verdict import (  # noqa: F401
    classify_error, error_response, header_value, refusal, refusal_response,
    verdict, watch_stream)

# xgrammar's converter defaults additionalProperties to true when it is not
# strict (measured: a bare {"type":"object"} admitted arbitrary keys), but
# saying it explicitly keeps "any JSON object" true if that default flips.
ANY_OBJECT = {"type": "object", "additionalProperties": True}

# "Any JSON object" as a grammar rather than as the schema above, for one
# reason: xgrammar's schema converter admits unbounded whitespace between
# tokens, and a model that drifts into "\n" can then never be forced out of it
# -- measured, a constrained run that wrote `{"ok": true,` and 190 newlines
# until max_tokens. A schema leaves no knob for that (genai exposes none);
# this grammar allows one optional space after ':' and ',' and nothing else.
COMPACT_JSON_OBJECT_EBNF = r'''
root ::= object
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ( pair ( "," sp pair )* )? "}"
pair ::= string ":" sp value
array ::= "[" ( value ( "," sp value )* )? "]"
string ::= "\"" char* "\""
char ::= [^"\\\x00-\x1f] | "\\" ( ["\\/bfnrt] | "u" hex hex hex hex )
hex ::= [0-9a-fA-F]
number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [+-]? [0-9]+ )?
sp ::= " "?
'''

_JSON_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
_SUBSCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions",
                   "dependentSchemas")
_SUBSCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SUBSCHEMA_ONE = ("items", "additionalProperties", "not", "if", "then", "else",
                  "contains", "propertyNames", "unevaluatedProperties")


class StructuredRequestError(ValueError):
    """The request's format field is malformed: a 400, not a generation."""


class Spec:
    """What was asked for, and the schema text genai will compile."""

    def __init__(self, kind, schema, source):
        self.kind = kind            # "json_object" | "json_schema"
        self.schema = schema
        self.source = source        # field name, for error messages
        self.text = json.dumps(schema)

    def __repr__(self):
        return f"Spec({self.kind}, {self.text[:60]})"


def _check_schema(schema, where="schema"):
    """Catch the mistakes xgrammar would otherwise report from C++.

    Not a JSON Schema validator -- the compiler is the authority, and its
    errors are mapped to a 400 as well. This exists so the common mistakes
    (a typo'd type, a schema sent as a string) get a message that names the
    path instead of a jenkins source path.
    """
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise StructuredRequestError(
            f"{where} must be a JSON object (or true/false), got {type(schema).__name__}")
    kind = schema.get("type")
    if kind is not None:
        kinds = kind if isinstance(kind, list) else [kind]
        bad = [k for k in kinds if k not in _JSON_TYPES]
        if bad:
            raise StructuredRequestError(
                f"{where}.type {bad[0]!r} is not a JSON Schema type "
                f"(use one of {', '.join(sorted(_JSON_TYPES))})")
    ref = schema.get("$ref")
    if ref is not None and not (isinstance(ref, str) and ref.startswith("#")):
        raise StructuredRequestError(
            f"{where}.$ref {ref!r}: only local references ('#/...') can be "
            "resolved; this server fetches nothing")
    for key in _SUBSCHEMA_MAPS:
        sub = schema.get(key)
        if isinstance(sub, dict):
            for name, child in sub.items():
                _check_schema(child, f"{where}.{key}.{name}")
    for key in _SUBSCHEMA_LISTS:
        sub = schema.get(key)
        if isinstance(sub, list):
            for i, child in enumerate(sub):
                _check_schema(child, f"{where}.{key}[{i}]")
    for key in _SUBSCHEMA_ONE:
        if key in schema and isinstance(schema[key], (dict, bool)):
            _check_schema(schema[key], f"{where}.{key}")


def _schema_spec(schema, source):
    if isinstance(schema, str):
        # Some clients send the schema JSON-encoded. Accept it, but only if it
        # is actually JSON -- a prose string is a mistake to report.
        try:
            schema = json.loads(schema)
        except ValueError:
            raise StructuredRequestError(f"{source}: schema is a string that is not JSON")
    _check_schema(schema, f"{source}.schema")
    return Spec("json_schema", schema, source)


def from_openai(response_format):
    """OpenAI `response_format` -> Spec, or None for plain text."""
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise StructuredRequestError("response_format must be an object")
    kind = response_format.get("type")
    if kind in (None, "text"):
        return None
    if kind == "json_object":
        return Spec("json_object", ANY_OBJECT, "response_format")
    if kind == "json_schema":
        block = response_format.get("json_schema")
        if not isinstance(block, dict):
            raise StructuredRequestError(
                "response_format.json_schema must be an object with a 'schema'")
        if "schema" not in block:
            # OpenAI's own spec lets `schema` be omitted; the only honest
            # reading is "some JSON object", not "anything at all".
            return Spec("json_object", ANY_OBJECT, "response_format")
        return _schema_spec(block["schema"], "response_format.json_schema")
    raise StructuredRequestError(
        f"response_format.type {kind!r} is not supported "
        "(use 'text', 'json_object' or 'json_schema')")


def from_ollama(fmt):
    """Ollama `format` ("json" or a schema object) -> Spec, or None."""
    if fmt in (None, ""):
        return None
    if fmt == "json":
        return Spec("json_object", ANY_OBJECT, "format")
    if isinstance(fmt, (dict, str)):
        return _schema_spec(fmt, "format")
    raise StructuredRequestError("format must be \"json\" or a JSON Schema object")


def anthropic_to_response_format(body):
    """Anthropic structured outputs (`output_format`, or the newer
    `output_config.format`) -> an OpenAI response_format, or None."""
    fmt = body.get("output_format")
    if fmt is None and isinstance(body.get("output_config"), dict):
        fmt = body["output_config"].get("format")
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        return None
    return {"type": "json_schema", "json_schema": {"schema": fmt.get("schema", ANY_OBJECT)}}


def apply(gen, spec):
    """Attach the constraint to a GenerationConfig (None is a no-op).

    json_object takes the compact grammar, a real schema takes the schema.
    Measured on a random-weight model at temperature 1.5, 20 seeds each: the
    grammar finished 16/20 as valid objects, ANY_OBJECT as a schema 9/20 --
    the other runs spent max_tokens on whitespace. Every run that stopped on
    its own parsed, in both; the difference is only in how many get to stop.
    """
    if spec is None:
        return gen
    if spec.kind == "json_object":
        config = ovg.StructuredOutputConfig(grammar=COMPACT_JSON_OBJECT_EBNF)
    else:
        config = ovg.StructuredOutputConfig(json_schema=spec.text)
    gen.structured_output_config = config
    return gen
