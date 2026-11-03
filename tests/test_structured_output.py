"""Structured output: request mapping, refusals, and the verdict on the answer.

The slot is a fake: what is under test is how a format field reaches
GenerationConfig and how genai's failures reach the client, not xgrammar
itself (that was measured on a real pipeline -- see CLAUDE.md).
"""

import json
import unittest
from unittest import mock

import openvino_genai as ovg

from core.app import create_app, create_ollama_app
from core.chat import structured, structured_verdict
from core.genai.results import GeneratedText
from core.slots.proxy import _gen_to_openai

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}

# What genai 2026.4 actually raised, measured on a CPU pipeline.
SCHEMA_ERROR = RuntimeError(
    "[22:50:12] /home/jenkins/.../xgrammar-src/cpp/json_schema_converter.cc:957: "
    "Check failed: (...) is false: Cannot find field defs in {\"$ref\":\"#/defs/x\"}")
DEVICE_ERROR = RuntimeError("Structured output is not supported by this pipeline")


class FakeSlot:
    model_name = "fixture-8b"
    model_type = "llm"
    device_name = "NPU"

    def __init__(self, answers):
        self.answers = list(answers)
        self.configs = []

    def native_chat(self):
        return None

    def ensure_loaded(self):
        pass

    def generate_llm(self, raw_messages, gen, record_metric=True):
        self.configs.append(gen.structured_output_config)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class MappingTests(unittest.TestCase):
    def test_openai_shapes(self):
        self.assertIsNone(structured.from_openai(None))
        self.assertIsNone(structured.from_openai({"type": "text"}))
        spec = structured.from_openai({"type": "json_schema",
                                       "json_schema": {"name": "x", "schema": SCHEMA}})
        self.assertEqual((spec.kind, spec.schema), ("json_schema", SCHEMA))
        self.assertEqual(structured.from_openai({"type": "json_object"}).kind, "json_object")
        # OpenAI lets `schema` be omitted; that is "some object", not "anything".
        self.assertEqual(structured.from_openai(
            {"type": "json_schema", "json_schema": {"name": "x"}}).kind, "json_object")

    def test_ollama_and_anthropic_shapes(self):
        self.assertIsNone(structured.from_ollama(None))
        self.assertEqual(structured.from_ollama("json").kind, "json_object")
        self.assertEqual(structured.from_ollama(SCHEMA).schema, SCHEMA)
        self.assertEqual(structured.from_ollama(json.dumps(SCHEMA)).schema, SCHEMA)
        rf = structured.anthropic_to_response_format(
            {"output_format": {"type": "json_schema", "schema": SCHEMA}})
        self.assertEqual(structured.from_openai(rf).schema, SCHEMA)
        self.assertIsNone(structured.anthropic_to_response_format({}))

    def test_malformed_requests_name_the_path(self):
        bad = [({"type": "xml"}, "response_format.type"),
               ({"type": "json_schema", "json_schema": {"schema": {
                   "type": "object", "properties": {"a": {"type": "str"}}}}},
                "properties.a.type"),
               ({"type": "json_schema", "json_schema": {"schema": {
                   "$ref": "https://example.com/s.json"}}}, "only local references"),
               ({"type": "json_schema", "json_schema": {"schema": "prose"}}, "not JSON"),
               ("json_object", "must be an object")]
        for fmt, needle in bad:
            with self.assertRaises(structured.StructuredRequestError) as cm:
                structured.from_openai(fmt)
            self.assertIn(needle, str(cm.exception), fmt)
        with self.assertRaises(structured.StructuredRequestError):
            structured.from_ollama(["json"])

    def test_apply_uses_grammar_for_any_object_and_schema_otherwise(self):
        gen = structured.apply(ovg.GenerationConfig(), structured.from_ollama("json"))
        self.assertEqual(gen.structured_output_config.grammar,
                         structured.COMPACT_JSON_OBJECT_EBNF)
        self.assertIsNone(gen.structured_output_config.json_schema)
        gen = structured.apply(ovg.GenerationConfig(), structured.from_ollama(SCHEMA))
        self.assertEqual(json.loads(gen.structured_output_config.json_schema), SCHEMA)
        self.assertIsNone(structured.apply(ovg.GenerationConfig(), None)
                          .structured_output_config)

    def test_proxy_forwards_the_constraint(self):
        gen = structured.apply(ovg.GenerationConfig(), structured.from_ollama(SCHEMA))
        body = _gen_to_openai(gen, "m", [], False)
        self.assertEqual(body["response_format"]["json_schema"]["schema"], SCHEMA)
        gen = structured.apply(ovg.GenerationConfig(), structured.from_ollama("json"))
        self.assertEqual(_gen_to_openai(gen, "m", [], False)["response_format"],
                         {"type": "json_object"})
        self.assertNotIn("response_format", _gen_to_openai(ovg.GenerationConfig(), "m", [], False))


class EndpointTests(unittest.TestCase):
    def setUp(self):
        structured_verdict._unsupported.clear()
        self.addCleanup(structured_verdict._unsupported.clear)
        self.client = create_app().test_client()
        self.ollama = create_ollama_app().test_client()

    def _chat(self, slot, body, path="/v1/chat/completions", tools_ok=False, builtin=None):
        body = {"messages": [{"role": "user", "content": "hi"}], **body}
        builtin = builtin or mock.Mock(return_value=[])
        with (mock.patch("core.chat.turn.overall_status", return_value="ready"),
              mock.patch("core.chat.turn._route_request", return_value=slot),
              mock.patch("core.chat.turn._tool_capable", return_value=tools_ok),
              mock.patch("core.chat.turn.builtin_tools_for", builtin),
              mock.patch("core.chat.ollama.overall_status", return_value="ready"),
              mock.patch("core.chat.ollama._route_request", return_value=slot),
              mock.patch("core.chat.ollama_generate.overall_status", return_value="ready"),
              mock.patch("core.chat.ollama_generate._route_request", return_value=slot)):
            client = self.ollama if path.startswith("/api/") else self.client
            return client.post(path, json=body)

    def test_schema_reaches_generation_and_answer_is_judged(self):
        slot = FakeSlot([GeneratedText('{"ok": true}', "stop")])
        builtin = mock.Mock(return_value=[])
        r = self._chat(slot, {"response_format": {"type": "json_schema",
                                                  "json_schema": {"schema": SCHEMA}}},
                       builtin=builtin)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["X-Structured-Output"], "constrained")
        self.assertEqual(json.loads(slot.configs[0].json_schema), SCHEMA)
        # The server's own tools are not offered: the grammar forbids calling them.
        builtin.assert_not_called()

    def test_plain_turn_is_untouched(self):
        slot = FakeSlot([GeneratedText("hello", "stop")])
        r = self._chat(slot, {})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("X-Structured-Output", r.headers)
        self.assertIsNone(slot.configs[0])

    def test_json_object_and_truncation(self):
        slot = FakeSlot([GeneratedText('{"a": [1, 2', "length")])
        r = self._chat(slot, {"response_format": {"type": "json_object"}})
        self.assertEqual(r.headers["X-Structured-Output"], "truncated")
        self.assertEqual(slot.configs[0].grammar, structured.COMPACT_JSON_OBJECT_EBNF)
        self.assertIsNone(structured.refusal(slot))      # length is not a violation

    def test_invalid_schema_is_a_400_before_generation(self):
        slot = FakeSlot([])
        r = self._chat(slot, {"response_format": {"type": "json_schema", "json_schema": {
            "schema": {"type": "nonsense"}}}})
        self.assertEqual(r.status_code, 400)
        self.assertIn("'nonsense' is not a JSON Schema type", r.json["error"]["message"])
        self.assertEqual(slot.configs, [])

    def test_schema_the_compiler_rejects_is_a_400_not_a_500(self):
        slot = FakeSlot([SCHEMA_ERROR])
        r = self._chat(slot, {"response_format": {"type": "json_schema", "json_schema": {
            "schema": {"$ref": "#/defs/x"}}}})
        self.assertEqual(r.status_code, 400)
        message = r.json["error"]["message"]
        self.assertIn("Invalid JSON schema", message)
        self.assertIn("Cannot find field defs", message)
        self.assertNotIn("jenkins", message)
        # The request's fault, not the device's: the slot stays usable.
        self.assertIsNone(structured.refusal(slot))

    def test_device_that_cannot_constrain_is_refused_and_remembered(self):
        slot = FakeSlot([DEVICE_ERROR, GeneratedText("not reached")])
        fmt = {"response_format": {"type": "json_object"}}
        r = self._chat(slot, fmt)
        self.assertEqual(r.status_code, 400)
        self.assertIn("NPU (fixture-8b) cannot constrain output", r.json["error"]["message"])
        # Second request: refused before it generates, never retried unconstrained.
        r = self._chat(slot, fmt)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(len(slot.configs), 1)
        # Plain requests on the same slot still work.
        self.assertEqual(self._chat(slot, {}).status_code, 200)

    def test_silently_ignored_constraint_is_caught_by_the_verdict(self):
        slot = FakeSlot([GeneratedText("Sure! Here is JSON: {", "stop")])
        fmt = {"response_format": {"type": "json_object"}}
        r = self._chat(slot, fmt)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["X-Structured-Output"], "violated")
        self.assertEqual(self._chat(slot, fmt).status_code, 400)

    def test_tool_turn_keeps_tools_and_skips_format(self):
        slot = FakeSlot([GeneratedText("plain answer", "stop")])
        slot.device_name = "GPU"
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        r = self._chat(slot, {"tools": tools, "response_format": {"type": "json_object"}},
                       tools_ok=True)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["X-Structured-Output"], "skipped-tools")
        self.assertIsNone(slot.configs[0])

    def test_ollama_format(self):
        slot = FakeSlot([GeneratedText('{"ok": false}', "stop"),
                         GeneratedText('{"ok": true}', "stop")])
        r = self._chat(slot, {"format": SCHEMA, "stream": False}, path="/api/chat")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(json.loads(slot.configs[0].json_schema), SCHEMA)
        r = self._chat(slot, {"prompt": "x", "format": "json", "stream": False},
                       path="/api/generate")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(slot.configs[1].grammar, structured.COMPACT_JSON_OBJECT_EBNF)
        r = self._chat(slot, {"format": 5, "stream": False}, path="/api/chat")
        self.assertEqual(r.status_code, 400)

    def test_ollama_device_refusal(self):
        slot = FakeSlot([DEVICE_ERROR])
        r = self._chat(slot, {"format": "json", "stream": False}, path="/api/chat")
        self.assertEqual(r.status_code, 400)
        self.assertIn("cannot constrain", r.json["error"])
        r = self._chat(slot, {"prompt": "x", "format": "json", "stream": False},
                       path="/api/generate")
        self.assertEqual(r.status_code, 400)


class StreamTests(unittest.TestCase):
    def setUp(self):
        structured_verdict._unsupported.clear()
        self.addCleanup(structured_verdict._unsupported.clear)

    @staticmethod
    def _frames(pieces, finish):
        for p in pieces:
            yield "data: " + json.dumps({"choices": [{"delta": {"content": p},
                                                      "finish_reason": None}]}) + "\n\n"
        yield "data: " + json.dumps({"choices": [{"delta": {},
                                                  "finish_reason": finish}]}) + "\n\n"
        yield "data: [DONE]\n\n"

    def _watch(self, pieces, finish):
        slot = FakeSlot([])
        turn = {"slot": slot, "structured": structured.from_ollama("json")}
        frames = list(structured.watch_stream(self._frames(pieces, finish), turn))
        self.assertEqual(frames[-1], "data: [DONE]\n\n")    # passed through untouched
        return slot

    def test_valid_stream_leaves_slot_alone(self):
        self.assertIsNone(structured.refusal(self._watch(['{"a"', ": 1}"], "stop")))

    def test_unparsable_finished_stream_marks_slot(self):
        self.assertIsNotNone(structured.refusal(self._watch(["hello"], "stop")))

    def test_device_error_frame_marks_slot(self):
        slot = self._watch(["\n[error: xgrammar backend unavailable on NPU]"], "error")
        self.assertIn("xgrammar", structured.refusal(slot))


if __name__ == "__main__":
    unittest.main()
