"""Tool calls constrained with genai structural tags: choice, fallback, verdict.

What xgrammar does with the tags was measured on a real CPU pipeline (see
CLAUDE.md); here the slot is a fake and the subject is how a request's tools
and tool_choice become a config, and how a failing config steps down instead
of failing the turn.
"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

import openvino_genai as ovg

from core import config
from core.app import create_app, create_ollama_app
from core.genai.results import GeneratedText
from core.tools import constrain
from core.tools.parse import parse_tool_calls

WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"unit": {"enum": ["c", "f"]}, "days": {"type": "integer"}},
    "required": ["unit", "days"], "additionalProperties": False}}}
PING = {"type": "function", "function": {"name": "ping", "parameters": {}}}
TOOLS = [WEATHER, PING]

TOOLSET_ERROR = RuntimeError(
    "xgrammar-src/cpp/grammar_parser.cc:552: EBNF parser error at line 16, column 1")
DEVICE_ERROR = RuntimeError("Structured output is not supported by this pipeline")


def slot(device="CPU", model="m"):
    return SimpleNamespace(device_name=device, model_name=model)


def tags_of(cfg):
    return repr(cfg.structural_tags_config)


class Reset(unittest.TestCase):
    def setUp(self):
        constrain._device_off.clear()
        constrain._toolset_level.clear()
        self.addCleanup(constrain._device_off.clear)
        self.addCleanup(constrain._toolset_level.clear)


class ChoiceTests(Reset):
    def test_tool_choice_shapes(self):
        names = ["get_weather", "ping"]
        r = constrain.resolve_choice
        self.assertEqual(r(None, names), ("auto", names))
        self.assertEqual(r("auto", names), ("auto", names))
        self.assertEqual(r("none", names), ("none", []))
        self.assertEqual(r("required", names), ("required", names))
        self.assertEqual(r({"type": "function", "function": {"name": "ping"}}, names),
                         ("required", ["ping"]))
        # Anthropic's shapes
        self.assertEqual(r({"type": "any"}, names), ("required", names))
        self.assertEqual(r({"type": "tool", "name": "get_weather"}, names),
                         ("required", ["get_weather"]))
        self.assertEqual(r({"type": "none"}, names), ("none", []))
        # A function that was never declared cannot be forced; stay on auto.
        self.assertEqual(r({"type": "function", "function": {"name": "nope"}}, names),
                         ("auto", names))

    def test_auto_builds_both_tag_shapes_for_every_tool(self):
        tc = constrain.build(slot(), TOOLS)
        text = tags_of(tc.config())
        for begin in ("<function=get_weather>", '"name": "get_weather"',
                      "<function=ping>", '"name": "ping"'):
            self.assertIn(begin, text)
        self.assertIn("at_least_one=False", text)
        self.assertIn('triggers=["<tool_call>"]', text)
        # A tool with no properties gets an empty-allowing body, not QwenXML,
        # whose grammar for zero parameters does not compile.
        self.assertIn('Regex("\\n?")', text)

    def test_required_and_named_function_force_a_call(self):
        self.assertIn("at_least_one=True", tags_of(constrain.build(slot(), TOOLS, "required").config()))
        one = tags_of(constrain.build(slot(), TOOLS, {"type": "function",
                                                      "function": {"name": "ping"}}).config())
        self.assertIn("<function=ping>", one)
        self.assertNotIn("get_weather", one)

    def test_when_nothing_is_added(self):
        self.assertIsNone(constrain.build(slot(), TOOLS, "none"))
        self.assertIsNone(constrain.build(slot(), TOOLS, native=True))     # Gemma dialect
        self.assertIsNone(constrain.build(slot(), []))
        with mock.patch.object(config, "CONSTRAIN_TOOLS", False):
            self.assertIsNone(constrain.build(slot(), TOOLS))

        class ProxySlot:
            device_name, model_name = "REMOTE", "m"
        self.assertIsNone(constrain.build(ProxySlot(), TOOLS))


class FallbackTests(Reset):
    def test_toolset_errors_step_down_and_are_remembered(self):
        gen = ovg.GenerationConfig()
        tc = constrain.build(slot(), TOOLS)
        tc.apply(gen)
        self.assertIn("QwenXMLParametersFormat", tags_of(gen.structured_output_config))
        self.assertTrue(tc.failed(gen, str(TOOLSET_ERROR)))
        self.assertEqual(constrain.LEVELS[tc.level], "simplified")
        self.assertTrue(tc.failed(gen, str(TOOLSET_ERROR)))
        self.assertEqual(constrain.LEVELS[tc.level], "name")
        self.assertIn("AnyText", tags_of(gen.structured_output_config))
        self.assertTrue(tc.failed(gen, str(TOOLSET_ERROR)))
        self.assertIsNone(gen.structured_output_config)          # unconstrained, not a 400
        self.assertFalse(tc.failed(gen, str(TOOLSET_ERROR)))     # nothing left to drop
        # The same tool set on the same slot does not pay for it again...
        self.assertIsNone(constrain.build(slot(), TOOLS))
        # ...but a different tool set, or another model, starts fresh.
        self.assertIsNotNone(constrain.build(slot(), [WEATHER]))
        self.assertIsNotNone(constrain.build(slot(model="other"), TOOLS))

    def test_device_errors_turn_the_slot_off(self):
        gen = ovg.GenerationConfig()
        tc = constrain.build(slot("NPU"), TOOLS)
        self.assertTrue(tc.failed(gen, str(DEVICE_ERROR)))
        self.assertIsNone(gen.structured_output_config)
        self.assertIsNone(constrain.build(slot("NPU"), [WEATHER]))
        self.assertIsNotNone(constrain.build(slot("GPU"), TOOLS))

    def test_other_errors_are_not_ours(self):
        tc = constrain.build(slot(), TOOLS)
        self.assertFalse(tc.failed(ovg.GenerationConfig(), "Got unfinished GenerationStatus"))

    def test_run_retries_until_it_compiles(self):
        gen = ovg.GenerationConfig()
        tc = constrain.build(slot(), TOOLS)
        tc.apply(gen)
        calls = []

        def fn():
            calls.append(gen.structured_output_config)
            if len(calls) == 1:
                raise TOOLSET_ERROR
            return "ok"
        self.assertEqual(tc.run(gen, fn), "ok")
        self.assertEqual(len(calls), 2)
        with self.assertRaises(RuntimeError):
            tc.run(gen, mock.Mock(side_effect=RuntimeError("out of memory")))

    def test_an_ignored_constraint_is_noticed(self):
        tc = constrain.build(slot(), TOOLS)
        tc.judge("<tool_call>\n<function=get_weather>\n...", [], "stop")
        self.assertIsNone(constrain.build(slot(), TOOLS))

    def test_a_truncated_or_valid_call_is_not_evidence(self):
        tc = constrain.build(slot(), TOOLS)
        tc.judge("<tool_call>\n<function=get_we", [], "length")
        tc.judge("<tool_call>...", [{"function": {"name": "ping"}}], "stop")
        self.assertIsNotNone(constrain.build(slot(), TOOLS))


class ParseTests(unittest.TestCase):
    def test_constraint_output_parses_to_schema_types(self):
        # Exactly what QwenXMLParametersFormat produced on a CPU pipeline.
        text = ('<tool_call>\n<function=get_weather>\n<parameter=unit>\n\t "f"\t\n</parameter>\n'
                '<parameter=days>-91</parameter>\n</function>\n</tool_call>')
        _content, calls = parse_tool_calls(text, TOOLS)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"unit": "f", "days": -91})

    def test_type_lists_and_untyped_values(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"properties": {
            "a": {"type": ["string", "null"]}, "b": {"anyOf": [{"type": "integer"}]},
            "c": {"type": ["integer", "string"]}}}}}]
        text = ("<tool_call>\n<function=f>\n<parameter=a>null</parameter>\n"
                "<parameter=b>7</parameter>\n<parameter=c>seven</parameter>\n</function>\n</tool_call>")
        _content, calls = parse_tool_calls(text, tools)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"a": None, "b": 7, "c": "seven"})


class FakeSlot:
    model_name = "fixture-8b"
    model_type = "llm"
    device_name = "GPU"

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


class EndpointTests(Reset):
    def _chat(self, slot, body, path="/v1/chat/completions"):
        body = {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS, **body}
        with (mock.patch("core.chat.turn.overall_status", return_value="ready"),
              mock.patch("core.chat.turn._route_request", return_value=slot),
              mock.patch("core.chat.turn._tool_capable", return_value=True),
              mock.patch("core.chat.turn.builtin_tools_for", mock.Mock(return_value=[])),
              mock.patch("core.chat.ollama.overall_status", return_value="ready"),
              mock.patch("core.chat.ollama._route_request", return_value=slot),
              mock.patch("core.chat.ollama._tool_capable", return_value=True)):
            client = (create_ollama_app() if path.startswith("/api/") else create_app()).test_client()
            return client.post(path, json=body)

    def test_compile_failure_falls_back_instead_of_failing(self):
        call = '<tool_call>\n<function=ping>\n</function>\n</tool_call>'
        s = FakeSlot([TOOLSET_ERROR, GeneratedText(call, "stop")])
        r = self._chat(s, {"tool_choice": "required"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["choices"][0]["finish_reason"], "tool_calls")
        self.assertIn("at_least_one=True", tags_of(s.configs[0]))
        # Retried one level down -- still constrained, still forced.
        self.assertIsNotNone(s.configs[1])
        self.assertIn("at_least_one=True", tags_of(s.configs[1]))
        self.assertEqual(len(s.configs), 2)

    def test_tool_choice_none_adds_nothing(self):
        s = FakeSlot([GeneratedText("hello", "stop")])
        self.assertEqual(self._chat(s, {"tool_choice": "none"}).status_code, 200)
        self.assertIsNone(s.configs[0])

    def test_flag_off_adds_nothing(self):
        s = FakeSlot([GeneratedText("hello", "stop")])
        with mock.patch.object(config, "CONSTRAIN_TOOLS", False):
            self._chat(s, {})
        self.assertIsNone(s.configs[0])

    def test_anthropic_tool_choice_reaches_the_constraint(self):
        s = FakeSlot([GeneratedText('<tool_call>\n<function=ping>\n</function>\n</tool_call>', "stop")])
        r = self._chat(s, {"model": "fixture-8b", "max_tokens": 50,
                           "tools": [{"name": "ping", "input_schema": {"type": "object"}}],
                           "tool_choice": {"type": "tool", "name": "ping"}}, path="/v1/messages")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["stop_reason"], "tool_use")
        self.assertIn("at_least_one=True", tags_of(s.configs[0]))

    def test_ollama_tool_turn_is_constrained(self):
        s = FakeSlot([GeneratedText("hello", "stop")])
        r = self._chat(s, {"stream": False, "model": "fixture-8b"}, path="/api/chat")
        self.assertEqual(r.status_code, 200)
        self.assertIn("<function=get_weather>", tags_of(s.configs[0]))
