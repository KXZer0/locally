"""Gemma 4 native tool prompt / thought channel / drifted Qwen-style calls.

The raw strings below are captured outputs, not invented ones: the
`<parameter>` JSON forms from Hermes + Gemma 4 E4B runs against the Qwen-style
prompt (each used to parse to {} and go out as a call with no arguments), and
the `<|tool_call>` / `<|channel>` forms from the model's own dialect.
"""
import json
import os
import unittest

from core.chat.common import _tool_message
from core.tools.gemma import (GemmaChat, normalize_messages, simplify_tools,
                              split_reasoning, thinking_requested)
from core.tools.parse import parse_tool_calls

Q = '<|"|>'
TOOLS = [
    {"type": "function", "function": {
        "name": "mcp__scrapling__browser_fetch", "description": "fetch",
        "parameters": {"type": "object", "required": ["url", "session_id"], "properties": {
            "url": {"type": "string"}, "session_id": {"type": "string"},
            "main_content_only": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "mcp__scrapling__browser_open", "description": "open",
        "parameters": {"type": "object", "properties": {
            "session_id": {"anyOf": [{"type": "string"}, {"type": "null"}],
                           "default": None, "description": "Custom ID"}}}}},
]
MODEL_DIR = os.path.join(os.path.expanduser("~"), "models", "gemma-4-E4B-it-int4-ov")


def args(call):
    return json.loads(call["function"]["arguments"])


class DriftedQwenCalls(unittest.TestCase):
    """Captured from Gemma 4 E4B a few steps into a Hermes run."""

    def test_unnamed_parameter_holding_a_json_object(self):
        raw = ('<tool_call>\n<function=mcp__scrapling__browser_fetch>\n<parameter>\n'
               '{"session_id": "81b213a019c1", "url": "https://httpbin.org/forms/post", '
               '"main_content_only": true}\n</parameter>\n</function>\n</tool_call>')
        _, calls = parse_tool_calls(raw, TOOLS)
        self.assertEqual(args(calls[0]), {
            "session_id": "81b213a019c1", "url": "https://httpbin.org/forms/post",
            "main_content_only": True})

    def test_text_before_an_unnamed_parameter_call(self):
        raw = ('The previous calls failed due to missing arguments.\n\n<tool_call>\n'
               '<function=mcp__scrapling__browser_fetch>\n<parameter>\n'
               '{"url": "https://httpbin.org/forms/post", "session_id": "81b2"}\n'
               '</parameter>\n</function>\n</tool_call>')
        content, calls = parse_tool_calls(raw, TOOLS)
        self.assertEqual(args(calls[0])["url"], "https://httpbin.org/forms/post")
        self.assertIn("previous calls failed", content)

    def test_parameter_name_attribute_and_child_tags(self):
        a = ('<tool_call><function=mcp__scrapling__browser_fetch>'
             '<parameter name="url">https://example.com</parameter>'
             '<parameter name="session_id">abc</parameter></function></tool_call>')
        b = ('<function=mcp__scrapling__browser_fetch><parameter><url>https://example.com'
             '</url><session_id>abc</session_id></parameter></function>')
        for raw in (a, b):
            _, calls = parse_tool_calls(raw, TOOLS)
            self.assertEqual(args(calls[0]), {"url": "https://example.com", "session_id": "abc"})

    def test_the_requested_form_still_parses(self):
        raw = ('<tool_call>\n<function=mcp__scrapling__browser_fetch>\n'
               '<parameter=url>https://example.com</parameter>\n'
               '<parameter=session_id>52847d5c34d2</parameter>\n</function>\n</tool_call>')
        _, calls = parse_tool_calls(raw, TOOLS)
        self.assertEqual(args(calls[0]), {"url": "https://example.com",
                                          "session_id": "52847d5c34d2"})

    def test_a_really_empty_call_stays_empty(self):
        raw = '<tool_call>\n<function=mcp__scrapling__browser_open>\n</function>\n</tool_call>'
        _, calls = parse_tool_calls(raw, TOOLS)
        self.assertEqual(args(calls[0]), {})


class NativeOutput(unittest.TestCase):
    def test_tool_call_ending_in_tool_response_token(self):
        raw = ('<|tool_call>call:mcp__scrapling__browser_fetch{session_id:' + Q +
               '52847d5c34d2' + Q + ',url:' + Q + 'https://example.com' + Q +
               '}<tool_call|><|tool_response>')
        message, calls = _tool_message(raw, TOOLS)
        self.assertEqual(args(calls[0]), {"session_id": "52847d5c34d2",
                                          "url": "https://example.com"})
        self.assertIsNone(message["content"])
        self.assertNotIn("reasoning_content", message)

    def test_thought_channel_becomes_reasoning_content(self):
        raw = ("<|channel>thought\n1. **Analyze** the request.\n2. The `f` tool "
               "needs `city`.<channel|><|tool_call>call:f{city:" + Q + "Paris" + Q +
               "}<tool_call|><eos>")
        message, calls = _tool_message(raw, TOOLS)
        self.assertEqual(calls[0]["function"]["name"], "f")
        self.assertEqual(args(calls[0]), {"city": "Paris"})
        self.assertTrue(message["reasoning_content"].startswith("1. **Analyze**"))
        self.assertIsNone(message["content"])

    def test_answer_after_thought_keeps_thought_out_of_content(self):
        raw = ("<|channel>thought\nThe tool returned cloudy.<channel|>"
               "The weather in Paris is cloudy.<turn|>")
        message, calls = _tool_message(raw, TOOLS)
        self.assertEqual(calls, [])
        self.assertEqual(message["content"], "The weather in Paris is cloudy.")
        self.assertEqual(message["reasoning_content"], "The tool returned cloudy.")

    def test_unclosed_thought_is_all_reasoning(self):
        reasoning, rest = split_reasoning("<|channel>thought\nstill thinking when the budget ran")
        self.assertEqual(reasoning, "still thinking when the budget ran")
        self.assertEqual(rest, "")

    def test_plain_text_is_untouched(self):
        self.assertEqual(split_reasoning("Hello there."), ("", "Hello there."))

    def test_string_arguments_with_delimiters_survive(self):
        raw = ("<|tool_call>call:f{q:" + Q + "a, b} c:{d}" + Q + ",n:3}<tool_call|>")
        _, calls = parse_tool_calls(raw, TOOLS)
        self.assertEqual(args(calls[0]), {"q": "a, b} c:{d}", "n": 3})


class ThinkingRequest(unittest.TestCase):
    def test_off_by_default_and_for_none(self):
        self.assertFalse(thinking_requested({}))
        self.assertFalse(thinking_requested({"reasoning_effort": "none"}))
        self.assertFalse(thinking_requested({"reasoning_effort": ""}))
        self.assertFalse(thinking_requested({"thinking": {"type": "disabled"}}))

    def test_the_ways_a_client_asks(self):
        self.assertTrue(thinking_requested({"reasoning_effort": "low"}))
        self.assertTrue(thinking_requested({"reasoning_effort": "medium"}))
        self.assertTrue(thinking_requested({"reasoning": {"effort": "high"}}))
        self.assertTrue(thinking_requested({"enable_thinking": True}))
        self.assertTrue(thinking_requested({"chat_template_kwargs": {"enable_thinking": True}}))
        self.assertTrue(thinking_requested({"thinking": {"type": "enabled", "budget_tokens": 1000}}))
        self.assertFalse(thinking_requested({"reasoning_effort": "high",
                                             "chat_template_kwargs": {"enable_thinking": False}}))


class SchemaSimplifier(unittest.TestCase):
    def test_refs_nullable_and_tagged_unions(self):
        tool = {"type": "function", "function": {"name": "acts", "description": "d", "parameters": {
            "type": "object", "required": ["session_id", "actions"],
            "properties": {
                "session_id": {"type": "string"},
                "proxy": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                "actions": {"type": "array", "description": "Ordered actions", "items": {
                    "oneOf": [{"$ref": "#/$defs/_Click"}, {"$ref": "#/$defs/_Text"}],
                    "discriminator": {"propertyName": "type"}}}},
            "$defs": {
                "_Target": {"anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "selector"},
                "_Click": {"type": "object", "required": ["type"], "properties": {
                    "type": {"const": "click", "type": "string"},
                    "target": {"$ref": "#/$defs/_Target"}}},
                "_Text": {"type": "object", "required": ["type", "value"], "properties": {
                    "type": {"const": "textbox", "type": "string"},
                    "target": {"$ref": "#/$defs/_Target"},
                    "value": {"type": "string"}}}}}}}
        out = simplify_tools([tool])[0]["function"]["parameters"]
        self.assertNotIn("$defs", json.dumps(out))
        self.assertNotIn("$ref", json.dumps(out))
        self.assertNotIn("anyOf", json.dumps(out))
        self.assertEqual(out["properties"]["proxy"]["type"], "string")
        item = out["properties"]["actions"]["items"]
        self.assertEqual(item["type"], "object")
        self.assertEqual(item["properties"]["type"]["enum"], ["click", "textbox"])
        self.assertEqual(item["properties"]["target"]["description"], "selector")
        self.assertIn("textbox(value", item["description"])
        self.assertEqual(out["required"], ["session_id", "actions"])


class Messages(unittest.TestCase):
    def test_arguments_become_mappings_and_content_text(self):
        msgs = normalize_messages([
            {"role": "system", "content": "S"}, {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "reasoning_content": "hm",
             "tool_calls": [{"id": "c1", "type": "function", "function": {
                 "name": "f", "arguments": '{"a": 1}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "RES"},
            {"role": "system", "content": "late system"}])
        self.assertEqual(msgs[2]["content"], "")
        self.assertEqual(msgs[2]["tool_calls"][0]["function"]["arguments"], {"a": 1})
        self.assertEqual(msgs[2]["reasoning_content"], "hm")
        self.assertEqual(msgs[4]["role"], "user")


@unittest.skipUnless(os.path.isfile(os.path.join(MODEL_DIR, "chat_template.jinja")),
                     "Gemma 4 E4B is not on this machine")
class RealTemplate(unittest.TestCase):
    def setUp(self):
        self.chat = GemmaChat.for_model(MODEL_DIR)
        self.assertIsNotNone(self.chat)

    def test_tool_turn_is_rendered_in_the_native_dialect(self):
        prompt = self.chat.render([
            {"role": "system", "content": "SYS"}, {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function",
                "function": {"name": "f", "arguments": '{"city": "Paris"}'}}]},
            {"role": "tool", "tool_call_id": "c", "content": "cloudy"}],
            [{"type": "function", "function": {"name": "f", "description": "d",
              "parameters": {"type": "object", "required": ["city"],
                             "properties": {"city": {"type": "string"}}}}}], False)
        self.assertTrue(prompt.startswith("<bos><|turn>system\nSYS<|tool>declaration:f{"))
        self.assertIn("<|tool_call>call:f{city:" + Q + "Paris" + Q + "}<tool_call|>"
                      "<|tool_response>response:f{value:" + Q + "cloudy" + Q + "}<tool_response|>",
                      prompt)
        self.assertNotIn("<|think|>", prompt)
        self.assertFalse(prompt.endswith("<|turn>model\n"))      # the model turn continues

    def test_thinking_switch(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.assertIn("<|think|>", self.chat.render(msgs, [], True))
        self.assertNotIn("<|think|>", self.chat.render(msgs, [], False))
        self.assertTrue(self.chat.render(msgs, [], False).endswith("<|turn>model\n"))


if __name__ == "__main__":
    unittest.main()
