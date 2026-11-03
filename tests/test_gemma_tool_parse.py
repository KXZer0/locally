"""Gemma 4 native tool calls: <|tool_call>call:NAME{k:<|"|>v<|"|>,n:3}<tool_call|>."""
import json
import unittest

from core.tools.parse import parse_tool_calls

TOOLS = [{"type": "function", "function": {
    "name": "web_search", "description": "search",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}]
Q = '<|"|>'


class GemmaToolParse(unittest.TestCase):
    def args(self, call):
        return json.loads(call["function"]["arguments"])

    def test_string_and_number_arguments(self):
        text = f"ok<|tool_call>call:web_search{{limit:5,query:{Q}OpenVINO, release{Q}}}<tool_call|>"
        content, calls = parse_tool_calls(text, TOOLS)
        self.assertEqual(content, "ok")
        self.assertEqual(calls[0]["function"]["name"], "web_search")
        self.assertEqual(self.args(calls[0]),
                         {"limit": 5, "query": "OpenVINO, release"})

    def test_nested_values_and_literals(self):
        text = (f"<|tool_call>call:f{{a:{{b:[1,2.5,{Q}x{Q}]}},ok:true,none:null}}"
                "<tool_call|>")
        _, calls = parse_tool_calls(text, TOOLS)
        self.assertEqual(self.args(calls[0]),
                         {"a": {"b": [1, 2.5, "x"]}, "ok": True, "none": None})

    def test_parallel_and_empty_arguments(self):
        text = ("<|tool_call>call:a{}<tool_call|><|tool_call>call:b{x:1}<tool_call|>")
        _, calls = parse_tool_calls(text, TOOLS)
        self.assertEqual([c["function"]["name"] for c in calls], ["a", "b"])
        self.assertEqual(self.args(calls[0]), {})

    def test_plain_text_is_untouched(self):
        content, calls = parse_tool_calls("no tools here", TOOLS)
        self.assertEqual((content, calls), ("no tools here", []))


if __name__ == "__main__":
    unittest.main()
