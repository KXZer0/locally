"""Liquid LFM2/LFM2.5 emit Pythonic tool calls between special tokens."""
import json
import unittest

from core.tools.parse import parse_tool_calls

TOOLS = [{"type": "function", "function": {
    "name": "web_search", "description": "search",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string"},
                                  "limit": {"type": "integer"}}}}}]


class LfmToolParse(unittest.TestCase):
    def test_single_call_with_keyword_args(self):
        text = ("<think>look it up</think>\n<|tool_call_start|>"
                "[web_search(query='OpenVINO release notes', limit=3)]"
                "<|tool_call_end|>")
        content, calls = parse_tool_calls(text, TOOLS)
        self.assertEqual(len(calls), 1)
        fn = calls[0]["function"]
        self.assertEqual(fn["name"], "web_search")
        self.assertEqual(json.loads(fn["arguments"]),
                         {"query": "OpenVINO release notes", "limit": 3})
        self.assertNotIn("tool_call", content)

    def test_parallel_calls(self):
        text = ('<|tool_call_start|>[web_search(query="a"), '
                'web_search(query="b")]<|tool_call_end|>')
        _, calls = parse_tool_calls(text, TOOLS)
        self.assertEqual([json.loads(c["function"]["arguments"])["query"]
                          for c in calls], ["a", "b"])

    def test_malformed_block_is_left_as_text(self):
        text = "<|tool_call_start|>[web_search(query=]<|tool_call_end|>"
        _, calls = parse_tool_calls(text, TOOLS)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
