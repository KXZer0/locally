import unittest

from core import config
from core.chat.anthropic import _anthropic_to_openai
from core.chat.turn import _with_no_think


class AnthropicAdapterTests(unittest.TestCase):
    def setUp(self):
        self._map = config.ANTHROPIC_MODEL_MAP
        config.ANTHROPIC_MODEL_MAP = {}

    def tearDown(self):
        config.ANTHROPIC_MODEL_MAP = self._map

    @staticmethod
    def _req(**kw):
        return dict({"model": "claude-opus-5", "max_tokens": 50,
                     "messages": [{"role": "user", "content": "hi"}]}, **kw)

    def test_server_tools_are_not_client_tools(self):
        out = _anthropic_to_openai(self._req(tools=[
            {"type": "web_search_20260209", "name": "web_search", "max_uses": 5},
            {"type": "code_execution_20250825", "name": "code_execution"},
            {"name": "open_app", "description": "d", "input_schema": {}}]))
        self.assertEqual([t["function"]["name"] for t in out["tools"]], ["open_app"])
        self.assertTrue(out["web_search"])

    def test_only_server_tools_leaves_no_tools(self):
        out = _anthropic_to_openai(self._req(tools=[
            {"type": "web_search_20260209", "name": "web_search"}]))
        self.assertNotIn("tools", out)

    def test_model_mapping(self):
        self.assertEqual(_anthropic_to_openai(self._req())["model"], "")
        config.ANTHROPIC_MODEL_MAP = {"opus": "gemma-4-26b-a4b-it@GPU"}
        self.assertEqual(_anthropic_to_openai(self._req())["model"],
                         "gemma-4-26b-a4b-it@GPU")
        out = _anthropic_to_openai(self._req(model="claude-haiku-5"))
        self.assertEqual(out["model"], "")
        out = _anthropic_to_openai(self._req(model="Qwen3-8B-int4-cw"))
        self.assertEqual(out["model"], "Qwen3-8B-int4-cw")

    def test_thinking_flag(self):
        self.assertTrue(_anthropic_to_openai(self._req())["no_think"])
        self.assertTrue(_anthropic_to_openai(
            self._req(thinking={"type": "disabled"}))["no_think"])
        self.assertNotIn("no_think", _anthropic_to_openai(
            self._req(thinking={"type": "enabled", "budget_tokens": 1024})))

    def test_no_think_goes_on_a_copy(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = _with_no_think(msgs)
        self.assertEqual(out[0]["content"], "hi /no_think")
        self.assertEqual(msgs[0]["content"], "hi")


if __name__ == "__main__":
    unittest.main()


class AnthropicThinkingTests(unittest.TestCase):
    def test_enabled_thinking_reaches_gemma_and_blocks_order(self):
        from core.chat.anthropic import _anthropic_blocks
        from core.tools.gemma import thinking_requested
        base = {"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]}
        on = _anthropic_to_openai(dict(base, thinking={"type": "enabled", "budget_tokens": 2048}))
        off = _anthropic_to_openai(dict(base, thinking={"type": "disabled"}))
        none = _anthropic_to_openai(base)
        self.assertTrue(thinking_requested(on))
        self.assertNotIn("no_think", on)
        for o in (off, none):
            self.assertFalse(thinking_requested(o))
            self.assertTrue(o["no_think"])
        blocks = _anthropic_blocks("answer", [], "because")
        self.assertEqual([b["type"] for b in blocks], ["thinking", "text"])
        self.assertEqual(blocks[1]["text"], "answer")
        self.assertEqual([b["type"] for b in _anthropic_blocks("answer", [])], ["text"])
