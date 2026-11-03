import unittest
from unittest.mock import MagicMock, patch
from flask import Flask
from core.chat.anthropic import _anthropic_to_openai, anthropic_messages
from core.chat.turn import _prepare_turn


class RamblerNativeTests(unittest.TestCase):
    def test_explicit_search_off_survives_anthropic_translation(self):
        request = {"messages":[{"role":"user","content":"um hello"}],
                   "web_search":False,"thinking":{"type":"disabled"}}
        out = _anthropic_to_openai(request)
        self.assertIs(out["web_search"], False)
        self.assertNotIn("thinking", out)
        self.assertNotIn("web_search", _anthropic_to_openai({"messages":request["messages"]}))

    def test_plain_no_think_turn_uses_native_template(self):
        slot = MagicMock()
        slot.model_name = "gemma-4-E4B-it"
        slot.model_type = "vlm"
        slot.device_name = "GPU"
        slot.native_chat.return_value.render.return_value = "<bos>native cleanup<|turn>model\n"
        messages = [{"role":"system","content":"Clean this text"},
                    {"role":"user","content":"um hello"}]
        with patch("core.chat.turn.overall_status", return_value="ready"), \
             patch("core.chat.turn._route_request", return_value=slot), \
             patch("core.chat.turn.builtin_tools_for", return_value=[]):
            turn = _prepare_turn({"messages":messages,"max_tokens":128,"web_search":False}, native=True)
        self.assertEqual(turn["native_prompt"], "<bos>native cleanup<|turn>model\n")
        self.assertFalse(turn["thinking"])
        slot.native_chat.return_value.render.assert_called_once_with(messages, [], False)

    def test_buffered_anthropic_calls_native_without_enabling_thinking(self):
        app = Flask(__name__)
        slot = MagicMock()
        slot.model_name = "gemma-4-E4B-it"
        slot.generate_native.return_value = "Hello world.<turn|>"
        turn = {"slot":slot,"gen":MagicMock(),"t0":0,"text_prompt":"cleanup",
                "python_tool_active":False,"stream":False,"tools_active":False,
                "thinking":False,"native_prompt":"native cleanup"}
        with app.test_request_context("/v1/messages",method="POST",json={
                "messages":[{"role":"user","content":"um hello"}],"web_search":False}), \
             patch("core.chat.anthropic._prepare_turn",return_value=turn) as prepare, \
             patch("core.chat.anthropic._count_tokens",return_value=2), \
             patch("core.chat.anthropic._turn_headers",return_value={}):
            response = anthropic_messages()
        self.assertTrue(prepare.call_args.kwargs["native"])
        slot.generate_native.assert_called_once()
        slot.generate_vlm.assert_not_called()
        self.assertEqual(response.get_json()["content"], [{"type":"text","text":"Hello world."}])


if __name__ == "__main__":
    unittest.main()
