import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console
from core.terminal import (ApiError, Chat, NO_THINK, OwnedServer, Truncated,
                           make_prompt_session, multipart, read_stream)


def stream_bytes(text, finish="stop", done=True):
    frames = [": heartbeat\n\n"]
    for part in (text[:7], text[7:]):
        frames.append("data: " + json.dumps({"choices": [{"delta": {"content": part}}]}) + "\n\n")
    frames.append("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": finish}]}) + "\n\n")
    if done:
        frames.append("data: [DONE]\n\n")
    return io.BytesIO("".join(frames).encode("utf-8"))


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.api = mock.Mock()
        self.console = Console(file=io.StringIO(), width=80, color_system=None)
        self.chat = Chat(self.api, self.console)
        self.markdown = "# Résumé\n\n**Answer:** π = 3.14\n\n```python\nprint('hi')\n```\n"

    def test_stream_preserves_markdown_and_ignores_heartbeat(self):
        self.assertEqual("".join(read_stream(stream_bytes(self.markdown))), self.markdown)

    def test_truncation_and_error_are_not_success(self):
        for stream in (stream_bytes("partial", done=False), stream_bytes("partial", finish="error"),
                       stream_bytes("partial", finish="length")):
            with self.assertRaises(ApiError):
                list(read_stream(stream))

    def test_truncated_answer_is_kept_and_said_to_be_incomplete(self):
        # Raised only after the tokens are yielded: the budget running out is
        # the ordinary end of a long answer, not a reason to lose it.
        self.api.stream.side_effect = lambda *a: read_stream(
            stream_bytes(self.markdown, finish="length"))
        self.chat.ask("question")
        self.assertEqual(self.chat.last_answer, self.markdown)
        self.assertIn("incomplete", self.console.file.getvalue())
        self.assertTrue(issubclass(Truncated, ApiError))

    def test_reasoning_is_stripped_from_the_answer_and_from_history(self):
        answer = "The answer is 4." + chr(10)
        self.api.stream.side_effect = lambda *a: read_stream(
            stream_bytes("<think>weigh the options</think>" + answer))
        self.chat.ask("2 + 2")
        self.assertEqual(self.chat.last_answer, answer)
        self.assertEqual(self.chat.messages[-1], {"role": "assistant", "content": answer})

    def test_a_turn_that_was_all_reasoning_shows_the_reasoning(self):
        self.api.stream.side_effect = lambda *a: read_stream(
            stream_bytes("<think>still weighing the options"))
        self.chat.ask("2 + 2")
        self.assertIn("still weighing", self.chat.last_answer)

    def test_interrupting_a_turn_asks_the_server_to_stop(self):
        def interrupted(*args):
            yield "partial"
            raise KeyboardInterrupt
        self.api.stream = interrupted
        with self.assertRaises(KeyboardInterrupt):
            self.chat.ask("question")
        self.api.cancel.assert_called_once_with()
        self.assertEqual(len(self.chat.messages), 1)

    def test_no_think_goes_out_but_never_into_the_conversation(self):
        self.api.stream.side_effect = lambda *a: read_stream(stream_bytes("4\n"))
        self.chat.command("/think off")
        self.chat.ask("2 + 2")
        sent = self.api.stream.call_args[0][0]
        self.assertTrue(sent[-1]["content"].endswith(NO_THINK))
        # History keeps the question the user asked. A control token stored
        # here is re-sent every turn and lands in /save.
        self.assertEqual(self.chat.messages[1]["content"], "2 + 2")
        self.api.stream.side_effect = lambda *a: read_stream(stream_bytes("4\n"))
        self.chat.command("/think on")
        self.chat.ask("3 + 3")
        self.assertNotIn(NO_THINK, self.api.stream.call_args[0][0][-1]["content"])

    def test_read_puts_the_document_in_the_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assignment.md"
            path.write_text("# Task 3" + chr(10) + "Integrate x^2.", encoding="utf-8")
            self.api.upload.return_value = {"markdown": "# Task 3", "engine": None}
            self.chat.command(f'/read "{path}"')
            endpoint, fields, files = self.api.upload.call_args[0]
            self.assertEqual(endpoint, "/v1/util/read")
            self.assertEqual(files[0][0], "file")
            self.assertIn("# Task 3", self.chat.messages[-1]["content"])
            self.assertEqual(self.chat.messages[-1]["role"], "user")

    def test_util_off_frees_only_the_utility_engines(self):
        self.api.json.side_effect = [
            {"util": {"npu": {"status": "ready"}}},          # wait_engines
            {"unloaded": ["util-npu"]},                       # the unload
        ]
        self.chat.command("/util off")
        self.assertEqual(self.api.json.call_args[0],
                         ("/v1/models/unload", {"scope": "util"}))
        self.assertIn("chat model is untouched", self.console.file.getvalue())

    def test_multipart_body_is_well_formed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.txt"
            path.write_bytes(b"hello")
            body, content_type = multipart({"engine": "npu"}, [("file", path)])
        boundary = content_type.split("boundary=")[1]
        self.assertTrue(body.startswith(("--" + boundary).encode()))
        self.assertTrue(body.endswith(("--" + boundary + "--\r\n").encode()))
        self.assertIn(b'name="file"; filename="a.txt"', body)
        self.assertIn(b"\r\n\r\nhello\r\n", body)

    def test_copy_and_save_use_exact_markdown_not_rendered_text(self):
        self.chat.last_answer = self.markdown
        with mock.patch("core.terminal.clipboard") as copy:
            self.chat.command("/copy")
            copy.assert_called_once_with(self.markdown)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "my answer.md"
            self.chat.command(f'/save "{path}"')
            self.assertEqual(path.read_bytes(), self.markdown.encode("utf-8"))
            with self.assertRaises(FileExistsError):
                self.chat.command(f'/save "{path}"')

    def test_new_clears_last_answer_and_history(self):
        self.chat.messages.append({"role": "user", "content": "old"})
        self.chat.last_answer = "old"
        self.chat.command("/new")
        self.assertEqual(len(self.chat.messages), 1)
        with self.assertRaises(ApiError):
            self.chat.command("/copy")

    def test_load_uses_actual_placement_and_clears_previous_model_context(self):
        self.api.json.return_value = {"model": "example", "device": "GPU", "placement": "NPU incompatible"}
        self.chat.messages.append({"role": "user", "content": "old"})
        self.chat.command("/load example@NPU")
        self.api.json.assert_called_once_with("/v1/models/load", {"model": "example@NPU"})
        self.assertEqual(self.chat.model, "example@GPU")
        self.assertEqual(len(self.chat.messages), 1)

    def test_failed_turn_does_not_poison_history_or_replace_complete_answer(self):
        def failure(*args):
            yield "partial"
            raise ApiError("disconnected")
        self.api.stream = failure
        self.chat.last_answer = "previous"
        with self.assertRaises(ApiError):
            self.chat.ask("question")
        self.assertEqual(len(self.chat.messages), 1)
        self.assertEqual(self.chat.last_answer, "previous")

    def test_successful_turn_preserves_pasted_newlines(self):
        self.api.stream.side_effect = lambda *a: read_stream(stream_bytes(self.markdown))
        self.chat.ask("Question one\nQuestion two")
        self.assertEqual(self.chat.messages[1]["content"], "Question one\nQuestion two")
        self.assertEqual(self.chat.last_answer, self.markdown)

    def test_exit_never_calls_a_remote_shutdown_or_unload(self):
        self.assertFalse(self.chat.command("/exit"))
        self.api.json.assert_not_called()

    def test_occupied_port_does_not_start_or_stop_any_process(self):
        import socket
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            server = OwnedServer(listener.getsockname()[1], [])
            with mock.patch("core.terminal.subprocess.Popen") as popen:
                with self.assertRaises(ApiError) as caught:
                    server.start()
                self.assertIn(str(listener.getsockname()[1]), str(caught.exception))
                server.close()
                popen.assert_not_called()

    def test_multiline_paste_is_one_editable_prompt(self):
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        with create_pipe_input() as pipe:
            session = make_prompt_session(input=pipe, output=DummyOutput())
            pipe.send_text("\x1b[200~first line\nsecond line\x1b[201~\r")
            self.assertEqual(session.prompt(), "first line\nsecond line")

    def test_alt_enter_adds_a_line_and_enter_sends(self):
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        with create_pipe_input() as pipe:
            session = make_prompt_session(input=pipe, output=DummyOutput())
            pipe.send_text("first\x1b\rsecond\r")
            self.assertEqual(session.prompt(), "first\nsecond")

    def test_load_completes_models_the_server_reports_not_a_baked_in_list(self):
        from prompt_toolkit.completion import CompleteEvent
        from prompt_toolkit.document import Document
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        catalog = [{"name": "Qwen3-8B-abliterated-int4-cw", "type": "llm",
                    "loaded_on": "NPU"},
                   {"name": "gemma-4-26b-a4b-it", "type": "vlm"},
                   {"name": "broken.gguf", "type": "llm",
                    "reason": "unsupported architecture"}]
        calls = []
        def fetch():
            calls.append(1)
            return catalog
        with create_pipe_input() as pipe:
            session = make_prompt_session(fetch_models=fetch, input=pipe,
                                          output=DummyOutput())
            def complete(text):
                return list(session.completer.get_completions(Document(text),
                                                              CompleteEvent()))
            # Substring, because "abl" sits in the middle of the name.
            self.assertEqual([c.text for c in complete("/load abl")],
                             ["Qwen3-8B-abliterated-int4-cw"])
            self.assertIn("loaded on NPU", complete("/load abl")[0].display_meta_text)
            # Why it cannot be loaded outranks what it is.
            self.assertEqual(complete("/load broken")[0].display_meta_text,
                             "unsupported architecture")
            # Everything on an empty argument, unloadable files included --
            # the server lists those on purpose, with the reason attached.
            self.assertEqual([c.text for c in complete("/load ")],
                             ["broken.gguf", "gemma-4-26b-a4b-it",
                              "Qwen3-8B-abliterated-int4-cw"])
            # A prefix match ranks above a mid-name one.
            self.assertEqual([c.text for c in complete("/load g")],
                             ["gemma-4-26b-a4b-it", "broken.gguf"])
            self.assertEqual([c.text for c in complete("/load gemma@n")], ["NPU"])
            # One fetch for all of that, and none for an unrelated command.
            self.assertEqual(len(calls), 1)
            self.assertEqual([c.text for c in complete("/unload n")], ["NPU"])
            # A model deleted between fetches disappears; nothing is cached in
            # this file, only the server's own answer.
            catalog.pop(0)
            session.completer._fetched_at = None
            self.assertEqual([c.text for c in complete("/load abl")], [])

    def test_a_server_that_stops_answering_never_breaks_the_prompt(self):
        from prompt_toolkit.completion import CompleteEvent
        from prompt_toolkit.document import Document
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        state = {"up": True}
        def fetch():
            if not state["up"]:
                raise ApiError("Cannot reach the server")
            return [{"name": "Qwen3-8B-int4-cw", "type": "llm"}]
        with create_pipe_input() as pipe:
            session = make_prompt_session(fetch_models=fetch, input=pipe,
                                          output=DummyOutput())
            def complete(text):
                return list(session.completer.get_completions(Document(text),
                                                              CompleteEvent()))
            self.assertEqual(len(complete("/load qwen")), 1)
            state["up"] = False
            session.completer._fetched_at = None
            self.assertEqual(len(complete("/load qwen")), 1)  # last good list
            self.assertEqual([c.text for c in complete("/mo")], ["/models"])

    def test_slash_completions_filter_commands_and_arguments_only(self):
        from prompt_toolkit.completion import CompleteEvent
        from prompt_toolkit.document import Document
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        with create_pipe_input() as pipe:
            session = make_prompt_session(input=pipe, output=DummyOutput())
            def complete(text):
                return list(session.completer.get_completions(Document(text), CompleteEvent()))
            matches = complete("/mo")
            self.assertEqual([item.text for item in matches], ["/models"])
            self.assertTrue(matches[0].display_meta_text)
            self.assertEqual([item.text for item in complete("/markdown o")], ["on", "off"])
            self.assertEqual([item.text for item in complete("/unload n")], ["NPU"])
            for text in ("", "Explain /mo", "/load C:/models/", "/mo\nassignment"):
                self.assertEqual(complete(text), [])


if __name__ == "__main__":
    unittest.main()
