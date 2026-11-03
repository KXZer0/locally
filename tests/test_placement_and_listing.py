"""Four bugs that each made a working feature unreachable, and stayed hidden.

They have nothing in common except how they hid. None of them was a wrong
decision -- every one of them was a decision that was never consulted:

  * the plugin property that streams MoE experts was spelled
    "config.OFFLOAD_RATIO", so OpenVINO refused every load that needed it;
  * `--device auto` picked a device from the device list without ever looking
    at the model, so a VLM went to the NPU and died in the vpux compiler;
  * `/api/tags` advertised only the resident slot, so a picker on port 11434
    offered one choice on a server that could load six;
  * the launcher passed no utility flag at all, so document reading, OCR and
    local search were off on every start while the models sat on disk.

Reading the code finds none of these -- the string looks like a name, the
device list really does contain an NPU, the tags loop really does list what it
was asked to, and a missing flag is invisible. Only running it does, which is
what these tests are for.
"""
import ast
import io
import os
import re
import types
import unittest
from pathlib import Path
from unittest import mock

from core import runtime
from core.app import create_app, create_ollama_app


ROOT = Path(__file__).resolve().parents[1]


def slot(name, device="NPU", status="ready", model_type="llm"):
    return types.SimpleNamespace(status=status, model_name=name,
                                 device_name=device, model_type=model_type,
                                 model_dir=None, info={})


class OffloadPropertyNameTests(unittest.TestCase):
    """The MoE offload property is "OFFLOAD_RATIO" and nothing else.

    It became "config.OFFLOAD_RATIO" when the one-file split rewrote bare
    module globals to `config.<NAME>` and the rewrite reached inside a string
    literal. OpenVINO answers an unknown property with

        Check 'it != m_options_map.end()' failed ... Option not found

    12-25 s into the load, naming the option and nothing else -- so every model
    big enough to need offload (gemma-4-26b, Qwen3-Coder-30B) stopped loading on
    the GPU, and the message pointed at the plugin rather than at us. The
    feature has no cheap runtime test: reaching it means compiling a 14 GB MoE
    onto a GPU. Reading the source is what is affordable, so that is what this
    does.
    """

    def test_property_key_is_not_namespaced(self):
        source = (ROOT / "core" / "slots" / "device.py").read_text(encoding="utf-8")
        keys = re.findall(r'offload\[\s*"([^"]+)"\s*\]', source)
        self.assertIn("OFFLOAD_RATIO", keys)
        for key in keys:
            self.assertFalse(
                key.startswith("config."),
                f'{key!r} is a Python module path, not an OpenVINO property. '
                f'OpenVINO rejects it with "Option not found".')

    def test_no_python_module_path_survives_inside_a_string(self):
        """The same rewrite could have hit any other literal in core/.

        Every legitimate `config.NAME` inside a string in this codebase is an
        f-string interpolation, so it is an AST FormattedValue rather than a
        constant. A `config.<UPPER>` sitting in constant text is the bug --
        except in a docstring, where naming the setting is the whole point, so
        docstrings are collected up front and skipped.
        """
        bad = []
        for path in sorted(ROOT.joinpath("core").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            prose = set()
            for holder in ast.walk(tree):
                if isinstance(holder, (ast.Module, ast.ClassDef,
                                       ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and ast.get_docstring(holder, clean=False) is not None:
                    prose.add(id(holder.body[0].value))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if id(node) in prose:
                    continue
                if re.search(r"\bconfig\.[A-Z][A-Z0-9_]+", node.value):
                    bad.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.value!r}")
        self.assertEqual(bad, [], "module paths left inside string literals:\n"
                                  + "\n".join(bad))


class AutoDeviceTests(unittest.TestCase):
    """`--device auto` at startup asks the model, not just the machine.

    It used to answer "NPU if there is one", which is right for the text model
    the launcher usually picks and wrong for every other kind. Starting with
    gemma-4-26b -- a VLM -- put it on the NPU and got

        Chunking is not implemented for Gemma model family yet

    out of the vpux compiler, under locally's own "Is another process using the
    NPU?". _choose_device had known the answer ("NPU has no working vision
    path") since dynamic placement was written; only POST /v1/models/load ever
    asked it.
    """

    def setUp(self):
        from core.launch import slots
        self.slots = slots
        self.devices = {"NPU": {"id": "NPU", "name": "AI Boost"},
                        "GPU": {"id": "GPU", "name": "Arc"},
                        "CPU": {"id": "CPU", "name": "Core Ultra"}}

    def args(self, **kw):
        base = dict(proxy_url=None, gpu_model_dir=None)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_vlm_is_kept_off_the_npu(self):
        with mock.patch.object(self.slots.os.path, "isdir", return_value=True), \
             mock.patch("core.models.discovery._choose_device",
                        return_value=("GPU", "GPU", "ruled out NPU: NPU has no "
                                                    "working vision path; GPU: fits")):
            self.assertEqual(
                self.slots._auto_device(self.args(), "/models/gemma", self.devices),
                "GPU")

    def test_text_model_still_lands_on_the_npu(self):
        """The NPU-first default is the point of the project; keep it."""
        with mock.patch.object(self.slots.os.path, "isdir", return_value=True), \
             mock.patch("core.models.discovery._choose_device",
                        return_value=("NPU", "NPU", "NPU: no budget reported")):
            self.assertEqual(
                self.slots._auto_device(self.args(), "/models/qwen", self.devices),
                "NPU")

    def test_no_model_on_disk_keeps_the_old_preference_order(self):
        """A server started with no model has nothing to read, and still starts."""
        with mock.patch.object(self.slots.os.path, "isdir", return_value=False):
            self.assertEqual(
                self.slots._auto_device(self.args(), "/models/nothing", self.devices),
                "NPU")

    def test_a_proxy_slot_reads_no_ir(self):
        with mock.patch("core.models.discovery._choose_device") as choose:
            self.slots._auto_device(self.args(proxy_url="http://x/v1"),
                                    "/models/whatever", self.devices)
            choose.assert_not_called()

    def test_dual_mode_asks_before_putting_the_primary_on_the_npu(self):
        with mock.patch.object(self.slots.os.path, "isdir", return_value=True), \
             mock.patch("core.models.discovery._device_can_host",
                        return_value=(False, "NPU has no working vision path")):
            self.assertEqual(
                self.slots._auto_device(self.args(gpu_model_dir="/models/vlm"),
                                        "/models/gemma", self.devices),
                "CPU")

    def test_every_device_refusing_still_starts_a_server(self):
        """A refusal here is a prediction. The driver's verdict outranks it."""
        with mock.patch.object(self.slots.os.path, "isdir", return_value=True), \
             mock.patch("core.models.discovery._choose_device",
                        return_value=(None, None, "nothing fits")):
            self.assertEqual(
                self.slots._auto_device(self.args(), "/models/huge", self.devices),
                "NPU")


class OllamaListingTests(unittest.TestCase):
    """Port 11434 advertises what port 8000 advertises.

    Both fixes /v1/models got -- list an idle-unloaded slot, list what is on
    disk -- were made on the OpenAI surface alone, and many harnesses'
    best-trodden path is the Ollama one. A picker built from /api/tags therefore showed a
    single model on a server holding six loadable ones.
    """

    def setUp(self):
        self.app = create_ollama_app().test_client()
        self._saved = (runtime.primary, runtime.secondary, runtime.MODELS_DIR)
        runtime.primary = slot("Qwen3-8B-abliterated-int4-cw")
        runtime.secondary = None
        self.available = [
            {"name": "gemma-4-26b-a4b-it", "path": "/m/gemma", "type": "vlm"},
            {"name": "Qwen3.5-9B", "path": "/m/q35", "type": "vlm"},
            # Listed by /v1/models/available with a reason, never offered as a
            # thing to pick: choosing it can only produce a refusal.
            {"name": "broken", "path": "/m/broken.gguf", "type": "llm",
             "loadable": False, "reason": "unsupported architecture"},
        ]

    def tearDown(self):
        runtime.primary, runtime.secondary, runtime.MODELS_DIR = self._saved

    def tags(self):
        with mock.patch("core.models.discovery._available_models",
                        return_value=self.available):
            return [m["name"] for m in self.app.get("/api/tags").get_json()["models"]]

    def test_models_on_disk_are_offered(self):
        self.assertEqual(
            self.tags(),
            ["Qwen3-8B-abliterated-int4-cw", "gemma-4-26b-a4b-it", "Qwen3.5-9B"])

    def test_an_idle_unloaded_slot_still_appears(self):
        """It still serves -- the chat path reloads it -- so it must still list.

        An empty listing is read by a polling client as "backend offline",
        which is what a quiet fifteen minutes used to look like.
        """
        runtime.primary = slot("Qwen3-8B-abliterated-int4-cw", status="idle_unloaded")
        self.assertIn("Qwen3-8B-abliterated-int4-cw", self.tags())

    def test_an_unloadable_model_is_not_offered(self):
        self.assertNotIn("broken", self.tags())

    def test_no_device_suffix_on_the_ollama_surface(self):
        """An Ollama client treats the tag as a name, not an address."""
        for name in self.tags():
            self.assertNotIn("@", name)

    def test_the_resident_model_is_listed_once(self):
        self.available.append({"name": "Qwen3-8B-abliterated-int4-cw",
                               "path": "/m/qwen", "type": "llm"})
        names = self.tags()
        self.assertEqual(names.count("Qwen3-8B-abliterated-int4-cw"), 1)

    def test_v1_models_answers_on_the_ollama_port(self):
        """`http://host:11434/v1` is a base URL people configure.

        The port already served /v1/chat/completions, so a client could chat
        and could not enumerate -- a 404 from the one call a model picker makes
        first, against a server that answers everything else.
        """
        with mock.patch("core.models.discovery._available_models",
                        return_value=self.available):
            body = self.app.get("/v1/models").get_json()
        self.assertEqual(
            [m["id"] for m in body["data"]],
            ["Qwen3-8B-abliterated-int4-cw@NPU", "gemma-4-26b-a4b-it", "Qwen3.5-9B"])

    def test_show_resolves_an_id_carrying_a_device_suffix(self):
        """/v1/models hands out "name@GPU"; /api/show has to accept it back."""
        runtime.primary = slot("gemma-4-26b-a4b-it", device="GPU")
        body = self.app.post("/api/show",
                             json={"model": "gemma-4-26b-a4b-it@GPU"}).get_json()
        self.assertIn("tools", body["capabilities"])


class UsageBlockTests(unittest.TestCase):
    """A non-streaming reply reports the counts the pipeline measured.

    Every one of them used to say `{-1, -1, -1}`. OpenAI's schema makes those
    fields non-negative integers, so -1 is not "unknown" to a client -- it is a
    number, and clients size their next request from it. The counts existed all
    along: _record_turn reads them off the same result to write /v1/metrics.
    """

    def block(self, **usage):
        from core.genai.results import GeneratedText, usage_block
        return usage_block(GeneratedText("hi", "stop", usage or None))

    def test_native_counts_are_reported(self):
        self.assertEqual(
            self.block(prompt_tokens=18, completion_tokens=7),
            {"prompt_tokens": 18, "completion_tokens": 7, "total_tokens": 25})

    def test_a_runtime_with_no_perf_metrics_reports_zero_not_minus_one(self):
        self.assertEqual(self.block(),
                         {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0})

    def test_a_plain_string_answer_does_not_raise(self):
        """The proxy slot and the tool loop hand back ordinary strings."""
        from core.genai.results import usage_block
        self.assertEqual(usage_block("just a string")["total_tokens"], 0)

    def test_the_builtin_tool_loop_keeps_the_counts_and_the_finish_reason(self):
        """`parse_tool_calls` returns a plain str and drops both.

        This mattered far more than it looks: the built-in tools are OFFERED
        rather than requested, so with `--search-url` or `--python-tool` set,
        EVERY ordinary chat turn goes through this loop. It reported `usage`
        of zero and `finish_reason: "stop"` on all of them -- including an
        answer cut off at max_tokens, which is precisely the mislabelling
        `extract_finish_reason` was written to end on the streaming path.
        """
        from core.chat.turn import _builtin_tool_turn
        from core.genai.results import GeneratedText, usage_block

        def generate(_messages):
            return GeneratedText("a truncated answer", "length",
                                 {"prompt_tokens": 262, "completion_tokens": 12})

        answer, runs = _builtin_tool_turn(generate, [{"role": "user", "content": "hi"}],
                                          [])
        self.assertEqual(str(answer), "a truncated answer")
        self.assertEqual(answer.finish_reason, "length")
        self.assertEqual(usage_block(answer)["total_tokens"], 274)
        self.assertEqual(runs, [])

    def test_no_negative_usage_is_left_anywhere(self):
        for path in sorted(ROOT.joinpath("core").rglob("*.py")):
            src = path.read_text(encoding="utf-8")
            self.assertNotRegex(
                src, r'"(?:prompt|completion|total)_tokens"\s*:\s*-1',
                f"{path.relative_to(ROOT)} still reports a negative token count")


class LauncherUtilityDefaultTests(unittest.TestCase):
    """The launcher enables the utilities when their models are actually there.

    Deliberately unlike the audio engines, which stay off until named. Audio
    slots compile eagerly and sit in the voice critical path; utility slots
    wait on the chat model's load event and compile on first use, so the whole
    startup cost is OCR's ~160 MB (measured: 9.7 s to chat-ready with them on,
    9.6 s with them off). Requiring a flag put `/health.util` back to `null` on
    an ordinary start, with document reading and OCR silently dead while the
    models sat on disk -- see the CLAUDE.md bullet, which this regressed twice.
    """

    def setUp(self):
        self.source = (ROOT / "scripts" / "locally-launch.ps1").read_text(
            encoding="utf-8")

    def test_an_explicit_directory_is_forwarded(self):
        self.assertIn("--util-models-dir", self.source)

    def test_it_is_conditional_on_the_ocr_model_existing(self):
        """OCR is the minimum the utilities need, so its file is the test.

        Passing the flag unconditionally would fail startup on a machine that
        has no utility models, which is most of them.
        """
        self.assertRegex(
            self.source,
            r"ocr-det[\\/]+inference\.onnx[\s\S]{0,250}"
            r"ocr-rec[\\/]+inference\.onnx[\s\S]{0,250}"
            r"--util-models-dir")

    def test_the_conventional_directory_is_the_default(self):
        """`api.ps1` with no arguments has to find ~/models/util by itself.

        The OCR sentinel above is what keeps this safe on a machine that has
        no utility models: the default is a path, not an unconditional flag.
        """
        self.assertRegex(
            self.source,
            r"\[string\]\s*\$UtilModels\s*=\s*\"\$env:USERPROFILE\\models\\util\"")

    def test_positional_binding_is_off(self):
        """`api.ps1` forwards `--port N`, which PowerShell will bind positionally.

        With default binding that lands as `$ChatModels = '--port'` and
        `$ModelsDir = 'N'`, so the launcher searches the wrong directory and
        `/v1/models` advertises whatever happens to sit in the repo root
        instead of the user's models -- measured, six models down to two, with
        no error anywhere. `--port` also never reaches python, so `-Port 9000`
        served 8000. Both failures are silent, which is why this is a test and
        not a comment.
        """
        self.assertRegex(self.source,
                         r"\[CmdletBinding\(\s*PositionalBinding\s*=\s*\$false\s*\)\]"
                         r"\s*param\(")

    def test_a_forwarded_port_updates_the_preflight_check(self):
        """Argparse honours the forwarded flag; this script has to notice too.

        Otherwise it probes its own default port and can refuse to start over
        a listener that has nothing to do with the port being requested.
        """
        self.assertRegex(self.source, r"ExtraArgs\[\$i\]\s*-eq\s*'--port'")

    def test_only_a_named_directory_warns_when_ocr_is_missing(self):
        """A clean install has no utility models and must not be nagged.

        The warning is for a user who asked for a directory and got nothing;
        firing it for the default would print on every start of a fresh clone.
        """
        self.assertIn("PSBoundParameters.ContainsKey('UtilModels')", self.source)


if __name__ == "__main__":
    unittest.main()
