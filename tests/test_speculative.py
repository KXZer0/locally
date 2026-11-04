"""Speculative decoding: plan, tokenizer refusal, config mapping, device rules.

No model is loaded: pipelines are stubbed. The lossless check against real
(random-weight) IRs is scripts/speculative-check.py's job and is recorded in
CLAUDE.md.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import openvino_genai as ovg

from core import config
from core.genai import draft_check, speculative
from core.genai.speculative_cli import configure as configure_cli
from core.slots import device as device_mod
from core.slots.device import DeviceSlot


def _model(root, name, vocab=None, tok_bytes=b"tok", vocab_size=100, eagle=False,
           tokenizer=True, specials=None):
    d = os.path.join(root, name)
    os.makedirs(d)
    cfg = {"architectures": ["LlamaForCausalLMEagle3" if eagle else "Qwen3ForCausalLM"],
           "vocab_size": vocab_size}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    if tokenizer:
        for ext in ("xml", "bin"):
            with open(os.path.join(d, f"openvino_tokenizer.{ext}"), "wb") as f:
                f.write(tok_bytes + ext.encode())
        vocab = vocab or {"a": 0, "b": 1, "<eos>": 2}
        with open(os.path.join(d, "tokenizer.json"), "w") as f:
            json.dump({"model": {"vocab": vocab, "merges": []},
                       "added_tokens": [{"id": 2, "content": "<eos>"}]}, f)
    with open(os.path.join(d, "specials.json"), "w") as f:
        json.dump(specials or [2, -1, 0], f)
    return d


def _stub_loader(model_dir):
    with open(os.path.join(model_dir, "specials.json")) as f:
        eos, bos, pad = json.load(f)
    return SimpleNamespace(get_eos_token_id=lambda: eos, get_bos_token_id=lambda: bos,
                           get_pad_token_id=lambda: pad)


class _Config(unittest.TestCase):
    FIELDS = ("DRAFT_MODELS", "DRAFT_DEVICE", "NUM_ASSISTANT_TOKENS",
              "ASSISTANT_CONFIDENCE", "DRAFT_SAMPLING", "PROMPT_LOOKUP",
              "MAX_NGRAM_SIZE", "PROMPT_CACHE")

    def setUp(self):
        self._saved = {f: getattr(config, f) for f in self.FIELDS}
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        for f, v in self._saved.items():
            setattr(config, f, v)
        self.tmp.cleanup()


class TokenizerCheckTests(_Config):
    def test_identical_tokenizer_files_match_without_loading(self):
        a, b = _model(self.root, "main"), _model(self.root, "draft")
        loader = mock.Mock(side_effect=AssertionError("must not load"))
        self.assertEqual(speculative.check_tokenizers(a, b, load=loader), (None, None))

    def test_same_vocabulary_different_export_bytes_matches(self):
        a = _model(self.root, "main", tok_bytes=b"export-1")
        b = _model(self.root, "draft", tok_bytes=b"export-2")
        refusal, _ = speculative.check_tokenizers(a, b, load=_stub_loader)
        self.assertIsNone(refusal)

    def test_different_vocabulary_is_refused_in_plain_english(self):
        a = _model(self.root, "main", tok_bytes=b"1")
        b = _model(self.root, "gemma", tok_bytes=b"2", vocab={"x": 0, "y": 1, "<eos>": 2})
        refusal, _ = speculative.check_tokenizers(a, b, load=_stub_loader)
        self.assertIn("different tokenizer", refusal)
        self.assertIn("same model family", refusal)

    def test_different_eos_is_refused(self):
        a = _model(self.root, "main", tok_bytes=b"1")
        b = _model(self.root, "draft", tok_bytes=b"2", specials=[7, -1, 0])
        refusal, _ = speculative.check_tokenizers(a, b, load=_stub_loader)
        self.assertIn("eos token", refusal)

    def test_padded_vocab_size_is_a_warning_not_a_refusal(self):
        a = _model(self.root, "main", vocab_size=152064)
        b = _model(self.root, "draft", vocab_size=151936)
        refusal, warning = speculative.check_tokenizers(a, b, load=_stub_loader)
        self.assertIsNone(refusal)
        self.assertIn("vocab_size differs", warning)

    def test_plain_draft_without_tokenizer_is_refused(self):
        a = _model(self.root, "main")
        b = _model(self.root, "draft", tokenizer=False)
        refusal, _ = speculative.check_tokenizers(a, b, load=_stub_loader)
        self.assertIn("no openvino_tokenizer", refusal)

    def test_eagle3_head_without_tokenizer_uses_the_main_one(self):
        a = _model(self.root, "main")
        b = _model(self.root, "eagle", tokenizer=False, eagle=True)
        self.assertEqual(speculative.draft_kind(b), "eagle3")
        self.assertEqual(speculative.check_tokenizers(a, b, load=_stub_loader), (None, None))


class PlanTests(_Config):
    def _slot(self, device="CPU", model="main"):
        s = DeviceSlot(device)
        s.model_dir = os.path.join(self.root, model)
        return s

    def test_nothing_asked_nothing_planned(self):
        config.DRAFT_MODELS, config.PROMPT_LOOKUP = [], False
        _model(self.root, "main")
        state = speculative.plan(self._slot(), vlm=False)
        self.assertIsNone(state["mode"])
        self.assertIsNone(speculative.info(state, None))

    def test_vlm_slot_is_skipped_with_a_reason(self):
        _model(self.root, "main")
        config.DRAFT_MODELS = [_model(self.root, "draft")]
        config.PROMPT_LOOKUP = True
        state = speculative.plan(self._slot(), vlm=True)
        self.assertIsNone(state["mode"])
        self.assertIn("VLM", state["skipped"])

    def test_npu_skips_prompt_lookup(self):
        _model(self.root, "main")
        config.DRAFT_MODELS, config.PROMPT_LOOKUP = [], True
        state = speculative.plan(self._slot("NPU"), vlm=False)
        self.assertIsNone(state["mode"])
        self.assertIn("NPU does not have", state["skipped"])

    def test_npu_draft_is_attempted_and_labelled_unverified(self):
        _model(self.root, "main")
        config.DRAFT_MODELS = [_model(self.root, "draft")]
        state = speculative.plan(self._slot("NPU"), vlm=False)
        self.assertEqual(state["mode"], "draft")
        self.assertEqual(state["draft_device"], "NPU")
        self.assertTrue(any("unverified" in n for n in state["notes"]))

    def test_gpu_names_the_continuous_batching_trap_and_keeps_cache_off(self):
        _model(self.root, "main")
        config.DRAFT_MODELS, config.PROMPT_CACHE = [_model(self.root, "draft")], None
        state = speculative.plan(self._slot("GPU"), vlm=False)
        note = " ".join(state["notes"])
        self.assertIn("continuous-batching", note)
        self.assertIn("stays off", note)

    @mock.patch.object(draft_check, "_load_tokenizer", _stub_loader)
    def test_first_compatible_draft_wins_and_mismatches_are_reported(self):
        _model(self.root, "main", tok_bytes=b"1")
        other = _model(self.root, "gemma", tok_bytes=b"2", vocab={"q": 0})
        good = _model(self.root, "qwen-small")
        config.DRAFT_MODELS = [other, good]
        state = speculative.plan(self._slot(), vlm=False)
        self.assertEqual(state["draft"], "qwen-small")
        config.DRAFT_MODELS = [other]
        state = speculative.plan(self._slot(), vlm=False)
        self.assertIsNone(state["mode"])
        self.assertIn("different tokenizer", state["skipped"])

    def test_draft_and_prompt_lookup_are_exclusive(self):
        _model(self.root, "main")
        config.DRAFT_MODELS, config.PROMPT_LOOKUP = [_model(self.root, "d")], True
        state = speculative.plan(self._slot(), vlm=False)
        self.assertEqual(state["mode"], "draft")
        self.assertTrue(any("not both" in n for n in state["notes"]))

    def test_a_model_is_not_its_own_draft(self):
        main = _model(self.root, "main")
        config.DRAFT_MODELS = [main]
        self.assertIsNone(speculative.plan(self._slot(), vlm=False)["mode"])


class ApplyTests(_Config):
    def test_plain_slot_clears_request_knobs(self):
        gen = ovg.GenerationConfig()
        speculative.request_overrides(gen, {"num_assistant_tokens": 4, "max_ngram_size": 3})
        speculative.apply(gen, {})
        self.assertFalse(gen.num_assistant_tokens)
        self.assertEqual(gen.max_ngram_size, 0)

    def test_prompt_lookup_always_carries_both_fields(self):
        config.NUM_ASSISTANT_TOKENS, config.MAX_NGRAM_SIZE = 5, 3
        gen = ovg.GenerationConfig()
        speculative.apply(gen, {"mode": "prompt_lookup", "kind": "prompt_lookup"})
        self.assertEqual((gen.num_assistant_tokens, gen.max_ngram_size), (5, 3))
        self.assertTrue(gen.is_prompt_lookup())

    def test_request_override_is_used_and_clamped(self):
        gen = ovg.GenerationConfig()
        speculative.request_overrides(gen, {"num_assistant_tokens": 99})
        speculative.apply(gen, {"mode": "draft", "kind": "draft"})
        self.assertEqual(gen.num_assistant_tokens, speculative.MAX_ASSISTANT_TOKENS)

    def test_invalid_override_is_ignored(self):
        gen = ovg.GenerationConfig()
        speculative.request_overrides(gen, {"num_assistant_tokens": "lots"})
        speculative.apply(gen, {"mode": "draft", "kind": "draft"})
        self.assertEqual(gen.num_assistant_tokens, config.NUM_ASSISTANT_TOKENS)

    def test_confidence_replaces_the_fixed_count(self):
        config.ASSISTANT_CONFIDENCE = 0.4
        gen = ovg.GenerationConfig()
        speculative.apply(gen, {"mode": "draft", "kind": "draft"})
        self.assertEqual(gen.num_assistant_tokens, 0)
        self.assertAlmostEqual(gen.assistant_confidence_threshold, 0.4)

    def test_sampled_turn_on_a_draft_slot_runs_greedy_unless_allowed(self):
        gen = ovg.GenerationConfig()
        gen.do_sample, gen.temperature = True, 0.8
        note = speculative.apply(gen, {"mode": "draft", "kind": "draft"})
        self.assertFalse(gen.do_sample)
        self.assertIn("greedy", note)
        config.DRAFT_SAMPLING = True
        gen.do_sample = True
        self.assertEqual(speculative.apply(gen, {"mode": "draft", "kind": "draft"}), "")
        self.assertTrue(gen.do_sample)

    def test_eagle3_is_greedy_and_fixed_count_regardless(self):
        config.DRAFT_SAMPLING, config.ASSISTANT_CONFIDENCE = True, 0.5
        gen = ovg.GenerationConfig()
        gen.do_sample = True
        speculative.apply(gen, {"mode": "draft", "kind": "eagle3"})
        self.assertFalse(gen.do_sample)
        self.assertEqual(gen.assistant_confidence_threshold, 0.0)
        self.assertGreater(gen.num_assistant_tokens, 0)

    def test_prompt_lookup_keeps_sampling(self):
        gen = ovg.GenerationConfig()
        gen.do_sample = True
        speculative.apply(gen, {"mode": "prompt_lookup", "kind": "prompt_lookup"})
        self.assertTrue(gen.do_sample)


class StatsTests(unittest.TestCase):
    def _result(self, generated, steps, ext=None):
        pm = SimpleNamespace(get_num_generated_tokens=lambda: generated,
                             raw_metrics=SimpleNamespace(token_infer_durations=[1.0] * steps))
        return SimpleNamespace(perf_metrics=pm, extended_perf_metrics=ext)

    def test_draft_acceptance_and_tokens_per_step(self):
        ext = SimpleNamespace(get_draft_acceptance_rate=lambda: 0.75,
                              get_num_accepted_tokens=lambda: 30,
                              get_num_draft_tokens=lambda: 40)
        s = speculative.stats(self._result(60, 20, ext))
        self.assertEqual(s, {"tokens_per_step": 3.0, "acceptance": 0.75,
                             "accepted": 30, "drafted": 40})
        self.assertIn("75%", speculative.log_suffix(s))

    def test_prompt_lookup_has_steps_but_no_acceptance(self):
        s = speculative.stats(self._result(60, 33))
        self.assertEqual(s, {"tokens_per_step": 1.82})

    def test_missing_metrics_is_none(self):
        self.assertIsNone(speculative.stats(object()))


class PipelineConstructionTests(_Config):
    def _slot(self, device):
        s = DeviceSlot(device)
        s.model_dir = os.path.join(self.root, "main")
        s.speculative = {"mode": "draft", "draft": "d", "draft_dir": "/d",
                         "draft_device": device, "kind": "draft", "notes": []}
        return s

    def test_draft_kwargs_reach_the_pipeline_and_npu_draft_gets_prompt_len(self):
        slot = self._slot("NPU")
        with mock.patch.object(device_mod.ovg, "LLMPipeline") as pipe, \
                mock.patch("openvino_genai.draft_model", return_value="DRAFT") as dm:
            slot._build_llm(slot.model_dir, {"CACHE_DIR": "/c"})
        dm.assert_called_once_with("/d", "NPU", CACHE_DIR="/c",
                                   MAX_PROMPT_LEN=config.NPU_MAX_PROMPT_LEN)
        self.assertEqual(pipe.call_args.kwargs["draft_model"], "DRAFT")

    def test_a_refused_draft_loads_the_model_without_it(self):
        slot = self._slot("CPU")
        config.PROMPT_CACHE = True
        calls = []

        def build(*a, **kw):
            calls.append(kw)
            if "draft_model" in kw:
                raise RuntimeError("Tokenizers for draft and main models are different!")
            return "PIPE"

        with mock.patch.object(device_mod.ovg, "LLMPipeline", side_effect=build), \
                mock.patch("openvino_genai.draft_model", return_value="DRAFT"):
            self.assertEqual(slot._build_llm(slot.model_dir, {}), "PIPE")
        # Exactly one failed attempt with the draft: it is not retried on the
        # plain pipeline, where genai would meet an already-rewritten model.
        self.assertEqual(sum("draft_model" in kw for kw in calls), 1)
        self.assertIsNone(slot.speculative["mode"])
        self.assertIn("genai could not build it", slot.speculative["skipped"])
        self.assertIsNotNone(slot.info["speculative"])

    def test_prompt_lookup_kwarg(self):
        slot = self._slot("CPU")
        slot.speculative = {"mode": "prompt_lookup", "kind": "prompt_lookup", "notes": []}
        config.PROMPT_CACHE = False
        with mock.patch.object(device_mod.ovg, "LLMPipeline") as pipe:
            slot._build_llm(slot.model_dir, {})
        self.assertIs(pipe.call_args.kwargs["prompt_lookup"], True)
        self.assertNotIn("scheduler_config", pipe.call_args.kwargs)


class CliTests(_Config):
    def test_flags_map_onto_config(self):
        from core.cli import parse_args
        d = _model(self.root, "draft")
        args = parse_args(["--draft-model", d, "--draft-model", "/no/such/dir",
                           "--num-assistant-tokens", "40", "--assistant-confidence", "2",
                           "--prompt-lookup", "--max-ngram-size", "4",
                           "--draft-device", "GPU"])
        configure_cli(args)
        self.assertEqual(config.DRAFT_MODELS, [os.path.abspath(d)])
        self.assertEqual(config.NUM_ASSISTANT_TOKENS, 16)
        self.assertEqual(config.ASSISTANT_CONFIDENCE, 1.0)
        self.assertEqual((config.PROMPT_LOOKUP, config.MAX_NGRAM_SIZE), (True, 4))
        self.assertEqual(config.DRAFT_DEVICE, "GPU")
        self.assertFalse(config.DRAFT_SAMPLING)

    def test_defaults_are_off(self):
        from core.cli import parse_args
        configure_cli(parse_args([]))
        self.assertEqual((config.DRAFT_MODELS, config.PROMPT_LOOKUP), ([], False))


if __name__ == "__main__":
    unittest.main()
