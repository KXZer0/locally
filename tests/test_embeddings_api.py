"""`POST /v1/embeddings` (OpenAI shape) and `/api/embed` (Ollama shape) over
the utility engine's search embedder. The engine is stubbed: these guard the
wire format, the bounds and the slot lifecycle, not MiniLM's numbers.
"""

import base64
import unittest
from unittest import mock

import numpy as np

from core import runtime
from core.app import create_app, create_ollama_app
from core.util import embeddings as emb


class _Engine:
    """Stands in for UtilityEngine: 4-dim vectors, one token per word + 2."""

    def __init__(self):
        self.calls = []

    def token_counts(self, task, texts):
        self.calls.append(("token_counts", task))
        return [len(t.split()) + 2 for t in texts]

    def embed(self, texts):
        self.calls.append(("embed", len(texts)))
        vecs = np.array([[float(len(t)), 1.0, 0.0, -0.5] for t in texts],
                        dtype=np.float32)
        return vecs, {"infer": 1.0}


class _Slot:
    def __init__(self, device="NPU", status="ready"):
        self.device_name = device
        self.status = status
        self.utility = _Engine()
        self.loaded = 0

    def ensure_loaded(self):
        self.loaded += 1

    def run_utility(self, name, *args, **kwargs):
        return getattr(self.utility, name)(*args, **kwargs)


class EmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()
        self.slot = _Slot()
        for name, value in (("util_npu", self.slot), ("util_gpu", None),
                            ("util_cpu", None)):
            p = mock.patch.object(runtime, name, value)
            p.start()
            self.addCleanup(p.stop)

    def post(self, body, path="/v1/embeddings"):
        return self.client.post(path, json=body)

    def test_openai_shape_for_a_list(self):
        r = self.post({"model": "text-embedding-3-small",
                       "input": ["one two", "three"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["X-Device"], "NPU")
        body = r.json
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["model"], emb.EMBED_MODEL)
        self.assertEqual([d["index"] for d in body["data"]], [0, 1])
        self.assertTrue(all(d["object"] == "embedding" for d in body["data"]))
        self.assertEqual(body["data"][0]["embedding"], [7.0, 1.0, 0.0, -0.5])
        self.assertEqual(body["usage"], {"prompt_tokens": 4 + 3,
                                         "total_tokens": 7})
        self.assertEqual(self.slot.loaded, 1)       # woke an idle slot first

    def test_a_bare_string_is_one_input(self):
        r = self.post({"input": "hello"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json["data"]), 1)

    def test_any_model_name_is_accepted(self):
        for model in ("", "nomic-embed-text", "Qwen3-8B", None):
            self.assertEqual(self.post({"model": model, "input": "x"}).status_code,
                             200, model)

    def test_base64_is_little_endian_float32(self):
        r = self.post({"input": "abc", "encoding_format": "base64"})
        raw = base64.b64decode(r.json["data"][0]["embedding"])
        vec = np.frombuffer(raw, dtype="<f4")
        np.testing.assert_array_equal(vec, [3.0, 1.0, 0.0, -0.5])

    def test_bad_encoding_format_is_400(self):
        self.assertEqual(self.post({"input": "x", "encoding_format": "int8"})
                         .status_code, 400)

    def test_inputs_are_bounded(self):
        cases = [
            {},
            {"input": []},
            {"input": ""},
            {"input": ["ok", "   "]},
            {"input": [[101, 2023]]},
            {"input": 5},
            {"input": ["x"] * (emb.MAX_INPUTS + 1)},
            {"input": "a" * (emb.MAX_INPUT_CHARS + 1)},
        ]
        for body in cases:
            r = self.post(body)
            self.assertEqual(r.status_code, 400, body if len(str(body)) < 80 else "big")
            self.assertIn("message", r.json["error"])
        # Nothing reached the engine for any of them.
        self.assertEqual(self.slot.utility.calls, [])

    def test_long_input_is_truncated_and_reported(self):
        long = " ".join(["w"] * 300)                 # 302 "tokens" > 256
        r = self.post({"input": ["short", long]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["X-Truncated-Inputs"], "1")
        self.assertEqual(r.json["usage"]["prompt_tokens"], 3 + 256)

    def test_truncate_false_refuses_instead(self):
        long = " ".join(["w"] * 300)
        r = self.post({"input": long, "truncate": False})
        self.assertEqual(r.status_code, 400)
        self.assertIn("302 tokens", r.json["error"]["message"])
        self.assertNotIn(("embed", 1), self.slot.utility.calls)

    def test_dimensions_must_match_the_model(self):
        self.assertEqual(self.post({"input": "x", "dimensions": 4}).status_code, 200)
        self.assertEqual(self.post({"input": "x", "dimensions": 2}).status_code, 400)

    def test_compiling_engine_is_503_with_retry_after(self):
        self.slot.status = "loading"
        r = self.post({"input": "x"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.headers.get("Retry-After"), "5")
        self.assertIn("still compiling", r.json["error"]["message"])

    def test_no_utility_engine_is_an_error_not_a_crash(self):
        with mock.patch.object(runtime, "util_npu", None):
            r = self.post({"input": "x"})
        self.assertIn(r.status_code, (400, 503))
        self.assertIn("error", r.json)

    def test_unavailable_model_is_503(self):
        from utility_pipeline import UtilityUnavailable

        def missing(*_a, **_k):
            raise UtilityUnavailable("The search utility is unavailable on NPU")
        self.slot.utility.token_counts = missing
        r = self.post({"input": "x"})
        self.assertEqual(r.status_code, 503)
        # Not installed: waiting fixes nothing, so no invitation to retry.
        self.assertIsNone(r.headers.get("Retry-After"))

    def test_explicit_engine_is_honoured(self):
        cpu = _Slot("CPU")
        with mock.patch.object(runtime, "util_cpu", cpu):
            r = self.post({"input": "x", "engine": "cpu"})
        self.assertEqual(r.headers["X-Device"], "CPU")

    def test_not_listed_in_v1_models(self):
        ids = [m["id"] for m in self.client.get("/v1/models").json["data"]]
        self.assertNotIn(emb.EMBED_MODEL, ids)


class EngineTokenCountTests(unittest.TestCase):
    """The route calls UtilityEngine.token_counts by name, through run_utility;
    this pins that the real method exists with that signature."""

    def test_token_counts_uses_the_task_tokenizer_untruncated(self):
        import tempfile
        from pathlib import Path
        import utility_pipeline as up

        with tempfile.TemporaryDirectory() as root:
            model = Path(root, up.TIERS["search"])
            model.parent.mkdir(parents=True)
            model.write_text("<net/>")
            engine = up.UtilityEngine(root, device="CPU")
            seen = {}

            def tokenizer(text, **kwargs):
                seen.update(kwargs)
                return {"input_ids": [0] * (len(text.split()) + 2)}
            engine._tokenizers["search"] = tokenizer
            self.assertEqual(engine.token_counts("search", ["a b c", "d"]), [5, 3])
            self.assertNotIn("truncation", seen)


class OllamaEmbedTests(unittest.TestCase):
    def setUp(self):
        self.client = create_ollama_app().test_client()
        p = mock.patch.object(runtime, "util_npu", _Slot())
        p.start()
        self.addCleanup(p.stop)

    def test_api_embed_shape(self):
        r = self.client.post("/api/embed", json={"model": "all-minilm",
                                                 "input": ["a b", "c"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json["model"], "all-minilm")
        self.assertEqual(len(r.json["embeddings"]), 2)
        self.assertEqual(r.json["prompt_eval_count"], 4 + 3)
        self.assertIsInstance(r.json["total_duration"], int)

    def test_api_embed_error_is_ollama_shaped(self):
        r = self.client.post("/api/embed", json={"input": []})
        self.assertEqual(r.status_code, 400)
        self.assertIsInstance(r.json["error"], str)

    def test_openai_route_on_the_ollama_port(self):
        r = self.client.post("/v1/embeddings", json={"input": "x"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json["object"], "list")


if __name__ == "__main__":
    unittest.main()
