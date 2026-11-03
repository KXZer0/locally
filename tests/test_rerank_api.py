"""`POST /v1/rerank` (Cohere/Jina shape) over the search cross-encoder, with
the engine stubbed: shape, ordering, normalisation and bounds."""

import unittest
from unittest import mock

from core import runtime
from core.app import create_app
from core.util import rerank as rr


class _Slot:
    device_name = "NPU"
    status = "ready"

    def __init__(self, logits):
        self.logits = logits
        self.calls = []

    def ensure_loaded(self):
        pass

    def run_utility(self, name, query, docs):
        self.calls.append((name, query, list(docs)))
        return [self.logits[d] for d in docs], {"infer": 1.0}


DOCS = ["cats purr", "the NPU is a low-power engine", "bread recipe"]
LOGITS = {"cats purr": -3.0, "the NPU is a low-power engine": 4.0,
          "bread recipe": -8.0}


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()
        self.slot = _Slot(LOGITS)
        for name, value in (("util_npu", self.slot), ("util_gpu", None),
                            ("util_cpu", None)):
            p = mock.patch.object(runtime, name, value)
            p.start()
            self.addCleanup(p.stop)

    def post(self, body):
        return self.client.post("/v1/rerank", json=body)

    def test_results_are_best_first_with_original_indexes(self):
        r = self.post({"model": "anything", "query": "what is an NPU",
                       "documents": DOCS})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["X-Device"], "NPU")
        self.assertEqual(r.json["model"], rr.RERANK_MODEL)
        self.assertEqual([h["index"] for h in r.json["results"]], [1, 0, 2])
        scores = [h["relevance_score"] for h in r.json["results"]]
        self.assertTrue(all(0.0 < s < 1.0 for s in scores))
        self.assertAlmostEqual(scores[0], 0.982014, places=5)   # sigmoid(4)
        self.assertNotIn("document", r.json["results"][0])
        self.assertEqual(self.slot.calls[0][0], "rerank")

    def test_top_n_and_return_documents(self):
        r = self.post({"query": "q", "documents": [{"text": d} for d in DOCS],
                       "top_n": 1, "return_documents": True})
        self.assertEqual(r.json["results"],
                         [{"index": 1, "relevance_score": 0.982014,
                           "document": {"text": DOCS[1]}}])

    def test_top_n_larger_than_documents_returns_all(self):
        r = self.post({"query": "q", "documents": DOCS, "top_n": 50})
        self.assertEqual(len(r.json["results"]), 3)

    def test_sigmoid_is_stable_at_extremes(self):
        self.assertEqual(rr._sigmoid(-1000.0), 0.0)
        self.assertEqual(rr._sigmoid(1000.0), 1.0)

    def test_bad_requests_are_400_and_never_reach_the_engine(self):
        for body in ({}, {"query": "q"}, {"query": "", "documents": DOCS},
                     {"query": "q", "documents": []},
                     {"query": "q", "documents": ["ok", ""]},
                     {"query": "q", "documents": [{"txt": "x"}]},
                     {"query": "q", "documents": DOCS, "top_n": 0},
                     {"query": "q", "documents": DOCS, "top_n": "2"},
                     {"query": "q", "documents": ["x"] * (rr.MAX_DOCUMENTS + 1)}):
            r = self.post(body)
            self.assertEqual(r.status_code, 400, str(body)[:80])
        self.assertEqual(self.slot.calls, [])

    def test_missing_reranker_is_503(self):
        from utility_pipeline import UtilityUnavailable

        def missing(*_a, **_k):
            raise UtilityUnavailable("The rerank utility is unavailable on NPU")
        self.slot.run_utility = missing
        r = self.post({"query": "q", "documents": DOCS})
        self.assertEqual(r.status_code, 503)


if __name__ == "__main__":
    unittest.main()
