import unittest
from unittest import mock

from core.tools import builtin


class BuiltinSearchToolTests(unittest.TestCase):
    def _run(self, payload, query='"latest version"'):
        seen = {}

        def fake(body):
            seen.update(body)
            yield "done", payload

        with mock.patch.object(builtin, "_web_search_run", fake), \
                mock.patch.object(builtin.web_search_mod, "WEB_SEARCH_URL", "http://x"):
            return builtin._run_web_search_tool({"query": query}), seen

    def test_quotes_stripped_and_one_passage_per_page(self):
        payload = {"results": [
            {"title": "A", "url": "u1", "text": "t1"},
            {"title": "A", "url": "u1", "text": "t1b"},
            {"title": "B", "url": "u2", "text": "t2"}],
            "snippets": {"u1": "openvino 2026.4.1"}, "sources": []}
        out, seen = self._run(payload)
        self.assertEqual(seen["query"], "latest version")
        self.assertEqual(seen["per_url"], 1)
        self.assertIn("openvino 2026.4.1", out)
        self.assertNotIn("t1b", out)
        self.assertIn("[2] B", out)


if __name__ == "__main__":
    unittest.main()
