"""An image sent to an NPU text model is read with OCR and the text trimmed to
fit the NPU's prompt cap. The cap is a flag (--npu-prompt-len, default 8192);
the trim used a literal 4096 that outlived the raise, so every screenshot was
cut to about half the room the model actually had.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from core import config, runtime
from core.chat import images


def _words(n):
    return " ".join(["word"] * n)


def _count(_slot, text):
    """One token per whitespace-separated word: easy arithmetic, real shape."""
    return len(text.split())


class _Util:
    device_name = "NPU"

    def __init__(self, text):
        self.text = text

    def ensure_loaded(self):
        pass

    def read(self, _img):
        return {"text": self.text}


IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}


def _messages(question="What does this say?", system=None):
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user",
                 "content": [{"type": "text", "text": question}, IMAGE]})
    return msgs


class NpuOcrBudgetTests(unittest.TestCase):
    def setUp(self):
        self.limit = config.NPU_MAX_PROMPT_LEN
        self.npu = SimpleNamespace(device_name="NPU")
        patches = [mock.patch.object(images, "_count_tokens", _count),
                   mock.patch.object(images, "load_image", lambda *_a: object()),
                   mock.patch.object(images, "_slot_serviceable", lambda _s: True)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        config.NPU_MAX_PROMPT_LEN = self.limit

    def _ocr(self, ocr_words, messages, slot=None):
        util = _Util(_words(ocr_words))
        with mock.patch.object(runtime, "util_npu", util), \
                mock.patch.object(runtime, "util_gpu", None), \
                mock.patch.object(runtime, "util_cpu", None):
            return images._ocr_for_text_model(messages, slot or self.npu)

    def test_budget_follows_the_configured_limit(self):
        msgs = _messages()          # 4 words of question
        config.NPU_MAX_PROMPT_LEN = 8192
        self.assertEqual(images._npu_ocr_budget(self.npu, msgs),
                         8192 - 4 - images._NPU_OCR_RESERVE)
        config.NPU_MAX_PROMPT_LEN = 4096
        self.assertEqual(images._npu_ocr_budget(self.npu, msgs),
                         4096 - 4 - images._NPU_OCR_RESERVE)

    def test_the_rest_of_the_conversation_is_charged(self):
        config.NPU_MAX_PROMPT_LEN = 8192
        msgs = _messages(system=_words(220))
        self.assertEqual(images._npu_ocr_budget(self.npu, msgs),
                         8192 - 224 - images._NPU_OCR_RESERVE)

    def test_uncountable_conversation_falls_back_to_the_fixed_reserve(self):
        config.NPU_MAX_PROMPT_LEN = 8192
        with mock.patch.object(images, "_count_tokens", lambda *_a: None):
            self.assertEqual(images._npu_ocr_budget(self.npu, _messages()),
                             8192 - images._NPU_OCR_FALLBACK_RESERVE)

    def test_budget_never_goes_below_the_floor(self):
        config.NPU_MAX_PROMPT_LEN = 1024
        msgs = _messages(question=_words(5000))
        self.assertEqual(images._npu_ocr_budget(self.npu, msgs),
                         images._NPU_OCR_MIN_BUDGET)

    def test_a_6000_token_page_fits_the_8192_window_untrimmed(self):
        # Under the old literal (4096 - 700 = 3396) this was cut almost in half.
        config.NPU_MAX_PROMPT_LEN = 8192
        block, note = self._ocr(6000, _messages())
        self.assertIsNone(note)
        self.assertNotIn("[Truncated", block)

    def test_an_oversized_page_is_trimmed_and_says_so(self):
        config.NPU_MAX_PROMPT_LEN = 8192
        block, note = self._ocr(12000, _messages())
        self.assertIn("[Truncated", block)
        budget = 8192 - 4 - images._NPU_OCR_RESERVE
        self.assertIn(f"~{budget}", note)
        # Proportional cut: the kept text is close to the budget, not half of it.
        kept = len(block.split("[Truncated")[0].split())
        self.assertGreater(kept, budget * 0.9)
        self.assertLessEqual(kept, budget + 20)

    def test_a_lowered_flag_trims_harder(self):
        config.NPU_MAX_PROMPT_LEN = 4096
        _block, note = self._ocr(6000, _messages())
        self.assertIsNotNone(note)

    def test_gpu_slots_are_not_trimmed(self):
        config.NPU_MAX_PROMPT_LEN = 1024
        block, note = self._ocr(12000, _messages(),
                                slot=SimpleNamespace(device_name="GPU"))
        self.assertIsNone(note)
        self.assertNotIn("[Truncated", block)


if __name__ == "__main__":
    unittest.main()
