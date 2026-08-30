"""Counting tokens on a slot's own tokenizer.

Claude Code budgets context with this, so a real count beats chars/4 -- but
it must never be the reason a request fails, hence the quiet None."""

import openvino_genai as ovg


def _count_tokens(slot, text):
    """Token count for `text` on this slot's tokenizer. None if unavailable.

    Claude Code budgets context with this, so a real count beats chars/4 —
    but it must never be the reason a request fails, hence the quiet None.
    """
    try:
        if slot.tokenizer is None:
            slot.tokenizer = ovg.Tokenizer(str(slot.model_dir))
        return int(len(slot.tokenizer.encode(text).input_ids.data[0]))
    except Exception:
        return None
