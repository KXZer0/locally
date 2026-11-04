"""Does a draft speak the main model's tokens? Checked before genai sees it.

genai 2026.4.1 compares draft and main tokenizers on some paths and not
others: a draft whose logits were 400 wide against a 385-token tokenizer
loaded and ran with no complaint, every proposal silently rejected. So the
vocabularies are compared here and a mismatch is refused in words.
"""

import hashlib
import json
import os


def draft_kind(draft_dir):
    """'eagle3', 'mtp' or 'draft' -- genai picks its strategy from the files."""
    if os.path.isfile(os.path.join(draft_dir, "openvino_mtp_model.xml")):
        return "mtp"
    try:
        with open(os.path.join(draft_dir, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        names = " ".join(cfg.get("architectures") or []) + " " + str(cfg.get("model_type", ""))
        if "eagle" in names.lower():
            return "eagle3"
    except (OSError, ValueError):
        pass
    return "draft"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _vocab_size(model_dir):
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg = cfg.get("text_config", cfg)
        return int(cfg.get("vocab_size") or 0) or None
    except (OSError, ValueError, TypeError):
        return None


def _load_tokenizer(model_dir):
    import openvino_genai as ovg
    return ovg.Tokenizer(str(model_dir))


def check_tokenizers(main_dir, draft_dir, load=None):
    """(refusal, warning): refusal is a plain-English reason, or None.

    Identical tokenizer IRs settle it without loading anything. Otherwise the
    vocabularies are compared string by string, plus the special-token ids,
    because two exports of one tokenizer can differ byte-wise and still agree
    on every id -- and that agreement is the only thing speculation needs.
    """
    load = load or _load_tokenizer
    name = os.path.basename(os.path.normpath(str(draft_dir)))
    files = ("openvino_tokenizer.xml", "openvino_tokenizer.bin")
    mains = [os.path.join(main_dir, f) for f in files]
    drafts = [os.path.join(draft_dir, f) for f in files]
    if not all(map(os.path.isfile, drafts)):
        if draft_kind(draft_dir) != "draft":
            return None, None   # EAGLE-3/MTP heads use the main model's tokenizer
        return (f"draft {name} has no openvino_tokenizer.xml/.bin, so there is "
                f"no way to check it speaks the same tokens as the main model"), None
    if not all(map(os.path.isfile, mains)):
        return f"the main model has no openvino_tokenizer to compare {name} against", None
    warning = None
    mv, dv = _vocab_size(main_dir), _vocab_size(draft_dir)
    if mv and dv and mv != dv:
        # Common and harmless when it is only padding (Qwen2.5-0.5B 151936 vs
        # 7B 152064): ids past the real vocabulary are never produced.
        warning = (f"vocab_size differs ({mv} main vs {dv} draft); fine if it "
                   f"is padding, which the tokenizer check below says it is")
    if all(_sha(a) == _sha(b) for a, b in zip(mains, drafts)):
        return None, warning
    hf = _hf_vocab(main_dir), _hf_vocab(draft_dir)
    try:
        a, b = load(main_dir), load(draft_dir)
        special = [(f, getattr(a, f)(), getattr(b, f)()) for f in
                   ("get_eos_token_id", "get_bos_token_id", "get_pad_token_id")]
        if None in hf:
            # No tokenizer.json on one side: the converted IRs are all there
            # is. The vocabulary vector is exact when it decodes (it does not
            # for raw byte tokens), and fixed probes are the fallback.
            try:
                hf = list(a.get_vocab_vector()), list(b.get_vocab_vector())
            except Exception:
                hf = ([list(a.encode(p).input_ids.data.flatten()) for p in _PROBES],
                      [list(b.encode(p).input_ids.data.flatten()) for p in _PROBES])
    except Exception as e:
        return f"could not load both tokenizers to compare them ({e})", None
    va, vb = hf
    if va != vb:
        return (f"draft {name} uses a different tokenizer from the main model "
                f"(the token ids do not line up), so every proposal would be "
                f"wrong. A draft must come from the same model family, e.g. "
                f"Qwen3-0.6B for Qwen3-8B"), None
    for field, x, y in special:
        if x != y:
            return (f"draft {name} has the same vocabulary but a different "
                    f"{field[4:-3].replace('_', ' ')} ({y} vs {x})"), None
    return None, warning


_PROBES = ("Hello, world! The quick brown fox jumps over the lazy dog.",
           "def f(x):\n    return {'a': [1, 2.5e-3, None]}  # comment",
           "Ünïcödé — 中文 日本語 русский العربية 😀",
           "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>")


def _hf_vocab(model_dir):
    """The decisive comparison when both sides ship tokenizer.json: the
    id->token map, the merges and the added (special) tokens."""
    try:
        with open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8") as f:
            tj = json.load(f)
        model = tj.get("model") or {}
        return (sorted((model.get("vocab") or {}).items(), key=lambda kv: kv[1]),
                [list(m) if isinstance(m, list) else m for m in model.get("merges") or []],
                sorted((t.get("id"), t.get("content")) for t in tj.get("added_tokens") or []))
    except (OSError, ValueError, AttributeError, TypeError):
        return None
