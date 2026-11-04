"""Speculative decoding on an LLM slot: a draft model, or prompt lookup.

Both are lossless under greedy decoding -- the main model verifies every
proposed token, so the answer is the one it would have written alone (checked
here on random-weight IRs: identical output with and without, on the plain and
the prefix-caching backend). What they buy is fewer main-model steps, which is
what a bandwidth-bound decode is paying for. What they cost is decided by the
backend, and that is the part this module exists to get right:

* genai 2026.4.1 builds BOTH on the continuous-batching (paged attention)
  backend, even when no scheduler_config is passed. On a GPU slot that is the
  backend CLAUDE.md records as prefilling ~10x slower on the B390. Prefix
  caching stays OFF there unless --prompt-cache, so the measured hang is not
  re-enabled, but the slow prefill may come with the backend itself. Nothing
  here claims a GPU speedup; scripts/speculative-check.py measures it.
* Draft + sampling crashed genai 2026.4.1 on every rejected draft
  ("content_length <= prompt_ids.size() + m_generated_ids.size()"), and the
  pipeline then silently stopped drafting for the rest of its life. So a
  sampled turn on a draft slot runs greedy unless --draft-sampling. EAGLE-3
  and MTP drafts are greedy-only in genai anyway. Prompt lookup sampled fine.
* Prompt lookup refuses any request that lacks num_assistant_tokens AND
  max_ngram_size, and a plain pipeline refuses max_ngram_size != 0, so every
  GenerationConfig goes through apply() on its way to generate().
* genai checks draft/main tokenizers on some paths and not others: a draft
  whose logits were 400 wide against a 385-token tokenizer loaded and ran with
  no complaint. check_tokenizers() compares the vocabularies themselves.
"""

import os

from core import config
from core.genai.draft_check import check_tokenizers, draft_kind  # noqa: F401 (re-exported)

MAX_ASSISTANT_TOKENS = 16


def plan(slot, vlm):
    """Decide this load's speculation. Returns the state dict /health shows."""
    state = {"mode": None, "draft": None, "draft_dir": None, "draft_device": None,
             "kind": None, "skipped": None, "notes": [], "draft_bytes": 0}
    want_draft, want_pl = bool(config.DRAFT_MODELS), bool(config.PROMPT_LOOKUP)
    if not (want_draft or want_pl):
        return state
    model_dir = str(slot.model_dir)
    if vlm:
        state["skipped"] = ("VLM slot: genai's only VLM speculative path is DFlash, "
                            "which needs a purpose-built DFlash draft")
        return state
    if model_dir.lower().endswith(".gguf"):
        state["skipped"] = "GGUF model: speculation is offered for OpenVINO IR models only"
        return state
    if want_draft:
        refusals = []
        for d in config.DRAFT_MODELS:
            if os.path.normpath(d) == os.path.normpath(model_dir):
                continue   # a model is not its own draft
            refusal, warning = check_tokenizers(model_dir, d)
            if refusal:
                refusals.append(refusal)
                continue
            from core.models.integrity import _dir_size_bytes
            state.update(mode="draft", draft=os.path.basename(os.path.normpath(d)),
                         draft_dir=d, kind=draft_kind(d),
                         draft_device=config.DRAFT_DEVICE or slot.device_name,
                         draft_bytes=_dir_size_bytes(d) or 0)
            if warning:
                state["notes"].append(warning)
            break
        if state["mode"] is None:
            state["skipped"] = "; ".join(refusals) or "no usable draft"
    if want_pl and state["mode"] == "draft":
        state["notes"].append("--prompt-lookup ignored: genai runs a draft OR prompt lookup, not both")
    elif want_pl:
        refused = state["skipped"]
        if slot.device_name == "NPU":
            state["skipped"] = "; ".join(filter(None, [refused, (
                "prompt lookup runs on the continuous-batching backend, "
                "which the NPU does not have")]))
            return state
        state.update(mode="prompt_lookup", skipped=None, kind="prompt_lookup")
        if refused:
            state["notes"].append(f"no draft: {refused}")
    if state["mode"] and slot.device_name == "GPU":
        state["notes"].append(
            "GPU: speculation runs on the continuous-batching backend, which "
            "prefilled ~10x slower than the plain pipeline on the B390 (prefix "
            "caching stays " + ("on" if slot._prompt_cache_on() else "off") +
            "); measure with scripts/speculative-check.py")
    if state["mode"] == "draft" and "NPU" in (slot.device_name, state["draft_device"]):
        state["notes"].append("NPU speculation is genai's stateful path and is "
                              "unverified here; a failed build falls back to no draft")
    return state


def pipeline_kwargs(state, device_props):
    """The LLMPipeline kwargs for this plan. Built fresh on every load: genai
    transforms the draft model in place, so a reused draft_model() object
    fails the second pipeline with 'No ScaledDotProductAttention ...'."""
    if state.get("mode") == "prompt_lookup":
        return {"prompt_lookup": True}
    if state.get("mode") != "draft":
        return {}
    import openvino_genai as ovg
    props = {}
    if device_props.get("CACHE_DIR"):
        props["CACHE_DIR"] = device_props["CACHE_DIR"]
    if state["draft_device"] == "NPU":
        props["MAX_PROMPT_LEN"] = config.NPU_MAX_PROMPT_LEN
    return {"draft_model": ovg.draft_model(state["draft_dir"], state["draft_device"], **props)}


def _clamp_tokens(n):
    return max(1, min(MAX_ASSISTANT_TOKENS, int(n)))


def request_overrides(gen, options):
    """Copy per-request knobs from a request body / Ollama options onto gen.

    All three are lossless (they change how much is proposed, never what is
    accepted), so a client may set them. Invalid values are ignored, and on a
    slot without speculation apply() clears them again.
    """
    if not isinstance(options, dict):
        return
    try:
        if options.get("num_assistant_tokens") is not None:
            gen.num_assistant_tokens = _clamp_tokens(options["num_assistant_tokens"])
        if options.get("assistant_confidence_threshold") is not None:
            gen.assistant_confidence_threshold = min(1.0, max(0.0, float(
                options["assistant_confidence_threshold"])))
        if options.get("max_ngram_size") is not None:
            gen.max_ngram_size = max(1, min(8, int(options["max_ngram_size"])))
    except (TypeError, ValueError):
        pass


def apply(gen, state):
    """Make gen valid for this slot's pipeline. Returns a note, or ''."""
    mode = (state or {}).get("mode")
    if not mode:
        if any(getattr(gen, f, None) for f in
               ("max_ngram_size", "num_assistant_tokens", "assistant_confidence_threshold")):
            gen.max_ngram_size = 0
            gen.num_assistant_tokens = 0
            gen.assistant_confidence_threshold = 0.0
        return ""
    n = gen.num_assistant_tokens or config.NUM_ASSISTANT_TOKENS
    conf = gen.assistant_confidence_threshold or config.ASSISTANT_CONFIDENCE
    note = ""
    if mode == "prompt_lookup":
        gen.num_assistant_tokens = _clamp_tokens(n)
        gen.assistant_confidence_threshold = 0.0
        gen.max_ngram_size = gen.max_ngram_size or config.MAX_NGRAM_SIZE
        return note
    gen.max_ngram_size = 0
    if state.get("kind") in ("eagle3", "mtp"):
        conf = 0.0          # genai: static candidate count only
    if conf > 0:
        gen.num_assistant_tokens = 0
        gen.assistant_confidence_threshold = float(conf)
    else:
        gen.assistant_confidence_threshold = 0.0
        gen.num_assistant_tokens = _clamp_tokens(n)
    if gen.do_sample and (state.get("kind") != "draft" or not config.DRAFT_SAMPLING):
        gen.do_sample = False
        gen.top_k = 1
        note = ("sampled request run greedy: draft decoding with sampling crashes "
                "genai 2026.4.1 on a rejected draft (--draft-sampling to allow)"
                if state.get("kind") == "draft" else
                f"sampled request run greedy: {state.get('kind')} drafts are greedy-only")
    return note


def stats(result):
    """{'acceptance', 'tokens_per_step', ...} from a result, or None.

    tokens_per_step works for both modes (generated tokens / main-model
    inference calls); 1.0 means speculation bought nothing on this turn.
    Acceptance exists only for a draft model -- prompt lookup reports none.
    """
    try:
        pm = result.perf_metrics
        generated = int(pm.get_num_generated_tokens())
        steps = len(pm.raw_metrics.token_infer_durations)
    except Exception:
        return None
    out = {"tokens_per_step": round(generated / steps, 2) if steps else None}
    ext = getattr(result, "extended_perf_metrics", None)
    if ext is not None and hasattr(ext, "get_draft_acceptance_rate"):
        try:
            rate = float(ext.get_draft_acceptance_rate())
            out["acceptance"] = round(rate, 3) if rate == rate else None
            out["accepted"] = int(ext.get_num_accepted_tokens())
            out["drafted"] = int(ext.get_num_draft_tokens())
        except Exception:
            pass
    return out


def log_suffix(s):
    if not s:
        return ""
    parts = []
    if s.get("acceptance") is not None:
        parts.append(f"draft acceptance {s['acceptance'] * 100:.0f}% "
                     f"({s.get('accepted')}/{s.get('drafted')})")
    if s.get("tokens_per_step") is not None:
        parts.append(f"{s['tokens_per_step']:.2f} tok/step")
    return (", " + ", ".join(parts)) if parts else ""


def info(state, last):
    """What /health shows for a slot."""
    if not state or not (state.get("mode") or state.get("skipped")):
        return None
    return {"mode": state.get("mode"), "draft": state.get("draft"),
            "draft_device": state.get("draft_device"), "kind": state.get("kind"),
            "skipped": state.get("skipped"), "notes": state.get("notes") or None,
            "last": last}
