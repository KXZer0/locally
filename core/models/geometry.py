"""Shapes read out of config.json: context length, KV bytes per token, and how
much of a MoE's weight is expert weight.

VLM configs nest their geometry under text_config, which is why _text_config
exists -- without it the KV half of the memory preflight silently no-ops on
every VLM."""

import json
import os


def _model_max_context(model_dir):
    """The model's own context ceiling, from config.json.

    This is the one context number every model has regardless of which backend
    serves it. The KV pool is a *separate*, additional constraint that only
    exists when prefix caching is on -- which is why deriving the context
    window from the pool alone left GPU slots reporting nothing at all.
    """
    try:
        return int(_text_config(model_dir).get("max_position_embeddings") or 0) or None
    except Exception:
        return None


def _text_config(model_dir):
    """config.json for the language model — VLMs nest it under text_config."""
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    nested = cfg.get("text_config")
    return nested if isinstance(nested, dict) else cfg


def _moe_expert_fraction(model_dir):
    """Share of a model's weights that live in MoE experts, 0-1. None if dense.

    This is what makes disk offload worth anything: `--offload-ratio` streams
    *expert* weights only, so on a dense model it is a no-op no matter what
    number you pass (TODONT.md). On an A3B MoE the experts are almost the
    whole model — 48 layers x 128 experts x 3 x 2048 x 768 is ~29B of
    Qwen3-Coder-30B's ~30.5B — which is why streaming a slice of them buys so
    much resident memory for so little speed.
    """
    try:
        cfg = _text_config(model_dir)
        experts = (cfg.get("num_experts") or cfg.get("num_local_experts")
                   or cfg.get("n_routed_experts"))
        moe_dim = (cfg.get("moe_intermediate_size")
                   or cfg.get("expert_intermediate_size"))
        if not experts or not moe_dim:
            return None
        layers = cfg["num_hidden_layers"]
        hidden = cfg["hidden_size"]
        heads = cfg["num_attention_heads"]
        kv_heads = cfg.get("num_key_value_heads") or heads
        head_dim = cfg.get("head_dim") or hidden // heads
        expert_params = layers * experts * 3 * hidden * moe_dim
        attn_params = layers * hidden * head_dim * (2 * heads + 2 * kv_heads)
        embed_params = 2 * cfg.get("vocab_size", 0) * hidden
        total = expert_params + attn_params + embed_params
        return expert_params / total if total else None
    except Exception:
        return None


def _kv_bytes_per_token(model_dir):
    """KV-cache bytes per token from config.json geometry (K+V, fp16).

    E.g. Qwen2.5-Coder-7B (28 layers x 4 KV heads x 128 head-dim) ≈ 57 KB/tok;
    Qwen3-Coder-30B ≈ 96 KB/tok. None when the geometry can't be read.

    VLM configs nest the language model under "text_config" (the top level
    holds only the vision/text split), so read through that when present —
    otherwise every VLM silently skipped the KV half of the preflight.
    """
    try:
        cfg = _text_config(model_dir)
        layers = cfg["num_hidden_layers"]
        heads = cfg["num_attention_heads"]
        kv_heads = cfg.get("num_key_value_heads") or heads
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
        return 2 * layers * kv_heads * head_dim * 2
    except Exception:
        return None
