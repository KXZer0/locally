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


def _kv_attention_layers(cfg):
    """How many layers actually hold a KV cache.

    A hybrid model interleaves linear-attention layers (Gated DeltaNet and
    friends) with full-attention ones, and a linear layer keeps a fixed-size
    recurrent state, not a per-token cache — it contributes nothing that grows
    with context. `layer_types` names them one by one, so count only the
    full-attention entries.

    This matters by a factor of four on the models this project is moving to.
    Qwen3.5-9B is 24 linear + 8 full: counting all 32 reports 128 KB/token when
    the truth is 32 (16 at u8). Everything downstream is sized off this number —
    `--context-tokens` converts it into a pool, `_device_fits` decides placement
    with it — so the over-count asks for four times the memory a context needs
    and rules the model out of a device it fits in comfortably. The same shape
    covers Qwen3.6 and Muse-Glimmer; a dense model has no `layer_types` and
    falls through to the full count, unchanged.
    """
    layers = cfg["num_hidden_layers"]
    types = cfg.get("layer_types")
    if not isinstance(types, list) or not types:
        return layers
    full = sum(1 for t in types if isinstance(t, str) and "full" in t)
    # A config that lists types but names none of them full-attention is more
    # likely a spelling this code does not know than a model with no KV cache.
    return full or layers


def _kv_layer_split(cfg):
    """(full, sliding, window) — the three things that decide how KV grows.

    Two different things hide behind "not full attention" and they must not be
    treated alike:

    * `linear_attention` (Qwen3.5, Qwen3.6) keeps a fixed recurrent state and
      no per-token cache at all. Excluding it is right at every context length.
    * `sliding_attention` (gemma-4) keeps a real KV cache, capped at
      `sliding_window`. Excluding it under-counts by a CONSTANT — 25 layers x
      1024 tokens on gemma-4-26b — and under-counting is the direction that
      hard-fails a generation with "Got unfinished GenerationStatus" (#21).

    So sliding layers are counted, but as a fixed cost rather than a rate.
    """
    layers = cfg["num_hidden_layers"]
    types = cfg.get("layer_types")
    window = int(cfg.get("sliding_window") or 0)
    if not isinstance(types, list) or not types:
        return layers, 0, 0
    full = sum(1 for t in types if isinstance(t, str) and "full" in t)
    sliding = sum(1 for t in types if isinstance(t, str) and "sliding" in t)
    if not full:
        return layers, 0, 0
    return full, (sliding if window else 0), window


def _kv_bytes_per_layer_token(cfg):
    """K+V bytes for one token in one attention layer, fp16."""
    heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads") or heads
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
    return 2 * kv_heads * head_dim * 2


def _kv_bytes_for_context(model_dir, tokens):
    """Total KV bytes to hold `tokens` of context, fp16.

    Not `tokens * _kv_bytes_per_token`: a sliding-window layer stops growing at
    its window, so the total is a line plus a constant, not a line through the
    origin. Sizing a pool off the rate alone is what makes a gemma-style model
    ask for the wrong number.
    """
    try:
        cfg = _text_config(model_dir)
        per_layer = _kv_bytes_per_layer_token(cfg)
        full, sliding, window = _kv_layer_split(cfg)
        return per_layer * (full * tokens + sliding * min(tokens, window))
    except Exception:
        return None


def _kv_bytes_per_token(model_dir):
    """KV-cache bytes per token from config.json geometry (K+V, fp16).

    This is the GROWING rate — what one more token of context costs. E.g.
    Qwen2.5-Coder-7B (28 layers x 4 KV heads x 128 head-dim) ≈ 57 KB/tok;
    Qwen3-Coder-30B ≈ 96 KB/tok. None when the geometry can't be read. For the
    total a given context needs, including the fixed part a sliding-window
    layer contributes, use `_kv_bytes_for_context`.

    VLM configs nest the language model under "text_config" (the top level
    holds only the vision/text split), so read through that when present —
    otherwise every VLM silently skipped the KV half of the preflight.
    """
    try:
        cfg = _text_config(model_dir)
        return _kv_attention_layers(cfg) * _kv_bytes_per_layer_token(cfg)
    except Exception:
        return None
