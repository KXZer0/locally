"""Live GPU fit verdicts for the on-disk model catalog.

The driver ceiling is not free memory on an integrated GPU. One system-memory
snapshot is taken for the whole catalog request, then every model is evaluated
against the same `min(ceiling, free - reserve)` budget used at load time. Static
model facts may take several filesystem stats; live memory is never re-read in
the per-model loop.
"""

import math

from core import config
from core.hardware.devices import (_gpu_has_xmx, _gpu_shares_system_ram,
                                   _usable_gpu_bytes)
from core.hardware.memory import _mem_status
from core.models.geometry import (_effective_kv_bytes_per_token,
                                  _model_max_context, _moe_expert_fraction)
from core.models.integrity import _dir_size_bytes


_GIB = 2 ** 30
_MIB = 2 ** 20


def _mb(value):
    return None if value is None else round(value / _MIB)


def _configured_kv_bytes(model_dir, vlm, ceiling, weights):
    """Fixed KV allocation this model would receive on a GPU load."""
    per_token = _effective_kv_bytes_per_token(model_dir)
    if vlm or config.PROMPT_CACHE is not True:
        return 0, per_token
    if config.CONTEXT_TOKENS is None or not per_token:
        return max(0, config.PROMPT_CACHE_GB) * _GIB, per_token

    max_context = _model_max_context(model_dir) or 0
    if config.CONTEXT_TOKENS == "auto":
        if not ceiling or not weights:
            return max(0, config.PROMPT_CACHE_GB) * _GIB, per_token
        pool_gb = max(1, int((ceiling - weights * 1.1 - 2 * _GIB) // _GIB))
    else:
        pool_gb = max(1, math.ceil(config.CONTEXT_TOKENS * per_token / _GIB))

    if max_context:
        pool_gb = min(pool_gb, max(1, math.ceil(max_context * per_token / _GIB)))
    return pool_gb * _GIB, per_token


def _offload_ratio(weights, need, budget, expert_fraction, has_xmx):
    """Estimated configured ratio, or None when offload cannot make it fit."""
    if (not weights or not budget or not expert_fraction or not has_xmx
            or config.OFFLOAD_RATIO == 0):
        return None
    if config.OFFLOAD_RATIO == "auto":
        overflow = max(0, need - budget)
        ratio = int(100 * overflow / (weights * expert_fraction)) + 5
        ratio = max(1, min(90, ratio))
    else:
        ratio = int(config.OFFLOAD_RATIO)
    resident_need = need - weights * expert_fraction * ratio / 100
    return ratio if resident_need <= budget else None


def _fit_one(model, budget, ceiling, free, shares_ram, has_xmx):
    weights = _dir_size_bytes(model["path"])
    kv, per_token = _configured_kv_bytes(
        model["path"], model.get("type") == "vlm", ceiling, weights)
    need = weights * 1.05 + kv + _GIB if weights is not None else None
    fits = bool(need is not None and budget is not None and need <= budget)
    expert_fraction = _moe_expert_fraction(model["path"])
    offload_ratio = (None if fits or need is None else
                     _offload_ratio(weights, need, budget, expert_fraction, has_xmx))
    would_offload = offload_ratio is not None

    close_bytes = None
    if (not fits and shares_ram and need is not None and free is not None
            and ceiling is not None and need <= ceiling):
        close_bytes = max(0, need + config.GPU_RESERVE_BYTES - free)

    if need is None or budget is None:
        verdict = "unknown"
        message = "fit unknown — model size or GPU budget is unavailable"
    elif fits:
        verdict = "fits_now"
        message = (f"fits now ({need / _GIB:.1f} GB needed, "
                   f"{budget / _GIB:.1f} GB budget)")
    elif would_offload:
        verdict = "would_offload_now"
        message = f"would offload now ({offload_ratio}% of expert weights)"
        if close_bytes:
            message += f" — close ~{math.ceil(close_bytes / _GIB)} GB to fix"
    else:
        verdict = "does_not_fit_now"
        message = (f"does not fit now ({need / _GIB:.1f} GB needed, "
                   f"{budget / _GIB:.1f} GB budget)")
        if close_bytes:
            message += f" — close ~{math.ceil(close_bytes / _GIB)} GB to fix"
        elif ceiling is not None and need > ceiling:
            message += (f" — the driver's {ceiling / _GIB:.1f} GB ceiling "
                        "must also be raised")

    return {
        "verdict": verdict,
        "message": message,
        "fits_now": fits,
        "would_offload_now": would_offload,
        "estimated_offload_ratio": offload_ratio,
        "weights_mb": _mb(weights),
        "kv_mb": _mb(kv),
        "kv_cache_mb": _mb(kv),
        "kv_precision": config.KV_PRECISION or "f16",
        "kv_bytes_per_token": per_token,
        "estimated_need_mb": _mb(need),
        "budget_mb": _mb(budget),
        "free_mb": _mb(free),
        "system_available_mb": _mb(free),
        "driver_ceiling_mb": _mb(ceiling),
        "reserve_mb": _mb(config.GPU_RESERVE_BYTES),
        "close_to_fit_mb": (math.ceil(close_bytes / _MIB)
                            if close_bytes is not None else None),
    }


def add_live_fit_data(models, devices):
    """Copy catalog entries and attach one coherent request-time fit snapshot."""
    _total, free = _mem_status()  # exactly one live memory read per request
    gpu = devices.get("GPU")
    if gpu:
        gpu_id = gpu.get("id", "GPU")
        budget, ceiling = _usable_gpu_bytes("GPU", gpu_id, free)
        shares_ram = _gpu_shares_system_ram(gpu_id)
        has_xmx = _gpu_has_xmx(gpu_id)
        # `_usable_gpu_bytes` deliberately falls back to the ceiling when a
        # load-time memory read fails: trying the load is preferable to making
        # the server unusable. A catalog claim is different. Without a live
        # reading there is no honest "fits now" verdict for shared memory.
        if shares_ram and free is None:
            budget = None
    else:
        budget = ceiling = None
        shares_ram = has_xmx = False

    out = []
    for model in models:
        entry = dict(model)
        entry["fit"] = _fit_one(
            entry, budget, ceiling, free, shares_ram, has_xmx)
        out.append(entry)
    return out
