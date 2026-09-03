"""How much memory and context a slot actually gets.

Six methods that run only during `load()` and answer one question, kept apart
from DeviceSlot so that the class is about SERVING a model and this is about
SIZING it. A mixin rather than free functions: every one of them reads several
fields of `self`, and threading those through as arguments would make the call
sites worse to read than the split makes the file.

The arithmetic here decides more about how the server feels than anything else
in it. Auto-offload measures against LIVE FREE RAM, not the driver's advertised
ceiling: on an integrated GPU that ceiling is a policy share of the same system
RAM everything else is using, and trusting it once put a 20.2 GB model on a
machine with 0.9 GB free, which then decoded out of the pagefile at 0.5 tok/s.
It fit the policy, not the machine.
"""
import gc

import openvino as ov

from core import config
from core.hardware.devices import _device_mem_bytes, _gpu_has_xmx, _usable_gpu_bytes
from core.hardware.memory import _mem_status
from core.models.geometry import (_effective_kv_bytes_per_token,
                                  _model_max_context, _moe_expert_fraction,
                                  _text_config)
from core.models.integrity import _dir_size_bytes


class MemoryPlanning:
    """Sizing decisions for a DeviceSlot. Mixed in, never instantiated."""

    def _prompt_cache_on(self):
        """Whether this slot uses the prefix-caching (CB) backend.

        Unset means per-device: CPU yes, GPU no. See config.PROMPT_CACHE for the
        measurements — the same feature is a large win on one device and a
        10x regression plus a hang on the other, so "default on" was never a
        answerable question globally.
        """
        if config.PROMPT_CACHE is not None:
            return config.PROMPT_CACHE
        return self.device_name == "CPU"

    def _resolve_kv_pool_gb(self, vlm):
        """How many GB of KV pool this slot gets, and why.

        The KV pool *is* the usable context window — exceed it and generation
        hard-fails mid-request (issue #21), so it is the number that decides
        whether an agent session survives. GB is the wrong unit to ask a user
        for, though: what they know is "I need 90k tokens of context". So
        --context-tokens takes the number they actually have, and the model's
        own geometry converts it. "auto" fills whatever the device has left
        after weights, capped at the model's real context length — a pool
        bigger than the model can address is just memory nobody can use.

        Returns (gb, note). gb of 0 means no pool: VLM and NPU slots use the
        plain pipeline, which has no continuous-batching cache to size.
        """
        if not config.PROMPT_CACHE or vlm or self.device_name not in ("GPU", "CPU"):
            return 0, ""
        per_tok = _effective_kv_bytes_per_token(self.model_dir)
        gib = 2 ** 30
        if config.CONTEXT_TOKENS is None or not per_tok:
            return config.PROMPT_CACHE_GB, ""

        max_ctx = 0
        try:
            cfg = _text_config(self.model_dir)
            max_ctx = int(cfg.get("max_position_embeddings") or 0)
        except Exception:
            pass

        if config.CONTEXT_TOKENS == "auto":
            mem = _device_mem_bytes(self.device_name, self.device_id)
            weights = _dir_size_bytes(self.model_dir)
            if not mem or not weights:
                return config.PROMPT_CACHE_GB, " (auto unavailable: no device budget)"
            # Leave headroom for activations and for the rest of the machine;
            # on a unified-memory laptop the "device budget" is RAM other
            # programs are also using.
            spare = mem - weights * 1.1 - 2 * gib
            gb = max(1, int(spare // gib))
            note = " (auto)"
        else:
            gb = max(1, -(-config.CONTEXT_TOKENS * per_tok // gib))  # ceil
            note = f" (for --context-tokens {config.CONTEXT_TOKENS})"

        if max_ctx:
            cap = max(1, -(-max_ctx * per_tok // gib))
            if gb > cap:
                gb, note = cap, note + f", capped at the model's {max_ctx // 1024}k context"
        return int(gb), note

    def _resolve_offload_ratio(self):
        """How much of the expert weights to stream from disk. 0 = none.

        Offload is a knob nobody should have to turn by hand: the right value
        is whatever makes the model fit, and that is arithmetic — weights + KV
        pool against the device budget, divided by the share of the weights
        that are actually offloadable. Too low and the load OOMs; too high and
        decode crawls for no reason (measured on a 140V: ratio 30 -> 25.3
        tok/s, ratio 90 -> 5.1 for the same model). So: compute the smallest
        ratio that fits, add a small margin, and say so.

        Silent no-ops are the trap this has to avoid (TODONT.md), so it only
        engages where offload does anything at all: GPU + XMX + MoE.
        """
        if config.OFFLOAD_RATIO != "auto":
            return int(config.OFFLOAD_RATIO or 0), ""
        if self.device_name != "GPU":
            return 0, ""
        fraction = _moe_expert_fraction(self.model_dir)
        if not fraction:
            return 0, ""                      # dense: offload cannot help
        if not _gpu_has_xmx(self.device_id):
            return 0, ""                      # no systolic HW: silent no-op
        # What the machine can actually back, not what the driver advertises:
        # on a shared-memory iGPU those are different numbers, and believing
        # the advertised one is what put an 18.3 GB model in the pagefile.
        budget, ceiling = _usable_gpu_bytes(self.device_name, self.device_id)
        weights = _dir_size_bytes(self.model_dir)
        if not budget or not weights:
            return 0, ""
        gib = 2 ** 30
        need = weights * 1.05 + self.kv_pool_gb * gib + gib   # +1 GB runtime
        if need <= budget:
            return 0, ""
        overflow = need - budget
        ratio = int(100 * overflow / (weights * fraction)) + 5   # margin
        ratio = max(1, min(90, ratio))
        note = (f" (auto: {weights / gib:.1f} GB weights + "
                f"{self.kv_pool_gb} GB KV exceed the "
                f"{budget / gib:.1f} GB usable")
        # When free RAM is the binding constraint, say so and name the other
        # number — otherwise the log reads as if the GPU were too small, and
        # the actual fix (close something, or lower the Shared Memory
        # Override so this stops lying) is invisible.
        if ceiling and ceiling > budget * 1.05:
            note += (f"; driver advertises {ceiling / gib:.1f} GB, but this "
                     f"iGPU shares system RAM and that much is not free")
        return ratio, note + ")"

    def _preflight_memory(self, vlm):
        """Sanity-check model weights + KV pool against the device's memory
        budget before loading. Warns and keeps going — the numbers are
        estimates, and on 16 GB cards OpenVINO's silent CPU fallback (or a
        'Got unfinished GenerationStatus' abort mid-request) is far worse
        than a false-positive warning here.
        """
        gib = 2 ** 30
        mem = _device_mem_bytes(self.device_name, self.device_id)
        weights = _dir_size_bytes(self.model_dir)
        if not mem or not weights:
            return  # can't estimate — stay quiet rather than guess
        kv_pool = self.kv_pool_gb * gib
        need = (weights + kv_pool) * 1.1  # ~10% runtime/activation overhead
        if need > mem:
            if self.offload_ratio and self.device_name == "GPU":
                # MoE disk offload keeps only part of the expert weights
                # resident; the estimate above ignores that (expert share
                # isn't knowable from config geometry alone). Inform, don't
                # cry wolf — a 15.2 GB MoE serves fine on a 16 GB iGPU at
                # --offload-ratio 30 (measured).
                print(f"  [{self.device_name}] model (~{weights / gib:.1f} GB)"
                      f"{f' + KV pool ({kv_pool // gib} GB)' if kv_pool else ''} "
                      f"exceeds the {mem / gib:.1f} GB device budget — "
                      f"offloading {self.offload_ratio}% of the expert weights "
                      f"to disk to fit", flush=True)
            else:
                hint = ("use a smaller quant or lower --cache-size-gb"
                        if self.device_name == "CPU" else
                        "raise the iGPU budget (Intel Graphics Software -> Shared GPU "
                        "Memory Override), use a smaller quant, or lower --cache-size-gb")
                print(f"  [{self.device_name}] WARNING: model (~{weights / gib:.1f} GB)"
                      f"{f' + KV pool ({kv_pool // gib} GB)' if kv_pool else ''} needs "
                      f"~{need / gib:.1f} GB but the device budget is {mem / gib:.1f} GB "
                      f"— this will likely NOT work ({hint})", flush=True)
        pool_capacity = None
        if kv_pool:
            per_tok = _effective_kv_bytes_per_token(self.model_dir)
            if per_tok:
                pool_capacity = int(kv_pool // per_tok)
                line = (f"  [{self.device_name}] KV pool {self.kv_pool_gb} GB"
                        f"{self.kv_pool_note} ~ {pool_capacity // 1000}k tokens of "
                        f"context ({per_tok // 1024} KB/token)")
                if pool_capacity < 32768:
                    line += (" — agent prompts (20k+ tokens) will exhaust it; "
                             "raise --context-tokens")
                print(line, flush=True)

    def _kv_pool_capacity(self):
        """How many tokens the KV pool holds, or None when there is no pool."""
        if not self.kv_pool_gb:
            return None
        per_tok = _effective_kv_bytes_per_token(self.model_dir)
        if not per_tok:
            return None
        return int((self.kv_pool_gb * (2 ** 30)) // per_tok)

    def _resolve_context_window(self):
        """Decide the usable context window, and record what is binding it.

        This used to live inside `if kv_pool:` in the memory preflight, so it
        only ran when prefix caching built a pool. The launcher passes
        --no-prompt-cache on GPU (the CB backend is a trap there, see
        TODONT.md), so GPU slots reported no context at all and the UI had no
        denominator to draw a ring against — "only the NPU tracks context" was
        that, not a client bug.

        Every candidate constraint is collected and the smallest wins, so the
        reported number is always one a request can actually reach. Naming the
        winner matters: 32k on a 262k model is not a mistake if the pool is
        what is binding, but it looks like one without the reason attached.
        """
        pool_capacity = self._kv_pool_capacity()
        limits = []
        model_max = _model_max_context(self.model_dir)
        if model_max:
            limits.append((model_max, "model"))
        if self.device_name == "NPU":
            limits.append((config.NPU_MAX_PROMPT_LEN, "npu prompt cap"))
        if pool_capacity:
            limits.append((pool_capacity, "kv pool"))
        if not limits:
            self.context_tokens = None
            self.context_limit_by = None
            return
        self.context_tokens, self.context_limit_by = min(limits, key=lambda x: x[0])
        if len(limits) > 1:
            print(f"  [{self.device_name}] context window "
                  f"{self.context_tokens // 1000}k tokens "
                  f"(limited by {self.context_limit_by})", flush=True)
