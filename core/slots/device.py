"""One model loaded on one device.

Everything about owning weights lives here: sizing the KV pool, choosing an
offload ratio, resolving the real context window, and the two generate paths
(buffered and streamed) for both LLM and VLM pipelines."""

import gc
import json
import openvino_genai as ovg
import threading
import time
from datetime import datetime
from queue import Empty, Queue
from core import config
from core.genai.results import explain_genai_error, extract_perf, extract_text
from core.genai.tokens import _count_tokens
from core.hardware.devices import _device_mem_bytes, _gpu_has_xmx, _usable_gpu_bytes
from core.metrics import record_turn
from core.models.geometry import _kv_bytes_per_token, _model_max_context, _moe_expert_fraction, _text_config
from core.models.identity import is_vlm, model_display_name
from core.models.integrity import _dir_size_bytes, _verify_weights_integrity
from core.slots.capability import _tools_supported
from core.slots.locks import _device_lock
from core.slots.planning import MemoryPlanning


class DeviceSlot(MemoryPlanning):
    """One loaded model on one device."""

    def __init__(self, device_name, device_id=None):
        self.device_name = device_name   # canonical "NPU", "GPU", "CPU" (display + routing)
        self.device_id = device_id or device_name  # OpenVINO id (may be "GPU.1" on multi-GPU)
        self.device_full = ""            # "Intel(R) AI Boost"
        self.pipe = None
        self.model_name = ""
        self.model_type = ""             # "vlm" or "llm"
        self.status = "not_configured"   # not_configured -> loading -> warming_up -> ready / error / idle_unloaded
        self.lock = _device_lock(self.device_name)
        self._cancel = threading.Event()  # signal to stop generation
        self.last_used = time.time()     # for idle-unload watchdog
        self.model_dir = None            # remembered so we can reload after unload
        self.last_ttft_ms = None         # last request's time-to-first-token (prefix-cache hit ≈ low)
        self.prewarmed = False           # did _prewarm_slot succeed for this load
        # Lazily built, and only for /v1/messages/count_tokens — Claude Code
        # budgets context with real counts. Tied to model_dir, so a swap must
        # drop it or the next count is from the previous model's vocabulary.
        self.tokenizer = None
        self.kv_pool_gb = 0              # resolved per model in load()
        self.kv_pool_note = ""           # how that number was arrived at
        self.context_tokens = None       # usable context window, in tokens
        self.context_limit_by = None     # which constraint is binding it
        self.offload_ratio = 0           # % of expert weights streamed
        self.offload_note = ""

    def load(self, model_dir):
        """Load model, auto-detecting VLM vs LLM."""
        self.status = "loading"
        # A model that just loaded has NOT been idle since whenever it was last
        # *used*. Without this the idle watchdog reads an hours-old timestamp
        # on the next 30 s pass and unloads a model the user swapped in ten
        # seconds ago — then the first request reloads it, back-to-back, before
        # the GPU driver has finished releasing the previous copy. Observed on
        # a swap into Qwen3-Coder-30B: loaded in 72.8 s, unloaded immediately,
        # reloaded on the next request, machine down to 4 GB free.
        self.last_used = time.time()
        self.tokenizer = None
        self.context_tokens = None
        self.context_limit_by = None
        self.model_dir = model_dir
        self.model_name = model_display_name(model_dir)
        vlm = is_vlm(model_dir)
        self.model_type = "vlm" if vlm else "llm"

        print(f"  [{self.device_name}] Detected: {self.model_type.upper()} ({self.model_name})")
        integrity_err = _verify_weights_integrity(model_dir)
        if integrity_err:
            raise RuntimeError(integrity_err)
        # Sized per model, not per process: KV bytes/token vary ~3x across the
        # models one slot may hold over its life, so a single global GB figure
        # means a different context window after every swap.
        self.kv_pool_gb, self.kv_pool_note = self._resolve_kv_pool_gb(vlm)
        # Offload depends on the pool size, so it is resolved after it.
        self.offload_ratio, self.offload_note = self._resolve_offload_ratio()
        self._preflight_memory(vlm)
        # Deliberately NOT inside _preflight_memory: that returns early when the
        # device reports no memory budget, which is the normal case on the NPU
        # (it allocates from system RAM). Nesting context resolution there meant
        # the NPU silently got no context window at all.
        self._resolve_context_window()
        print(f"  [{self.device_name}] Loading...", flush=True)

        # MoE disk offload (--offload-ratio): GPU-only plugin property, and it
        # only does anything on XMX hardware (Arc dGPU, Lunar Lake 140V+) —
        # verified: Qwen3-30B-A3B int4 runs in 2.35 GB resident at ratio 90 on
        # a 140V, while non-XMX iGPUs silently ignore it (see TODONT.md).
        offload = {}
        # int8 KV halves cache bytes/token (96 KB -> ~48 KB on a 30B), which is
        # the difference between a 100k-token session fitting this GPU budget
        # and not. It is also a far gentler compromise than the int4 weights
        # already in use — KV quantization costs much less quality per bit.
        if config.KV_PRECISION and self.device_name != "NPU":
            offload["KV_CACHE_PRECISION"] = config.KV_PRECISION
            print(f"  [{self.device_name}] KV cache precision {config.KV_PRECISION} "
                  f"(~half the bytes/token of f16)", flush=True)
        if self.offload_ratio > 0 and self.device_name == "GPU":
            # Add, don't replace. A reassignment here silently discarded
            # KV_CACHE_PRECISION above, and `--kv-precision u8 --offload-ratio N`
            # is precisely the pairing a large MoE on a small device needs —
            # so the one case that most wanted half-size KV was the one case
            # that never got it.
            offload["config.OFFLOAD_RATIO"] = self.offload_ratio
            print(f"  [{self.device_name}] MoE disk offload on "
                  f"({self.offload_ratio}% of expert weights streamed)"
                  f"{self.offload_note}", flush=True)
            if self.offload_ratio >= 40:
                # Offload costs decode, but it costs *prefill* far more —
                # every expert a long prompt touches is streamed from disk.
                # That is invisible until an agent's 35k-token prompt takes
                # a quarter of an hour, so say it while the numbers are in
                # front of the user and the choice is still theirs.
                print(f"  [{self.device_name}] NOTE: at {self.offload_ratio}% "
                      f"the first long prompt will prefill slowly (minutes for "
                      f"an agent-sized prompt). Lower --context-tokens, or "
                      f"raise the iGPU Shared Memory Override, to cut the "
                      f"ratio.", flush=True)

        # Compile cache. Loading a model is mostly *compiling* it for the
        # target device — the NPU especially rebuilds its whole graph every
        # start, which is what makes a cold launch feel broken on a machine
        # you open twenty times a day. All three devices here advertise
        # EXPORT_IMPORT, so the compiled blob can be reused: first load pays
        # the compile, every later load reads it back.
        if config.MODEL_CACHE_DIR:
            offload["CACHE_DIR"] = config.MODEL_CACHE_DIR

        if vlm:
            VLMPipe = getattr(ovg, "VLMPipeline", None)
            if VLMPipe is None:
                raise RuntimeError("No VLMPipeline in this openvino_genai build.")
            self.pipe = VLMPipe(str(model_dir), device=self.device_id, **offload)
        else:
            # NPU has a default prompt limit of 1024 tokens — raise it
            if self.device_name == "NPU":
                self.pipe = ovg.LLMPipeline(
                    str(model_dir), device=self.device_id,
                    MAX_PROMPT_LEN=config.NPU_MAX_PROMPT_LEN, **offload,
                )
            elif self._prompt_cache_on():
                # GPU/CPU: enable prefix (KV) caching via the continuous-batching
                # backend so a repeated prompt prefix — e.g. an agent's fixed
                # system prompt + tool schemas, identical every turn — is
                # prefilled once instead of every turn. Auto-invalidated by any
                # prefix change (no staleness). Opt out with --no-prompt-cache.
                try:
                    sc = ovg.SchedulerConfig()
                    sc.enable_prefix_caching = True
                    sc.cache_size = self.kv_pool_gb
                    self.pipe = ovg.LLMPipeline(
                        str(model_dir), device=self.device_id, scheduler_config=sc,
                        **offload,
                    )
                    print(f"  [{self.device_name}] prefix caching on "
                          f"({self.kv_pool_gb} GB KV pool)", flush=True)
                except Exception as e:
                    print(f"  [{self.device_name}] prefix caching unavailable "
                          f"({e}); using plain pipeline", flush=True)
                    self.pipe = ovg.LLMPipeline(str(model_dir), device=self.device_id,
                                                **offload)
            else:
                self.pipe = ovg.LLMPipeline(str(model_dir), device=self.device_id,
                                            **offload)

    def prefill_note(self, text_prompt):
        """' (prefill 2043 tok/s)' for the log line, or ''.

        The number that matters most on a small device, and the one nothing
        reported: an agent turn is dominated by prefilling its prompt, so a
        backend that prefills 10x slower looks like "the model is slow" or
        "the prompt is too big" — both of which cost this project a day to
        rule out. prompt_tokens / TTFT states it outright.
        """
        if not self.last_ttft_ms or not text_prompt:
            return ""
        n = _count_tokens(self, text_prompt)
        # Below a few hundred tokens TTFT is mostly fixed overhead, so the
        # ratio says more about the round trip than the prefill — a 3-token
        # "Say OK" reads as 9 tok/s on a device doing 2,000. Report only where
        # the number means what it claims.
        if not n or n < 200:
            return ""
        return f", prefill {n / (self.last_ttft_ms / 1000):.0f} tok/s"

    @staticmethod
    def _message_text(raw_messages):
        return "\n".join(str(msg.get("content") or "") for msg in raw_messages)

    def _record_turn(self, prompt_text, completion_text, completion_tokens,
                     started, finish_reason="stop", ttft_ms=None):
        """Turn the measurements already used by log lines into JSON data."""
        total_ms = (time.perf_counter() - started) * 1000
        prompt_tokens = _count_tokens(self, prompt_text)
        if completion_tokens is None:
            completion_tokens = _count_tokens(self, completion_text)
        record_turn(
            device=self.device_name,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            finish_reason=finish_reason,
        )


    def warmup(self):
        self.status = "warming_up"
        print(f"  [{self.device_name}] Warmup...", end="", flush=True)
        t0 = time.perf_counter()
        gen = ovg.GenerationConfig()
        gen.max_new_tokens = 5
        gen.do_sample = False
        gen.top_k = 1
        try:
            if self.model_type == "vlm":
                self.pipe.generate(prompt="Hello", generation_config=gen)
            else:
                history = ovg.ChatHistory()
                history.append({"role": "user", "content": "Hi"})
                self.pipe.generate(history, gen)
            elapsed = time.perf_counter() - t0
            print(f" done ({elapsed:.1f}s)", flush=True)
            self.status = "ready"
        except Exception as e:
            print(f" failed: {e}", flush=True)
            self.status = "error"

    def unload(self):
        """Release the loaded pipeline. Caller must hold self.lock."""
        if self.pipe is None:
            return
        print(f"  [{self.device_name}] Idle — unloading {self.model_name}", flush=True)
        self.pipe = None
        self.status = "idle_unloaded"
        import gc
        gc.collect()

    def ensure_loaded(self):
        """Reload pipeline if it was unloaded. Blocks until ready."""
        if self.pipe is not None and self.status == "ready":
            return
        with self.lock:
            if self.pipe is not None and self.status == "ready":
                return  # someone else loaded it while we waited
            if self.model_dir is None:
                raise RuntimeError(f"Slot {self.device_name} has no model_dir")
            print(f"  [{self.device_name}] Reloading {self.model_name}...", flush=True)
            self.load(self.model_dir)
            self.warmup()

    def generate_vlm(self, text_prompt, images, gen):
        """VLM generate — images optional."""
        self.last_ttft_ms = None
        with self.lock:
            started = time.perf_counter()
            if images:
                imgs = images[0] if len(images) == 1 else images
                result = self.pipe.generate(
                    prompt=text_prompt, images=imgs, generation_config=gen,
                )
            else:
                result = self.pipe.generate(
                    prompt=text_prompt, generation_config=gen,
                )
            self.last_used = time.time()
        text = extract_text(result)
        ttft_ms, _ = extract_perf(result)
        self.last_ttft_ms = ttft_ms
        self._record_turn(text_prompt, text, None, started, ttft_ms=ttft_ms)
        return text

    def generate_llm(self, raw_messages, gen, record_metric=True):
        """LLM generate — non-streaming."""
        self.last_ttft_ms = None
        history = ovg.ChatHistory()
        for msg in raw_messages:
            history.append({"role": msg["role"], "content": msg["content"]})
        with self.lock:
            started = time.perf_counter()
            result = self.pipe.generate(history, gen)
            self.last_used = time.time()
        ttft_ms, _ = extract_perf(result)
        self.last_ttft_ms = ttft_ms
        text = extract_text(result)
        if record_metric:
            self._record_turn(self._message_text(raw_messages), text, None,
                              started, ttft_ms=ttft_ms)
        return text

    def cancel(self):
        """Signal the current generation to stop."""
        self._cancel.set()

    def stream_vlm(self, text_prompt, images, gen, completion_id, created, t0):
        """VLM generate — SSE streaming. openvino-genai 2026.1+."""
        token_queue = Queue()
        token_count = 0
        gen_error = [None]
        ttft_ms = None
        self.last_ttft_ms = None

        def streamer_callback(token):
            if self._cancel.is_set():
                return True
            token_queue.put(token)
            return False

        def _generate():
            try:
                with self.lock:
                    self._cancel.clear()
                    if images:
                        imgs = images[0] if len(images) == 1 else images
                        self.pipe.generate(
                            prompt=text_prompt, images=imgs,
                            generation_config=gen, streamer=streamer_callback,
                        )
                    else:
                        self.pipe.generate(
                            prompt=text_prompt, generation_config=gen,
                            streamer=streamer_callback,
                        )
                    self.last_used = time.time()
            except Exception as e:
                gen_error[0] = e
                print(f"{datetime.now():%H:%M:%S} !! [{self.device_name}] "
                      f"VLM generate error: {e}", flush=True)
            finally:
                token_queue.put(None)

        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        try:
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": self.model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            while True:
                try:
                    token = token_queue.get(timeout=180)
                except Empty:
                    break
                if token is None:
                    break
                if token_count == 0:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    self.last_ttft_ms = ttft_ms
                token_count += 1
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            was_cancelled = self._cancel.is_set()
            if gen_error[0] is not None:
                err_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {
                        "content": f"\n[error: {gen_error[0]}]"
                    }, "finish_reason": "error"}],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
            else:
                finish_reason = "cancelled" if was_cancelled else "stop"
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            self._cancel.set()

        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        tag = " (cancelled)" if was_cancelled else (" (error)" if gen_error[0] else "")
        print(f"{datetime.now():%H:%M:%S} -> [{self.device_name}] "
              f"VLM {token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s){tag}",
              flush=True)
        self._record_turn(text_prompt, "", token_count, t0,
                          "cancelled" if was_cancelled else
                          ("error" if gen_error[0] else "stop"),
                          ttft_ms=ttft_ms)

    def stream_llm(self, raw_messages, gen, completion_id, created, t0):
        """LLM generate — SSE streaming."""
        history = ovg.ChatHistory()
        for msg in raw_messages:
            history.append({"role": msg["role"], "content": msg["content"]})

        token_queue = Queue()
        token_count = 0
        cancelled = False
        ttft_ms = None
        self.last_ttft_ms = None

        def streamer_callback(token):
            if self._cancel.is_set():
                return True  # stop generation
            token_queue.put(token)
            return False

        gen_error = [None]  # captured from generate thread

        def _generate():
            try:
                with self.lock:
                    # Clear inside the lock, just before generation, to avoid
                    # racing with the previous request's finally: _cancel.set()
                    self._cancel.clear()
                    self.pipe.generate(history, gen, streamer_callback)
                    self.last_used = time.time()
            except Exception as e:
                gen_error[0] = e
                print(f"{datetime.now():%H:%M:%S} !! [{self.device_name}] "
                      f"generate error: {explain_genai_error(e)}", flush=True)
            finally:
                token_queue.put(None)

        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        try:
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": self.model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            while True:
                try:
                    token = token_queue.get(timeout=config.HEARTBEAT_SECS)
                except Empty:
                    # No token yet — likely a long prefill on a big prompt.
                    # Emit an empty-content delta to keep the client's idle
                    # watchdog from aborting (a real chunk resets content-based
                    # watchdogs, not just byte-based ones; empty string is a
                    # no-op for assembly). The background thread delivers tokens
                    # or the None sentinel when ready.
                    if not t.is_alive():
                        break
                    ka = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": self.model_name,
                        "choices": [{"index": 0, "delta": {"content": ""},
                                     "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(ka)}\n\n"
                    continue
                if token is None:
                    break
                if token_count == 0:
                    # Wall-clock TTFT: prefill is over when the first token lands.
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    self.last_ttft_ms = ttft_ms
                token_count += 1
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Capture state BEFORE the finally-block safety-net sets _cancel
            was_cancelled = self._cancel.is_set()
            if gen_error[0] is not None:
                finish_reason = "error"
                err_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {
                        "content": f"\n[error: {explain_genai_error(gen_error[0])}]"
                    }, "finish_reason": "error"}],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
            else:
                finish_reason = "cancelled" if was_cancelled else "stop"
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Safety net: if client disconnects, stop generation
            self._cancel.set()

        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        tag = " (cancelled)" if was_cancelled else (" (error)" if gen_error[0] else "")
        ttft = (f", TTFT {self.last_ttft_ms:.0f}ms" if token_count and
                self.last_ttft_ms is not None else "")
        print(f"{datetime.now():%H:%M:%S} -> [{self.device_name}] "
              f"{token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s{ttft}){tag}",
              flush=True)
        self._record_turn(self._message_text(raw_messages), "", token_count, t0,
                          finish_reason, ttft_ms=ttft_ms)

    @property
    def info(self):
        return {
            "status": self.status,
            "model": self.model_name,
            "type": self.model_type,
            "device": self.device_full,
            "device_name": self.device_name,   # canonical NPU/GPU/CPU (routing/checks)
            "tools": _tools_supported(self),    # can this slot drive an agent loop?
            "prewarmed": self.prewarmed,        # did the --prewarm prefill succeed
            "last_ttft_ms": (round(self.last_ttft_ms)
                             if self.last_ttft_ms is not None else None),
            # The usable context window: the smallest of the model's own
            # max_position_embeddings, the NPU prompt cap, and the KV pool
            # capacity when prefix caching built one. Reported on every device,
            # so the UI always has a denominator.
            "context_tokens": self.context_tokens,
            # Which constraint produced that number — "model", "kv pool" or
            # "npu prompt cap". Without it, a small figure on a large model
            # reads as a bug rather than as the pool doing its job.
            "context_limit_by": self.context_limit_by,
            "kv_pool_gb": self.kv_pool_gb or None,
            # What fraction of the expert weights is being streamed from disk,
            # and why. This decides decode speed more than anything else on a
            # MoE (ratio 30 -> 25.3 tok/s, ratio 90 -> 5.1 on the same model),
            # and nothing reported it — so a slot silently pinned at 90 by a
            # bad memory reading looked like "the model got slow" with no way
            # to tell from the outside. 0 means fully resident.
            "offload_ratio": self.offload_ratio,
            "offload_note": self.offload_note.strip() or None,
        }
