"""A slot backed by another OpenAI-compatible server, not a local pipeline.

locally is OpenVINO, and OpenVINO is Intel. On a machine with an NVIDIA card
there is no device to place a generative model on, so everything else the app
is -- utilities, voice, the whole UI -- was unreachable there for want of a
chat model. This makes one program run on both machines.

It implements DeviceSlot's surface rather than introducing a second slot
protocol, so every caller keeps working untouched. What it refuses to do is
invent the half that is about owning weights: no KV pool, no offload, no
prefill rate. Those report None, because a fabricated figure in the memory
HUD is worse than an absent one."""

import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from queue import Empty, Queue
from core import config
from core.slots.capability import _tools_supported
from core.slots.locks import _device_lock


def _gen_to_openai(gen, model, messages, stream):
    """Translate an ovg.GenerationConfig into an OpenAI request body.

    Read defensively with getattr: GenerationConfig's attribute set moves
    between genai releases, and a proxy slot must not be the thing that breaks
    on an upgrade it has no stake in.

    `repetition_penalty` is deliberately dropped rather than mapped onto
    frequency_penalty. They are different functions — one rescales logits for
    any token already present, the other subtracts a count-weighted constant —
    and silently substituting one for the other would make the same request
    sample differently on a proxied slot than on a native one, which is exactly
    the class of bug that is impossible to find from the outside.
    """
    body = {"model": model, "messages": messages, "stream": stream}
    max_new = getattr(gen, "max_new_tokens", None)
    if max_new:
        body["max_tokens"] = int(max_new)
    if getattr(gen, "do_sample", False):
        # Rounded because GenerationConfig stores these as float32: a client
        # that asked for 0.7 gets 0.699999988079071 back out, and forwarding
        # that verbatim puts a number in the upstream request that nobody
        # typed. Six places is far below any sampler's sensitivity and
        # restores exactly what was sent.
        body["temperature"] = round(float(getattr(gen, "temperature", 1.0)), 6)
        top_p = getattr(gen, "top_p", None)
        if top_p is not None:
            body["top_p"] = round(float(top_p), 6)
    else:
        # do_sample False is greedy. Upstream has no top_k=1 knob in the OpenAI
        # shape, and temperature 0 is how every server spells the same thing.
        body["temperature"] = 0.0
    for attr in ("frequency_penalty", "presence_penalty"):
        val = getattr(gen, attr, None)
        if val:
            body[attr] = float(val)
    return body


class ProxySlot:
    """A slot backed by another OpenAI-compatible server, not a local pipeline.

    Why this exists: locally is OpenVINO, and OpenVINO is Intel. On a machine
    with an NVIDIA card there is no device to place a generative model on, so
    everything else the app is — the utilities, the memory HUD, the voice path,
    the whole web UI — was unreachable there for want of a chat model. A slot
    that forwards to Ollama / vLLM / llama.cpp makes one program run on both
    machines: native OpenVINO where there is Intel silicon, proxied where there
    is not.

    It implements the DeviceSlot surface (load / warmup / unload /
    ensure_loaded / generate_llm / stream_llm / cancel / info) rather than
    introducing a second slot protocol, so every caller — both chat paths, the
    Ollama and Anthropic shims, /health, the idle watchdog — keeps working
    untouched.

    What it cannot do is the half that is about owning weights: there is no KV
    pool to size, no expert offload to compute, no memory to preflight. Those
    report None rather than a plausible-looking zero, because an invented
    figure in the memory HUD is worse than an absent one.

    Only the standard library is used for HTTP. Adding `requests` for one
    POST would be the first CDN-shaped dependency in a project that self-hosts
    its own fonts to survive a flight.
    """

    # Generous, because upstream prefill on a big agent prompt is legitimately
    # slow and the heartbeat — not the socket timeout — is what keeps clients
    # alive. Too tight a value here turns a slow answer into a failed one.
    READ_TIMEOUT = 600
    PROBE_TIMEOUT = 10

    def __init__(self, base_url, model=None, api_key=None, device_name="REMOTE"):
        self.device_name = device_name
        self.device_id = device_name
        self.base_url = base_url.rstrip("/")
        self.device_full = self.base_url
        self.api_key = api_key
        self.pipe = None                 # never a pipeline; kept for duck-typing
        self.model_name = model or ""
        self.requested_model = model     # what the user asked for, before probing
        self.model_type = "llm"          # a proxied VLM is not routed here (see load)
        self.status = "not_configured"
        self.lock = _device_lock(self.device_name)
        self._cancel = threading.Event()
        self.last_used = time.time()
        self.model_dir = None            # no weights on this machine
        self.last_ttft_ms = None
        self.prewarmed = False
        self.tokenizer = None
        self.kv_pool_gb = 0
        self.kv_pool_note = ""
        self.context_tokens = None
        self.context_limit_by = None
        self.offload_ratio = 0
        self.offload_note = ""
        self.upstream_models = []

    # -- HTTP helpers -------------------------------------------------------

    def _request(self, path, payload=None, timeout=None, stream=False):
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data else "GET")
        resp = urllib.request.urlopen(req, timeout=timeout or self.READ_TIMEOUT)
        if stream:
            return resp
        with resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    def _explain(self, e):
        """Upstream failures in the words of someone who has to go fix them."""
        if isinstance(e, urllib.error.HTTPError):
            try:
                detail = json.loads(e.read().decode("utf-8", "replace"))
                msg = (detail.get("error", {}).get("message")
                       if isinstance(detail.get("error"), dict)
                       else detail.get("error")) or str(detail)
            except Exception:
                msg = e.reason
            if e.code == 401:
                return f"upstream rejected the API key (401): {msg}"
            if e.code == 404:
                return (f"upstream has no such model or endpoint (404): {msg}. "
                        f"Check --proxy-model against what it actually serves.")
            return f"upstream returned HTTP {e.code}: {msg}"
        if isinstance(e, urllib.error.URLError):
            return (f"cannot reach {self.base_url}: {e.reason}. "
                    f"Is the server running?")
        return str(e)

    # -- DeviceSlot surface -------------------------------------------------

    def load(self, model_dir=None):
        """'Loading' a proxy slot means confirming the upstream can serve it.

        Kept synchronous and strict for the same reason POST /v1/models/load is:
        reporting success on a slot that cannot actually answer just moves the
        failure into the user's first message, where it is far harder to read.
        """
        self.status = "loading"
        self.last_used = time.time()
        try:
            listing = self._request("/v1/models", timeout=self.PROBE_TIMEOUT)
            self.upstream_models = [m.get("id") for m in listing.get("data", [])
                                    if m.get("id")]
        except Exception as e:
            self.status = "error"
            raise RuntimeError(f"proxy: {self._explain(e)}") from e

        if self.requested_model:
            if (self.upstream_models
                    and self.requested_model not in self.upstream_models):
                self.status = "error"
                raise RuntimeError(
                    f"proxy: upstream does not serve '{self.requested_model}'. "
                    f"It offers: {', '.join(self.upstream_models[:8])}"
                    + (" ..." if len(self.upstream_models) > 8 else ""))
            self.model_name = self.requested_model
        elif self.upstream_models:
            # No model named: take the first the upstream advertises, which is
            # what a single-model Ollama or vLLM host means by "the model".
            self.model_name = self.upstream_models[0]
        else:
            self.status = "error"
            raise RuntimeError("proxy: upstream advertises no models and "
                               "--proxy-model was not given")
        print(f"  [{self.device_name}] Proxying {self.model_name} via {self.base_url}",
              flush=True)

    def warmup(self):
        """One token, to prove the path end to end before serving anyone.

        Cheap upstream and it catches the failures a /v1/models probe cannot:
        a model listed but not loadable, a wrong key on the completions route,
        a reverse proxy that passes GETs and blocks POSTs.
        """
        self.status = "warming_up"
        print(f"  [{self.device_name}] Warmup...", end="", flush=True)
        t0 = time.perf_counter()
        try:
            self._request("/v1/chat/completions", {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1, "temperature": 0.0, "stream": False,
            }, timeout=self.PROBE_TIMEOUT * 6)
            print(f" done ({time.perf_counter() - t0:.1f}s)", flush=True)
            self.status = "ready"
        except Exception as e:
            print(f" failed: {self._explain(e)}", flush=True)
            self.status = "error"

    def unload(self):
        """No-op: this process holds no weights for this slot.

        The idle watchdog calls unload() on every slot it walks, so this has to
        exist and has to leave the slot serviceable — returning it to
        idle_unloaded would make ensure_loaded re-probe the upstream on the
        next request for no gain. Whether the far end unloads on idle is the
        far end's policy, and reaching over to force it would be locally
        managing a server it does not own.
        """
        return

    def ensure_loaded(self):
        if self.status == "ready":
            return
        with self.lock:
            if self.status == "ready":
                return
            self.load()
            self.warmup()

    def cancel(self):
        self._cancel.set()

    def generate_llm(self, raw_messages, gen):
        messages = [{"role": m["role"], "content": m["content"]} for m in raw_messages]
        body = _gen_to_openai(gen, self.model_name, messages, stream=False)
        t0 = time.perf_counter()
        with self.lock:
            try:
                result = self._request("/v1/chat/completions", body)
            except Exception as e:
                raise RuntimeError(f"proxy: {self._explain(e)}") from e
            self.last_used = time.time()
        self.last_ttft_ms = (time.perf_counter() - t0) * 1000
        choices = result.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content", "") or ""

    def prefill_note(self, text_prompt):
        """Always empty, on purpose.

        DeviceSlot reports prompt_tokens / TTFT because on a local pipeline
        that ratio is the prefill rate. Here TTFT is a round trip — network,
        upstream queueing, and someone else's prefill — and this process never
        tokenizes the prompt, so any figure would be an invented measurement of
        hardware it cannot see. prefill_note's own rule is to report only where
        the number means what it claims, and here it never does.
        """
        return ""

    def generate_vlm(self, text_prompt, images, gen):
        """Images are not routed to a proxy slot — see _choose_device.

        Present so the duck type is complete; reaching it means routing chose
        wrongly, and saying so beats a confusing upstream 400.
        """
        raise RuntimeError("proxy slots do not serve vision requests")

    def stream_vlm(self, text_prompt, images, gen, completion_id, created, t0):
        """See generate_vlm — vision is not proxied."""
        raise RuntimeError("proxy slots do not serve vision requests")

    def stream_llm(self, raw_messages, gen, completion_id, created, t0):
        """Relay the upstream SSE stream as our own.

        The structure deliberately mirrors DeviceSlot.stream_llm — background
        thread, token Queue, config.HEARTBEAT_SECS keep-alive on an empty read, TTFT
        stamped on the first token — because the client-visible behaviour has
        to be identical whether a turn was served locally or proxied. The web
        UI, Copilot and Claude Code all sit behind the same frames.

        Upstream chunks are re-emitted rather than forwarded verbatim: their
        `model` field names the far end's id and their `id` is the far end's
        completion, and letting either leak would make cancel and the request
        log refer to something this server has never heard of.
        """
        messages = [{"role": m["role"], "content": m["content"]} for m in raw_messages]
        body = _gen_to_openai(gen, self.model_name, messages, stream=True)

        token_queue = Queue()
        token_count = 0
        gen_error = [None]
        upstream_finish = [None]

        def _pump():
            resp = None
            try:
                with self.lock:
                    self._cancel.clear()
                    resp = self._request("/v1/chat/completions", body, stream=True)
                    for raw in resp:
                        if self._cancel.is_set():
                            break
                        line = raw.decode("utf-8", "replace").strip()
                        if not line or line.startswith(":"):
                            continue        # SSE comment / keep-alive from upstream
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue        # a partial frame is not fatal
                        choice = (chunk.get("choices") or [{}])[0]
                        if choice.get("finish_reason"):
                            upstream_finish[0] = choice["finish_reason"]
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            token_queue.put(piece)
                    self.last_used = time.time()
            except Exception as e:
                gen_error[0] = e
                print(f"{datetime.now():%H:%M:%S} !! [{self.device_name}] "
                      f"proxy error: {self._explain(e)}", flush=True)
            finally:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                token_queue.put(None)

        t = threading.Thread(target=_pump, daemon=True)
        t.start()

        def frame(delta, finish=None):
            return "data: " + json.dumps({
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": self.model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }) + "\n\n"

        was_cancelled = False
        try:
            yield frame({"role": "assistant"})
            while True:
                try:
                    token = token_queue.get(timeout=config.HEARTBEAT_SECS)
                except Empty:
                    if not t.is_alive():
                        break
                    yield frame({"content": ""})    # keep the client's watchdog quiet
                    continue
                if token is None:
                    break
                if token_count == 0:
                    self.last_ttft_ms = (time.perf_counter() - t0) * 1000
                token_count += 1
                yield frame({"content": token})

            was_cancelled = self._cancel.is_set()
            if gen_error[0] is not None:
                yield frame({"content": f"\n[error: {self._explain(gen_error[0])}]"},
                            "error")
            else:
                yield frame({}, "cancelled" if was_cancelled
                            else (upstream_finish[0] or "stop"))
            yield "data: [DONE]\n\n"
        finally:
            self._cancel.set()

        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        tag = " (cancelled)" if was_cancelled else (" (error)" if gen_error[0] else "")
        ttft = (f", TTFT {self.last_ttft_ms:.0f}ms"
                if token_count and self.last_ttft_ms is not None else "")
        print(f"{datetime.now():%H:%M:%S} -> [{self.device_name}] "
              f"{token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s{ttft}){tag}",
              flush=True)

    @property
    def info(self):
        return {
            "status": self.status,
            "model": self.model_name,
            "type": self.model_type,
            "device": self.device_full,
            "device_name": self.device_name,
            "tools": _tools_supported(self),
            "prewarmed": self.prewarmed,
            "last_ttft_ms": (round(self.last_ttft_ms)
                             if self.last_ttft_ms is not None else None),
            # Everything below is a property of weights this process does not
            # hold. None means "not ours to know" — distinct from 0, which
            # would read as a measurement.
            "context_tokens": self.context_tokens,
            "context_limit_by": self.context_limit_by,
            "kv_pool_gb": None,
            "offload_ratio": None,
            "offload_note": None,
            # Proxy-only, so the UI can say where an answer actually came from
            # rather than showing a device that is not doing the work.
            "proxy": {"upstream": self.base_url,
                      "models": self.upstream_models[:12]},
        }
