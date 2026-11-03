"""OpenAI `/v1/embeddings` and Ollama `/api/embed` over the search embedder.

The model is the one `/v1/util/search` already runs (MiniLM-L6 INT8, NPU by
default), so this adds a wire format, not a model: a RAG client that speaks
OpenAI embeddings can index against the same low-power engine the server's
own search uses.

`model` in the request is accepted and ignored. There is exactly one embedder
per engine, clients differ wildly in what they send ("text-embedding-3-small"
hard-coded, the chat model's name, an empty string), and refusing a name we
could not have honoured anyway buys nothing. The response names the model that
actually ran.
"""

import base64
import time

from flask import jsonify, request

from utility_pipeline import TEXT_MAX_TOKENS, TEXT_MODEL_NAMES, UtilityUnavailable
from core.errors import _TurnError, openai_error
from core.slots.select import _default_util_engine, _util_slot, _util_slots

# Bounds. MiniLM reads one sequence at a time (~10 ms each on the NPU), so 256
# inputs is a couple of seconds of a lock chat shares; 20,000 characters is
# far past the 256 tokens the model can read and bounds the tokenizer's work.
MAX_INPUTS = 256
MAX_INPUT_CHARS = 20_000

EMBED_MODEL = TEXT_MODEL_NAMES["search"]


def _inputs(value):
    """Normalise `input` to a list of non-empty strings, or raise a 400."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise _TurnError(openai_error(
            "'input' is required: a string or a list of strings."))
    if len(value) > MAX_INPUTS:
        raise _TurnError(openai_error(
            f"Too many inputs ({len(value)}); the maximum is {MAX_INPUTS} "
            f"per request."))
    for i, text in enumerate(value):
        if not isinstance(text, str):
            # OpenAI also takes token-id arrays, but they are ids in OpenAI's
            # tokenizer and mean nothing to this model's vocabulary.
            raise _TurnError(openai_error(
                f"input[{i}] is not a string. Send text; token-id arrays "
                f"are not supported."))
        if not text.strip():
            raise _TurnError(openai_error(f"input[{i}] is empty."))
        if len(text) > MAX_INPUT_CHARS:
            raise _TurnError(openai_error(
                f"input[{i}] is {len(text)} characters; the maximum is "
                f"{MAX_INPUT_CHARS}. The model reads at most "
                f"{TEXT_MAX_TOKENS} tokens, so chunk the text first."))
    return value


def ready_slot(engine):
    """The utility slot for `engine`, woken if idle-unloaded.

    A slot still compiling is a 503 from _util_slot; this adds Retry-After to
    THAT one, because the OpenAI SDKs honour it and retry, and the first
    seconds of a server's life are exactly when the utility engines compile.
    A missing model is also 503 but gets no Retry-After: waiting fixes nothing.
    """
    want = (engine or _default_util_engine()).lower()
    try:
        slot = _util_slot(want)
    except _TurnError as e:
        pending = _util_slots().get(want)
        if pending is not None and pending.status in ("loading", "warming_up"):
            body, status = e.response[:2]
            raise _TurnError((body, status, {"Retry-After": "5"}))
        raise
    slot.ensure_loaded()
    return slot


def embed_texts(texts, engine=None, truncate=True):
    """Embed `texts` on a utility slot. Returns (vectors, read_tokens, slot, trimmed).

    `read_tokens` is per input and capped at what the model actually read.
    Over-long input is truncated to TEXT_MAX_TOKENS by default, matching
    sentence-transformers' own behaviour for this model; `truncate=False`
    makes it a 400 instead, for a client that would rather chunk than embed
    the first 256 tokens of a passage without knowing it.
    """
    slot = ready_slot(engine)
    counts = slot.run_utility("token_counts", "search", texts)
    trimmed = [i for i, n in enumerate(counts) if n > TEXT_MAX_TOKENS]
    if trimmed and not truncate:
        i = trimmed[0]
        raise _TurnError(openai_error(
            f"input[{i}] is {counts[i]} tokens; {EMBED_MODEL} reads at most "
            f"{TEXT_MAX_TOKENS}. Chunk it, or omit 'truncate': false to embed "
            f"the first {TEXT_MAX_TOKENS}."))
    vectors, _timings = slot.run_utility("embed", texts)
    return vectors, [min(n, TEXT_MAX_TOKENS) for n in counts], slot, trimmed


def _run(body):
    """Shared request handling. Returns (vectors, tokens, slot, trimmed)."""
    texts = _inputs(body.get("input"))
    try:
        return embed_texts(texts, body.get("engine"),
                           truncate=body.get("truncate", True) is not False)
    except UtilityUnavailable as e:
        # Not installed / not offered is 503, as everywhere in /v1/util.
        raise _TurnError(openai_error(str(e), "server_error", 503))
    except _TurnError:
        raise
    except Exception as e:
        raise _TurnError(openai_error(f"Embedding failed: {e}",
                                      "server_error", 500))


def _headers(slot, trimmed):
    headers = {"X-Device": slot.device_name}
    if trimmed:
        # Not part of the OpenAI shape, so a header: a client that cares can
        # see that some inputs were read only in part, and one that does not
        # is unaffected.
        headers["X-Truncated-Inputs"] = ",".join(str(i) for i in trimmed)
    return headers


def embeddings():
    """POST /v1/embeddings in the OpenAI shape."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return openai_error("Request body must be a JSON object")
    fmt = body.get("encoding_format") or "float"
    if fmt not in ("float", "base64"):
        return openai_error("'encoding_format' must be 'float' or 'base64'.")
    try:
        vectors, tokens, slot, trimmed = _run(body)
    except _TurnError as e:
        return e.response
    dims = body.get("dimensions")
    if dims is not None and dims != vectors.shape[1]:
        # MiniLM is not a Matryoshka model: cutting its vector short is not a
        # smaller embedding, it is a broken one.
        return openai_error(
            f"'dimensions' must be {vectors.shape[1]} for {EMBED_MODEL}; it "
            f"cannot produce shortened embeddings.")
    data = []
    for i, vec in enumerate(vectors):
        if fmt == "base64":
            # Little-endian float32, which is what the OpenAI SDKs decode.
            emb = base64.b64encode(vec.astype("<f4").tobytes()).decode("ascii")
        else:
            emb = [float(x) for x in vec]
        data.append({"object": "embedding", "index": i, "embedding": emb})
    used = sum(tokens)
    return jsonify({
        "object": "list", "data": data, "model": EMBED_MODEL,
        "usage": {"prompt_tokens": used, "total_tokens": used},
    }), 200, _headers(slot, trimmed)


def ollama_embed():
    """POST /api/embed in Ollama's shape: {model, input, truncate?}."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400
    t0 = time.perf_counter_ns()
    try:
        vectors, tokens, slot, trimmed = _run(body)
    except _TurnError as e:
        err, status = e.response[:2]
        return jsonify({"error": err.get_json()["error"]["message"]}), status
    return jsonify({
        "model": body.get("model") or EMBED_MODEL,
        "embeddings": [[float(x) for x in vec] for vec in vectors],
        "total_duration": time.perf_counter_ns() - t0,
        "load_duration": 0,
        "prompt_eval_count": sum(tokens),
    }), 200, _headers(slot, trimmed)
