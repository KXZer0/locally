"""`POST /v1/rerank` in the Cohere/Jina shape over the search reranker.

The model is the cross-encoder `/v1/util/search` already uses to reorder its
candidates (bge-reranker-base INT8). Request: {"model"?, "query", "documents":
[str | {"text": str}], "top_n"?, "return_documents"?}. Response: {"model",
"results": [{"index", "relevance_score", "document"?}]}, best first.

`relevance_score` is the sigmoid of the model's logit, which is how the BGE
rerankers' own library normalises them (and what Cohere/Jina clients expect:
a number in 0..1). It still orders documents for ONE query; across queries it
is not a calibrated probability, so sort on it rather than threshold it.
`model` is accepted and ignored, for the reason core/util/embeddings.py gives.
"""

import math

from flask import jsonify, request

from utility_pipeline import TEXT_MODEL_NAMES, UtilityUnavailable
from core.errors import _TurnError, openai_error
from core.util.embeddings import MAX_INPUT_CHARS, ready_slot

# One cross-encoder inference per document, under the lock chat shares.
MAX_DOCUMENTS = 256

RERANK_MODEL = TEXT_MODEL_NAMES["rerank"]


def _text(value, where):
    if isinstance(value, dict):
        value = value.get("text")
    if not isinstance(value, str) or not value.strip():
        raise _TurnError(openai_error(f"{where} must be a non-empty string."))
    if len(value) > MAX_INPUT_CHARS:
        raise _TurnError(openai_error(
            f"{where} is {len(value)} characters; the maximum is "
            f"{MAX_INPUT_CHARS}."))
    return value


def _parse(body):
    query = _text(body.get("query"), "'query'")
    docs = body.get("documents")
    if not isinstance(docs, list) or not docs:
        raise _TurnError(openai_error(
            "'documents' is required: a list of strings or {\"text\": ...}."))
    if len(docs) > MAX_DOCUMENTS:
        raise _TurnError(openai_error(
            f"Too many documents ({len(docs)}); the maximum is "
            f"{MAX_DOCUMENTS} per request."))
    docs = [_text(d, f"documents[{i}]") for i, d in enumerate(docs)]
    top_n = body.get("top_n")
    if top_n is None:
        top_n = len(docs)
    elif isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise _TurnError(openai_error("'top_n' must be a positive integer."))
    return query, docs, min(top_n, len(docs))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def rerank():
    """POST /v1/rerank."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return openai_error("Request body must be a JSON object")
    try:
        query, docs, top_n = _parse(body)
        slot = ready_slot(body.get("engine"))
        scores, _timings = slot.run_utility("rerank", query, docs)
    except _TurnError as e:
        return e.response
    except UtilityUnavailable as e:
        return openai_error(str(e), "server_error", 503)
    except Exception as e:
        return openai_error(f"Rerank failed: {e}", "server_error", 500)

    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    results = []
    for i in order[:top_n]:
        hit = {"index": i, "relevance_score": round(_sigmoid(scores[i]), 6)}
        if body.get("return_documents"):
            hit["document"] = {"text": docs[i]}
        results.append(hit)
    return jsonify({"model": RERANK_MODEL, "results": results}), 200, \
        {"X-Device": slot.device_name}
