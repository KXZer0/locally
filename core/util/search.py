"""Bounded in-memory semantic indexes for uploaded documents."""

import threading
import time
import uuid
from datetime import datetime

import numpy as np
from flask import jsonify, request

from utility_pipeline import UtilityUnavailable
from core.documents.inputs import (_UTIL_SEARCH_MAX_CHUNKS, _util_chunks, _util_item)
from core.documents.read import _document_result
from core.errors import _TurnError, openai_error
from core.slots.select import _default_util_engine, _util_slot
from core.web.fetch_page_cfg import _UTIL_RERANK_POOL

_util_search_indexes = {}
_util_search_lock = threading.Lock()
_UTIL_SEARCH_MAX_INDEXES = 8
_UTIL_SEARCH_MAX_FILES = 20

def util_index():
    """Build a bounded in-memory semantic index from local uploads."""
    uploads = request.files.getlist("files") or request.files.getlist("file")
    if not uploads:
        return openai_error("Send one or more multipart files in 'files'.")
    if len(uploads) > _UTIL_SEARCH_MAX_FILES:
        return openai_error(
            f"Too many files (maximum {_UTIL_SEARCH_MAX_FILES} per index).")
    engine = request.form.get("engine") or _default_util_engine()
    try:
        slot = _util_slot(engine)
        slot.ensure_loaded()
        chunks = []
        for upload in uploads:
            item = _util_item(upload.read(), upload.filename, upload.mimetype)
            if item["kind"] == "image":
                read = slot.read(item["image"])
                text = read["markdown"]
            else:
                text = _document_result(item, engine)["markdown"]
            chunks.extend(_util_chunks(text, item["filename"]))
            if len(chunks) >= _UTIL_SEARCH_MAX_CHUNKS:
                chunks = chunks[:_UTIL_SEARCH_MAX_CHUNKS]
                break
        if not chunks:
            return openai_error("No readable text was found in those files.")
        vectors, timings = slot.run_utility(
            "embed", [chunk["text"] for chunk in chunks])
    except _TurnError as e:
        return e.response
    except UtilityUnavailable as e:
        # Deliberately unavailable is 503, not 500 — see _require().
        return openai_error(str(e), "server_error", 503)
    except Exception as e:
        return openai_error(f"Indexing failed: {e}", "server_error", 500)

    index_id = uuid.uuid4().hex
    with _util_search_lock:
        while len(_util_search_indexes) >= _UTIL_SEARCH_MAX_INDEXES:
            oldest = next(iter(_util_search_indexes))
            del _util_search_indexes[oldest]
        _util_search_indexes[index_id] = {
            "chunks": chunks, "vectors": vectors, "created": time.time(),
            "engine": slot.device_name.lower(),
        }
    return jsonify({
        "index_id": index_id, "files": len(uploads), "chunks": len(chunks),
        "engine": slot.device_name.lower(), "timings_ms": timings,
        "storage": "memory",
    }), 200, {"X-Device": slot.device_name}


def util_search():
    """Search an in-memory utility index with a MiniLM query embedding."""
    body = request.get_json(silent=True) or {}
    index_id = str(body.get("index_id") or "")
    query = str(body.get("query") or "").strip()
    if not index_id or not query:
        return openai_error("'index_id' and 'query' are required")
    with _util_search_lock:
        index = _util_search_indexes.get(index_id)
    if index is None:
        return openai_error("That in-memory index no longer exists.", status=404)
    try:
        slot = _util_slot(index["engine"])
        slot.ensure_loaded()
        query_vector, timings = slot.run_utility("embed", [query])
    except _TurnError as e:
        return e.response
    except UtilityUnavailable as e:
        # Deliberately unavailable is 503, not 500 — see _require().
        return openai_error(str(e), "server_error", 503)
    except Exception as e:
        return openai_error(f"Search failed: {e}", "server_error", 500)
    scores = np.asarray(index["vectors"]) @ query_vector[0]
    top_k = min(10, max(1, int(body.get("top_k", 5))))

    # Retrieve wide, then let the cross-encoder settle the order. Measured on
    # this corpus: the bi-encoder puts the right chunk in the top 5 every time
    # (recall@5 1.000) but ranks it first only 7 times in 9; reranking that
    # pool took top-1 to 8/9 and MRR from 0.889 to 0.944. Widening retrieval is
    # therefore free accuracy — the work is in the ordering, not the recall.
    pool_size = min(len(index["chunks"]), max(top_k, _UTIL_RERANK_POOL))
    pool = list(np.argsort(scores)[::-1][:pool_size])
    order, reranked = pool[:top_k], False

    cross_scores = {}
    if body.get("rerank", True):
        try:
            passages = [index["chunks"][int(i)]["text"] for i in pool]
            rerank_scores, rerank_timings = slot.run_utility(
                "rerank", query, passages)
            cross_scores = {int(i): s for s, i in zip(rerank_scores, pool)}
            ranked = sorted(zip(rerank_scores, pool),
                            key=lambda pair: pair[0], reverse=True)
            order = [i for _score, i in ranked[:top_k]]
            timings = {**timings, **{f"rerank_{k}": v
                                     for k, v in rerank_timings.items()}}
            reranked = True
        except UtilityUnavailable:
            # Reranking is an improvement, not a dependency: without the model
            # installed, search still answers in embedding order rather than
            # 503-ing on a feature the caller never asked for by name.
            pass
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
                  f"rerank failed, using embedding order: {e}", flush=True)

    # Carry both scores when reranking ran. The embedding cosine no longer
    # explains the order once a cross-encoder has reordered it — results come
    # back with a lower cosine above a higher one, which reads as a bug unless
    # the number the sort actually used is visible next to it.
    results = []
    for i in order:
        hit = {**index["chunks"][int(i)], "score": round(float(scores[i]), 4)}
        if int(i) in cross_scores:
            hit["rerank_score"] = round(cross_scores[int(i)], 3)
        results.append(hit)
    return jsonify({
        "results": results, "engine": slot.device_name.lower(),
        "timings_ms": timings,
        # Which model decided this order. `score` stays the embedding cosine so
        # it means the same thing in both modes; `rerank_score` is what the
        # sort used when this is true.
        "reranked": reranked,
    }), 200, {"X-Device": slot.device_name}


