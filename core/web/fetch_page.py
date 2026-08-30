"""Turning a search result list into text a model can answer from.

Fetch the top hits, chunk them, and rank the chunks lexically before any of
it reaches the prompt -- on an 8k NPU window the budget is spent long before
the pages run out, so WHICH passages survive matters more than how many."""

import concurrent.futures
import re
import threading
import time

from core import config
from core.web.reader import _html_to_markdown, _http_fetcher
from core.web.search import _WEB_UA, _searx, _web_grounded_blocks
from core.documents.read import _native_document_markdown
from core.genai.results import explain_genai_error
from core.tools.text import strip_thinking
from core.web.search import WEB_SEARCH_URL
import json
import openvino_genai as ovg
from core.web.fetch_page_cfg import _WEB_ANALYZE_DEFAULT_MAX_TOKENS
from core.web.fetch_page_cfg import _WEB_FETCH_TIMEOUT
from core.web.fetch_page_cfg import _WEB_MAX_BYTES
from core.web.fetch_page_cfg import _WEB_PREFILTER_KEEP
from core.web.fetch_page_cfg import _WEB_PREFILTER_MIN_PER_SOURCE
from core.web.fetch_page_cfg import _login_wall
from core.web.fetch_page_cfg import _web_prepare_history
from core.web.fetch_page_cfg import _web_sources_for_results
from core.slots.route import _route_request
from core.documents.inputs import _util_chunks
from core.errors import _TurnError
from core.slots.select import _util_slot
from core.web.fetch_page_cfg import _UTIL_RERANK_POOL
import numpy as np

try:
    from utility_pipeline import UtilityUnavailable
except ImportError:
    class UtilityUnavailable(RuntimeError):
        """Stand-in so the except clauses resolve."""


_WEB_MAX_RESULTS = 10          # SearXNG hits considered


_WEB_MAX_FETCH = 7             # of those, how many pages we actually read


_WEB_MAX_CHUNKS = 400          # across all pages, before ranking


# Applied to a search snippet's cross-encoder score. Not a ban: a snippet is
# still the only thing we have for a script-rendered page, and dropping it
# would lose that source entirely. This just stops a 200-character summary
# out-ranking a real passage from a page we did read.
#
# 3.0, not the 1.5 tried first: measured on a live query, snippets still scored
# 5.38/5.21/5.08 after a 1.5 penalty against pages at 4.65/3.78/3.04, so they
# were merely knocked from first place to second. The rule this encodes is that
# a snippet should be cited only when it is CLEARLY better than anything that
# was actually read, never when it is marginally better -- because unlike a
# page, its claim cannot be checked against surrounding context.
_WEB_SNIPPET_PENALTY = 3.0


def _fetch_page_text(url):
    """Fetch one page and convert it to Markdown. Returns (text, reason).

    text is None when the page could not be read, and reason says why -- which
    the first version did not, so a high snippet-only rate was visible but
    undiagnosable. Measured causes, in order of how often they actually fire:

    * **JS shells.** reddit.com returns an 8 KB skeleton and chip.computer 3.8 KB;
      both convert to zero characters because the article is assembled by
      script. Nothing short of a headless browser reads those, and that is a
      dependency this project will not take.
    * **Truncation.** urllib does not request compression, so a page arrives at
      full size: tomshardware returned exactly the old 2 MB cap, i.e. it was cut
      mid-HTML. Asking for gzip makes the same article roughly a fifth the size,
      so it now fits with room to spare.

    A browser User-Agent was measured and REJECTED: it fetched nothing extra
    across eight real URLs and actively broke intel.com, which served 200 to the
    honest agent and 403 to the browser one.
    """
    import gzip
    import urllib.error
    import urllib.request
    headers = {
        "User-Agent": _WEB_UA,
        # The single biggest win here. Without it every page arrives
        # uncompressed and large articles hit the size cap mid-document.
        "Accept-Encoding": "gzip",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    raw = ctype = None
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=_WEB_FETCH_TIMEOUT) as r:
                if not r.geturl().startswith(("http://", "https://")):
                    return None, "redirected off the web"
                ctype = (r.headers.get("Content-Type") or "").lower()
                if not any(t in ctype for t in ("html", "text", "xml", "pdf")):
                    return None, f"unsupported type {ctype.split(';')[0]}"
                body = r.read(_WEB_MAX_BYTES)
                if (r.headers.get("Content-Encoding") or "").lower() == "gzip":
                    try:
                        body = gzip.decompress(body)
                    except Exception:
                        pass          # truncated gzip: fall through, let the
                                      # converter salvage what it can
                raw = body
            break
        except Exception as e:
            # One retry, and only for the transient shapes. A 404 or a 403 will
            # say the same thing twice and just costs the user another timeout.
            transient = (isinstance(e, (TimeoutError, urllib.error.URLError))
                         and not isinstance(e, urllib.error.HTTPError))
            if attempt == 0 and transient:
                continue
            return None, f"{type(e).__name__}: {str(e)[:60]}"

    ext = ".pdf" if "pdf" in ctype else ".html"
    try:
        # _native_document_markdown owns the MarkItDown singleton, its lock and
        # the plugins-disabled policy. Going around it would fork all three.
        text = _native_document_markdown(raw, url, ext, ctype.split(";")[0])
    except Exception as e:
        return None, f"convert failed: {type(e).__name__}"
    text = (text or "").strip()
    if not text:
        return None, "no readable text (page is script-rendered)"
    wall = _login_wall(text)
    if wall:
        return None, wall
    return text, None


def _lexical_prefilter(query, chunks, keep=_WEB_PREFILTER_KEEP):
    """Cheap term-overlap stage before the bi-encoder.

    The embedder runs one inference per chunk (the NPU needs a static shape, so
    there is no batching to reach for), which made embedding the single most
    expensive phase of a search turn: 316 chunks x ~13 ms = 4.2 s, more than the
    search and fetch put together.

    So this is the classic three-stage retrieval pipeline rather than an
    optimisation hack: a lexical filter picks candidates, the bi-encoder ranks
    them, the cross-encoder settles the order. Each stage is more accurate and
    more expensive than the last, and each sees fewer items.

    Vocabulary mismatch is the known weakness of lexical matching, so it is
    deliberately generous: chunks scoring zero are kept as filler when there is
    room, and every source is guaranteed a few slots so one long page cannot
    crowd the others out entirely.
    """
    if len(chunks) <= keep:
        return chunks, False
    terms = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    if not terms:
        return chunks[:keep], True

    scored = []
    for i, c in enumerate(chunks):
        words = re.findall(r"[a-z0-9]+", c["text"].lower())
        if not words:
            scored.append((0.0, i))
            continue
        hits = sum(1 for w in words if w in terms)
        # Normalise by length so a long chunk does not win on volume alone.
        scored.append((hits / (len(words) ** 0.5), i))
    scored.sort(key=lambda p: p[0], reverse=True)

    picked, per_source = [], {}
    for _score, i in scored:
        src = chunks[i].get("url") or chunks[i].get("source")
        if len(picked) >= keep:
            break
        per_source[src] = per_source.get(src, 0) + 1
        picked.append(i)
    # Guarantee each source a floor, so a page whose wording differs from the
    # query is still represented when the cross-encoder gets its turn.
    for src in {(c.get("url") or c.get("source")) for c in chunks}:
        have = per_source.get(src, 0)
        if have >= _WEB_PREFILTER_MIN_PER_SOURCE:
            continue
        for i, c in enumerate(chunks):
            if (c.get("url") or c.get("source")) != src or i in picked:
                continue
            picked.append(i)
            have += 1
            if have >= _WEB_PREFILTER_MIN_PER_SOURCE:
                break
    picked = sorted(set(picked))
    return [chunks[i] for i in picked], True


def _web_search_hits(query, count):
    """Ask SearXNG for results. Returns [{url,title,snippet}]."""
    import urllib.parse
    import urllib.request
    base = WEB_SEARCH_URL.rstrip("/")
    url = f"{base}/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "safesearch": "0"})
    req = urllib.request.Request(url, headers={"User-Agent": _WEB_UA})
    with urllib.request.urlopen(req, timeout=_WEB_FETCH_TIMEOUT) as r:
        data = json.loads(r.read(4 * 1024 * 1024).decode("utf-8", "replace"))
    hits = []
    for item in (data.get("results") or [])[:count]:
        link = item.get("url") or ""
        if not link.startswith(("http://", "https://")):
            continue          # never follow a non-web scheme out of a result
        hits.append({"url": link,
                     "title": (item.get("title") or link)[:300],
                     "snippet": (item.get("content") or "")[:500]})
    return hits


def _web_reused_payload(body, query, engine):
    """Build an answer payload from passages already ranked by an earlier turn."""
    results = []
    for item in body.get("passages") or []:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        results.append({
            "text": str(item.get("text")),
            "title": str(item.get("title") or item.get("source") or "Source"),
            "url": item.get("url"),
            "score": item.get("score"),
            "rerank_score": item.get("rerank_score"),
        })
    sources = _web_sources_for_results(results, body.get("sources"))
    return {
        "query": query,
        "results": results,
        "sources": sources,
        "reranked": True,
        "engine": engine,
        "pages_read": sum(1 for s in sources if s.get("read", True)),
        "pages_found": len(sources),
        "chunks_ranked": len(results),
        "chunks_found": len(results),
        "fetch_failures": {},
        "timings_ms": {},
    }


def _web_search_answer(payload, body, mode):
    """Generate the grounded answer after retrieval, or from reused passages."""
    messages = body.get("messages")
    sources = payload.get("sources", [])
    if not body.get("answer"):
        yield "done", payload
        return

    yield "stage", {"stage": "answering",
                    "detail": f"writing an answer from {len(sources)} sources",
                    "sources": sources}
    try:
        t4 = time.perf_counter()
        chat_slot = _route_request(False, "")
        chat_slot.ensure_loaded()
        gen = ovg.GenerationConfig()
        requested = int(body.get("max_tokens", 512))
        if mode == "analyze":
            gen.max_new_tokens = max(_WEB_ANALYZE_DEFAULT_MAX_TOKENS, requested)
            # Analysis is allowed to sample; brief mode retains today's greedy
            # behavior so a legacy request remains byte-for-byte equivalent.
            gen.do_sample = True
            gen.temperature = float(body.get("temperature", 0.7))
            gen.top_p = float(body.get("top_p", 0.9))
        else:
            gen.max_new_tokens = requested
            gen.do_sample = False

        if messages is None and mode == "brief":
            src_index = {src["url"]: n for n, src in enumerate(sources, 1)}
            by_source = {}
            for result in payload.get("results", []):
                n = src_index.get(result.get("url"))
                if n:
                    by_source.setdefault(n, []).append(result["text"])
            blocks = ["[" + str(n) + "] " + sources[n - 1]["title"] + "\n"
                      + "\n".join(by_source[n]) for n in sorted(by_source)]
            cited = "\n\n".join(blocks)
            prompt = (
                "Answer the question using ONLY the sources below. Cite them inline as "
                "[1], [2] and so on. If the sources do not contain the answer, say so "
                "plainly and do not guess.\n\nSources:\n" + cited
                + "\n\nQuestion: " + str(body.get("query") or "") + " /no_think")
            history = ovg.ChatHistory()
            history.append({"role": "user", "content": prompt})
        else:
            history_messages, meta = _web_prepare_history(
                chat_slot, messages or [{"role": "user", "content":
                                         str(body.get("query") or "")}],
                payload.get("results", []), sources,
                str(body.get("query") or ""), mode)
            history = ovg.ChatHistory()
            for message in history_messages:
                history.append(message)
            payload.update(meta)

        with chat_slot.lock:
            out = chat_slot.pipe.generate(history, generation_config=gen)
        raw = str(out)
        answer = strip_thinking(raw).strip()
        payload["answer"] = answer or raw.strip() or None
        payload["answer_was_only_reasoning"] = not answer and bool(raw.strip())
        payload["timings_ms"]["answer"] = round((time.perf_counter() - t4) * 1000)
    except Exception as e:
        payload["answer"] = None
        payload["answer_error"] = explain_genai_error(e)
    yield "done", payload


def _web_search_run(body):
    """The search pipeline as a generator of stage events.

    Yields ("stage", {...}) as each phase begins, then exactly one ("done",
    payload) or ("error", (message, status)). The route below either drains it
    (JSON) or forwards each event as SSE.

    It is a generator so the UI can say what is happening. A search turn is
    20-33 s against 12 s for a plain answer, and for most of that the old
    version showed a single blinking dot -- indistinguishable from a hung
    request. The stages are real measurements, not a timer pretending: each is
    emitted when that phase actually starts.
    """
    query = str(body.get("query") or "").strip()
    if not query:
        yield "error", ("'query' is required", 400)
        return
    top_k = min(10, max(1, int(body.get("top_k", 5))))
    engine = str(body.get("engine") or "npu").lower()
    mode = str(body.get("mode") or "brief").lower()
    if mode not in ("brief", "analyze"):
        mode = "brief"

    # Analyse is a synthesis pass over the ranked passages in the preceding
    # search turn. It deliberately never wakes SearXNG; an empty cache falls
    # through to the normal retrieval path below.
    if mode == "analyze" and body.get("passages"):
        payload = _web_reused_payload(body, query, engine)
        yield from _web_search_answer(payload, body, mode)
        return

    if not _searx.listening() and _searx.can_start():
        yield "stage", {"stage": "starting", "detail": "starting the search backend"}
    ok, why = _searx.ensure_running()
    if not ok:
        yield "error", (why, 503)
        return

    yield "stage", {"stage": "searching", "detail": "searching the web"}
    t0 = time.perf_counter()
    try:
        hits = _web_search_hits(query, _WEB_MAX_RESULTS)
    except Exception as e:
        yield "error", (f"Could not reach the search backend at "
                        f"{WEB_SEARCH_URL}: {e}", 502)
        return
    t_search = time.perf_counter() - t0
    if not hits:
        yield "done", {"query": query, "results": [], "sources": [],
                       "note": "the search backend returned nothing"}
        return

    n_fetch = min(len(hits), _WEB_MAX_FETCH)
    yield "stage", {"stage": "reading", "detail": f"reading {n_fetch} pages",
                    "pages": n_fetch}
    t1 = time.perf_counter()
    pages, fetch_failures = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=_WEB_MAX_FETCH) as pool:
        futures = {pool.submit(_fetch_page_text, h["url"]): h
                   for h in hits[:_WEB_MAX_FETCH]}
        for fut in concurrent.futures.as_completed(futures):
            hit = futures[fut]
            try:
                text, reason = fut.result()
            except Exception as e:
                text, reason = None, f"{type(e).__name__}"
            if text:
                pages[hit["url"]] = text
            else:
                fetch_failures[hit["url"]] = reason
                print(f"  [search] skipped {hit['url'][:70]} - {reason}",
                      flush=True)
    t_fetch = time.perf_counter() - t1

    # A page that failed still contributes its search snippet rather than
    # vanishing: paywalled and JS-only pages are common, and a short honest
    # snippet beats pretending the result did not exist.
    chunks = []
    for hit in hits:
        text = pages.get(hit["url"])
        if text:
            for c in _util_chunks(text, hit["title"]):
                c["url"] = hit["url"]
                chunks.append(c)
        elif hit["snippet"]:
            chunks.append({"source": hit["title"], "url": hit["url"],
                           "text": hit["snippet"], "snippet": True})
        if len(chunks) >= _WEB_MAX_CHUNKS:
            break
    chunks = chunks[:_WEB_MAX_CHUNKS]
    if not chunks:
        yield "done", {"query": query, "results": [],
                       "sources": [{"url": h["url"], "title": h["title"]}
                                   for h in hits],
                       "note": "found results but could not read any of them"}
        return

    found = len(chunks)
    chunks, prefiltered = _lexical_prefilter(query, chunks)
    detail = f"ranking {len(chunks)} passages on {engine.upper()}"
    if prefiltered:
        detail += f" (of {found})"
    yield "stage", {"stage": "ranking", "detail": detail,
                    "chunks": len(chunks), "chunks_found": found}
    try:
        slot = _util_slot(engine)
        slot.ensure_loaded()
        t2 = time.perf_counter()
        vectors, _ = slot.run_utility("embed", [c["text"] for c in chunks])
        query_vector, _ = slot.run_utility("embed", [query])
        t_embed = time.perf_counter() - t2
    except _TurnError:
        yield "error", ("No utility engine is available to rank results.", 503)
        return
    except UtilityUnavailable as e:
        yield "error", (str(e), 503)
        return
    except Exception as e:
        yield "error", (f"Ranking failed: {e}", 500)
        return

    scores = np.asarray(vectors) @ query_vector[0]
    pool_size = min(len(chunks), max(top_k, _UTIL_RERANK_POOL))
    pool_idx = [int(i) for i in np.argsort(scores)[::-1][:pool_size]]
    order, reranked, t_rerank = pool_idx[:top_k], False, 0.0
    cross = {}
    try:
        t3 = time.perf_counter()
        rr, _ = slot.run_utility("rerank", query,
                                 [chunks[i]["text"] for i in pool_idx])
        t_rerank = time.perf_counter() - t3
        cross = {i: float(s) for s, i in zip(rr, pool_idx)}
        ranked = [(float(sc) - (_WEB_SNIPPET_PENALTY
                                if chunks[i].get("snippet") else 0.0), i)
                  for sc, i in zip(rr, pool_idx)]
        order = [i for _s, i in sorted(ranked, key=lambda p: p[0],
                                       reverse=True)[:top_k]]
        reranked = True
    except UtilityUnavailable:
        pass          # ordering is an improvement, not a dependency

    results = [{
        "text": chunks[i]["text"],
        "title": chunks[i]["source"],
        "url": chunks[i].get("url"),
        # Both numbers: a reranked list shows a lower cosine above a higher one
        # and that reads as a bug without the score the sort actually used.
        "score": float(scores[i]),
        "rerank_score": cross.get(i),
    } for i in order]

    # A source the fetcher could not read still appears -- its search snippet
    # was used, and hiding it would misrepresent what the answer drew on. But
    # it must SAY so: listing a page we only saw a two-line snippet of as
    # though we read it is the difference between a citation and a guess.
    seen, sources = set(), []
    for r in results:
        if r["url"] and r["url"] not in seen:
            seen.add(r["url"])
            sources.append({"url": r["url"], "title": r["title"],
                            "read": r["url"] in pages})

    payload = {
        "query": query, "results": results, "sources": sources,
        "reranked": reranked, "engine": engine,
        "pages_read": len(pages), "pages_found": len(hits),
        "chunks_ranked": len(chunks), "chunks_found": found,
        "fetch_failures": fetch_failures,
        "timings_ms": {k: round(v * 1000) for k, v in
                       (("search", t_search), ("fetch", t_fetch),
                        ("embed", t_embed), ("rerank", t_rerank))},
    }
    yield from _web_search_answer(payload, body, mode)
