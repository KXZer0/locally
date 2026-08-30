"""Limits and shaping for a web-grounded answer.

Kept beside fetch_page.py rather than inside it: these are the knobs, and
they are read from three places.
"""
import re

from core import config
from core.genai.tokens import _count_tokens
from core.web.search import _WEB_ANSWER_RESERVE
from core.web.search import _web_grounded_blocks
from core.web.history import _LOGIN_MARKERS
from core.web.history import _LOGIN_WALL_MAX_CHARS
from core.web.history import _web_history_text
from core.web.history import _web_message_copy


_WEB_ANALYZE_DEFAULT_MAX_TOKENS = 1024


_WEB_FETCH_TIMEOUT = 8         # seconds per page


_WEB_MAX_BYTES = 8 * 1024 * 1024   # post-gzip this is generous; pre-gzip 2 MB truncated real articles


_WEB_PREFILTER_KEEP = 96      # chunks that reach the embedder


_WEB_PREFILTER_MIN_PER_SOURCE = 6


def _login_wall(text):
    """Return a reason if this looks like a login/paywall, else None.

    Deliberately requires BOTH a marker and a short body: a real article about
    authentication mentions "sign in" constantly, and rejecting it would be a
    worse failure than the one this prevents. A page with 40,000 characters is
    a page, whatever words it contains.
    """
    if len(text) > _LOGIN_WALL_MAX_CHARS:
        return None
    # POSITION is the real signal, not length. An article about authentication
    # says "sign in" constantly, but it does not OPEN with it; a wall does,
    # because the prompt is the whole page. x.com's wall ran to 5,921
    # characters -- mostly navigation -- so a length-only test missed it.
    head = text[:400]
    if not _LOGIN_MARKERS.search(head):
        return None
    return ("needs a login (the site returned a sign-in wall, not the page) - "
            "open it in your browser where you are already signed in")


def _web_prepare_history(slot, messages, results, sources, query, mode):
    """Fit a message-aware grounded turn into the model's hard context window."""
    history = _web_message_copy(messages)
    if not history:
        history = [{"role": "user", "content": query}]
    user_i = next((i for i in range(len(history) - 1, -1, -1)
                   if history[i]["role"] == "user"), None)
    if user_i is None:
        history.append({"role": "user", "content": query})
        user_i = len(history) - 1

    passages = [dict(r) for r in results]
    dropped_turns, dropped_passages = [], []
    truncation_note = None
    instruction = (
        "Answer the question using ONLY the sources below. Cite them inline as "
        "[1], [2] and so on. If the sources do not contain the answer, say so "
        "plainly and do not guess."
        if mode == "brief" else
        "Analyze the question and synthesize a useful answer from the sources "
        "below. Cite every factual claim inline as [1], [2] and so on. Reconcile "
        "disagreements, call out uncertainty, and do not add facts that are not "
        "supported by the sources."
    )

    def compose():
        cited = _web_grounded_blocks(passages, sources)
        text = instruction + "\n\nSources:\n" + cited
        if truncation_note:
            text += "\n\n[Truncated: " + truncation_note + "]"
        text += "\n\nQuestion: " + query
        if mode == "brief":
            text += " /no_think"
        current = [dict(m) for m in history]
        current[user_i] = dict(current[user_i])
        current[user_i]["content"] = (current[user_i]["content"].rstrip()
                                       + "\n\n" + text)
        return current

    budget = (slot.context_tokens - _WEB_ANSWER_RESERVE
              if slot.context_tokens else None)
    if budget and budget > 0:
        count = lambda: _count_tokens(slot, _web_history_text(compose()))
        n = count()
        while n and n > budget:
            candidates = [i for i, m in enumerate(history)
                          if m["role"] != "system" and i != user_i]
            if not candidates:
                break
            start = candidates[0]
            end = start + 1
            if (end < len(history) and history[end]["role"] == "assistant"
                    and end != user_i):
                end += 1
            dropped_turns.extend(history[start:end])
            del history[start:end]
            user_i = next(i for i in range(len(history) - 1, -1, -1)
                          if history[i]["role"] == "user")
            n = count()

        while n and n > budget and len(passages) > 1:
            removed = passages.pop()
            source_n = next((i for i, src in enumerate(sources, 1)
                             if src.get("url") == removed.get("url")), None)
            dropped_passages.append({"source": source_n,
                                     "title": removed.get("title", "Source")})
            n = count()

        if n and n > budget and passages:
            original = passages[-1].get("text", "")
            lo, hi = 0, len(original)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                passages[-1]["text"] = original[:mid].rstrip()
                if (count() or 0) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            passages[-1]["text"] = original[:lo].rstrip()
            if lo < len(original):
                source_n = next((i for i, src in enumerate(sources, 1)
                                 if src.get("url") == passages[-1].get("url")), None)
                dropped_passages.append({"source": source_n,
                                         "title": passages[-1].get("title", "Source"),
                                         "trimmed": True})

        if dropped_turns or dropped_passages:
            turn_count = len(dropped_turns)
            passage_names = ", ".join(
                f"[{x['source']}] {x['title']}" if x.get("source") else x["title"]
                for x in dropped_passages)
            truncation_note = (f"dropped {turn_count} earlier message(s)"
                               if turn_count else "")
            if passage_names:
                truncation_note += ("; passages omitted from the synthesis: "
                                    if truncation_note else
                                    "passages omitted from the synthesis: ") + passage_names

    prepared = compose()
    meta = {"passages_used": passages, "passages_dropped": dropped_passages}
    if truncation_note:
        meta["truncation_note"] = truncation_note
    return prepared, meta


def _web_sources_for_results(results, sources=None):
    """Return the stable source order used by inline citations and the panel."""
    out = []
    seen = set()
    for src in sources or []:
        url = src.get("url")
        if url and url not in seen:
            out.append(dict(src))
            seen.add(url)
    for result in results:
        url = result.get("url")
        if url and url not in seen:
            out.append({"url": url, "title": result.get("title") or url,
                        "read": result.get("read", True)})
            seen.add(url)
    return out


# How many bi-encoder candidates the cross-encoder re-scores. Reranking costs
# ~13 ms per passage on the NPU, so this is a latency budget as much as a
# quality knob: 20 is ~260 ms worst case, and recall@5 was already 1.000 on the
# measured corpus — so a deeper pool buys ordering confidence, not recall.
_UTIL_RERANK_POOL = 20
