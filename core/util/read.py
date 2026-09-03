"""Read uploaded documents, images, and public web pages."""

import time
from datetime import datetime

from flask import jsonify, request

from utility_pipeline import UtilityUnavailable
from core.documents.inputs import _util_input_from_request
from core.documents.read import _document_result, _layout_markdown
from core.errors import _TurnError, openai_error
from core.slots.select import _default_util_engine, _util_slot
from core.web.reader import (BLOCKED_STATUSES, _guard_public_url, _html_to_markdown,
                             _http_fetcher, _stealth_fetch, stealth_available)

_UTIL_URL_TIMEOUT_S = 20
_UTIL_URL_MAX_BYTES = 8 * 1024 * 1024
_UTIL_URL_MAX_REDIRECTS = 5
_UTIL_URL_MAX_CHARS = 200_000

def _url_result(url):
    """Fetch one public page and return it in the document response shape.

    No accelerator touches this — it is an HTTP request and an HTML-to-Markdown
    conversion — so `engine` is None and the header says LOCAL, exactly as the
    MarkItDown path does. Claiming a device that did not run is the one thing
    this endpoint's callers cannot check for themselves.
    """
    creq = _http_fetcher()
    _guard_public_url(url)

    t0 = time.perf_counter()
    try:
        # impersonate gives a real Chrome TLS/JA3 fingerprint, which is the
        # difference between a page answering and refusing python-requests.
        page = creq.get(url, timeout=_UTIL_URL_TIMEOUT_S,
                        impersonate="chrome", allow_redirects=True,
                        max_redirects=_UTIL_URL_MAX_REDIRECTS)
    except _TurnError:
        raise
    except Exception as e:
        raise _TurnError(openai_error(f"Failed to fetch {url}: {e}"))
    fetch_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Re-check where we actually landed. A public hostname that 302s to
    # 127.0.0.1 would otherwise smuggle a LAN service's response back out.
    for hop in list(getattr(page, "history", None) or []) + [page]:
        hop_url = getattr(hop, "url", hop)
        if hop_url and str(hop_url) != url:
            _guard_public_url(str(hop_url))

    status = int(getattr(page, "status_code", 0) or 0)
    stealth_html = None
    if status in BLOCKED_STATUSES:
        # A bot check, not a missing page. curl_cffi already sends a real Chrome
        # TLS/JA3 fingerprint, so a refusal here means the site wants an actual
        # browser -- which is exactly what Scrapling provides and the only case
        # that justifies its cost. Escalate when it is installed; otherwise say
        # so, rather than reporting a bare HTTP 403 the user cannot act on.
        try:
            stealth_html = _stealth_fetch(url, _UTIL_URL_TIMEOUT_S)
        except Exception as e:
            stealth_html = None
            print(f"  [web] stealth fetch failed for {url}: {e}", flush=True)
        if stealth_html:
            fetch_ms = round((time.perf_counter() - t0) * 1000, 1)
    if stealth_html is None and status >= 400:
        hint = ""
        if status in BLOCKED_STATUSES and not stealth_available():
            hint = (" This looks like a bot check. `pip install scrapling` "
                    "then `scrapling install` adds a real-browser fallback for "
                    "exactly this case.")
        raise _TurnError(openai_error(
            (f"{url} returned HTTP {status} "
             f"{getattr(page, 'reason', '') or ''}").strip() + hint))

    content_type = (page.headers.get("content-type") or "").split(";")[0].strip().lower()
    if stealth_html is not None:
        content_type = "text/html"      # a browser rendered it; it is a page
    if content_type and not (content_type.startswith("text/")
                             or content_type.endswith(("html", "xml"))):
        raise _TurnError(openai_error(
            f"{url} served '{content_type}', which is not a web page. Download "
            f"it and send it as a file — /v1/util/read reads PDFs and Office "
            f"documents from an upload."))

    if stealth_html is not None:
        body = stealth_html.encode("utf-8", "replace")
    else:
        body = getattr(page, "content", b"") or b""
    size = len(body if isinstance(body, (bytes, bytearray))
               else body.encode("utf-8", "ignore"))
    if size > _UTIL_URL_MAX_BYTES:
        raise _TurnError(openai_error(
            f"{url} is {size / 1e6:.1f} MB, over the {_UTIL_URL_MAX_BYTES / 1e6:.0f} MB "
            f"page limit."))

    try:
        markdown = _html_to_markdown(page.text)
    except _TurnError:
        raise
    except Exception as e:
        raise _TurnError(openai_error(f"Failed to convert {url} to Markdown: {e}",
                                      "server_error", 500))

    truncated = len(markdown) > _UTIL_URL_MAX_CHARS
    if truncated:
        markdown = markdown[:_UTIL_URL_MAX_CHARS].rstrip() + "\n\n*[truncated]*"

    return {
        "text": markdown, "markdown": markdown, "blocks": [],
        "regions": 0, "dropped_low_conf": 0,
        "source": "url", "engine": None,
        "timings_ms": {"fetch": fetch_ms},
        "pages_total": None, "pages_processed": None, "truncated": truncated,
        "url": str(getattr(page, "url", url)), "status": status,
        "bytes": size,
    }


def util_read():
    """Extract Markdown from an image, a local document, or a public web page.

    Images and scanned PDF pages use OCR on the requested engine (NPU default).
    Text-bearing Office/PDF/HTML/EPUB/tabular files use a native parser —
    inference would be slower and less accurate than reading their embedded
    XML/text. A `url` is fetched over plain HTTP and converted to Markdown, so
    a page can be handed to a model as prose instead of as tag soup. User
    uploads stay in memory; nothing here is written to disk.
    """
    body = request.get_json(silent=True) or {}
    engine = request.form.get("engine") or body.get("engine") or _default_util_engine()
    if engine.lower() not in ("npu", "gpu", "cpu"):
        return openai_error(f"Unknown engine '{engine}'. Use 'npu', 'gpu' or 'cpu'.")

    url = str(request.form.get("url") or body.get("url") or "").strip()
    if url:
        print(f"\n{datetime.now():%H:%M:%S} <- [LOCAL] util/read {url}", flush=True)
        try:
            result = _url_result(url)
        except _TurnError as e:
            return e.response
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [LOCAL] util/read error: {e}",
                  flush=True)
            return openai_error(f"Read failed: {e}", "server_error", 500)
        print(f"{datetime.now():%H:%M:%S} -> [LOCAL] util/read "
              f"{len(result['markdown'])} chars from {result['bytes']} bytes "
              f"({result['timings_ms']['fetch']}ms)", flush=True)
        return jsonify(result), 200, {"X-Device": "LOCAL",
                                      "X-Document-Source": result["source"]}

    try:
        item = _util_input_from_request()
    except _TurnError as e:
        return e.response

    if item["kind"] == "document":
        print(f"\n{datetime.now():%H:%M:%S} <- [LOCAL] util/read "
              f"{item['filename']}", flush=True)
        try:
            result = _document_result(item, engine)
        except _TurnError as e:
            return e.response
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [LOCAL] util/read error: {e}",
                  flush=True)
            return openai_error(f"Read failed: {e}", "server_error", 500)
        print(f"{datetime.now():%H:%M:%S} -> [LOCAL] util/read "
              f"{len(result['markdown'])} chars via {result['source']}", flush=True)
        headers = {"X-Device": (result["engine"] or "LOCAL").upper(),
                   "X-Document-Source": result["source"]}
        return jsonify(result), 200, headers

    try:
        slot = _util_slot(engine)
        img = item["image"]
    except _TurnError as e:
        return e.response
    try:
        slot.ensure_loaded()
    except Exception as e:
        return openai_error(f"Failed to load utility models: {e}",
                            "server_error", 500)

    print(f"\n{datetime.now():%H:%M:%S} <- [{slot.device_name}] "
          f"util/read {img.width}x{img.height}", flush=True)
    try:
        result = slot.read(img)
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
              f"util/read error: {e}", flush=True)
        return openai_error(f"Read failed: {e}", "server_error", 500)

    # Structure the flat OCR reading when a layout model is installed. This is
    # strictly additive: any failure leaves the caller with exactly the text
    # they would have had, because losing a correct OCR result to a layout
    # model's bad day is a far worse trade than returning unstructured text.
    layout_regions = []
    try:
        layout_regions, layout_timings = slot.run_utility("layout", img)
        structured = _layout_markdown(result["blocks"], layout_regions)
        if structured:
            result["markdown"] = structured
        result["timings_ms"]["layout"] = layout_timings["infer"]
    except UtilityUnavailable:
        pass                       # not installed on this engine; fine
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
              f"layout failed, returning unstructured text: {e}", flush=True)

    t = result["timings_ms"]
    layout_note = (f", layout {len(layout_regions)} regions "
                   f"({t['layout']}ms)" if layout_regions else "")
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
          f"util/read {len(result['text'])} chars, {result['regions']} regions "
          f"(detect {t['detect']}ms, read {t['recognise']}ms)"
          f"{layout_note}", flush=True)

    return jsonify({
        "text": result["text"],
        "markdown": result["markdown"],
        "blocks": result["blocks"],
        "regions": result["regions"],
        # Labelled page regions in reading order, when a layout model ran.
        "layout": layout_regions,
        "dropped_low_conf": result["dropped_low_conf"],
        "source": "ocr",
        "engine": slot.device_name.lower(),
        "timings_ms": t,
        "pages_total": 1,
        "pages_processed": 1,
        "truncated": False,
    }), 200, {"X-Device": slot.device_name}


