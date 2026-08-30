"""Turning a document into Markdown.

PP-DocLayoutV3 segments the page and supplies a reading order -- the 7th
output column, which matters because sorting by y alone interleaves the
columns of a two-column page. Headings become headings; table, chart,
figure and formula regions are FLAGGED rather than inlined as prose, which
is honest: without a table-structure model their cell order really is
scrambled. Strictly additive -- any layout failure falls back to the plain
reading rather than costing text OCR already read correctly."""

import io
import time

from core.errors import _TurnError, openai_error
from core.slots.select import _util_slot
import threading

try:
    from utility_pipeline import UtilityUnavailable
except ImportError:
    class UtilityUnavailable(RuntimeError):
        pass

_UTIL_PDF_OCR_PAGE_LIMIT = 50


# Page furniture: correctly detected, but repeating it in the body is noise.
_LAYOUT_SKIP = {"header", "footer", "number"}


# Layout labels that carry structure rather than body prose. Headings become
# Markdown headings; the "note" labels are regions whose *content* we cannot
# currently reconstruct properly — a table's cell structure needs SLANeXt and a
# formula needs FormulaNet, and both are blocked upstream (TODONT.md). Flagging
# them is honest: the text is still there, and the reader is told it came out of
# a table rather than being silently presented as a paragraph.
_LAYOUT_HEADINGS = {"doc_title": "#", "paragraph_title": "##",
                    "figure_title": "###", "abstract": "##"}


_LAYOUT_NOTES = {"table": "table", "chart": "chart", "image": "figure",
                 "display_formula": "formula", "seal": "seal",
                 "header_image": "figure", "footer_image": "figure"}


_markitdown_lock = threading.Lock()



def _pdf_pages(raw):
    """Inspect every page, rendering at most the OCR safety-limit pages.

    PDFium's scale 1.0 is only 72 DPI, which is visibly too soft for small
    invoice text. 2.0 gives OCR a 144-DPI page while the render cap prevents a
    large scanned book from being expanded into hundreds of PIL images at once.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise RuntimeError(
            "PDF rendering is not installed. Run `pip install -r requirements.txt`.")

    pdf = pdfium.PdfDocument(raw)
    pages = []
    rendered_scans = 0
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = None
            try:
                textpage = page.get_textpage()
                text = textpage.get_text_range() or ""
                has_text = sum(ch.isalnum() for ch in text) >= 20
                image = None
                if not has_text and rendered_scans < _UTIL_PDF_OCR_PAGE_LIMIT:
                    image = page.render(scale=2.0).to_pil().convert("RGB")
                    rendered_scans += 1
                pages.append((i + 1, has_text, image))
            finally:
                if textpage is not None:
                    textpage.close()
                page.close()
    finally:
        pdf.close()
    return len(pages), pages


def _ocr_pages(slot, pages):
    """OCR rendered PDF pages and combine them into page-labelled Markdown."""
    sections, blocks = [], []
    regions = dropped = 0
    timings = {"detect": 0.0, "recognise": 0.0}
    for page_number, image in pages:
        result = slot.read(image)
        sections.append(f"## Page {page_number}\n\n{result['markdown']}")
        for block in result["blocks"]:
            blocks.append({**block, "page": page_number})
        regions += result["regions"]
        dropped += result["dropped_low_conf"]
        for key in timings:
            timings[key] += result["timings_ms"].get(key, 0)
    return {
        "markdown": "\n\n".join(sections).strip(),
        "text": "\n\n".join(sections).strip(),
        "blocks": blocks,
        "regions": regions,
        "dropped_low_conf": dropped,
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
    }


def _box_contains_fraction(inner, outer):
    """How much of `inner`'s area falls inside `outer`, 0-1."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    w = max(0.0, min(ix1, ox1) - max(ix0, ox0))
    h = max(0.0, min(iy1, oy1) - max(iy0, oy0))
    area = max(1e-9, (ix1 - ix0) * (iy1 - iy0))
    return (w * h) / area


def _layout_markdown(blocks, regions):
    """Assemble OCR text into structured Markdown using layout regions.

    OCR alone returns a flat wall of lines, because a text detector finds lines
    and has no idea which are headings or which belong to a table. PP-DocLayoutV3
    supplies both the region types and a reading order — and the reading order is
    the part that matters most: sorting text by y-coordinate interleaves the
    columns of a two-column page, while the model's own order does not.

    Blocks matching no region are appended rather than dropped. A layout model
    that misses a region must not be able to delete text that OCR read
    correctly.
    """
    kept = [b for b in blocks if b.get("kept")]
    if not kept or not regions:
        return None

    buckets = {id(r): [] for r in regions}
    orphans = []
    for block in kept:
        best, best_overlap = None, 0.25   # must be mostly inside to count
        for region in regions:
            overlap = _box_contains_fraction(block["bbox"], region["box"])
            if overlap > best_overlap:
                best, best_overlap = region, overlap
        (buckets[id(best)] if best is not None else orphans).append(block)

    out = []
    for region in regions:                      # already in reading order
        group = buckets[id(region)]
        if not group or region["label"] in _LAYOUT_SKIP:
            continue
        # Within a region, reading order is simply top-to-bottom then left.
        group.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        body = " ".join(b["text"].strip() for b in group).strip()
        if not body:
            continue
        label = region["label"]
        if label in _LAYOUT_HEADINGS:
            out.append(f"{_LAYOUT_HEADINGS[label]} {body}")
        elif label in _LAYOUT_NOTES:
            out.append(f"**[{_LAYOUT_NOTES[label]}]** {body}")
        else:
            out.append(body)
    if orphans:
        orphans.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        out.append(" ".join(b["text"].strip() for b in orphans).strip())
    return "\n\n".join(part for part in out if part).strip() or None


def _document_result(item, engine):
    """Read a native document; OCR only PDF pages without a text layer."""
    t0 = time.perf_counter()
    markdown = _native_document_markdown(
        item["raw"], item["filename"], item["extension"], item["mimetype"])
    native_ms = round((time.perf_counter() - t0) * 1000, 1)

    if item["extension"] != ".pdf":
        return {
            "text": markdown, "markdown": markdown, "blocks": [],
            "regions": 0, "dropped_low_conf": 0,
            "source": "native", "engine": None,
            "timings_ms": {"parse": native_ms},
            "pages_total": None, "pages_processed": None, "truncated": False,
        }

    page_count, page_info = _pdf_pages(item["raw"])
    scanned = [(number, image) for number, has_text, image in page_info if not has_text]
    if not scanned:
        return {
            "text": markdown, "markdown": markdown, "blocks": [],
            "regions": 0, "dropped_low_conf": 0,
            "source": "native", "engine": None,
            "timings_ms": {"parse": native_ms},
            "pages_total": page_count, "pages_processed": page_count,
            "truncated": False,
        }

    slot = _util_slot(engine)
    slot.ensure_loaded()
    chosen = scanned[:_UTIL_PDF_OCR_PAGE_LIMIT]
    ocr = _ocr_pages(slot, chosen)
    all_scanned = len(scanned) == page_count
    if all_scanned:
        combined = ocr["markdown"]
        source = "ocr"
    else:
        combined = markdown.rstrip()
        if combined:
            combined += "\n\n---\n\n# Scanned pages\n\n"
        combined += ocr["markdown"]
        source = "native+ocr"
    ocr.update({
        "text": combined, "markdown": combined, "source": source,
        "engine": slot.device_name.lower(),
        "timings_ms": {"parse": native_ms, **ocr["timings_ms"]},
        "pages_total": page_count,
        "pages_processed": page_count - len(scanned) + len(chosen),
        "truncated": len(scanned) > len(chosen),
    })
    return ocr


def _native_document_markdown(raw, filename, extension, mimetype):
    """Convert a text-bearing local document to Markdown in memory."""
    global _markitdown_instance
    try:
        from markitdown import MarkItDown, StreamInfo
    except ImportError:
        raise RuntimeError(
            "Document parsing is not installed. Run `pip install -r requirements.txt`.")

    info = StreamInfo(mimetype=mimetype or None, extension=extension,
                      filename=filename)
    # MarkItDown owns a Magika classifier and converter registry. Build it once
    # (the first construction costs seconds), and serialize conversions because
    # converter thread-safety is not part of its public contract.
    with _markitdown_lock:
        if _markitdown_instance is None:
            _markitdown_instance = MarkItDown(enable_plugins=False)
        converted = _markitdown_instance.convert_stream(
            io.BytesIO(raw), stream_info=info, file_extension=extension)
    return (converted.text_content or "").strip()
