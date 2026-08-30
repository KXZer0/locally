"""Getting a document or image out of a request, and back into a response."""

from PIL import Image
import base64
import io
import os
import re
from core.errors import _TurnError, openai_error
from core.slots.select import _util_slot

from flask import request

from core.media.images import load_image
from core.errors import _TurnError, openai_error

_UTIL_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


_UTIL_NATIVE_EXTS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm",
    ".epub", ".csv", ".tsv", ".txt", ".md", ".json", ".xml",
    ".yaml", ".yml",
}


_UTIL_SEARCH_MAX_CHUNKS = 500



def _util_item(raw, filename, mimetype=""):
    """Classify one in-memory utility upload and decode images."""
    if not raw:
        raise _TurnError(openai_error("Uploaded file is empty"))
    filename = os.path.basename((filename or "document").replace("\0", ""))
    ext = os.path.splitext(filename)[1].lower()
    mimetype = (mimetype or "").lower()
    if ext in _UTIL_IMAGE_EXTS or (not ext and mimetype.startswith("image/")):
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise _TurnError(openai_error(f"Failed to read image: {e}"))
        return {"kind": "image", "image": img, "filename": filename,
                "mimetype": mimetype}
    if ext not in _UTIL_NATIVE_EXTS:
        supported = ", ".join(sorted(_UTIL_IMAGE_EXTS | _UTIL_NATIVE_EXTS))
        raise _TurnError(openai_error(
            f"Unsupported file type '{ext or 'unknown'}'. Supported: {supported}"))
    return {"kind": "document", "raw": raw, "filename": filename,
            "extension": ext, "mimetype": mimetype}


def _util_input_from_request():
    """Read one utility upload without writing user content to disk.

    MarkItDown's stream API lets Office/PDF inputs stay in memory. The
    extension is allowlisted before the converter sees anything — this is a
    document reader, not a generic archive/URL opener.
    """
    if "file" in request.files:
        upload = request.files["file"]
        return _util_item(upload.read(), upload.filename, upload.mimetype)

    body = request.get_json(silent=True) or {}
    url = body.get("image") or body.get("image_url") or ""
    if not url:
        raise _TurnError(openai_error(
            "Send an image as multipart 'file', or JSON {\"image\": "
            "\"data:image/png;base64,...\"}."))
    try:
        # load_image's max_dim is unused (only pil_to_tensor resizes), which is
        # what we want here — see the docstring on util_read.
        return {"kind": "image", "image": load_image(url, 0),
                "filename": "image", "mimetype": "image/*"}
    except Exception as e:
        raise _TurnError(openai_error(f"Failed to read image: {e}"))


def _pil_data_url(image, image_format="PNG"):
    buf = io.BytesIO()
    image.save(buf, format=image_format)
    mime = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _util_chunks(text, source, target=900, overlap=120):
    """Make bounded, overlapping text chunks without splitting every paragraph."""
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text) and len(chunks) < _UTIL_SEARCH_MAX_CHUNKS:
        end = min(len(text), start + target)
        if end < len(text):
            boundary = max(text.rfind("\n\n", start + target // 2, end),
                           text.rfind(". ", start + target // 2, end))
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"source": source, "text": chunk})
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks
