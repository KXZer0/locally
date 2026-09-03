"""OCR fallback for image turns served by text-only models."""

from datetime import datetime

from core import runtime
from core.genai.tokens import _count_tokens
from core.media.images import load_image
from core.slots.select import _slot_serviceable

def _image_urls_in(messages):
    """Every image URL in an OpenAI message list, in order."""
    urls = []
    for msg in messages or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "image_url":
                url = (block.get("image_url") or {}).get("url", "")
                if url:
                    urls.append(url)
    return urls


def _ocr_for_text_model(messages, slot):
    """Read attached images with the NPU utilities so a text model can 'see'.

    The NPU has no working vision path, so an image sent to an NPU chat model
    used to be a hard error. Reading it with the OCR utilities and handing the
    model the text is not a substitute for vision — it cannot describe a
    photograph — but for the things people actually paste at a local model
    (screenshots, receipts, scanned pages, error dialogs) it is *better* than a
    small VLM, which is the finding this whole feature rests on.

    Returns (text_block, truncated_note) or (None, None) when OCR isn't
    possible, so the caller can fall back to its original error.
    """
    util = next((u for u in (runtime.util_npu, runtime.util_gpu, runtime.util_cpu)
                 if u and _slot_serviceable(u)), None)
    if util is None:
        return None, None

    urls = _image_urls_in(messages)
    if not urls:
        return None, None

    try:
        util.ensure_loaded()
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} !! [{util.device_name}] "
              f"util load failed, cannot read images: {e}", flush=True)
        return None, None

    parts = []
    for i, url in enumerate(urls):
        try:
            # Full resolution on purpose: max_dim exists to fit a VLM's budget,
            # and downscaling a screenshot is how fine print stops being legible.
            img = load_image(url, 0)
            result = util.read(img)
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [{util.device_name}] "
                  f"OCR failed on image {i + 1}: {e}", flush=True)
            continue
        text = (result.get("text") or "").strip()
        label = f"Image {i + 1}" if len(urls) > 1 else "Image"
        parts.append(f"{label}:\n{text}" if text
                     else f"{label}: (no readable text found)")

    if not parts:
        return None, None

    block = ("Text read from the attached image(s) by local OCR:\n\n"
             + "\n\n".join(parts))

    # The NPU slot is built with MAX_PROMPT_LEN=4096 and that is a hard cap, so
    # a dense multi-page scan cannot simply be pasted in. Trim the OCR text —
    # never the user's own question — and say so, because an answer drawn from
    # half a document without a word of warning is the worst outcome here.
    note = None
    if slot is not None and slot.device_name == "NPU":
        budget = 4096 - 700            # leaves room for system prompt + reply
        n = _count_tokens(slot, block)
        if n and n > budget:
            keep = max(200, int(len(block) * budget / n))
            block = block[:keep].rstrip()
            note = f"{n} tokens of OCR text trimmed to ~{budget}"
            block += ("\n\n[Truncated: the page did not fit this model's "
                      "prompt limit. Switch the chat model to the GPU for the "
                      "whole document.]")
    return block, note


