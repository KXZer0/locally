"""Image utility request handlers."""

from flask import jsonify, request

from utility_pipeline import UtilityUnavailable
from core.documents.inputs import _pil_data_url, _util_input_from_request
from core.errors import _TurnError, openai_error
from core.slots.select import _default_util_engine, _util_slot

def _util_image_turn(operation):
    """Resolve an image utility request to a ready slot and PIL image."""
    body = request.get_json(silent=True) or {}
    engine = request.form.get("engine") or body.get("engine") or _default_util_engine()
    item = _util_input_from_request()
    if item["kind"] != "image":
        raise _TurnError(openai_error(
            f"{operation} accepts PNG, JPEG, WebP, GIF, BMP, or TIFF images."))
    slot = _util_slot(engine)
    try:
        slot.ensure_loaded()
    except Exception as e:
        raise _TurnError(openai_error(
            f"Failed to load utility models: {e}", "server_error", 500))
    return slot, item["image"]


def util_background():
    """Return a transparent PNG with the foreground retained."""
    try:
        slot, image = _util_image_turn("Background removal")
        result, timings = slot.run_utility("remove_background", image)
    except _TurnError as e:
        return e.response
    except UtilityUnavailable as e:
        # Deliberately unavailable is 503, not 500 — see _require().
        return openai_error(str(e), "server_error", 503)
    except Exception as e:
        return openai_error(f"Background removal failed: {e}", "server_error", 500)
    return jsonify({
        "image": _pil_data_url(result), "engine": slot.device_name.lower(),
        "width": result.width, "height": result.height,
        "timings_ms": timings,
    }), 200, {"X-Device": slot.device_name}


def util_upscale():
    """Upscale an image through the static 3x Open Model Zoo network."""
    try:
        slot, image = _util_image_turn("Upscaling")
        result, details = slot.run_utility("upscale", image)
    except _TurnError as e:
        return e.response
    except UtilityUnavailable as e:
        # Deliberately unavailable is 503, not 500 — see _require().
        return openai_error(str(e), "server_error", 503)
    except Exception as e:
        return openai_error(f"Upscale failed: {e}", "server_error", 500)
    timings = {"infer": details.pop("infer")}
    return jsonify({
        "image": _pil_data_url(result), "engine": slot.device_name.lower(),
        "timings_ms": timings, **details,
    }), 200, {"X-Device": slot.device_name}


def util_detect():
    """Detect common COCO objects and return an annotated image."""
    try:
        slot, image = _util_image_turn("Object detection")
        # RF-DETR's exported scores have a long high-confidence tail. A 0.5
        # default floods a normal street image with hundreds of tiny false
        # positives; 0.99 retained the bus + four people in the probe image.
        threshold = float(request.form.get("threshold") or 0.99)
        threshold = min(0.999, max(0.5, threshold))
        annotated, detections, timings = slot.run_utility(
            "detect", image, threshold=threshold)
    except _TurnError as e:
        return e.response
    except UtilityUnavailable as e:
        # Deliberately unavailable is 503, not 500 — see _require().
        return openai_error(str(e), "server_error", 503)
    except Exception as e:
        return openai_error(f"Detection failed: {e}", "server_error", 500)
    return jsonify({
        "image": _pil_data_url(annotated), "detections": detections,
        "engine": slot.device_name.lower(), "timings_ms": timings,
    }), 200, {"X-Device": slot.device_name}




def util_generate():
    """Generate a 512px image when a complete local GenAI pipeline exists."""
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return openai_error("'prompt' is required")
    if len(prompt) > 2000:
        return openai_error("'prompt' is too long (maximum 2000 characters)")
    try:
        slot = _util_slot(body.get("engine") or _default_util_engine())
        slot.ensure_loaded()
        steps = min(30, max(1, int(body.get("steps", 12))))
        seed = int(body.get("seed", 0))
        image, timings = slot.run_utility(
            "generate", prompt, steps=steps, seed=seed)
    except _TurnError as e:
        return e.response
    except UtilityUnavailable as e:
        # Deliberately unavailable is 503, not 500 — see _require().
        return openai_error(str(e), "server_error", 503)
    except Exception as e:
        return openai_error(f"Image generation failed: {e}", "server_error", 500)
    return jsonify({
        "image": _pil_data_url(image), "engine": slot.device_name.lower(),
        "width": image.width, "height": image.height,
        "steps": steps, "seed": seed, "timings_ms": timings,
    }), 200, {"X-Device": slot.device_name}


