"""Ollama single-prompt generation and compatibility stubs."""

import json
import threading
import time
from datetime import datetime
from queue import Empty, Queue

import openvino_genai as ovg
from flask import Response, jsonify, request

from core import config
from core.chat.common import apply_penalties, overall_status
from core.chat.openai import chat_completions
from core.genai.results import explain_genai_error
from core.media.images import load_image, pil_to_tensor
from core.slots.route import _route_request

def ollama_generate():
    """Single-turn completion (no chat history)."""
    if overall_status() != "ready":
        return jsonify({"error": "model not ready"}), 503

    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "")
    stream = body.get("stream", True)
    requested_model = body.get("model", "")
    max_tokens = body.get("options", {}).get("num_predict", 2048)
    temperature = body.get("options", {}).get("temperature", 0.0)

    # Images in generate endpoint
    images_b64 = body.get("images", [])
    has_images = bool(images_b64)

    slot = _route_request(has_images, requested_model)
    if slot is None:
        return jsonify({"error": "no model ready"}), 503

    try:
        slot.ensure_loaded()
    except Exception as e:
        return jsonify({"error": f"Failed to reload model: {e}"}), 500

    gen = ovg.GenerationConfig()
    gen.max_new_tokens = max_tokens
    if temperature and temperature > 0.01:
        gen.do_sample = True
        gen.temperature = temperature
    else:
        gen.do_sample = False
        gen.top_k = 1
    _opts = body.get("options", {})
    apply_penalties(gen,
                    repetition=_opts.get("repeat_penalty"),
                    frequency=_opts.get("frequency_penalty"),
                    presence=_opts.get("presence_penalty"))

    t0 = time.perf_counter()

    # VLM with images
    if has_images and slot.model_type == "vlm":
        img_tensors = []
        for b64 in images_b64:
            img = load_image(f"data:image/jpeg;base64,{b64}", config.MAX_IMAGE_DIM)
            img_tensors.append(pil_to_tensor(img, config.MAX_IMAGE_DIM))
        try:
            text = slot.generate_vlm(prompt, img_tensors, gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        elapsed = time.perf_counter() - t0
        return jsonify({
            "model": slot.model_name,
            "response": text,
            "done": True,
            "total_duration": int(elapsed * 1e9),
        })

    if has_images and slot.model_type == "llm":
        return jsonify({"error": "model does not support images"}), 400

    # Text-only generate → wrap as single-turn chat
    raw_messages = [{"role": "user", "content": prompt}]

    if stream and slot.model_type == "llm":
        return Response(
            _ollama_stream_generate(slot, raw_messages, gen, t0),
            mimetype="application/x-ndjson",
        )

    # Non-streaming
    if slot.model_type == "vlm":
        try:
            text = slot.generate_vlm(prompt, [], gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        try:
            text = slot.generate_llm(raw_messages, gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elapsed = time.perf_counter() - t0
    return jsonify({
        "model": slot.model_name,
        "response": text,
        "done": True,
        "total_duration": int(elapsed * 1e9),
    })


def _ollama_stream_generate(slot, raw_messages, gen, t0):
    """Ollama /api/generate streaming."""
    history = ovg.ChatHistory()
    for msg in raw_messages:
        history.append({"role": msg["role"], "content": msg["content"]})

    token_queue = Queue()
    token_count = 0

    def streamer_callback(token):
        if slot._cancel.is_set():
            return True
        token_queue.put(token)
        return False

    def _generate():
        try:
            with slot.lock:
                slot._cancel.clear()
                slot.pipe.generate(history, gen, streamer_callback)
                slot.last_used = time.time()
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] [Ollama] "
                  f"generate error: {explain_genai_error(e)}", flush=True)
        finally:
            token_queue.put(None)

    t = threading.Thread(target=_generate, daemon=True)
    t.start()

    try:
        while True:
            try:
                token = token_queue.get(timeout=120)
            except Empty:
                break
            if token is None:
                break
            token_count += 1
            yield json.dumps({
                "model": slot.model_name,
                "response": token,
                "done": False,
            }) + "\n"

        elapsed = time.perf_counter() - t0
        yield json.dumps({
            "model": slot.model_name,
            "response": "",
            "done": True,
            "total_duration": int(elapsed * 1e9),
            "eval_count": token_count,
        }) + "\n"
    finally:
        slot._cancel.set()


# Stubs — clients expect these to exist
def ollama_pull():
    return jsonify({"status": "success"})


def ollama_delete():
    return "", 200


def ollama_copy():
    return "", 200


# Copilot Chat 0.53+ sends actual chat via /v1/chat/completions on the Ollama
# port rather than /api/chat — delegate to the same handler.
def ollama_v1_chat_completions():
    return chat_completions()
