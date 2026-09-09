"""Bounded, process-local performance history for completed model turns."""

import threading
from collections import deque
from datetime import datetime, timezone


_turns = deque(maxlen=50)
_lock = threading.Lock()


def _rounded(value):
    return None if value is None else round(float(value), 1)


def record_turn(*, device, model, prompt_tokens, completion_tokens,
                ttft_ms, total_ms, finish_reason="stop", source="openvino",
                native_decode_tps=None, token_count_source="tokenizer"):
    """Append one completed inference without ever growing process memory."""
    prompt_tokens = (int(prompt_tokens) if prompt_tokens is not None else None)
    completion_tokens = (int(completion_tokens)
                         if completion_tokens is not None else None)
    ttft_ms = float(ttft_ms) if ttft_ms is not None else None
    total_ms = float(total_ms) if total_ms is not None else None
    prefill_tps = (prompt_tokens / (ttft_ms / 1000)
                   if prompt_tokens is not None and ttft_ms and ttft_ms > 0
                   else None)
    decode_ms = (max(0.0, total_ms - ttft_ms)
                 if total_ms is not None and ttft_ms is not None else None)
    decode_tps = (max(0, completion_tokens - 1) / (decode_ms / 1000)
                  if completion_tokens is not None and decode_ms
                  and decode_ms > 0 else None)
    if native_decode_tps is not None:
        decode_tps = native_decode_tps
    item = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                     .replace("+00:00", "Z"),
        "device": device,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": _rounded(ttft_ms),
        "prefill_tokens_per_second": _rounded(prefill_tps),
        "decode_tokens_per_second": _rounded(decode_tps),
        "decode_rate_source": "native" if native_decode_tps is not None else "estimated",
        "token_count_source": token_count_source,
        "end_to_end_tokens_per_second": _rounded(
            completion_tokens / (total_ms / 1000)
            if completion_tokens is not None and total_ms and total_ms > 0 else None),
        "total_ms": _rounded(total_ms),
        "finish_reason": finish_reason,
        "source": source,
    }
    with _lock:
        _turns.append(item)
    return item


def metrics_snapshot():
    """Oldest-to-newest copy of the last 50 turns."""
    with _lock:
        return [dict(item) for item in _turns]


def clear_metrics():
    """Test helper; a server process otherwise keeps history for its lifetime."""
    with _lock:
        _turns.clear()


def register_metrics_route(app):
    """Register GET /v1/metrics without putting endpoint state in locally.py."""
    from flask import jsonify

    def get_metrics():
        return jsonify(metrics_snapshot())

    app.add_url_rule("/v1/metrics", "turn_metrics", get_metrics,
                     methods=["GET"])
