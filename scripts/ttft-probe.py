#!/usr/bin/env python
"""ttft-probe.py — TTFT and decode rate at a given prompt length, over the API.

AGENT TASK B asks for "decode tok/s and TTFT at 8k and at 64k". Measuring it
through /v1/chat/completions rather than against the pipeline is deliberate:
that is the path a client takes, and it includes the prompt rendering, the
tokenizer and the SSE framing that a bare `pipe.generate()` skips.

TTFT is wall-clock to the first token of CONTENT. The first SSE frame is a role
delta carrying no text, and counting that would report a latency no user ever
experiences.

Usage:
    python scripts/ttft-probe.py <base-url> <approx-prompt-tokens> [more ...]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

# Filler that tokenizes at roughly one token per word rather than one per
# character, so "8k tokens" means about 8k tokens and not 2k.
FILLER = (
    "The runtime compiles the model graph for the device before any token is "
    "produced, and the compiled graph is cached so the second load is cheap. "
    "Prefill processes the whole prompt at once and is bound by memory "
    "bandwidth; decode then emits one token at a time and is bound by how fast "
    "weights and cache can be moved. "
)


def build_prompt(target_tokens: int) -> str:
    # English prose runs about 1.33 tokens per word, so a target in TOKENS is
    # about 0.75x that many words. Dividing instead of multiplying (the first
    # cut of this) overshoots by 1.8x, and a run labelled "8k" was really 14k.
    words = int(target_tokens * 0.75)
    text = (FILLER * (words // len(FILLER.split()) + 2)).split()
    return " ".join(text[:words])


def probe(base: str, target: int) -> None:
    prompt = build_prompt(target)
    body = json.dumps({
        "model": "any",
        "messages": [
            {"role": "user",
             "content": prompt + "\n\nIn one short sentence, what is prefill?"}
        ],
        "max_tokens": 64,
        "stream": True,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                tokens += 1
    total = time.perf_counter() - t0
    decode = tokens / (total - ttft) if ttft and total > ttft else 0
    approx_prompt = len(prompt.split())
    print(f"  ~{target // 1000}k prompt ({approx_prompt:,} words): "
          f"TTFT {ttft:6.2f} s | prefill ~{approx_prompt / ttft:7.0f} word/s | "
          f"decode {decode:5.1f} tok/s over {tokens} tokens")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    base = sys.argv[1]
    for target in (int(a) for a in sys.argv[2:]):
        probe(base, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
