#!/usr/bin/env python
"""npu-ab.py — load two NPU models in turn and score them on the same prompts.

AGENT TASK A asks for "load time on NPU, decode tok/s, and a 20-prompt factual
score against the current Qwen3-8B-int4-cw". This is that, and it is
deliberately a *paired* run: the two models see identical prompts, identical
generation config and the same machine minutes apart, because a tok/s figure
from one session and a quality impression from another are not a comparison.

Scoring is exact-substring against a list of accepted answers. That is a blunt
instrument and it is the right one here: the question is not "is this model
good", it is "did abliteration cost anything measurable versus the export we
already trust". A blunt scorer applied identically to both is a fair difference
even when it is an unfair absolute.

Usage:
    python scripts/npu-ab.py <model-dir> [<model-dir> ...]
"""
from __future__ import annotations

import gc
import sys
import time

import openvino_genai as ov_genai

# Kept short on purpose: the NPU's prompt cap is 8192 and every token of prompt
# competes with the answer. These are one-line questions with one-word answers.
PROMPTS = [
    ("What is the capital city of Australia? Answer with the city name only.",
     ["canberra"]),
    ("How many bits are in one byte? Answer with the number only.", ["8", "eight"]),
    ("What gas do plants absorb from the air for photosynthesis? Two words.",
     ["carbon dioxide", "co2"]),
    ("Who wrote the play Hamlet? Surname only.", ["shakespeare"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("In what year did the Second World War end? Year only.", ["1945"]),
    ("What is the largest planet in the Solar System?", ["jupiter"]),
    ("What does CPU stand for?", ["central processing unit"]),
    ("What is 17 multiplied by 3? Number only.", ["51"]),
    ("Which ocean lies between Africa and Australia?", ["indian"]),
    ("What language is primarily spoken in Brazil?", ["portuguese"]),
    ("What is the freezing point of water in Celsius? Number only.", ["0", "zero"]),
    ("Which element has the atomic number 1?", ["hydrogen"]),
    ("What is the square root of 144? Number only.", ["12", "twelve"]),
    ("Who painted the Mona Lisa? Surname only.", ["vinci", "leonardo"]),
    ("What is the smallest prime number? Number only.", ["2", "two"]),
    ("What organ pumps blood around the human body?", ["heart"]),
    ("How many continents are there? Number only.", ["7", "seven"]),
    ("What is the currency of Japan?", ["yen"]),
    ("Which planet is known as the Red Planet?", ["mars"]),
]

# Qwen3 is a thinking model; /no_think is the control token that keeps it out of
# a <think> block it would otherwise spend the whole budget inside. CLAUDE.md
# measured 210 tokens/13.4 s become 31 tokens/3.1 s with it.
SUFFIX = " /no_think"
MAX_NEW = 48


def run(model_dir: str) -> dict:
    print(f"\n=== {model_dir}")
    t0 = time.perf_counter()
    pipe = ov_genai.LLMPipeline(model_dir, "NPU", MAX_PROMPT_LEN=8192)
    load_s = time.perf_counter() - t0
    print(f"  load           {load_s:6.1f} s")

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = MAX_NEW
    cfg.do_sample = False

    hits, misses, tok_total, gen_s = 0, [], 0, 0.0
    for question, accepted in PROMPTS:
        prompt = f"<|im_start|>user\n{question}{SUFFIX}<|im_end|>\n<|im_start|>assistant\n"
        t = time.perf_counter()
        out = pipe.generate(prompt, cfg)
        dt = time.perf_counter() - t
        text = str(out)
        # Strip any think block the model opened anyway.
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        text = text.replace("<think>", " ").strip()
        n = len(out.tokens[0]) if getattr(out, "tokens", None) else MAX_NEW
        tok_total += n
        gen_s += dt
        low = text.lower()
        if any(a in low for a in accepted):
            hits += 1
        else:
            misses.append((question, text[:70].replace("\n", " ")))

    tps = tok_total / gen_s if gen_s else 0
    print(f"  decode         {tps:6.1f} tok/s   ({tok_total} tokens in {gen_s:.1f} s)")
    print(f"  factual        {hits}/{len(PROMPTS)}")
    for q, got in misses:
        print(f"     miss: {q[:52]:<54} -> {got!r}")

    del pipe
    gc.collect()
    return {"dir": model_dir, "load_s": load_s, "tok_s": tps,
            "score": hits, "of": len(PROMPTS)}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    results = [run(d) for d in sys.argv[1:]]
    print("\n" + "=" * 72)
    print(f"{'model':<44}{'load':>8}{'tok/s':>9}{'factual':>10}")
    for r in results:
        name = r["dir"].rstrip("\\/").split("\\")[-1].split("/")[-1]
        print(f"{name:<44}{r['load_s']:7.1f}s{r['tok_s']:8.1f}{r['score']:>7}/{r['of']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
