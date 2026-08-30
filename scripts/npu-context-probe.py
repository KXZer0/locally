"""Probe how far MAX_PROMPT_LEN can be raised on the NPU.

Nothing gets built into locally.py until it passes here — same rule as
scripts/npu-probe.py.

Why this exists: MAX_PROMPT_LEN has been 4096 since it was lifted off the
driver's 1024 default, and nobody has ever tested a higher value. The common
claim online is that the cap is a *memory* limit and that KV-cache compression
lifts it. On this model that premise is measurably wrong -- Qwen3-8B is 144 KB
of KV per token, so 4096 tokens is 0.56 GB, not the ~4 GB such advice assumes.

MAX_PROMPT_LEN is a *static shape* the graph is compiled against, because the
NPU rejects dynamic dimensions (see TODONT.md). So the real question is not
"how do we save memory" but "what does the vpux compiler accept", which is a
question only a compile can answer.

Each step reports: compile time, whether generation runs, whether the answer is
CORRECT (compiling and running is not the same as working -- the repeated
lesson in NPU-UTIL-PHASE0.md), and process memory delta.

Usage:
    venv/Scripts/python.exe scripts/npu-context-probe.py [--model DIR]
                                                         [--lens 4096,8192,16384]
                                                         [--kv-precision u8]
Stop the server first: the NPU serves one process at a time.
"""

import argparse
import ctypes
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _process_mb():
    """PrivateUsage, not RSS. An idle process has already had its working set
    trimmed by Windows, which is how 'unload frees nothing' was once wrongly
    concluded (TODONT.md)."""
    class C(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]
    c = C(); c.cb = ctypes.sizeof(C)
    try:
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PrivateUsage / 2 ** 20
    except Exception:
        return 0.0


# A needle far enough in that a truncated context cannot answer it. If the cap
# silently truncates rather than erroring, this is what catches it -- which is
# the failure the user actually feels as "the NPU hallucinates".
NEEDLE = "The access code for the north gate is SPARROW-7241."
QUESTION = ("What is the access code for the north gate? "
            "Answer with the code only. /no_think")


def build_prompt(target_tokens, tokenizer):
    """Filler + needle + question, sized to land just UNDER target_tokens.

    Built by trimming the token list rather than guessing with a text loop: the
    pipeline hard-errors when input_ids exceeds MAX_PROMPT_LEN (it does not
    truncate), so overshooting by even one token loses the whole measurement.
    """
    filler = ("The maintenance log records routine inspections of the facility. "
              "Each entry notes the date, the technician, and the outcome. ")
    tail = chr(10) + chr(10) + QUESTION
    tail_n = len(tokenizer.encode(tail))
    # Needle first: recalling it at the end proves the whole window was
    # attended to, not just the tail.
    head = NEEDLE + " "
    budget = target_tokens - tail_n - 32          # leave room for BOS/template
    ids = tokenizer.encode(head + filler * 400)
    while len(ids) < budget:
        ids = tokenizer.encode(head + filler * (400 + len(ids)))
    body = tokenizer.decode(ids[:budget], skip_special_tokens=True)
    return body + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(
        "~/models/Qwen3-8B-int4-cw-ov"))
    ap.add_argument("--lens", default="4096,8192,16384")
    ap.add_argument("--kv-precision", default=None, choices=["u8", "f16"])
    ap.add_argument("--cache-dir", default=".ov-cache")
    args = ap.parse_args()

    import openvino_genai as ovg
    import openvino as ov

    devs = ov.Core().get_available_devices()
    if not any(d.startswith("NPU") for d in devs):
        print(f"FAIL: no NPU in {devs}")
        return 1

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"model      : {args.model}")
    print(f"kv precision: {args.kv_precision or 'f16 (default)'}")
    print(f"lengths    : {args.lens}\n")

    results = []
    for length in [int(x) for x in args.lens.split(",")]:
        print(f"--- MAX_PROMPT_LEN = {length} " + "-" * 40)
        opts = {"MAX_PROMPT_LEN": length, "CACHE_DIR": args.cache_dir}
        if args.kv_precision:
            opts["KV_CACHE_PRECISION"] = args.kv_precision

        before = _process_mb()
        row = {"len": length, "compile_s": None, "mem_mb": None,
               "ran": False, "correct": None, "error": None}
        pipe = None
        try:
            t0 = time.perf_counter()
            pipe = ovg.LLMPipeline(args.model, device="NPU", **opts)
            row["compile_s"] = round(time.perf_counter() - t0, 1)
            row["mem_mb"] = round(_process_mb() - before)
            print(f"  compiled in {row['compile_s']}s  (+{row['mem_mb']} MB)")

            prompt = build_prompt(length, tok)
            ntok = len(tok.encode(prompt))
            print(f"  prompt is {ntok} tokens; generating...")

            gen = ovg.GenerationConfig()
            gen.max_new_tokens = 96
            gen.do_sample = False
            t1 = time.perf_counter()
            out = pipe.generate(prompt, gen)
            dt = time.perf_counter() - t1
            text = str(out).strip()
            row["ran"] = True
            row["correct"] = "SPARROW-7241" in text.upper()
            row["prompt_tokens"] = ntok
            print(f"  generated in {dt:.1f}s -> {text[-90:]!r}")
            print(f"  RECALL: {'PASS' if row['correct'] else 'FAIL (needle lost)'}")
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  FAIL {row['error']}")
        finally:
            del pipe
            gc.collect()
            time.sleep(1.5)
        results.append(row)
        print()

    print("=" * 62)
    print(f"{'MAX_PROMPT_LEN':>15} {'compile':>9} {'mem':>8} {'ran':>5} {'recall':>7}")
    for r in results:
        if r["error"]:
            print(f"{r['len']:>15} {'-':>9} {'-':>8} {'FAIL':>5}  {r['error'][:40]}")
        else:
            print(f"{r['len']:>15} {r['compile_s']:>8}s {r['mem_mb']:>6}MB "
                  f"{'yes':>5} {'PASS' if r['correct'] else 'FAIL':>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
