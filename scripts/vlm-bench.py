"""VLM text-only bench with the honest methodology (warm-up + median, real tokens).

Usage: python vlm_bench.py <model_dir> <device>
"""
import sys, time, traceback
import openvino_genai as og

MODEL, DEVICE = sys.argv[1], sys.argv[2]
print(f"genai {og.__version__}, {DEVICE}, VLMPipeline, {MODEL}", flush=True)
t0 = time.time()
try:
    pipe = og.VLMPipeline(MODEL, DEVICE)
    print(f"OK: loaded in {time.time()-t0:.1f}s", flush=True)
except Exception:
    print(f"LOAD FAILED after {time.time()-t0:.1f}s:", flush=True)
    traceback.print_exc()
    sys.exit(1)

cfg = og.GenerationConfig()
cfg.max_new_tokens = 64
rates = []
try:
    for i in range(3):
        stamps, chunks = [], []

        def meter(sub):
            stamps.append(time.time())
            chunks.append(sub)
            return False

        t1 = time.time()
        pipe.generate("Explain what a hash map is and when to use one.",
                      generation_config=cfg, streamer=meter)
        n = len(stamps)
        rate = (n - 1) / (stamps[-1] - stamps[0]) if n > 1 else 0
        label = "warm-up" if i == 0 else f"run {i}"
        print(f"  {label}: {n} tokens in {time.time()-t1:.1f}s -> {rate:.1f} tok/s", flush=True)
        if i > 0:
            rates.append(rate)
    steady = sorted(rates)[len(rates) // 2]
    print(f"OK: steady-state {steady:.1f} tok/s", flush=True)
    print("OUTPUT:", "".join(chunks)[:160].replace(chr(10), " "), flush=True)
    print("RESULT: succeeded", flush=True)
except Exception:
    print("GENERATION FAILED:", flush=True)
    traceback.print_exc()
    sys.exit(1)
