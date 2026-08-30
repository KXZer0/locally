"""Test OpenVINO's MoE disk offload (OFFLOAD_RATIO) on a GPU.

Usage:
    python scripts/offload-test.py <ratio> <model_dir> [device]

Run twice — ratio 0, then 90 — and compare `mem usm_device`:
  - drops sharply at 90  -> offload engaged (XMX GPU, fusable MoE IR)
  - identical            -> offload not engaged (most likely no XMX:
    OPTIMIZATION_CAPABILITIES must list GPU_HW_MATMUL — see TODONT.md)

Requires openvino + openvino-genai >= 2026.3. The property only ever
applies to MoE expert weights; dense models are unaffected.
"""
import sys, time, traceback

import openvino as ov
import openvino_genai as og

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(2)

RATIO = int(sys.argv[1])
MODEL = sys.argv[2]
DEVICE = sys.argv[3] if len(sys.argv) > 3 else "GPU"

print(f"genai {og.__version__}, {DEVICE}, OFFLOAD_RATIO={RATIO}", flush=True)
t0 = time.time()
try:
    if RATIO > 0:
        pipe = og.LLMPipeline(MODEL, DEVICE, OFFLOAD_RATIO=RATIO)
    else:
        pipe = og.LLMPipeline(MODEL, DEVICE)
    print(f"OK: loaded in {time.time()-t0:.1f}s", flush=True)
except Exception:
    print(f"LOAD FAILED after {time.time()-t0:.1f}s:", flush=True)
    traceback.print_exc()
    sys.exit(1)

def report_memory(when):
    if DEVICE.startswith("GPU"):
        try:
            stats = ov.Core().get_property(DEVICE, "GPU_MEMORY_STATISTICS")
            for k, v in sorted(stats.items()):
                print(f"  mem {k} ({when}): {v/2**30:.2f} GB", flush=True)
        except Exception as e:
            print(f"  (GPU memory statistics unavailable: {e})", flush=True)
        return
    # CPU has no plugin memory-statistics property; report what the process
    # actually holds instead. Note: CPU weights are mapped lazily, so the
    # post-load resident number is small — post-generate is the honest one.
    try:
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes as wt

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [
                    (n, ctypes.c_size_t) for n in (
                        "PeakWorkingSetSize", "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage")]

            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
            ctypes.windll.kernel32.GetCurrentProcess.restype = wt.HANDLE
            pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
            psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                       ctypes.byref(pmc), pmc.cb)
            print(f"  mem resident ({when}): {pmc.WorkingSetSize/2**30:.2f} GB "
                  f"(committed {pmc.PagefileUsage/2**30:.2f} GB)", flush=True)
        else:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith(("VmRSS", "VmSwap")):
                        print(f"  mem {line.strip()} ({when})", flush=True)
    except Exception as e:
        print(f"  (process memory info unavailable: {e})", flush=True)


report_memory("post-load")

cfg = og.GenerationConfig()
cfg.max_new_tokens = 64
# Run 1 pays the bills — CPU faults weights in lazily, and offloaded GPU
# runs start with a cold expert LRU — so warm-up must be excluded from the
# verdict. Two strategies:
#
# - Offload ACTIVE (GPU + ratio>0): ONE long generate, per-token timing via
#   streamer; first third is warm-up, steady-state = rate over the last
#   half. (A second generate() on an offload-active pipeline hangs in
#   native code — observed on Arc 140V, genai 2026.3, Qwen3-30B ratio 50,
#   uninterruptible by Ctrl-C — so multi-run is not an option there.)
# - Otherwise: 3 generates; first labeled warm-up, verdict = median of the
#   rest.
try:
    if DEVICE.startswith("GPU") and RATIO > 0:
        cfg.max_new_tokens = 192
        stamps = []
        chunks = []

        def meter(sub):
            stamps.append(time.time())
            chunks.append(sub)
            return False

        t1 = time.time()
        pipe.generate("Explain what a hash map is and when to use one.", cfg, meter)
        out = "".join(chunks)
        n = len(stamps)
        if n >= 32:
            third, half = n // 3, n // 2
            warm = third / (stamps[third - 1] - t1)
            steady = (n - half) / (stamps[-1] - stamps[half - 1])
            print(f"  warm-up (first {third} tokens): {warm:.1f} tok/s", flush=True)
            print(f"OK: steady-state {steady:.1f} tok/s "
                  f"(last {n - half} of {n} tokens, single pass)", flush=True)
        else:
            print(f"OK: {n} tokens in {time.time()-t1:.1f}s "
                  f"-> {n/(time.time()-t1):.1f} tok/s (too short to split)", flush=True)
    else:
        # Count real tokens via streamer — a model that answers briefly and
        # hits EOS would otherwise inflate the rate (64/dt for a 4-token
        # "Hello!" reported 645 tok/s once). The prompt is chosen to
        # reliably out-generate the 64-token budget.
        rates = []
        out = ""
        for i in range(3):
            stamps = []
            chunks = []

            def meter(sub):
                stamps.append(time.time())
                chunks.append(sub)
                return False

            t1 = time.time()
            pipe.generate("Explain what a hash map is and when to use one.", cfg, meter)
            out = "".join(chunks)
            n = len(stamps)
            dt = time.time() - t1
            rate = (n - 1) / (stamps[-1] - stamps[0]) if n > 1 else 0
            label = "warm-up" if i == 0 else f"run {i}"
            print(f"  {label}: {n} tokens in {dt:.1f}s -> {rate:.1f} tok/s", flush=True)
            if i > 0:
                rates.append(rate)
        steady = sorted(rates)[len(rates) // 2]
        print(f"OK: steady-state {steady:.1f} tok/s "
              f"(median of {len(rates)} post-warm-up runs)", flush=True)
    report_memory("post-generate")
    print("OUTPUT:", str(out)[:200], flush=True)
    print(f"RESULT: ratio={RATIO} load+generate succeeded", flush=True)
except Exception:
    print("GENERATION FAILED:", flush=True)
    traceback.print_exc()
    sys.exit(1)
