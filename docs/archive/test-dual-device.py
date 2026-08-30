#!/usr/bin/env python3
"""Test running GPU (VLM) and NPU (LLM) simultaneously.

Loads the VLM on the ARC GPU and a small LLM on the NPU, then runs
both at the same time to see if they can coexist.

Usage:
    python test-dual-device.py

Requires:
    - model/              (VLM, e.g. Qwen2.5-VL-3B — loaded on GPU)
    - ~/models/deepseek-r1-1.5b-npu-ov/  (LLM — loaded on NPU)
"""

import os
import sys
import time
import threading
from pathlib import Path

import openvino as ov
import openvino_genai as ovg
import numpy as np
from PIL import Image

# --- Config ---
SCRIPT_DIR = Path(__file__).parent
VLM_DIR = str(SCRIPT_DIR / "model")
NPU_DIR = str(Path.home() / "models" / "deepseek-r1-1.5b-npu-ov")
TEST_IMAGE = None  # will find one automatically

# Find a test image from compare-thumbs
THUMB_ROOT = SCRIPT_DIR.parent / "compare-thumbs"
if THUMB_ROOT.exists():
    for img in THUMB_ROOT.rglob("S1_*.jpg"):
        TEST_IMAGE = str(img)
        break


def load_tensor(path, max_dim=768):
    img = Image.open(path).convert("RGB")
    if max(img.width, img.height) > max_dim:
        ratio = max_dim / max(img.width, img.height)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[None, ...])
    return ov.Tensor(arr)


def run_gpu_vlm(results):
    """Load VLM on GPU, run an image inference."""
    try:
        print("[GPU] Loading VLM...", flush=True)
        t0 = time.perf_counter()
        pipe = ovg.VLMPipeline(VLM_DIR, device="GPU")
        load_time = time.perf_counter() - t0
        print(f"[GPU] VLM loaded in {load_time:.1f}s", flush=True)

        gen = ovg.GenerationConfig()
        gen.max_new_tokens = 30
        gen.do_sample = False
        gen.top_k = 1

        # Warmup
        print("[GPU] Warmup...", flush=True)
        pipe.generate(prompt="Hello", generation_config=gen)

        # Text-only test
        print("[GPU] Text-only inference...", flush=True)
        t0 = time.perf_counter()
        res = pipe.generate(prompt="What is 2+2? One word.", generation_config=gen)
        text = res if isinstance(res, str) else getattr(res, 'texts', [str(res)])[0]
        elapsed = time.perf_counter() - t0
        print(f"[GPU] Text result: {text.strip()!r} ({elapsed:.2f}s)", flush=True)

        # Image test
        if TEST_IMAGE:
            print(f"[GPU] Image inference: {os.path.basename(TEST_IMAGE)}", flush=True)
            tensor = load_tensor(TEST_IMAGE)
            t0 = time.perf_counter()
            res = pipe.generate(
                prompt='What vehicle? Reply JSON: {"vehicle":"car","colour":"..."}',
                images=tensor, generation_config=gen,
            )
            text = res if isinstance(res, str) else getattr(res, 'texts', [str(res)])[0]
            elapsed = time.perf_counter() - t0
            print(f"[GPU] Image result: {text.strip()!r} ({elapsed:.2f}s)", flush=True)
        else:
            print("[GPU] No test image found, skipping image test", flush=True)

        results["gpu"] = "OK"
    except Exception as e:
        print(f"[GPU] ERROR: {e}", flush=True)
        results["gpu"] = f"FAILED: {e}"


def run_npu_llm(results):
    """Load LLM on NPU, run a text inference."""
    try:
        print("[NPU] Loading LLM...", flush=True)
        t0 = time.perf_counter()
        pipe = ovg.LLMPipeline(NPU_DIR, device="NPU")
        load_time = time.perf_counter() - t0
        print(f"[NPU] LLM loaded in {load_time:.1f}s", flush=True)

        gen = ovg.GenerationConfig()
        gen.max_new_tokens = 50
        gen.do_sample = False
        gen.top_k = 1

        # Warmup
        print("[NPU] Warmup...", flush=True)
        history = ovg.ChatHistory()
        history.append({"role": "user", "content": "Hi"})
        pipe.generate(history, gen)

        # Real test
        print("[NPU] Inference...", flush=True)
        history = ovg.ChatHistory()
        history.append({"role": "user", "content": "What is the capital of Norway? One sentence."})
        t0 = time.perf_counter()
        res = pipe.generate(history, gen)
        text = res if isinstance(res, str) else getattr(res, 'texts', [str(res)])[0]
        elapsed = time.perf_counter() - t0
        print(f"[NPU] Result: {text.strip()!r} ({elapsed:.2f}s)", flush=True)

        results["npu"] = "OK"
    except Exception as e:
        print(f"[NPU] ERROR: {e}", flush=True)
        results["npu"] = f"FAILED: {e}"


def main():
    print("=" * 60)
    print("  Dual Device Test: GPU (VLM) + NPU (LLM)")
    print("=" * 60)
    print()

    # Check devices
    core = ov.Core()
    print(f"  CPU: {core.get_property('CPU', 'FULL_DEVICE_NAME')}")
    print(f"  GPU: {core.get_property('GPU', 'FULL_DEVICE_NAME')}")
    print(f"  NPU: {core.get_property('NPU', 'FULL_DEVICE_NAME')}")
    print()

    # Check model directories
    if not os.path.isdir(VLM_DIR):
        print(f"ERROR: VLM model not found at {VLM_DIR}")
        sys.exit(1)
    if not os.path.isdir(NPU_DIR):
        print(f"ERROR: NPU model not found at {NPU_DIR}")
        print(f"Download it with:")
        print(f"  huggingface-cli download OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov --local-dir {NPU_DIR}")
        sys.exit(1)

    print(f"  VLM: {VLM_DIR}")
    print(f"  NPU: {NPU_DIR}")
    if TEST_IMAGE:
        print(f"  Test image: {TEST_IMAGE}")
    print()

    # --- Phase 1: Test each device alone ---
    print("--- Phase 1: Test each device alone ---")
    print()

    results = {}

    print("Testing NPU alone...")
    run_npu_llm(results)
    print()

    print("Testing GPU alone...")
    run_gpu_vlm(results)
    print()

    if results.get("npu", "").startswith("FAILED") or results.get("gpu", "").startswith("FAILED"):
        print("--- One or both devices failed. Skipping concurrent test. ---")
        print()
        print_summary(results)
        return

    # --- Phase 2: Run both simultaneously ---
    print("--- Phase 2: Run both simultaneously ---")
    print()

    results_concurrent = {}
    t0 = time.perf_counter()

    gpu_thread = threading.Thread(target=run_gpu_vlm, args=(results_concurrent,))
    npu_thread = threading.Thread(target=run_npu_llm, args=(results_concurrent,))

    gpu_thread.start()
    npu_thread.start()

    gpu_thread.join(timeout=120)
    npu_thread.join(timeout=120)

    total = time.perf_counter() - t0
    print()
    print(f"--- Concurrent test completed in {total:.1f}s ---")
    print()

    results["gpu_concurrent"] = results_concurrent.get("gpu", "TIMEOUT")
    results["npu_concurrent"] = results_concurrent.get("npu", "TIMEOUT")

    print_summary(results)


def print_summary(results):
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for key, val in results.items():
        status = "PASS" if val == "OK" else "FAIL"
        icon = "+" if val == "OK" else "!"
        print(f"  [{icon}] {key:20s} {val}")
    print()

    if all(v == "OK" for v in results.values()):
        print("  All tests passed! GPU and NPU can run simultaneously.")
    else:
        print("  Some tests failed. Check output above.")
    print()


if __name__ == "__main__":
    main()
