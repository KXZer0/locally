# NEXT STEPS (2026-08-06/07, OpenVINO 2026.3 testing session)

## 2026-08-07 HEADLINE — model-bigger-than-RAM works on CPU

Qwen3-Coder-Next int8 (74.4 GB, Dmitriy's HF upload, integrity verified
here) on the 64 GB 285K desktop, plain CPU, no flags: loads in ~75-85 s,
**steady-state 8.8-11.5 tok/s depending on OS page-cache warmth** (cold
cache after eviction → 8.8; warm after download → 11.5), resident
stabilizes at ~29-35 GB (committed 78).
Only the hot experts (10 of 512 active/token) stay in RAM; cold experts
page from SSD. The A3B access pattern makes >RAM models viable on CPU —
faster than the laptop's GPU-offload on a model HALF the size. This
supersedes the 08-06 working-set-cap caveat: real scarcity with real
access patterns behaves BETTER than the artificial cap suggested.
Full-day device matrix (same-family MoE): desktop CPU 23.7 (30B, fits) /
11.5 (74B, doesn't fit); laptop CPU 9.1 (30B); laptop GPU offload 25.3
(30B ratio 30); non-XMX iGPU: nothing.

## EVENING RESOLUTION — offload CONFIRMED on Arc 140V (laptop)

Qwen3-30B-A3B int4 runs in **2.35 GB resident** at ratio 90 (2.5 tok/s);
LFM2-8B 4.10→0.70 GB. Old IRs fuse fine on XMX — vintage never mattered,
only hardware. Shipped same evening: `--offload-ratio` flag (25541ae),
loop defenses + UI stop button (f2131dd), six models published to HF
(aweussom/: SmolLM3 ×2, LFM2-1.2B, LFM2.5-1.2B, LFM2-8B-A1B, Qwen2.5-VL-3B),
registry updated, #19 updated with measured numbers.

Ratio sweep, Qwen3-30B on 140V — STEADY-STATE (2026-08-07 correction; the
08-06 single-shot numbers were cold-LRU, 2-5× too low): 30 → 10.79 GB @
**25.3 tok/s (interactive!)**; 50 → 8.05 GB @ 22.1; 90 → 2.35 GB @ 5.1.
Reframed: offload at moderate ratios IS interactive on XMX laptops.
LFM2-8B resident on 140V: 86.8 tok/s (earlier 197/645 were a token-count
bug — model EOS'd at 4 tokens, script assumed 64).

VERIFIED 2026-08-07 ~08:46: the CB backend does NOT share the plain-
pipeline second-generate hang — locally.py --offload-ratio 30 on the 140V
served two sequential chat requests (355 tok @ 12.5 tok/s, then 242 tok @
15.9 tok/s, TTFT 8.0s → 1.9s thanks to prefix caching). The flag is
production-safe. The plain-pipeline hang remains upstream-repro-worthy
(only affects scripts using LLMPipeline without scheduler_config).
Preflight memory warning now acknowledges offload instead of crying wolf.

LATE-NIGHT ADDENDUM — big MoE on CPU (285K, 2026-08-07 00:xx):
- Qwen3-30B-A3B int4 on the 285K CPU: **23.7 tok/s, TTFT 458 ms** — fully
  interactive, 4.4× the laptop GPU offload. On a desktop with RAM ≥ model,
  plain CPU is the best big-MoE device in the house. (A3B decode only
  touches ~3B active params/token — bandwidth cost of a small model.)
- Under an 8 GB hard working-set cap (15.2 GB model): 12.3 tok/s — the A3B
  access pattern tolerates eviction well. Caveat: pagefile use 17.9 GB
  shows OpenVINO CPU repacks weights into anonymous memory (no llama.cpp-
  style file-backed mmap streaming), and with 64 GB physical RAM the
  evicted pages stayed in the standby list (soft faults) — a genuinely
  RAM-poor machine would do worse. Scripts: scratchpad cpu_pressure_bench.py.
- Recommendation matrix now: desktop w/ RAM → CPU; XMX laptop, tight
  memory → GPU + --offload-ratio; non-XMX iGPU → model must fit.

OPEN: Qwen3.5-4B vision verdict for the registry note; SmolLM3 registry
notes could mention thinking-mode + /no_think; desktop swap-raise is NOT
needed (offload can't work there — no XMX).

## OFFLOAD INVESTIGATION: CLOSED — root cause found, swap raise NOT needed

`OFFLOAD_RATIO` requires an **XMX-capable GPU**: the MoE fusion it depends on is gated
`if (device_info.supports_immad && oneDNN)` in the GPU plugin
(transformations_pipeline.cpp). The 285K's Xe-LPG iGPU has no XMX
(OPTIMIZATION_CAPABILITIES lists no GPU_HW_MATMUL — verified) → silent no-op on this
machine, regardless of export vintage, pagefile, or ratio. Proven end-to-end: fresh
2026.3-stack LFM2-8B-A1B export (tiled expert constants confirmed in the IR) loads and
runs 27 tok/s on the iGPU with byte-identical 14.91 GB device memory at ratio 0 and 90.

Consequences:
- **Do NOT bother raising swap for the 30B re-export** — it cannot offload here anyway.
- The **Arc 140V laptop HAS XMX** — that's the machine to validate offload on (fresh-stack
  export + ratio 0 vs 90 + GPU_MEMORY_STATISTICS). The LFM2-8B-A1B-int4-2026.3 export
  (~4.5 GB, `~/models`) is the ready-made test artifact to copy over.
- For #19: ask Dmitriy what GPU his laptop has before promising anything — 128 GB RAM
  suggests Meteor/Arrow Lake-H (no XMX → no offload for him either). Draft updated.
- Big-MoE loads OOM in staging on non-XMX iGPUs (11.6 GB model on 33 GB device fails);
  that's the same missing fusion, not a memory setting. Full log in TODONT.md.

## Waiting on Tommy

1. **Upload go-ahead**: 4 validated builds with model cards + LICENSEs ready in
   `~/models/{SmolLM3-3B-int4-cw, SmolLM3-3B-int8-cw, LFM2-1.2B-int4-cw, LFM2.5-1.2B-Instruct-int4-cw}`
   → `hf upload` under your namespace, then add to `models.json` (npu category).
2. **#19 comment**: draft at scratchpad `issue19-comment-draft.md` — update with the
   LFM2-8B-A1B result before posting.
3. **Revoke unattended permissions**: delete `"PowerShell"`, `"Bash"`, `"WebFetch"` from
   `.claude/settings.local.json`.
4. **Disk cleanup** (~40 GB of experiment leftovers in `~/models`, nothing deleted):
   `LFM2-1.2B-int8-cw`, `LFM2.5-1.2B-Instruct-int8-cw` (NPU garbage), `LFM2-1.2B-int8-asym`,
   `LFM2-1.2B-int8-new`, `LFM2-1.2B-int4-cw-v2` (experiments), `SmolLM3-3B` (superseded),
   `Qwen3-30B-A3B-int4-ov`, `LFM2-24B-A2B-int4-ov` (failed controls, ~27 GB);
   keep `_src-Qwen3-30B-A3B` until the export succeeds.

## Session verdicts (full detail in TODONT.md + CLAUDE.md)

- NPU: SmolLM3-3B / LFM2-1.2B / LFM2.5-1.2B all work on 2026.3 — but ONLY channel-wise
  exports (`download-model.ps1 -Weight int4-cw|int8-cw`); default int4 crashes the vpux
  compiler. LFM int8 is a trap (sym=garbage, asym=1.4 tok/s). LFM builds are NPU-only.
- EAGLE-3: works; +6% GPU, +14% CPU on Qwen3-8B. Draft export needs venv-2026.3 stack.
- OFFLOAD_RATIO: not validated on this box — every big-MoE load dies in USM Host staging
  before offload matters (even 11.6 GB on the 33 GB iGPU); old-vintage IRs additionally
  can't fuse to MOECompressed. Mechanism notes in TODONT.md.
- Qwen3.6-35B-A3B: NPU dead (shape inference), iGPU dead (staging OOM). Parked.

CUDA column (2026-08-07): Ollama qwen3-coder-next Q4 (53 GB) on RTX 5090
32 GB + CPU auto-split (58/42): **70-75 tok/s decode**, prefill ~265 tok/s
warm. 6-8x the OpenVINO-CPU route for the same model family (different
quant: Q4 vs int8 — best-route-per-stack, not controlled A/B). Untuned;
llama-server with --n-cpu-moe expert pinning would likely add more.
