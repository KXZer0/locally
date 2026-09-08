# AGENT TASKS A, B, C — results (2026-09-03)

Answers to the three tasks in `docs/REBUILD-PLAN.md`. Every number here was
measured on this machine (Core Ultra X7 358H, Arc B390 iGPU, 31.5 GB) today.

---

## TASK A — abliterated 8B for the NPU slot: **works, adopt it**

`huihui-ai/Huihui-Qwen3-8B-abliterated-v2` downloaded (15.3 GB, 4/4 shards, no
pickle, no remote code, so no `-Trust`) and exported channel-wise:

```bash
optimum-cli export openvino -m <local-source-dir> --task text-generation-with-past \
  --weight-format int4 --sym --group-size -1 --ratio 1.0 \
  C:\Users\zeror\models\Qwen3-8B-abliterated-int4-cw-ov
```

`--task` is required and the plan's command omits it: optimum refuses to infer a
task from a **local directory**, and the export dies in seconds with a wall of
valid task names. Exporting from the HF id infers it; exporting from the copy you
already downloaded does not.

`--scan` confirms **INT4 (symmetric, channel-wise)**, 4.4 GB, integrity complete —
the one quantization the vpux compiler accepts.

Paired against the incumbent, same prompts, same greedy config, minutes apart
(`scripts/npu-ab.py`):

| | Qwen3-8B-int4-cw | **abliterated** |
|---|---|---|
| load, warm compile cache | 5.9 s | **5.0 s** |
| load, cold (first compile) | — | 164.2 s |
| decode | 20.1 tok/s | **21.6 – 22.0 tok/s** |
| 20-prompt factual | 20/20 | **20/20** |

**Verdict: a drop-in.** Same size, same quality on this set, marginally faster,
compiles and runs on the NPU. The factual set is blunt by design — 20/20 both
sides means it found no difference, not that either model is perfect — but a
blunt scorer applied identically to both is a fair *difference*.

Two things that cost time and are worth knowing:

* **Pass `CACHE_DIR`.** Without it every load is a cold vpux compile of the whole
  graph. The first measurement said 168 s and meant nothing; locally passes the
  cache, so 5 s is the honest figure.
* **A crashed process keeps the NPU.** The first run died with
  `ZE_RESULT_ERROR_DEVICE_LOST`. The device was not broken — an 8.3 GB python
  process from an earlier killed server still held it. Kill the holder and
  `Get-PnpDevice` reports `Intel(R) NPU: OK` again. Do not reach for a driver
  reset before checking for an orphan.

---

## TASK B — Qwen3.5-9B on the GPU: **go/no-go passed, with one caveat**

`OpenVINO/Qwen3.5-9B-int8-ov`, 8.8 GB, loaded with `--kv-precision u8
--offload-ratio 0`.

**The go/no-go the plan specifies — resident bytes — passes.** `/v1/memory`:

```
GPU  Qwen3.5-9B  ready  9020 MB
gpu: driver_ceiling_mb 25941, reserve_mb 3072, usable_mb 8412
state: resident — "Models are fully resident."
```

9,020 MB against 8.8 GB on disk. **int8 did not inflate to fp16** (that would have
been ~19 GB), which was the specific failure the task was written to catch.

Throughput, over `/v1/chat/completions` with streaming — the path a client
actually takes, not `pipe.generate` (`scripts/ttft-probe.py`):

| prompt | TTFT | decode |
|---|---|---|
| ~0.5k tokens | 0.65 s | 11.9 tok/s |
| ~8k tokens (6,000 words) | **0.72 s** | 11.8 tok/s |
| ~64k tokens (48,000 words) | **48.6 s** | 7.7 tok/s |

**The caveat is prefill, not memory.** From 8k to 64k the prompt grows 8× and
prefill time grows **67×** (0.72 s → 48.6 s); the effective rate falls from 8,314
to 988 words/s. Decode degrades far more gently, 11.8 → 7.7 tok/s. So this model
is comfortable for chat and for agent turns up to roughly 8–16k, and a 64k turn
costs the better part of a minute before the first token — on a client with an
idle watchdog that is the thing to watch, not the 9 GB of weights.

It is a **VLM** (`Qwen3_5ForConditionalGeneration`), so it takes the plain
VLMPipeline and gets no prefix cache and no `--context-tokens` pool. A repeated
agent prompt is therefore re-prefilled every turn, which the 64k figure above
prices exactly.

### It could not stream at all until this was fixed

`stream_llm` called `pipe.generate(history, gen, streamer_callback)`
positionally. LLMPipeline accepts that; **VLMPipeline does not** — its
`ChatHistory` form is `(history, **kwargs)` with no positional variant — so every
streamed text turn on a VLM slot ended with 0 tokens and an "incompatible
function arguments" error. The non-streaming path takes a different overload and
answered perfectly, which is exactly why nobody noticed: a plain POST worked, and
streaming is what the web UI and every agent client use. Fixed by passing them by
keyword (`59ee312`).

### The KV arithmetic was wrong by 4× on this model, and that blocked it

`_kv_bytes_per_token` multiplied by `num_hidden_layers`. Qwen3.5-9B is **24
`linear_attention` + 8 `full_attention`**, and a linear layer keeps a fixed
recurrent state, not a per-token cache — so locally reported 128 KB/token against
a real 32 (16 at u8). `--context-tokens 100000` asked for a 13 GB pool instead of
4, and `_device_fits` ruled the model out of a device it fits in with room to
spare. Fixed in `43e6198`; `--scan` now prints the split.

`sliding_attention` is the trap in that fix and is handled separately: gemma-4 has
25 such layers with a 1024 window and they **do** hold a cache, just a capped one.
Treating them like linear layers under-counts by a constant, and under-counting is
the direction that hard-fails a generation (#21) — measured on gemma-4-26b at a 2k
context, 0.27 GB is right and the rate alone says 0.08.

---

## TASK C — Podman sandbox: **proven, after a bug that made it a no-op**

`scripts/verify-sandbox.py` runs each escape through the real `execute_python`
path and asserts it fails. Measured on this machine with
`docker.io/library/python:3.12-alpine`:

| case | podman | subprocess |
|---|---|---|
| baseline `print(sum(range(10)))` | runs, `45` | runs, `45` |
| network egress | **blocked** — connect hangs to timeout | blocked (import refused) |
| write outside the mount | **blocked** — `[Errno 30] Read-only file system: /etc/locally-escape` | blocked (open refused) |
| 2 GB allocation | **killed, exit 137** (SIGKILL, OOM at `--memory=512m`) | **ESCAPED** — `ALLOCATED 2147483648` |
| fork bomb | **contained** — `BlockingIOError` at `--pids-limit=64` | blocked (`os.fork` refused) |
| `ctypes.CDLL(None)` | runs, and is **useless**: `getpid()` returns **1** | blocked (import refused) |

The two paths fail differently and that is the interesting part. The subprocess
audit hook refuses things **by name** — it has no memory cap at all, so the one
escape it lets through is the runaway allocation, which is also the likeliest
accident from a small model. The container caps memory for real and does not
bother forbidding ctypes: `CDLL(None)` loads, and `getpid()` answers **1**,
because it is looking at its own PID namespace and there is nothing of the
host's to reach.

**It did not work at all until `59ee312`+ fixed this**: the status-file path
baked into the generated `run.py` was built with `os.path.join` on a Windows
host, producing `/workspace\.locally-python-xxx\status.json` — one long
filename, opened in a read-only root. **Every calculation failed**, including
the arithmetic baseline. Five of the six cases still reported "contained",
because code that never runs never escapes; only the baseline caught it. That
is the argument for putting a positive control in a security test suite.

**Lifecycle, measured.** The machine is started on demand and stopped again:
`machine_running` is False before and False after, and `sandbox_status()` costs
**1.9 ms** with the VM down, so a `/health` read never pays the VM probe.

**The cost is real and is the argument against `auto` for interactive use.** A
warm container run is **~7 s**; the whole six-case suite including machine start
was 50 s. The subprocess path answers the same calculations in **75-90 ms**,
about 80x faster. For a tool whose purpose is a quick exact calculation inside a
chat turn, seven seconds is the difference between a tool and an interruption —
so `--python-sandbox` earning its three settings is the point, and which default
is right depends on whether the code being run is trusted or merely likely.
## gemma-4-26b at 100k: the wall is an allocation, not the budget

Asked directly: gemma-4-26b is already good, so why not run *it* at 100k? The
memory arithmetic says yes and the GPU plugin says no, and the reason is worth
writing down because it is not the reason anyone expects.

**The budget is fine.** Weights 14.3 GB, and with the corrected KV geometry
(5 full-attention + 25 sliding-window layers, 1024 window) a 100k context at
`--kv-precision u8` needs **2.01 GB** of cache — 16.3 GB total against a 20.7 GB
budget on this machine. `--offload-ratio auto` agreed and picked **0**: it loaded
fully resident at 14,647 MB with no expert streaming at all.

**What fails is a single allocation.** Both large prompts died with:

```
[GPU] Exceeded max size of memory object allocation:
  requested 32,664,649,728 bytes  (~30.4 GB, 32k prompt)
  requested 288,364,068,864 bytes (~268.6 GB, 100k prompt)
  max alloc size supported by device is 27,201,245,184 bytes (25.3 GB)
```

268 GB is not a cache — the KV for that context is 2 GB. It is the **attention
score buffer, and it is O(n²)**: the prompt grew 3.1x from 32k to 100k and the
request grew 8.8x, which is n² to within measurement error. This path materialises
the full attention matrix instead of chunking it.

**So the ceiling is per-allocation, not capacity, and no amount of free RAM moves
it.** The device refuses any single object over 25.3 GB. The error names
`ov::intel_gpu::hint::enable_large_allocations`, which is worth trying for the
32k case (30.4 GB is within reach of a raised limit); 268 GB is not reachable by
any setting.

**And this is exactly what hybrid attention is for.** Qwen3.5-9B *did* complete
64k on the same machine, in 48.6 s, because 24 of its 32 layers are
`linear_attention` and build no n² matrix at all — only 8 full-attention layers
pay the quadratic cost. gemma-4-26b has 5 full-attention layers with a **global**
span and 25 windowed at 1024: the windowed ones are cheap, but the 5 global ones
each materialise the full square. Fewer quadratic layers is why the smaller model
reaches the longer context.

**Practical answer: gemma-4-26b is a strong model with a short usable context on
this GPU path, and Qwen3.5-9B is a weaker model that actually reaches 64k.** If
the 100k agent is the goal, the model is not the variable to tune — the attention
implementation is.

## Also fixed along the way

**`.gitignore` was swallowing source.** `models/` was unanchored, so it matched
`core/models/` — this project's own package — and `git add -A` silently skipped
two new modules there. Everything passed in the worktree where the files exist and
failed the moment the commit was checked out. All four model patterns are now
anchored with a leading slash (`17b4297`).
