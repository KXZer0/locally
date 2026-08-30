# 140V book-run measurement — handoff for claude-code on the laptop

Written 2026-08-09 on the desktop (285K), for claude-code running on the
Lunar Lake laptop (Core Ultra 7 258V, Arc 140V 16 GB, 32 GB RAM). Everything
you need is in `C:\devel\delt\locally-140v-test\` plus this repo.

## Why this measurement exists

We are evaluating local models for whole-book literary analysis (secondreader:
one ~113k-token prompt per artifact) and, downstream, an offline "slush pile"
appliance. The desktop CPU measurement is done and the verdict is recorded in
TODONT.md: **decode is fine, cold prefill is not** (~45 tok/s at 10k tokens,
>90 min for the full 113k prompt — three timeout-retries piled into the
uncancellable engine and nothing ever finished).

The 140V has XMX. Prefill is compute-bound, so this GPU is the hypothesis:
**if whole-book prefill is minutes instead of hours on XMX-class hardware, the
appliance floor is an AI-PC; if not, it's dGPU/128GB-unified machines.** That
is the fork this measurement decides. The B60 24 GB arrives next week and gets
the same protocol.

## What's in the shared folder

| File | What |
|---|---|
| `Qwen3-30B-A3B-Instruct-2507-int4-ov\` | The model. Intel pre-export, integrity-verified. **Copy to a local NVMe path first** — offload streams expert weights from disk on every LRU miss; running it off a network share would measure the network. |
| `payload-testbok-facts.json` | Flow-check payload: 2-chapter mini book, 10,608 tokens, `max_tokens` 32000. Byte-identical to what secondreader sends. |
| `payload-oldgods-facts.json` | The real measurement: whole novel, 112,753 tokens, `max_tokens` **16000** (deliberate — see pool math below). |
| `testbok-2ch\` | The mini book itself, if you want to run via secondreader instead of curl. Not required. |
| `results\` | Put raw reply JSON + numbers here; the desktop scores them (strip_to_anchor + terra citation check need secondreader + keys, which live there). |

## Hard-won constraints — read before starting

1. **KV pool must hold prompt + max_tokens.** Geometry is 96 KB/token. The
   full-book payload needs (112,753 + 16,000) × 96 KB ≈ **12.4 GB** →
   `--cache-size-gb 13`. That is why its `max_tokens` is 16000, not 32000:
   13 GB pool + offloaded weights is what plausibly fits a ~16 GB GPU budget.
   A pool smaller than prompt+output can evict its own prefix mid-generation,
   or hard-fail with `Got unfinished GenerationStatus`.
2. **Never retry into a running generation.** OpenVINO cannot cancel; a dead
   client socket does not stop the sequence. Retries multiply the load until
   nothing finishes (measured on the desktop: 3 identical 113k sequences,
   permanent preemption). curl once, `--max-time 0`, wait.
3. **Offload needs the 2026.3 runtime** (`venv-2026.3`, already on this
   laptop — offload was first validated here 2026-08-06). The plain `venv`
   is 2026.1 and silently ignores nothing — it just can't do OFFLOAD_RATIO.
4. **Steady-state, not first-sentence.** The offload LRU needs ~60 tokens to
   warm. TTFT is the prefill number; compute decode tok/s from the tail.
5. `--idle-timeout 0` auto-captures the first big prompt to
   `prewarm-<port>.json` and re-prefills it at next startup. Fine in
   production, confusing in a benchmark — use `--no-prewarm`, or expect a
   long second startup.

## Protocol

```powershell
# 0. copy model to local NVMe, e.g. ~/models/
robocopy C:\devel\delt\locally-140v-test\Qwen3-30B-A3B-Instruct-2507-int4-ov `
         $HOME\models\Qwen3-30B-A3B-Instruct-2507-int4-ov /E

# 1. flow check (mini payload, small pool, ratio 30)
python locally.py --model-dir $HOME\models\Qwen3-30B-A3B-Instruct-2507-int4-ov `
    --device GPU --offload-ratio 30 --cache-size-gb 4 --port 8010 `
    --ollama-port 11435 --no-prewarm
curl -s --max-time 0 http://localhost:8010/v1/chat/completions `
    -H "Content-Type: application/json" `
    --data-binary "@C:\devel\delt\locally-140v-test\payload-testbok-facts.json" `
    -o C:\devel\delt\locally-140v-test\results\reply-testbok-140v.json
# server log prints:  -> [GPU] ~N tokens in Xs (Y tok/s, TTFT Zms)

# 2. the real one (restart server: big pool, ratio 90 first)
#    ratio 90 -> ~2.4 GB resident weights + 13 GB pool. If it loads and the
#    GPU budget shows headroom, a second pass at ratio 70/50 is a bonus —
#    decode was 5.1 tok/s @90 vs 22.1 @50 on this GPU (short-context bench).
python locally.py --model-dir $HOME\models\Qwen3-30B-A3B-Instruct-2507-int4-ov `
    --device GPU --offload-ratio 90 --cache-size-gb 13 --port 8010 `
    --ollama-port 11435 --no-prewarm
curl -s --max-time 0 http://localhost:8010/v1/chat/completions `
    -H "Content-Type: application/json" `
    --data-binary "@C:\devel\delt\locally-140v-test\payload-oldgods-facts.json" `
    -o C:\devel\delt\locally-140v-test\results\reply-oldgods-140v.json
```

If the big pool + weights won't co-load at any ratio, that is itself the
finding: record the failure mode and stop — the B60 retest covers it.

## Numbers to bring home (drop a markdown table in `results\`)

Per run: load time, offload ratio, `--cache-size-gb`, TTFT ms (= cold
prefill), decode tok/s (steady tail), total wall clock, peak GPU memory if
visible, and whether the reply is coherent prose that starts near
"# <title> — Continuity Bible". Desktop reference to beat: mini book
TTFT 234,181 ms / ~6 tok/s decode; full book TTFT **never** (>90 min).

Raw replies go back in `results\` — they get scored on the desktop with the
same terra extractor as the five-model cloud bake-off, so don't trim them.
