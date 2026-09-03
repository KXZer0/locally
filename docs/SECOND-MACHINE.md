# Pre-flight: `locally` on the RTX 4070 desktop

Machine B is an NVIDIA box. `locally` is OpenVINO, so it cannot drive a 4070 —
there it runs `--proxy-url` against Ollama for chat, and OpenVINO **CPU** for the
utility models. `scripts/locally-proxy.ps1` is the launcher and
[ODYSSEUS.md](ODYSSEUS.md) §3 covers the setup.

This file is the honest list of what is **not yet proven** on that machine. It is
not a to-do list of features; it is the set of things that have never been run
against real hardware or real upstream software, ordered by risk.

Everything marked **verified** was verified on **machine A** (Core Ultra X7 358H,
2026-08-30). Machine A is Intel, so a machine-A pass proves the code path, never
the NVIDIA/Ollama half.

---

## 1. The proxy slot has never talked to a real Ollama — highest risk

`ProxySlot` passes 26/26 against a mock built to the OpenAI spec. A mock built
from the spec cannot fail in the ways a real server does, so this stays the
single largest unknown.

**Not verified, and not verifiable here:** Ollama is not installed on machine A.
Port 11434 on this box does answer, but it is `locally`'s *own* Ollama-compatible
shim — confirmed by its `Server: Werkzeug/3.1.8 Python/3.14.7` header, not an
Ollama build string. Pointing the proxy at it would be `locally` proxying itself,
which proves nothing.

Concrete things to check first, from reading `core/slots/proxy.py`:

- **The URL must have no `/v1` suffix.** The slot appends `/v1/models` and
  `/v1/chat/completions` to `base_url` itself (`core/slots/proxy.py:127`).
  `--proxy-url http://localhost:11434` is right; the `.../11434/v1` form silently
  becomes `/v1/v1/models` and 404s. Every doc in this repo uses the correct form —
  the risk is copying the `/v1` form from Ollama's own OpenAI-compat page.
- **`--proxy-model` must match Ollama's id exactly, tag included** (`qwen3:8b`,
  not `qwen3`). `load()` hard-fails on a mismatch and lists what upstream offers,
  so this is loud rather than silent — but it will stop startup.
- **Warmup timeout is 60 s** (`PROBE_TIMEOUT = 10`, warmup uses `× 6`). Ollama
  loads a model on first request. A large model cold-loading onto a 12 GB 4070
  with partial CPU offload can exceed that, and the failure reads as "the upstream
  is broken" rather than "it was still loading". Pre-load with
  `ollama run <model> ""` before starting `locally`, or expect one retry.
- **`repetition_penalty` is dropped, not mapped onto `frequency_penalty`.**
  Deliberate and documented in the module; noted here so a sampling difference
  against machine A is not mistaken for a bug.

## 2. The utility engine defaults to NPU, on a machine with no NPU — verified broken

This is the item most likely to bite on day one, because on machine B the
utilities are the whole reason to run `locally` at all.

**The backend is fine.** Run with `--util-engines cpu`, `/health` reports
`util.cpu.status: ready` with `read`, `background`, `detect`, `upscale`, `search`
and `rerank` all true, and OCR works:

```
POST /v1/util/read   -F file=@page.png -F engine=cpu
→ HTTP 200, X-Device: CPU, text read correctly
```

**Verified** on machine A running `locally.py --util-engines cpu`.

Two things around it were not fine, both **verified failing** in that same run,
and both are now **fixed and re-verified** on machine A (2026-08-30):

- **A request that omitted `engine` got a 503 naming the wrong device.** The
  default was the hardcoded string `"npu"` — `locally.py:2314` for `read`, and
  the same literal again at `2422`, `2539` and `2568`. The reply was *"No utility
  models loaded on the NPU. … fetch them with `python scripts/npu-probe.py
  --all`"*: advice to provision an accelerator the machine does not have. Any
  client that did not explicitly send `engine` hit this, Odysseus included.
  Now all four sites call `_default_util_engine()` (`core/slots/select.py`),
  which returns the first *loaded* engine in NPU → GPU → CPU order and only
  falls back to the literal when nothing at all is loaded. An explicit `engine`
  is still authoritative. The 503 for an engine that really is absent now names
  the ones that are up (*"The CPU has utility models loaded; pass engine=cpu."*),
  and the npu-probe hint appears only when no engine is loaded.
- **The web UI could not select CPU at all.** `templates/index.html` emitted only
  `data-util-engine="npu"` and `"gpu"`, and `setUtilEngine` rejected anything else.
  With CPU utilities loaded and ready, the Tools tab showed both engine buttons
  disabled and read *"native docs ready · OCR models not loaded"* — models loaded,
  serving, and unreachable from the UI. There is now a third button, `UTIL_ENGINE_ORDER`
  is the single engine list the module works from, and the default falls through
  NPU → GPU → CPU without moving a user's explicit choice. Its dot follows the
  existing fill grammar (solid NPU, ring GPU, hollow CPU) — no new colour.

This was the same failure `CLAUDE.md` already records for the old NPU-only
`--util-engines` default: the capability was built, documented, and unreachable.

Verified by running it, `--util-engines cpu` on port 8125: `/health` →
`util.cpu.status: ready`; `POST /v1/util/read` with **no** `engine` → HTTP 200,
`X-Device: CPU`, correct text; `engine=cpu` → 200; `engine=npu` and `engine=gpu`
→ 503 naming CPU; the rendered Tools tab → CPU button active, *"CPU ready"*.

## 3. Speaker verification has never heard two real humans

`--speaker-threshold` defaults to **0.35**, and that number is a placeholder. The
right value depends on model, microphone and room, which is why nothing in the
code claims to know it.

**Current state on machine A: no profile is enrolled** (no `speaker-profile.json`).
Gating is therefore inactive and every turn is let through — the deliberate
`abstain` behaviour, not a failure, but it means the threshold has never been
exercised against a second voice.

To tune it, on each machine separately:

1. `POST /v1/audio/enroll` with several short recordings of the owner (≥6 s total).
2. `POST /v1/audio/verify` with a recording of the owner → note the cosine.
3. `POST /v1/audio/verify` with a recording of someone else → note the cosine.
4. Put `--speaker-threshold` in the gap between them.

`speaker-profile.json` is biometric data and is gitignored. It must not be copied
between machines: it is stamped with the model that produced it, and embeddings
from two models are not comparable.

## 4. Odysseus is not installed anywhere yet

[ODYSSEUS.md](ODYSSEUS.md) was written from the upstream repository, not from a
running deployment. Points marked unconfirmed there are still unconfirmed — verify
them against the version actually pulled, and correct that doc in place.

`locally` now shows an **Odysseus** entry in the sidebar, but only when the address
answers; it stays hidden otherwise, so an absent Odysseus costs nothing. The
address is set in Settings → Odysseus (default `http://localhost:7000`).
Reachability is probed **from the browser**, not from the server, because the
browser is what follows the link — a phone that cannot reach Odysseus is not shown
an entry that would dead-end for it.

It is a link and must stay one: Odysseus sends HSTS, scopes cookies to its own
hostname, and registers a service worker as an installable PWA. Each of the three
breaks an embed independently. Full reasoning in ODYSSEUS.md.

## 5. Unknown: whether the 4070 box's CPU is Intel or AMD

This changes how the CPU utility models feel, and nothing else. OpenVINO's CPU
plugin runs on AMD, so the utilities will work either way — but it is tuned for
Intel, and the OCR/upscale latencies quoted throughout this repo were all measured
on Intel parts. Treat them as upper bounds on an AMD CPU, not as expectations.

Worth capturing on first run: `python locally.py --scan` and the startup device
banner, so that machine has a baseline of its own.

---

## What was verified for this list, and what was not

| Claim | Verified? | How |
|---|---|---|
| CPU utility slot loads and serves OCR | yes, machine A | `--util-engines cpu`; `/health`; `/v1/util/read` → `X-Device: CPU` |
| A utility request without `engine` reaches the loaded engine | fixed, machine A | request with no `engine` field → 200, `X-Device: CPU` |
| Tools tab can select CPU | fixed, machine A | rendered page: CPU button active, "CPU ready" |
| Odysseus entry hides/shows on reachability | yes, machine A | stub server on :7000, toggled off and on |
| Real Ollama behind `--proxy-url` | **no** | Ollama is not installed on machine A |
| Speaker threshold 0.35 separates two people | **no** | no profile enrolled anywhere |
| Odysseus's own behaviour | **no** | not installed |
| Anything at all on machine B | **no** | machine B has not been touched |
