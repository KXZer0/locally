# Running Odysseus against `locally`

[Odysseus](https://github.com/odysseus-dev/odysseus) (AGPL-3.0) is the assistant
layer: chat, agents, reminders, calendar, email, deep research. It has no
inference engine of its own — it talks to one over HTTP.

`locally` is the inference engine: Intel NPU/iGPU under OpenVINO, plus the
utility studio (OCR, layout, upscale, matting, detect, embed/rerank search),
voice, and the memory HUD. It speaks three protocols on the wire — an
OpenAI-compatible API, an Ollama-compatible API, and the Anthropic Messages API.

The split is clean and worth stating once: **Odysseus owns the assistant,
`locally` owns the silicon.** Neither needs to know how the other works, and
each machine runs its own pair.

---

## 1. Architecture

One `locally` per machine. One Odysseus per machine. Odysseus talks to the
`locally` on the same box, over the OpenAI-compatible base URL. Nothing crosses
the network except your phone, and that goes over Tailscale (§6).

```
Machine A — Intel laptop (Core Ultra X7 358H)
  ┌──────────────────────────────────────────────────────────┐
  │  Odysseus (Podman Compose)      127.0.0.1:7000           │
  │    ├── chromadb  127.0.0.1:8100                          │
  │    ├── searxng   127.0.0.1:8080   ← port clash, see §5   │
  │    └── ntfy      127.0.0.1:8091                          │
  │            │                                             │
  │            │  http://172.17.96.1:8000/v1  ← §4, NOT      │
  │            │  localhost and NOT host.docker.internal     │
  │            ▼                                             │
  │  locally (native, venv)          0.0.0.0:8000           │
  │    ├── chat slot   → NPU   (Qwen3-8B int4-cw)           │
  │    └── util slots  → NPU + GPU (OCR, upscale, search)   │
  └──────────────────────────────────────────────────────────┘

Machine B — desktop, RTX 4070
  ┌──────────────────────────────────────────────────────────┐
  │  Odysseus (Docker Compose)      127.0.0.1:7000           │
  │            │  http://host.docker.internal:8000/v1        │
  │            ▼                                             │
  │  locally --proxy-url http://localhost:11434  0.0.0.0:8000│
  │            │                                             │
  │            ▼                                             │
  │  Ollama    127.0.0.1:11434  → CUDA → RTX 4070            │
  └──────────────────────────────────────────────────────────┘
```

### Why Odysseus is a LINK in the sidebar, never an iframe

`locally`'s sidebar can carry a link to Odysseus. It must not carry an iframe,
and this is not a matter of taste — three separate mechanisms in Odysseus each
break an embed independently:

1. **It refuses outright.** `GET /login` returns **`X-Frame-Options: DENY`**
   and a CSP containing **`frame-ancestors 'none'`**. Either alone ends the
   discussion; a browser will not paint the frame, and there is no header we
   can send from our side that overrides the framed site's own refusal.

   > Measured 2026-08-30 against a real stack (`podman compose up`, Odysseus +
   > chromadb + searxng + ntfy). An earlier version of this document claimed the
   > mechanism was **HSTS**. That was wrong: `Strict-Transport-Security` is
   > **not sent** on this deployment. The conclusion was right and the reason
   > was not, which is worth recording — a correct rule resting on a false
   > premise is one upstream change away from being quietly discarded.
   > Note `/` alone is a bare `302 -> /login` with no security headers at all,
   > so anyone re-checking this must look at `/login`, not the root.

2. **Its own hostname.** Odysseus sets `ALLOWED_ORIGINS` (the compose file
   defaults it to `http://localhost,http://127.0.0.1`) and issues cookies
   scoped to the host it thinks it is. Framed under `locally`'s origin, the
   login cookie is third-party — modern browsers partition or drop it, so you
   are logged out on every navigation inside the frame.

3. **A service worker.** Odysseus registers one and is an installable PWA. A
   service worker's scope is tied to its own origin and it expects to control a
   top-level page; inside a frame it either fails to register or serves a
   confusingly cached shell.

Fighting all three buys you a worse version of a `target="_blank"` link. Take the
link. On a phone, the PWA install is a *better* result than an embed would be.

> **Status (2026-08-30):** the sidebar link now exists. The address is a UI
> setting, not a server flag — Settings -> Odysseus, stored under
> `locally-odysseus-url`, default `http://localhost:7000`. The entry appears
> only while that address answers, and reachability is probed from the BROWSER
> rather than from the server: the browser is what follows the link, so a phone
> that cannot reach Odysseus is not shown an entry that dead-ends for it.
> See `static/js/ui/odysseus.js`.
>
> **locally can also start it** (`core/odysseus.py`, 2026-08-30). It looks for a
> checkout in `$ODYSSEUS_DIR`, then a sibling `odysseus/`, then `~/odysseus`, and
> runs `docker compose up -d` on a background thread at startup. Turn it on with
> the "Start it with locally" toggle in Settings (persisted in the gitignored
> `odysseus-autostart.json`) or with `--odysseus-autostart`; the flag wins over the
> toggle. `--odysseus-port` and `--odysseus-dir` override the defaults.
> Endpoints: `GET /v1/odysseus`, `POST /v1/odysseus/{start,stop,autostart}` — the
> POSTs are localhost-only. Stopping uses `docker compose stop`, never `down`.
>
> **Unverified:** Docker is not installed on the Intel laptop, so the real
> `compose up` / `stop` paths, the port poll after a successful start, and adoption
> of a live stack have never been exercised. What has been verified is every
> failure path that does not need Docker: no checkout (503 in 0.74 s with a
> sentence naming the three search locations), a checkout with no compose file,
> `docker` absent, and the autostart preference surviving a restart.

---

## 2. Machine A — Intel laptop, native `locally` on the NPU

`locally` runs natively here. There is no proxy and no Ollama.

```powershell
# From the repo root, with the venv active.
.\scripts\locally-launch.ps1
```

That launcher already does the right thing for this setup: NPU chat model,
Whisper on GPU, Kokoro TTS and Silero VAD on CPU, SearXNG on demand. For an
always-on assistant you want two changes from its defaults:

```powershell
.\scripts\locally-launch.ps1 -IdleTimeout 0 -SearxPort 8081
```

- **`-IdleTimeout 0`** → passes `--idle-timeout 0`. An assistant is queried in
  bursts all day; a 900-second idle unload means every burst after a quiet
  stretch pays a cold model load inside the user's first message. `0` also
  auto-enables `--prewarm`, so the prefix cache survives a restart.
- **`-SearxPort 8081`** → moves `locally`'s SearXNG off 8080, which Odysseus's
  bundled SearXNG already publishes on. See §5.

Verify the slot came up on the NPU:

```powershell
curl.exe -s http://127.0.0.1:8000/health | ConvertFrom-Json |
    Select-Object -ExpandProperty slots
curl.exe -s http://127.0.0.1:8000/v1/models
```

### Utilities

`--util-engines` defaults to `auto`, which loads a utility slot on **every Intel
engine detected** — NPU and GPU both, here. Nothing to configure. OCR loads
eagerly (~160 MB); everything else compiles on first use and idle-unloads.

---

## 3. Machine B — RTX 4070, `locally` in proxy mode

`locally` is OpenVINO, and OpenVINO's GPU plugin is Intel-only —
`detect_devices()` explicitly filters non-Intel GPUs out, because the plugin
enumerates any OpenCL device but its kernels only run on Intel. So on this box
`locally` cannot touch the 4070, and there is no point pretending otherwise.

Instead, **Ollama drives the 4070 and `locally` proxies to it.** You still get
one program, one URL, and one UI on both machines; the slot behind it just
happens to be someone else's.

```powershell
# Start Ollama first (its own service or `ollama serve`), then:
.\scripts\locally-proxy.ps1
```

That script (shipped alongside this doc) checks Ollama is reachable, picks the
model, and starts:

```
python locally.py `
    --proxy-url   http://localhost:11434 `
    --proxy-model qwen3-coder:30b `
    --port        8000 `
    --ollama-port 0 `
    --idle-timeout 0
```

The three proxy flags, exactly as they exist in `locally.py`:

| Flag | Meaning |
|---|---|
| `--proxy-url URL` | Serve the primary slot from an OpenAI-compatible server instead of a local model. Replaces `--model-dir`/`--device` for chat. `--model-dir` is not required in this mode. |
| `--proxy-model ID` | Model id to request upstream. **Optional** — omitted, the slot takes the first model the upstream advertises, which is what a single-model Ollama host means by "the model". |
| `--proxy-key TOKEN` | Bearer token for the upstream, if it needs one. Ollama does not. |

Two behaviours worth knowing, because they change how failures read:

- **Loading a proxy slot is a real probe, not a config read.** `ProxySlot.load()`
  hits the upstream's `/v1/models`; if you named a `--proxy-model` the upstream
  does not serve, it refuses at startup and *lists what the upstream does offer*.
  Then `warmup()` sends one real token through `/v1/chat/completions`, which
  catches the failures a model listing cannot — a model listed but not loadable,
  a wrong key on the completions route.
- **`--ollama-port 0` is not optional here.** `locally` runs its own
  Ollama-compatible shim on 11434 by default. On this machine the real Ollama
  owns that port. `0` disables the shim; without it `locally` fails to bind and
  warns, and you have two things claiming to be Ollama.

### Tool calling still works on Machine B

`_tools_supported()` returns true for `GPU`, `CPU`, and `REMOTE`. A proxy slot is
`REMOTE`, deliberately: every reason to restrict tools is about hardware *this*
process owns, and a proxy owns none of it. A 4070 running a 30B coder drives
agent loops fine, and the relaying machine having no Intel GPU is irrelevant to
that.

### Utilities on Machine B: yes, on the CPU

`--util-engines` accepts **`npu`, `gpu`, `cpu`, a comma-separated list, or
`auto`**. Under `auto`, CPU is a *fallback* rather than an addition: it is taken
only when no Intel accelerator is detected — which is exactly Machine B. A CPU
slot alongside an NPU would spend real memory duplicating models that already
have a faster home, so it is not created there.

So on Machine B you need no flag at all. `auto` finds no NPU and no GPU, and
loads the utility models on the CPU. Measured on a Core Ultra X7 358H:

| Task | On CPU |
|---|---|
| OCR (`/v1/util/read`) | detect 164 ms · layout 280 ms · recognise 34 ms |
| Upscale (`/v1/util/upscale`) | true **4.00×** (128×96 → 512×384), 3.8 s |

Everything except `generate` is available — and image generation is disabled on
every engine, not just this one.

One deliberate difference: the CPU's `upscale` tier is the **GPU's** Swin2SR,
not the NPU's OMZ model. What rules Swin2SR out on the NPU is the vpux compiler
rejecting it, which says nothing about x86 — so the CPU gets the better model
(true 4×, no input ceiling) even though it runs slower than the NPU's fixed 3×
capped at 640×360.

Expect CPU utilities to be slower than the NPU's, and to compete with whatever
else that machine is doing. They are not slower in a way that matters for
reading a document or cleaning up an image.

---

## 4. Pointing Odysseus at `locally`

### Where the setting lives

Model providers are configured **in Odysseus's Settings UI**, not primarily in
`.env`. Odysseus's own setup guide is explicit: open `http://localhost:7000`, log
in with the generated admin password, and configure the rest inside **Settings**.
The compose override in `scripts/odysseus-compose.override.yml` pre-seeds the env
vars so the Settings page starts from the right value, but Settings is the
authoritative place.

### The URL

On Machine A — Podman on Windows, which is the setup here:

```
http://172.17.96.1:8000/v1
```

Under **Docker Desktop** it is `http://host.docker.internal:8000/v1` instead.
Four things about that string, each of which is a way people get it wrong:

1. **Not `localhost`.** Inside a container, `localhost` is the container. This
   is the single most common failure and it was the last thing standing in the
   way here — measured 2026-09-08 from inside `odysseus_odysseus_1`:
   `localhost:8000` → HTTP 000, `172.17.96.1:8000` → HTTP 200.
2. **Not `host.docker.internal` under Podman.** Docker Desktop special-cases
   that name; Podman does not, and under Podman it resolves to the WSL VM's own
   gateway rather than the Windows host. Use the numeric address.
3. **Port 8000** is `locally`'s OpenAI API (`--port`, default 8000). Not 11434,
   unless you are deliberately using the Ollama shim (below).
4. **The `/v1` suffix is required.** `locally` serves `/v1/models` and
   `/v1/chat/completions`, so the base is `.../v1`.

`locally` binds `0.0.0.0` by default (`--host`, `core/cli.py:58`), which is *why*
a container can reach it at all.

That bind is not, by itself, a decision to serve the network. Since 2026-09-09
`--allow-from` (default `auto`) answers loopback and the private subnets of
virtual/container adapters and refuses everything else, so the wide bind the
container path needs does not also hand the model to the Wi-Fi. It discovers
the subnet live — on the development box, `172.17.96.0/20`, the same figure
written by hand into the rule below — because WSL assigns it per machine and
changes it, which is exactly why that rule could never be copied to a second
box. Nothing here needs configuring for Odysseus; see `core/netguard.py`. A 127.0.0.1-bound server is invisible from a
container no matter how the firewall is set — which is exactly what
`chat.ps1 --start` gives you, since `core/terminal.py:270` pins the host. Use
`api.ps1` or `scripts/locally-launch.ps1` for anything Odysseus will talk to.

### Setup, start to finish

Verified end to end on Machine A, 2026-09-08.

**1. Start `locally` so it survives and stays advertised.**

```powershell
pwsh -File scripts/locally-launch.ps1 -IdleTimeout 0
```

`-IdleTimeout 0` is no longer required for Odysseus to see the model — an
idle-unloaded slot is advertised now — but it keeps the prefix cache warm, which
is the real win for a client re-sending a fixed system prompt every turn.

**2. Open the firewall to the WSL subnet only, once, elevated.**

This is about *reachability*, not safety: without an inbound allow the packets
never arrive, and `--allow-from` cannot answer a request it never sees. The two
are complementary, and the split matters — a firewall rule that ends up too
broad (Windows writes a `python.exe` Any/Any rule from an ordinary permission
prompt, and it outranks this one) is no longer a hole, because the app still
refuses the source.

```powershell
New-NetFirewallRule -DisplayName 'locally API (WSL/Podman only)' -Direction Inbound `
  -Action Allow -Protocol TCP -LocalPort 8000 -RemoteAddress 172.17.96.0/20 -Profile Any
```

Then check nothing is *blocking* it, because block beats allow and a cancelled
Windows prompt leaves a block rule behind:

```powershell
$target = (Get-Process -Id (Get-NetTCPConnection -State Listen -LocalPort 8000).OwningProcess).Path
Get-NetFirewallRule -Direction Inbound -Action Block |
  Where-Object { ($_ | Get-NetFirewallApplicationFilter -EA SilentlyContinue).Program -eq $target }
```

Anything that returns must go. See §7 for why `$target` is read that way and not
from the command line.

**3. Point Odysseus at the gateway address.** Settings → Added Models → the
endpoint's URL:

```
http://172.17.96.1:8000/v1
```

Press **Probe**. It should report 1/1 and the model list should fill.

**4. Check the whole path in one command.**

```powershell
pwsh -File scripts/diagnose-odysseus-link.ps1
```

Green all the way down looks like this:

```
  WSL gateway                      172.17.96.1
  listening on :8000               0.0.0.0 (all interfaces)
  host via 127.0.0.1               HTTP 200
  host via WSL gateway             HTTP 200
  podman VM -> host                HTTP 200 in 0.54s
  Odysseus container -> host       HTTP 200 in 0.50s
  models advertised                Qwen3-8B-abliterated-int4-cw@NPU, gemma-4-26b-a4b-it, Ornith-1.5-9B-abliterated
  block rule on serving exe        none
  active allow rule for :8000      locally API (WSL/Podman only)
```

**5. Pick a model in Odysseus and type.** Every loadable model on disk is
advertised, and naming one that is not resident loads it — measured 19.1 s for a
cold NPU swap plus the answer, from inside the container. See "Changing models
from Odysseus" below.

### The model id must match what `/v1/models` advertises

`locally` advertises models as **`<model_name>@<DEVICE>`**:

```console
$ curl -s http://127.0.0.1:8000/v1/models
{"object":"list","data":[
  {"id":"Qwen3-8B-int4-cw-ov@NPU","object":"model","owned_by":"local-npu"}
]}
```

The `model_name` is the **model directory's name** — the directory name is
authoritative throughout `locally`; renaming the folder renames the model. On a
proxy slot the device is `REMOTE`, so you get e.g. `qwen3-coder:30b@REMOTE`.

`_route_request()` matches a requested model against **either** `model@DEVICE`
**or** the bare `model_name`. Both work. A name that matches something on disk
but is not loaded is loaded on demand — see "Changing models from Odysseus"
below. Anything else falls through to default routing rather than erroring.

Note the two protocols disagree on purpose:

| Endpoint | Advertised id |
|---|---|
| `GET /v1/models` (port 8000) | `Qwen3-8B-int4-cw-ov@NPU` |
| `GET /api/tags` (port 11434) | `Qwen3-8B-int4-cw-ov` |

The Ollama shim omits the device suffix because Ollama clients treat the tag as a
name, not an address.

### Changing models from Odysseus

`/v1/models` advertises the resident model **and every loadable model on disk**,
so a client's model picker has something to pick from. The two are told apart by
their id and their `owned_by`:

```console
$ curl -s http://127.0.0.1:8000/v1/models
{"object":"list","data":[
  {"id":"Qwen3-8B-abliterated-int4-cw@NPU","owned_by":"local-npu"},
  {"id":"gemma-4-26b-a4b-it","owned_by":"local-available"}
]}
```

The resident one carries `@DEVICE`. An unloaded one is named alone, deliberately:
placement is decided at load time by `_choose_device` reading the IR's `rt_info`,
so promising a device here would sometimes promise one the model cannot use — the
NPU has no vision path, and group-quantized int4 crashes its compiler.

**Selecting one and sending a message loads it.** `_load_on_demand`
(`core/slots/route.py`) matches the requested id against what is on disk and
swaps the slot before serving, the same bargain Ollama strikes. Nothing else is
needed on the Odysseus side: pick the model, type, wait once.

The cost is real and worth knowing before you switch mid-conversation:

- The swap is **synchronous**, so the turn that triggers it pays the whole load —
  9.3 s for Qwen3-8B on the NPU against a warm compile cache, 65 s cold, and
  longer for a 14 GB VLM onto the GPU.
- It is **one model at a time**, so the model you were using is unloaded. That is
  not a limitation of this path; it is how the server is deliberately run (the
  NPU allocates from the same RAM as the GPU, so a second resident model costs
  real memory for a model you cannot talk to concurrently anyway).

An id that matches nothing on disk still falls through to the resident model
rather than erroring, because clients send ids nobody configured — an
unconfigured default like `gpt-4` would otherwise break turns that work today.
That silence is the trap §7 describes; on-demand loading removes the half of it
that mattered, which was asking for a model you actually have and being answered
by a different one under the requested model's name.

### Alternative: the Ollama shim

Odysseus's Ollama path is its best-trodden one, and `locally`'s shim implements
`/api/tags`, `/api/show`, `/api/chat`, `/api/generate`, `/api/version` and
`/v1/chat/completions`. On Machine A (where nothing else wants 11434) you can
point Odysseus at `http://host.docker.internal:11434/v1` and it will look like
plain Ollama.

One caveat that matters for the NPU: `/api/show` advertises the `tools`
capability only for `_tools_supported` slots — GPU, CPU, REMOTE. An **NPU slot
advertises as completion-only**, on purpose: an advert is answered before any
tool set exists, so claiming `tools` would invite a client to pick the NPU model
for agent mode and then send it thirty schemas. Clients that simply *send*
`tools` without consulting the advert — **Odysseus does** — get them honored,
subject to §5. Use the `/v1` endpoint on port 8000 if you want the honest
advert.

---

## 5. The NPU tool-calling budget

This is the constraint Odysseus users on Machine A will actually hit, so here it
is with the numbers rather than as folklore.

The NPU has a hard prompt cap: `NPU_MAX_PROMPT_LEN = 8192` tokens. That is the
vpux compiler's ceiling, not a memory limit — 9216, 10240 and 12288 all fail
graph legalisation. Every tool schema a client sends is rendered into the system
prompt and spends part of that 8192.

Measured on Qwen3-8B-int4-cw's own tokenizer
(`scripts/measure-tool-budget.py` reproduces it):

| Tool set | Rendered cost | Share of the 8192 window | Result |
|---|---|---|---|
| Assistant set — calendar, tasks, notes, search, fetch, notify — **8 tools** | **735 tokens** | **9.0%** | accepted |
| Trimmed to 5 tools | ~515 tokens | ~6.3% | accepted |
| Coding agent, **30 tools** | **4,553 tokens** | **55.6%** | **refused** |

The gate is `_tool_capable(slot, tools)` = `_tools_supported(slot)` (device) OR
`_npu_tools_affordable(slot, tools)` (this request's rendered schema block
≤ `NPU_TOOL_BUDGET`, **1200 tokens**).

**So: Odysseus's assistant tools work on the NPU. A large tool catalogue will
not.** An eight-tool assistant costs under a tenth of the window and leaves 40%
headroom under the budget. A thirty-tool coding agent costs more than half the
window before you have typed anything, and is refused outright.

Two details, both measured on the 358H:

- The budget measures the **rendered block**, not the tool count. Two schemas
  with the same name can differ tenfold in size; it is bytes in the prompt that
  the cap is about. If the tokenizer is unavailable it estimates at chars/3.5 —
  biased high, so an unmeasurable prompt errs toward refusing.
- It is a ceiling on the **schema block only**. The conversation still has to fit
  under `MAX_PROMPT_LEN` separately. A tool set that fits does not license an
  unbounded chat history.

### If you hit the refusal

The turn is answered as **plain chat** — you get an answer, not an error — with a
note that names the number:

```
tools ignored: 30 schemas render to 4553 tokens, over the NPU's 1200-token
budget. Send fewer tools, or use --agent-tools to trim them here.
```

Three fixes, in order of preference:

1. **Turn off tools you do not need in Odysseus.** Fewer schemas is the honest
   fix and costs nothing.
2. **`--agent-tools NAMES`** trims the client's tool list server-side, before the
   prompt is built. Comma-separated names to *keep*; the rest are dropped. This is
   the only option for a client with no such setting.
3. **Move the slot to the GPU** (or to Machine B). The budget is an NPU
   constraint; GPU, CPU, and REMOTE slots have no schema ceiling.

Do not raise `--npu-prompt-len` past 8192 hoping to buy room — it is clamped to
`min(8192, max(1024, N))` because 8192 is the compiler's ceiling.

### Two things `locally` does to make NPU tool turns actually work

Both were measured on 2026-08-29 and both are automatic — listed here because
they explain behaviour you will see:

- **Reasoning is suppressed on NPU tool turns.** Asked "Remind me to call the
  dentist tomorrow" with 8 tools, Qwen3-8B spent its entire 400-token budget
  inside `<think>` and emitted no tool call at all (272 tokens, 23.2 s).
  Appending the `/no_think` control token: **23.2 s → 4.2 s and a correct
  `create_task`.** Scoped to the NPU — on the GPU, reasoning before a tool call
  is affordable.
- **The tool prompt carries today's date** (~15 tokens, all devices). Without a
  clock the model called `get_calendar` with its training cutoff (`2023-10-11`)
  and then told the user the results looked "way in the future".

Full assistant turn on the NPU — tool call, `tool_result`, answer:
**4.0 s + 4.7 s**, correct events, correct dates.

### SearXNG port clash

Odysseus's compose publishes its bundled SearXNG on `127.0.0.1:8080:8080`.
`locally`'s own `--search-url` / `--searxng-root` setup defaults to 8080 too
(`scripts/locally-launch.ps1 -SearxPort`). On a machine running both, move one.
Moving `locally`'s is easier — `-SearxPort 8081` — since Odysseus's is wired into
its compose network. Two SearXNG instances is not a problem; two on one port is.

---

## 6. Phone and tablet access — Tailscale, not port-forwarding

Odysseus binds **127.0.0.1 by default** (`APP_BIND=127.0.0.1`, and the compose
port mapping is `${APP_BIND:-127.0.0.1}:${APP_PORT:-7000}:7000`). ChromaDB, ntfy
and SearXNG are all bound the same way. That is a deliberate default and it is
the right one: this stack has your calendar, your email, and your notes in it.

**Do not change `APP_BIND` to `0.0.0.0` to reach it from your phone.** On any
network you do not fully control — a café, a hotel, an office, a flat with
guests — that publishes the whole assistant to everyone on the subnet. Port
forwarding on the router is worse: it publishes it to the internet.

Use Tailscale instead. It gives every device a private address, the traffic is
encrypted end to end, and nothing is exposed to the local subnet or the internet.

1. Install Tailscale on Machine A, Machine B, and the phone. Same tailnet.
2. `tailscale up` on each; note the machine names (`machine-a.tailXXXX.ts.net`).
3. On the machine running Odysseus, enable the Tailscale serve proxy so the
   127.0.0.1 binding stays intact:

   ```powershell
   tailscale serve --bg 7000
   ```

   This proxies your tailnet address to `127.0.0.1:7000`. Odysseus keeps binding
   loopback; Tailscale is the only thing that can reach it.
4. On the phone, open `https://machine-a.tailXXXX.ts.net/`. Tailscale terminates
   HTTPS with a real certificate, which is also what makes the **PWA install**
   work — service workers require a secure context, and this is the clean way to
   get one on a private network.

You may need to add the tailnet hostname to `ALLOWED_ORIGINS` in Odysseus's env.
Verify against your Odysseus version; the variable exists but its exact parsing
was not confirmed for this document.

**`locally` itself needs the same care and gets none from its own code.** It
binds `0.0.0.0` on ports 8000 and 11434 with no flag to restrict it. On a laptop
that leaves the house, block those ports in Windows Firewall for public networks
and let Tailscale be the only path in. `locally` has no authentication.

---

## 7. Troubleshooting

### `locally` not reachable from inside the Odysseus container

**This is the most common failure by a wide margin. Check it first.**

One command answers it, and it is read-only — it changes no rules and starts
nothing:

```powershell
pwsh -File scripts/diagnose-odysseus-link.ps1
```

It walks the ladder below, prints the layer that is broken, and exits non-zero
when something is wrong, so a launcher or a scheduled check can call it. The
rest of this section is what it is checking and why.

Symptoms: Odysseus's Settings page shows no models, model discovery times out, or
chat returns a connection error. Meanwhile `curl http://127.0.0.1:8000/v1/models`
on the host works perfectly.

Cause: you gave Odysseus `http://localhost:8000/v1`. Inside a container,
`localhost` is **the container itself** — it is not the machine. The container
has no `locally` in it, so nothing answers.

Fix, under **Docker Desktop** (Windows and macOS):

```
http://host.docker.internal:8000/v1
```

`host.docker.internal` is Docker's hostname for the host machine as seen from
inside a container. Docker Desktop special-cases it.

**Under Podman on Windows that name does not work**, and this is the setup on
Machine A. Podman's containers run inside a WSL2 VM and the name resolves to
`169.254.1.2` — the gateway of the VM, not the Windows host where `locally`
listens. Podman does not special-case it. Measured 2026-08-30:
`host.docker.internal` → HTTP 000, `172.17.96.1` → HTTP 200. Use the Windows
`vEthernet (WSL)` gateway address directly:

```
OLLAMA_BASE_URL=http://172.17.96.1:8000/v1
```

The `/v1` suffix is required — this is the OpenAI-compatible base URL, not the
server root. See "If the gateway address moved" below before assuming that
address is still current.

On **Linux** `host.docker.internal` is not automatic — the service needs:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

Odysseus's upstream `docker-compose.yml` already defines this on the `odysseus`
service (confirmed against the repo). The override file ships it anyway, harmless
and explicit, so the setting is visible where you are editing.

Verify from inside the container — note `podman exec` and the container's real
name, since `docker compose exec` is not what is running here:

```bash
podman exec odysseus_odysseus_1 curl -sS http://172.17.96.1:8000/v1/models
```

If that fails but the host `curl` works, work the ladder below rather than
guessing. Every rung is a one-line command and each one eliminates a layer.

#### The reachability ladder

Run these in order. The **timing** is the diagnosis, not the status code — every
rung returns HTTP 000 when it fails, and what separates a firewall from a dead
server is how long it takes to fail.

```powershell
# 1. Is the server up and serving, from the host itself?
curl -s http://127.0.0.1:8000/v1/models

# 2. Is it bound to all interfaces, or only loopback?
netstat -ano | Select-String ":8000"        # want 0.0.0.0:8000, not 127.0.0.1:8000

# 3. Does the host answer on the WSL gateway address?
curl -s http://172.17.96.1:8000/v1/models

# 4. Can the podman VM reach the host at all?
podman machine ssh "curl -s -o /dev/null -m 6 -w '%{http_code} %{time_total}s' http://172.17.96.1:8000/v1/models"

# 5. Can the Odysseus container reach it?
podman exec odysseus_odysseus_1 curl -s -o /dev/null -m 6 -w '%{http_code} %{time_total}s' http://172.17.96.1:8000/v1/models
```

**Reading rungs 4 and 5 — this is the whole trick:**

| Result | curl exit | Meaning |
|---|---|---|
| fails in **~2 ms** | 56 | The packet **reached the host** and something reset it. The network path is open; your problem is the server, the port, or the protocol. |
| fails after the **full timeout** | 28 | The packet was **silently dropped**. This is a firewall, every time. Nothing else drops without answering. |
| HTTP 200 | 0 | Working. |

Measured on this box 2026-09-08, with the API confirmed healthy on the host:

```
port 7070 (AnyDesk)  = 000  time=0.0017s  exit=56   -> path open, AnyDesk reset it
port 8000 (locally)  = 000  time=6.0032s  exit=28   -> DROPPED
port 445  (SMB)      = 000  time=6.0025s  exit=28   -> DROPPED (expected)
```

Port 7070 is the control: it proves WSL-to-host networking, the `172.17.96.1`
gateway and the podman NAT are all fine. Only port 8000 is being dropped.

Do **not** use `ping` as a rung. ICMP echo to the gateway fails on this machine
even when TCP works, because echo requests are blocked separately — a failed
ping proves nothing and sent an earlier debugging session down the wrong path.

#### Firewall rules: three traps, in the order they bite

**1. The command line is not the program.** Firewall rules match the **loaded
image path**, and on this box those two things disagree — which is what made this
bug take a whole session to find:

```
CommandLine : "C:\Projects\Nollama\venv\Scripts\python.exe" locally.py --device auto
ImagePath   : C:\Users\zeror\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

`venv\Scripts\python.exe` redirects to the base interpreter rather than being the
process, so as far as Windows Firewall is concerned the server is the **system
Python** — not the venv one the launcher names. Every tool that shows you a
command line, `Get-CimInstance Win32_Process` and Task Manager included, shows
the venv path and is useless here. Ask for the image path instead:

```powershell
(Get-Process -Id (Get-NetTCPConnection -State Listen -LocalPort 8000).OwningProcess).Path
```

Write firewall rules against *that*. A rule aimed at the venv path matches
nothing, and — the trap that actually bit — a stale **block** rule on the system
Python matches everything.

**2. Block beats Allow.** Windows evaluates block rules first, so one stale block
rule defeats any number of correct allow rules. Cancelling a Windows firewall
prompt silently creates one — this box accumulated two, both on the system
Python, Public profile, all ports. Find them:

```powershell
Get-NetFirewallRule -Enabled True -Direction Inbound -Action Block |
  ForEach-Object { [pscustomobject]@{ Name = $_.DisplayName
    Program = ($_ | Get-NetFirewallApplicationFilter).Program } }
```

**3. A rule can be valid, active, and still not take effect.** Confirm it reached
the enforced store, not just the persistent one — a rule present in
`PersistentStore` but absent from `ActiveStore` is not being enforced:

```powershell
Get-NetFirewallRule -PolicyStore ActiveStore -DisplayName 'locally API (WSL/Podman only)'
```

#### Scoping the allow rule

Scope by **port plus WSL subnet**, not by program. The program path changes
whenever the venv is rebuilt, and three different Pythons can plausibly serve
this; the subnet does not move. `172.17.96.0/20` is the WSL virtual switch only,
so the rule does not expose port 8000 to the Wi-Fi LAN — which matters, because
`locally` has no authentication and this machine's Wi-Fi profile is Public.

```powershell
New-NetFirewallRule -DisplayName 'locally API (WSL/Podman only)' -Direction Inbound `
  -Action Allow -Protocol TCP -LocalPort 8000 -RemoteAddress 172.17.96.0/20 -Profile Any
```

Requires an elevated PowerShell. Verify with rung 5, not by re-reading the rule.

#### If the gateway address moved

`172.17.96.1` is stable across container and machine restarts but **not** across
a WSL virtual-switch recreation (a Windows feature update, or `wsl --shutdown`
plus a network reset). It appears in three places that must agree — the host
adapter, the VM's default route, and `../odysseus/.env`:

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object InterfaceAlias -like '*WSL*'
podman machine ssh "ip route | head -1"
Select-String OLLAMA_BASE_URL ../odysseus/.env
```

The VM's own address sits elsewhere in that subnet (`172.17.109.78/20` here), and
it is the address the allow rule's `-RemoteAddress` must cover — not the gateway.

#### Never start the server with `chat.ps1 --start` for this

`core/terminal.py:270` passes `--host 127.0.0.1`, so a chat-owned server is
loopback-only and Odysseus can never reach it no matter what the firewall says.
Use `api.ps1`, which goes through the launcher and keeps the `0.0.0.0` default
from `core/cli.py:58`.

#### What this actually was, 2026-09-08 — resolved

**Three independent causes, stacked.** Each one on its own produces the identical
symptom — Odysseus offline, no models — which is why fixing one at a time looked
like it had changed nothing, twice. Anyone debugging this again should assume
there is more than one.

**Cause 1: two stale `Block` rules** on `pythoncore-3.14-64\python.exe`, Public
profile, all ports — the residue of cancelling a Windows firewall prompt at some
point. They beat the correct, active, correctly scoped allow rule for port 8000,
because block always wins.

They stayed hidden because every process listing showed the server running as
`venv\Scripts\python.exe`, which no block rule named. The venv path is the
command line; the image is the system Python (trap 1 above). The allow rule and
the block rules had been pointed at the same executable all along. Removing them,
elevated:

```powershell
$target = (Get-Process -Id (Get-NetTCPConnection -State Listen -LocalPort 8000).OwningProcess).Path
Get-NetFirewallRule -Direction Inbound -Action Block |
  Where-Object { ($_ | Get-NetFirewallApplicationFilter -EA SilentlyContinue).Program -eq $target } |
  Remove-NetFirewallRule
```

Confirmation is rung 4 of the ladder flipping from `DROPPED - timed out after
6.0s` to `HTTP 200 in 0.54s`. Do not confirm by re-reading the rule list; a rule
that exists is not a rule that is being matched, and that mistake cost most of a
session.

**Cause 2: the idle-unload listing bug.** `/v1/models` listed a slot only while
its status was `ready`, so after the default 900-second idle timeout the server
answered chat requests while advertising an empty list. Fixed — the endpoint now
asks `_slot_serviceable`, the same predicate routing uses. Full account under
"Reachable, but the model list is empty" below.

**Cause 3: Odysseus was pointed at `http://localhost:8000/v1`.** This was the
last one standing, and no amount of server-side work could have fixed it:
`localhost` inside a container is the container. Measured from inside
`odysseus_odysseus_1` with everything else already healthy:

```
container -> localhost:8000     = 000
container -> 172.17.96.1:8000   = 200
```

Changing the endpoint in Odysseus's Settings to the gateway address is what
finally made it work.

**What working looks like.** Six models advertised, the container getting a
byte-identical list, and an unloaded model loading on demand from a chat request
— 19.1 s for the swap plus the answer, correctly labelled with the requested
model. `scripts/diagnose-odysseus-link.ps1` reports green on every rung; §4 has
the expected output.

**If you are here again**, run the script first. It walks all three causes in
order and names the one that is broken, which is the thing nobody could do while
debugging this by hand.


### Reachable, but the model list is empty

Symptoms: identical to the firewall failure above — Odysseus says offline and
adds no models — but the ladder passes every rung and the container gets HTTP 200
from `/v1/models`. The difference is in the body:

```
{"data":[],"object":"list"}
```

Cause: **idle unload**. `_models_data()` (`core/models/discovery.py:15`) lists a
slot only when its status is `ready`, and the launcher's default
`-IdleTimeout 900` unloads the model after fifteen quiet minutes. The slot then
reports `idle_unloaded` and drops out of the listing.

The slot has not stopped working. `core/slots/select.py:14` and
`core/chat/common.py:123` both treat `idle_unloaded` as usable, and a chat
request reloads it on demand — verified from inside the container 2026-09-08,
which got a correct completion and left the slot `ready` and advertised again.
So the server will *answer* while advertising nothing, and any client that
discovers models by polling `/v1/models` — Odysseus does — concludes the backend
is gone. Fifteen minutes after you last spoke to it, which is exactly when you
come back to it.

Check for it directly:

```powershell
curl -s http://127.0.0.1:8000/health | ConvertFrom-Json |
  ForEach-Object { $_.devices.PSObject.Properties.Value } | Select-Object model, status
```

`status: idle_unloaded` with an empty `/v1/models` is this, not a network fault.

Fix: keep the model resident for an agent setup, which is what CLAUDE.md already
prescribes and what `install.ps1` writes for agent installs:

```powershell
.\scripts\locally-launch.ps1 -IdleTimeout 0
```

That also keeps the prefix cache alive, which is the larger win for an agent
client re-sending a fixed system prompt every turn.

The underlying inconsistency is worth fixing rather than working around: a slot
that will serve on demand should probably still be advertised, which is a
one-line change to the status test in `_models_data()`. The argument against is
that a client would then pick a model whose first turn pays a 10-40 s reload. Not
changed here — decide it deliberately.

### Model id mismatch

Symptoms: Odysseus's model dropdown is empty, or it sends a model name and gets
an answer from a model you did not pick.

`GET /v1/models` is the ground truth. Ask it:

```powershell
curl.exe -s http://127.0.0.1:8000/v1/models
```

Use exactly what it returns — `Qwen3-8B-int4-cw-ov@NPU`, or the bare
`Qwen3-8B-int4-cw-ov`; `_route_request()` accepts either.

The trap: **an unrecognised model id does not error.** `_route_request()` falls
through to default routing and answers from whichever slot is loaded. With one
slot resident — which is `locally`'s default, one model at a time — you always
get an answer, so a typo'd model id looks like it worked. If you swap the model
and Odysseus keeps sending the old name, it will keep being silently right until
you load a second slot, at which point it will be silently wrong.

Two things that change the advertised id, both easy to forget:

- **The directory name is the model name.** Rename `~/models/Qwen3-8B-int4-cw-ov`
  and the model id changes. There is deliberately no `--model-name` flag; the
  rename *is* the interface.
- **The device suffix changes with placement.** Load the same model on the GPU
  and `@NPU` becomes `@GPU`. A proxy slot is always `@REMOTE`.

If a model does not appear in `/v1/models` at all, its slot is not `ready`. Check
`/health` — a slot still compiling reports `loading`, and a 14 GB model takes
20–40 s (9.3 s on the NPU with the compile cache warm, 65.1 s cold).

### Tool-budget refusal

Symptom: Odysseus asks for a reminder or a calendar event, gets a chatty English
answer describing what it *would* do, and nothing is created.

Look for this in `locally`'s console:

```
tools ignored: 30 schemas render to 4553 tokens, over the NPU's 1200-token
budget. Send fewer tools, or use --agent-tools to trim them here.
```

That is the NPU budget, not a bug and not a model-quality problem. See §5 for the
numbers and the three fixes. If instead you see:

```
tools ignored (NPU slot)
```

with no token count, the tool list was empty or failed to render — check that
Odysseus is actually sending `tools` on that request.

### Proxy slot (Machine B) refuses to start

`ProxySlot` probes on load, so failures surface at startup with the reason:

| Message | Meaning |
|---|---|
| `proxy: cannot reach http://localhost:11434: ... Is the server running?` | Ollama is not up. `ollama serve`, or start the service. |
| `proxy: upstream does not serve 'X'. It offers: a, b, c` | `--proxy-model` typo. Use one of the listed ids, or drop the flag and take the first. |
| `proxy: upstream rejected the API key (401)` | `--proxy-key` wrong. Ollama does not need one — omit it. |
| `proxy: upstream has no such model or endpoint (404)` | The base URL is wrong, or a reverse proxy is in the way. `--proxy-url` wants the **root** (`http://localhost:11434`), not `.../v1`. |

Note the asymmetry, it catches people: `--proxy-url` takes the **root** because
`ProxySlot` appends `/v1/...` itself. The URL you give **Odysseus** ends in
`/v1`. Different layers, different conventions.

### Odysseus in an iframe shows a blank page

You embedded it. Don't — see §1. Use a link.

---

## 8. Confirmed vs. unverified

Everything about `locally` in this document was read out of `locally.py` and
`utility_pipeline.py` in this repo. Every flag named here exists; check with
`grep -n add_argument locally.py`.

**Confirmed against the Odysseus repo** (`odysseus-dev/odysseus`, `main`):

- Default web port **7000**; `APP_PORT` / `APP_BIND` (default `127.0.0.1`).
- Compose services: `odysseus`, `chromadb`, `searxng`, `ntfy`.
- Published ports: odysseus `${APP_BIND:-127.0.0.1}:${APP_PORT:-7000}:7000`,
  chromadb `127.0.0.1:8100:8000`, searxng `127.0.0.1:8080:8080`,
  ntfy `${NTFY_BIND:-127.0.0.1}:8091:80`.
- `SEARXNG_INSTANCE` defaults to `http://localhost:8080`, and `.env.example`
  states Compose overrides it to `http://searxng:8080` for in-network access.
- `OLLAMA_BASE_URL`, with the documented Docker form
  `http://host.docker.internal:11434/v1` — **the `/v1` suffix is required**.
- `LLM_HOST` (default `localhost`) and `LLM_HOSTS` (comma-separated, for model
  discovery; hostnames/IPs only, Odysseus scans common serve ports).
- Model providers are configured in the **Settings** UI after first login.
- The `odysseus` service already defines
  `extra_hosts: ["host.docker.internal:host-gateway"]`.
- The service builds from source (`build: .`), so there is no image tag to pin.
- **`OLLAMA_BASE_URL` in `.env` is NOT sufficient on its own** (measured
  2026-09-08). `.env` carried the correct `http://172.17.96.1:8000/v1` while the
  endpoint stored in Settings still said `http://localhost:8000/v1`, and
  Odysseus used the Settings value: it reported "Probed 0/1 endpoints; 1 failed"
  and listed no models against a server that was healthy and reachable from
  inside its own container. Settings is authoritative; treat the env var as a
  pre-seed for the *first* run only, and edit the endpoint in the UI when
  changing it later.

**Not confirmed — verify against your Odysseus version:**

- Whether `SEARXNG_INSTANCE` is set on the `odysseus` service in
  `docker-compose.yml` or applied elsewhere. `.env.example` says Compose
  overrides it; the override was not read directly.
- The exact parsing and format of `ALLOWED_ORIGINS` (comma-separated? scheme
  required?) for adding a Tailscale hostname.
- Whether Odysseus filters the model list by any capability advert. If it does,
  an NPU slot advertising completion-only on `/api/show` could be hidden from an
  agent picker — use the `/v1` endpoint on port 8000, whose `/v1/models` carries
  no capability field at all.
- `RESEARCH_LLM_ENDPOINT` exists and its example is a full
  `.../v1/chat/completions` path rather than a base URL. If you use deep research
  against `locally`, check that form against your version.
