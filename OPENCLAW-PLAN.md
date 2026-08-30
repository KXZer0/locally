# Running OpenClaw on locally (Intel iGPU/GPU) — Today

Goal: drive the **OpenClaw** terminal coding agent with a model served locally by
locally on the Intel Arc 140V (or any Intel GPU/iGPU), with tool calling working.

This needs **no new locally code** — OpenClaw speaks the OpenAI chat-completions
protocol, which locally already serves, and the GPU tool-calling path landed with
PR #9. The work is configuration + one launch convenience, plus verification.

> Claude Code is **not** covered here — it speaks the Anthropic Messages API
> (`POST /v1/messages`), which locally does not serve. That would be a separate
> feature (an Anthropic-Messages translation shim). OpenClaw is the cheap win.

---

## Why this works without code changes

| Requirement | OpenClaw | locally today |
| --- | --- | --- |
| Wire protocol | "any endpoint that implements the OpenAI chat completions API" (`api: "openai-completions"`) | `POST /v1/chat/completions`, `GET /v1/models` |
| Tool calling | sends `tools` every agent turn | parsed/emitted **GPU-only** (`_tools_supported`, PR #9) |
| Model selection | `provider/model-id` string | routes by requested model name |

---

## The one thing that will bite us: routing

locally's default routing sends **text → NPU**, but **tools are GPU-only**. If an
OpenClaw request lands on the NPU, locally silently ignores `tools` and answers as
plain chat — the agent loop breaks (no tool calls ever come back).

**Fix:** address the model by its `@GPU`-suffixed id. `_route_request`
(`locally.py:986-993`) treats an explicit `model@device` string as a hard override
and returns that exact slot, bypassing NPU-default routing:

```python
slot_full = f"{slot.model_name}@{slot.device_name}"   # e.g. "Qwen3-Coder@GPU"
if requested_model in (slot_full, slot.model_name):
    return slot
```

So OpenClaw's model id **must** be `<model_name>@GPU`, not the bare name (the bare
name would route to the NPU whenever an NPU slot is loaded).

---

## Step 1 — Run locally with a GPU LLM that can call tools

- Load an **LLM on the GPU** (not a VLM). Tool calling is gated to `device_name == "GPU"`.
- Prefer a **Qwen3-Coder** variant: locally renders the tool prompt in Qwen3-Coder
  native format, so format match maximizes reliability. `parse_tool_calls` also
  understands Hermes / Mistral `[TOOL_CALLS]` / Llama `<|python_tag|>` / DeepSeek /
  bare-JSON, so other instruct models can work — Qwen3-Coder is just the safest.
- Must fit 16 GB (Arc 140V): an INT4 7B-class coder model is the sweet spot.
  Qwen3-30B-A3B falls back to CPU silently on 16 GB — avoid for an interactive agent.

```powershell
# OpenAI API on :8000 (default). Whatever your generated start.ps1 already does is fine,
# as long as a GPU LLM is loaded. Example:
.\start.ps1
```

Confirm the GPU LLM is up and note its exact id:

```powershell
# Should list an entry whose id ends in @GPU, e.g. "Qwen3-Coder@GPU"
curl http://localhost:8000/v1/models
```

## Step 2 — Prove the OpenClaw-critical path with one request

Before involving OpenClaw, confirm a `tools` request addressed to `<model>@GPU`
comes back with a `tool_calls` array (this is the whole integration in one curl):

```bash
curl -s http://localhost:8000/v1/chat/completions -H "content-type: application/json" -d '{
  "model": "Qwen3-Coder@GPU",
  "messages": [{"role":"user","content":"What files are in the current directory? Use the tool."}],
  "tools": [{"type":"function","function":{
    "name":"list_dir",
    "description":"List files in a directory",
    "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
  }}]
}'
```

Expected: `choices[0].message.tool_calls` present, `finish_reason: "tool_calls"`.
If you instead get plain prose, you hit the NPU — re-check the `@GPU` suffix.
(Note: with `tools` present the response is **buffered, not streamed** — a known
locally limitation, not a bug.)

## Step 3 — Point OpenClaw at locally

Add a provider to `openclaw.json` (confirm the path OpenClaw reads — project dir or
`~/.config/openclaw/`; the onboarding wizard will tell you). locally does no auth,
so `apiKey` just needs to be non-empty.

```json5
{
  agents: {
    defaults: {
      model: { primary: "locally/Qwen3-Coder@GPU" }
    }
  },
  models: {
    providers: {
      locally: {
        baseUrl: "http://localhost:8000/v1",
        apiKey: "local-no-auth",
        api: "openai-completions",
        timeoutSeconds: 600,
        models: [
          {
            id: "Qwen3-Coder@GPU",
            name: "locally Qwen3-Coder (Arc GPU)",
            contextWindow: 32768,
            maxTokens: 8192
          }
        ]
      }
    }
  }
}
```

- `id` must be the `@GPU` form from `/v1/models`.
- Verify OpenClaw forwards the id **verbatim** (including `@GPU`) as the request's
  `model` — that's what the routing override matches on.
- `contextWindow` / `maxTokens` are starting guesses — tune to the model.

## Step 4 — Launch and verify the agent loop

```powershell
openclaw      # or `openclaw chat`, per its CLI
```

Give it a task that forces a tool call ("list the files in this folder", "read
README.md"). Success = OpenClaw invokes a tool, locally logs a tool turn on the
`[GPU]` slot, OpenClaw acts on the result.

Convenience launcher: **`start-openclaw.ps1`** (built) — the locally equivalent of
`ollama launch openclaw`. It reuses a locally already on the port, or starts one with the
agent flags (`--device`, `--idle-timeout 0`, `--prewarm`), waits for it to be ready, then
runs `openclaw`. Params: `-ModelDir`, `-Device CPU|GPU`, `-Port`, `-Prewarm`, `-Openclaw`.
The prefix-cache pre-fill itself is internal to locally (`--prewarm`); the script just wires
locally + OpenClaw together.

---

## Open items / risks

1. **`@` in the model id.** Confirm OpenClaw's `provider/model` parser keeps `@GPU`
   as part of the model id and sends it through. If it chokes on `@`, use the
   **text→GPU override** instead (preferred anyway — see below).

   ### text→GPU override (drops the `@GPU` requirement)

   Two ways, in order of robustness:

   - **Server-side (preferred, ~3 lines, the right fix):** in `_route_request`
     (or just before routing in `chat_completions`), *if the request carries
     `tools` and a GPU slot is serviceable, route to that GPU slot.* Tools are
     GPU-only regardless, so a tools-bearing request has no business on the NPU.
     This makes OpenClaw work with the **bare** model name — no `@GPU` trick, no
     reliance on OpenClaw forwarding `@` verbatim. This is the clean version of
     what the launcher would otherwise paper over.
   - **Wrapper-side (zero locally code):** the `launch-openclaw.ps1` wrapper just
     writes/ensures the `@GPU` id in `openclaw.json`. Works today, but fragile if
     OpenClaw mangles `@`.

   Recommendation: ship with the `@GPU` id to prove it end-to-end **today**, then
   add the server-side `tools⇒GPU` route as the durable fix.
2. **`openclaw.json` location** — confirm via the onboarding wizard.
3. **No streaming on tool turns** — expected; OpenClaw agent mode is always buffered.
4. **Model fit** — keep the GPU LLM under 16 GB or it silently falls back to CPU.

## Out of scope

- Claude Code support (needs an Anthropic `/v1/messages` shim — separate feature).
- NPU tool calling (NPU can't drive agent loops; intentionally GPU-gated).
