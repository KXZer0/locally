# Obsidian Copilot with locally

Obsidian's Copilot plugin talks to any OpenAI-compatible endpoint. Point it at
`locally` and every model on your disk becomes selectable from the plugin,
including ones that are not resident yet — selecting one and sending a message
loads it.

## Add the provider

In **Copilot → Settings → Model**:

1. Add a **custom model**.
2. Provider: **OpenAI Compatible** (some builds call it "3rd party OpenAI
   Format").
3. Base URL: `http://127.0.0.1:8000/v1`
4. API key: anything non-empty (`local` is fine) — `locally` does not validate
   it unless you started the server with `--api-key`.
5. Model name: an id from `GET http://127.0.0.1:8000/v1/models` (see below).

Add **one custom-model entry per model** you want in the picker. Copilot has no
concept of "list the provider's models"; each entry is a fixed `model` string it
will send on every request.

## Which model id to enter

`GET /v1/models` returns one entry per model — the resident one and every
loadable directory on disk:

```console
$ curl -s http://127.0.0.1:8000/v1/models
{"object":"list","data":[
  {"id":"Qwen3-8B-int4-cw-ov@NPU","owned_by":"local-npu"},
  {"id":"gemma-4-26b-a4b-it","owned_by":"local-available"}
]}
```

Use the **bare directory name** (`Qwen3-8B-int4-cw-ov`, `gemma-4-26b-a4b-it`) as
the Copilot model name. It is the stable identity: matching is case-insensitive
and ignores surrounding whitespace, so the entry keeps working across a model
swap, an idle-unload/reload, and a device fallback.

The `@DEVICE` suffix on the resident entry (`@NPU`, `@GPU`, `@CPU`, `@REMOTE`) is
accepted too, for older configs and for the rare case of two slots holding
models with the same name on different devices. It is not required, and a
suffix that has gone stale — the model fell back from NPU to GPU but your config
still says `@NPU` — still resolves to the resident model rather than forcing a
reload.

An id that matches nothing on disk is **not** an error: `locally` serves it from
whatever model is resident. That is deliberate — clients send unconfigured
defaults like `gpt-4` — but it means a typo in the Copilot model name looks like
it works while silently answering from the wrong model. Copy the id from
`/v1/models`.

## Copilot's selection vs. terminal `/models` and `/load`

Terminal chat still has `/models` (list local models, `/models npu` filters NPU
candidates) and `/load <name|path>[@NPU|@GPU|@CPU]` (swap the resident model).
These are unchanged and change the model for **every** connected client.

When Copilot sends a request, the `model` field in that request wins for that
turn:

- If it names a model on disk that is not resident, `locally` swaps to it
  before answering — overriding whatever `/load` last selected. The swap is
  synchronous (9–65 s for an NPU model against a warm compile cache), and it is
  one-at-a-time, so the model you had in the terminal is unloaded.
- If it names the already-resident model (bare or with any `@DEVICE` suffix),
  nothing reloads — the turn is served immediately.
- If it names nothing, or an id that isn't on disk, the terminal's selection
  stands.

So a terminal `/load` sets the default, and an explicit model in Copilot
overrides it on the next request. If you want the terminal and Copilot to stay
on the same model, either leave Copilot's entry pointed at that model or don't
switch models in the terminal mid-session.

## One model at a time

`locally` runs one generative model at a time by design (the NPU allocates from
the same RAM as the GPU, so a second resident model costs real memory for a
model you cannot talk to concurrently). Every Copilot entry that names a
different on-disk model is therefore a swap when you pick it, not an instant
switch. Keep a single entry for day-to-day use and add others only when you
actually need to move between them.

## Ollama surface

Copilot can also use the Ollama provider type against `http://127.0.0.1:11434`
(unless started with `--ollama-port 0`). That surface advertises the same models
**without** the `@DEVICE` suffix — Ollama clients treat the tag as a name — so
the bare id is the only form there.
