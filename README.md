# locally

A terminal-operated OpenAI-compatible model server for Intel NPU and OpenVINO models.
Use Odysseus or another API client as your main interface. Optional terminal chat is
available for quick questions, pasted assignments, and Markdown answers.

The custom web UI, static assets, and native browser shell have been removed.
`/` returns service information as JSON. There is no browser setup wizard.

## Run the API

From the repository in PowerShell:

```powershell
.\start.ps1
# Or choose a model explicitly:
.\venv\Scripts\python.exe locally.py --model-dir ~\models\Qwen3-8B-int4-cw-ov --device NPU
```

The default model is the existing `model` directory/link, with NPU-first device
selection. Only one chat slot starts unless `--gpu-model-dir` is supplied.
Press **Ctrl+C in the server terminal** to stop the API and release its model
allocations. Closing an attached chat client does not stop the server.

Audio and utility models are opt-in: pass `--whisper-dir`, `--tts-dir`,
`--util-models-dir`, or `--auto-util` for utility discovery. Previous browser setup
settings and saved companion autostart choices are no longer read. Explicit API
features such as `--search-url` and `--odysseus-autostart` remain available.

Idle models unload after 30 minutes by default. `--idle-timeout 0` keeps them resident.
An unloaded model can reload when a new API request needs it; stop the process when
you want its memory to stay released. Driver memory accounting may settle after exit.

## Connect Odysseus or another harness

Use an OpenAI-compatible provider with base URL **http://127.0.0.1:8000/v1**.
Choose an exact model ID from `GET /v1/models`, such as `model-name@NPU`.
If the client requires an API key field, use `local`; Locally does not authenticate it.
Ollama compatibility is available on port 11434 unless occupied or disabled with
`--ollama-port 0`. Anthropic Messages is available at `/v1/messages`.

For a containerized harness on this Windows host, start the API with `--host 0.0.0.0`
and use **http://host.docker.internal:8000/v1** in the container. The standalone server keeps its existing all-interface bind for container clients.
Use `--host 127.0.0.1` for a local-only server. The API has no authentication, so only expose it to trusted clients.
A firewall or container-specific host routing may need configuration.

Client tool calls run on GPU and CPU slots unconditionally. **An NPU slot takes
them when the request's rendered tool block fits `NPU_TOOL_BUDGET`** (1200 tokens
-- an 8-tool assistant set measures 735, a 30-tool coding agent 4,553), and an
over-budget set is refused with the token count and a fix rather than a bare
"GPU-only". Fitting is not answering well: a small model given thirty tools is a
harder problem than one given six, so test your harness and model together. The
NPU prompt cap is 8192 tokens, and a large agent system prompt spends it before
the user types anything.

## Optional terminal chat

Attach to an API you already started (exiting leaves that API running):

```powershell
.\venv\Scripts\python.exe locally.py chat
```

Or start a server owned by this one chat session:

```powershell
.\venv\Scripts\python.exe locally.py chat --start --device NPU
```

`--start` refuses an occupied port. `/exit`, Ctrl+D, or Ctrl+C at the prompt stops
its owned server process tree, including any in-flight native inference. It does
not stop a separately running harness. The owned server binds to loopback, disables
the secondary Ollama listener and companion autostart, and writes a temporary log
whose path is printed. If you forcibly kill the terminal application itself rather
than exiting the client, check that the server has stopped.

The prompt may appear while a model is compiling; `/status` shows its state.
Use `--port 8001` to run an independent server, or `--url http://host:8000` to attach
elsewhere. `--max-tokens 2048` changes the output budget. `--plain` displays raw
Markdown without colors.

Type `/` to open command suggestions with descriptions. Keep typing to filter;
Tab or the arrow keys select a completion. Arguments complete too: `/markdown on`,
`/unload NPU`, and **`/load` completes model names from the server itself**
(`GET /v1/models/available`), matching anywhere in the name — `/load abl` finds
`Qwen3-8B-abliterated-int4-cw` — and showing what each one is, where it is
loaded, or why it cannot be. `name@` then completes `NPU`/`GPU`/`CPU`. The list
is the server's, never a baked-in one, so a model you delete stops being offered
and one you add appears. Suggestions stay out of ordinary chat text.

**Enter sends. Alt+Enter inserts a newline.** Pasted multiline text remains editable
before sending in terminals supporting bracketed paste. Use Ctrl+V (or Ctrl+Shift+V,
depending on your terminal), or `/paste` to insert the clipboard for editing.
Mouse selection and normal terminal copy shortcuts remain available.

| Command | What it does |
| --- | --- |
| `/models` | List models on disk |
| `/models npu` | Show models passing local NPU capability checks |
| `/load model-name@NPU` | Load or swap a model; actual placement is reported |
| `/load C:\path with spaces\model@GPU` | Load a model by directory |
| `/unload` or `/unload NPU` | Release model slots while keeping the API up |
| `/status` | Show device states, context limits, TTFT, and available RAM |
| `/new` | Clear the conversation and last answer |
| `/copy` | Copy the last complete answer as original Markdown |
| `/save homework.md` | Save the last complete answer in UTF-8; never overwrite |
| `/markdown on` or `/markdown off` | Choose formatted or raw display |
| `/think on` or `/think off` | Reasoning before the answer (off sends Qwen3's `/no_think`) |
| `/read PATH` or `/read https://...` | Read a document, image or page **into the conversation** |
| `/index PATH [PATH ...]` | Build a local semantic index from files |
| `/find query` | Search that index; shows the embedding score and the rerank score |
| `/upscale IN [OUT]` | Upscale an image and write a PNG |
| `/util` / `/util off` | Utility engine state; `off` frees them, chat untouched |
| `/voice` | Talk instead of typing (needs `sounddevice` and the server's audio models) |
| `/paste` | Insert clipboard text into the next prompt |
| `/exit` | Exit the client; also stop its server if started with `--start` |

`/read` puts the extracted Markdown in the conversation, so the next question can
be about it; a URL is fetched server-side and a scanned page goes through OCR.
`/index` and `/find` use the utility engines — they retrieve wide and let a
cross-encoder settle the order, which is why a hit can show a lower embedding
score above a higher one. The engines compile on first use and stay resident;
`/util off` frees **only** them (`POST /v1/models/unload {"scope": "util"}`),
because on a one-model box the chat model is usually on the same device and
unloading by device took the conversation down with it. A utility command run
while the engines are still compiling waits instead of failing.

`/load` clears the conversation to avoid carrying the previous model's context.
Model load/unload commands affect all clients connected to that server.
Ctrl+C during a streamed response stops reading it and asks the server to cancel
(`POST /v1/cancel`). That request is best effort by construction: OpenVINO may be
blocked in native code and never look at the flag, so a native prefill can run to
completion. Exiting an owned session stops the process entirely.
Failed or interrupted turns are not appended to conversation history. An answer
that hits the `--max-tokens` budget is kept and labelled incomplete rather than
discarded — on the 1024-token default that is the ordinary end of a long answer.
A thinking model's `<think>` block is hidden: it is not the answer, and it would
otherwise land in `/copy` and `/save` as though it were. If a turn produces only
reasoning (a small model can spend its whole budget before closing the tag), the
reasoning is shown, because an empty answer reads as a hang.

Markdown headings, lists, tables, and code blocks render in the terminal. Copy/save
preserve the original Markdown rather than terminal decoration. LaTeX source stays
intact; this client does not typeset equations. Conversations live in memory and are
not saved automatically. Use `/new` when a long conversation reaches the model's
context budget. For attachments, rich math, and persistent conversations, use your
chosen harness.

### Voice in the terminal

`/voice` is push-to-talk: Enter opens the microphone, Enter closes it, an empty
turn leaves, Ctrl+C interrupts. A terminal cannot see a key being *released*, so
hold-to-talk is not available; the server's Silero VAD can take turns on its own
over `/v1/audio/stream`, but that needs a live socket and is not wired up here.

It needs `sounddevice` (`pip install sounddevice`) on the client, and the server
started with `--whisper-dir` and `--tts-dir`; without either it says so and the
rest of the client carries on. Both models are warmed on entry, because idle
unload is doing its job when it evicts them and the reload otherwise lands
inside your first spoken turn.

Spoken turns share the conversation with typed ones, and are shaped for speech:
a voice-only directive is appended last (2-3 plain sentences, no Markdown — a
table read aloud is gibberish), the budget is 220 tokens, and reasoning is
suppressed by default because it must never be spoken. Neither the directive nor
the control token is kept in the conversation. **Speech starts before the answer
finishes**: clips are peeled off the token stream a sentence at a time and clip
N+1 is synthesized while clip N plays, and each line is printed when its audio
starts, so what you read is what you are hearing.

Measured on this machine, through the client's own methods: Kokoro synthesized
3.2 s of audio in 1.9 s, and Whisper transcribed it back word for word.

## API surface

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Readiness and enabled model slots |
| `GET /v1/models` | Loaded model IDs |
| `GET /v1/models/available` | Local model discovery and placement information |
| `POST /v1/models/load` | `{ "model": "name", "device": "NPU" }` |
| `POST /v1/models/unload` | `{}` or `{ "device": "NPU" }` |
| `POST /v1/chat/completions` | OpenAI chat, including SSE streaming |
| `POST /v1/messages` | Anthropic Messages compatibility |
| `GET /v1/memory` | Available memory and offload state |
| `GET /v1/metrics` | Last 50 recorded turns |

Audio, document/OCR, and utility APIs remain available when their models are enabled.
Historical details are in [the previous README](docs/LEGACY-README.md); its UI and
setup instructions are retired. Models are stored outside the frontend and were not deleted.

## Install and verify

Starting Locally works in Windows PowerShell 5.1 (the Windows default) and
PowerShell 7. The installer itself still requires PowerShell 7. The hardware-key
launcher uses PowerShell 7 when present and falls back to Windows PowerShell; it
opens a chat window attached to a server that is already listening, and only
starts one (`chat --start`) when nothing answers on the port.
If an older generated `start.ps1` complains about a PowerShell version, replace
it with the repository's current `start.ps1` or rerun the installer once.

`./install.ps1` installs the environment and opens the terminal model chooser.
`-SkipModel` installs dependencies only. `-TerminalSetup` remains accepted for old scripts.
`python locally.py --help` lists inference settings; `python locally.py chat --help`
lists client settings. `python locally.py --scan` inspects models without loading
them, and `python locally.py --list-models [DIR]` prints the same discovery as
`path<TAB>name<TAB>llm|vlm` for scripts — which is how `scripts/locally-launch.ps1`
picks a model when the ones it prefers are no longer on disk, instead of naming
models by hand.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Unit tests use synthetic responses and no model loads. Hardware-specific notes in
`TODONT.md` remain relevant. Old browser tooling and integration source files are
retained as historical code, but no browser routes are registered by the server.
