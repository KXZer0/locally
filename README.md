# locally

`locally` is an OpenAI-compatible AI server for Intel hardware. It has no web dashboard. Run the API in PowerShell, then connect an API client or open the built-in terminal chat when you need it.

## Quick start

Open PowerShell in the project folder. In File Explorer, click the address bar, type `powershell`, and press Enter.

For a fresh clone:

```powershell
git clone https://github.com/KXZer0/locally.git
cd locally
pwsh -NoProfile -File .\install.ps1
```

The installer creates `venv`, installs dependencies, and helps you choose a model. It needs PowerShell 7 (`pwsh`). If that command is not found, install PowerShell 7 and reopen your terminal. If PowerShell blocks scripts, allow them for the installer process and retry:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

Then start the API and keep that PowerShell window open:

```powershell
.\api.ps1
```

The first model load can take a while because OpenVINO compiles it. When ready, the API is available at:

```
http://127.0.0.1:8000/v1
```

Press Ctrl+C in the API window when you want to stop the server and release model memory.
Press Shift+Enter in that same window to open terminal chat attached to the API;
leaving chat does not stop the server.

## Start and check the API

`api.ps1` is the simple foreground launcher. It uses the model and device settings saved in `start.ps1` by the installer and accepts a port when you need a separate server:

```powershell
.\api.ps1 -Port 8001
```

Check a running server from a different PowerShell window:

```powershell
.\venv\Scripts\python.exe locally.py status
```

For direct control over a model or device, start `locally.py` yourself:

```powershell
.\venv\Scripts\python.exe locally.py --model-dir "$HOME\models\Qwen3-8B-int4-cw-ov" --device NPU
```

Use `--device auto` to let the server choose compatible Intel hardware. The default server bind is `0.0.0.0` for local container clients. The source filter permits loopback and detected virtual/container subnets by default; optional `--api-key` authentication is available. Use `--host 127.0.0.1` when only this computer should connect.

After setup, launch and chat commands work in Windows PowerShell 5.1 and PowerShell 7. To install dependencies without choosing a model, use `.\install.ps1 -SkipModel`.

## Open terminal chat

For Obsidian Copilot model selection, see [Copilot setup](docs/COPILOT.md).
For the NPU speed investigation and reproducible probe, see
[token speed review](docs/TOKEN-SPEED-REVIEW.md).

Terminal chat is an HTTP client. It connects to the API; attaching to a server does not load OpenVINO or stop the server when you leave.

Start the API first with `.\api.ps1`, then open another PowerShell window in the repository and run:

```powershell
.\chat.ps1
```

That is the easiest chat command. It attaches to an API already listening on port 8000. If no API is listening, it starts one for this chat session and stops that temporary API when you leave.

To use another port:

```powershell
.\chat.ps1 -Port 8001
```

You can also attach explicitly:

```powershell
.\venv\Scripts\python.exe locally.py chat
```

For a one-off chat that always starts and owns its own API:

```powershell
.\venv\Scripts\python.exe locally.py chat --start --device NPU
```

`--start` owns the API it starts. `/exit`, Ctrl+D, or Ctrl+C at the prompt ends both chat and that API. It refuses an occupied port, so leave off `--start` when an API is already running.

Type a question and press Enter. Alt+Enter inserts a newline. Type `/` to browse commands. Use `/status` to inspect the server, `/new` to clear the conversation, and `/exit` to leave. Ctrl+C during an answer asks the server to cancel it.

| Command | Purpose |
| --- | --- |
| `/models` | List local chat models |
| `/models npu` | Show models that pass local NPU capability checks |
| `/load name@NPU` | Load or swap a model |
| `/unload` | Free the chat model while keeping the API alive |
| `/status` | Show readiness, devices, and memory |
| `/copy` | Copy the last answer as Markdown |
| `/save answer.md` | Save the last answer without overwriting a file |
| `/markdown on` or `/markdown off` | Choose formatted or raw Markdown display |
| `/think on` or `/think off` | Toggle Qwen3 reasoning mode |
| `/read PATH` | Add extracted document text to the conversation |
| `/index PATH` and `/find query` | Search local files with utility models |
| `/upscale IN [OUT]` | Upscale an image and write a PNG |
| `/util off` | Unload utility models without unloading chat |
| `/voice` | Start optional push-to-talk voice chat |
| `/paste` | Insert clipboard text into the current prompt |
| `/exit` | Leave chat; stop the API only when chat used `--start` |

`/load` clears the current conversation, and model load/unload affects every client connected to that API. `/copy` and `/save` preserve the original Markdown, not terminal formatting; `/save` never overwrites an existing file. Pasted multiline text remains editable before you send it in terminals that support bracketed paste. Use Ctrl+V or Ctrl+Shift+V, depending on your terminal, or use `/paste`.

`/read`, `/index`, `/find`, `/upscale`, and `/voice` need optional models. Start the API with `--whisper-dir`, `--tts-dir`, or `--util-models-dir` to enable those features. Voice also needs `sounddevice` on the client: `python -m pip install sounddevice`.

## Shortcut and Copilot key

Create a Start-menu shortcut, with an optional Desktop copy:

```powershell
.\scripts\locally-launch.ps1 -CreateShortcut -Desktop
```

The default shortcut opens a terminal for the persistent API: if an API is running, it shows its status; otherwise it starts the foreground API. This is the recommended target for a Copilot or hardware key, because the API stays available for your normal client and for terminal chat.

To make the shortcut open chat instead, use:

```powershell
.\scripts\locally-launch.ps1 -CreateShortcut -ShortcutMode chat
```

Chat mode attaches to the current API, or starts a temporary API that ends when chat exits. You can open either behavior directly, without a shortcut:

```powershell
.\scripts\locally-key.ps1 -Mode api
.\scripts\locally-key.ps1 -Mode chat
```

The shortcut command creates `locally.lnk` in the repository, puts a copy in the Windows Start menu, and adds one to the Desktop when `-Desktop` is supplied. Point a key-remapping tool at that installed shortcut.

Windows’ built-in **Customize Copilot key** option accepts signed MSIX apps instead of ordinary `.lnk` shortcuts. Use a key-remapping tool such as NewPilot or your keyboard/laptop vendor's utility to launch the shortcut. Microsoft documents the limitation in [Manage the Copilot key](https://learn.microsoft.com/en-us/windows/client-management/manage-windows-copilot).

## Connect another client

Use an OpenAI-compatible provider with the base URL `http://127.0.0.1:8000/v1`. Read loaded model IDs from `GET /v1/models`; they look like `model-name@NPU`. If your client insists on an API key, enter `local`; locally does not validate it.

Ollama compatibility is available on port 11434 unless disabled with `--ollama-port 0`. Anthropic Messages compatibility is available at `POST /v1/messages`.

For a client in a Docker container on this Windows computer, start the API with `--host 0.0.0.0` and use `http://host.docker.internal:8000/v1` inside the container.

### Who can reach the server

locally binds `0.0.0.0` by default, because a container cannot see a loopback socket — that bind is what makes the WSL, Podman and Docker path work at all. It is not a decision to serve the network: `--allow-from` (default `auto`) answers loopback and the private subnets of virtual/container adapters, and refuses everything else with `403`. The container subnet is discovered at startup (about 10 ms), so nothing is hardcoded and nothing needs configuring on a new machine.

The startup banner says the posture out loud, e.g. `Access: 127.0.0.0/8, ::1/128, 172.17.96.0/20`.

Add a subnet, or turn the filter off:

```powershell
python locally.py --model-dir model --allow-from auto,192.168.1.0/24
```

To expose the port deliberately — a real LAN, Tailscale, a second machine — a source address vouches for nobody, so add a key:

```powershell
python locally.py --model-dir model --allow-from any --api-key 'choose-a-long-random-string'
```

Clients then send `Authorization: Bearer <key>` or `X-Api-Key: <key>`. The server reads `$LOCALLY_API_KEY` when `--api-key` is absent, and the terminal client (`chat.ps1`) reads the same variable so it keeps working. Without `--api-key`, no key is required or checked.

On Windows the firewall is a separate matter, and it governs *reachability* rather than safety: without an inbound allow rule the packets never arrive. See `docs/ODYSSEUS.md` for the scoped rule.

### Clients that run in a browser (Obsidian, and anything embedded)

An Obsidian plugin runs inside a browser engine at the origin `app://obsidian.md`, so its call to locally is a *cross-origin* request. The browser discards a cross-origin response that carries no `Access-Control-Allow-Origin` header before the plugin sees it, and the plugin can only report `Failed to fetch` — no status, no message — while locally logs an ordinary `200`. Reaching the port from a browser address bar proves nothing here, because typing a URL is not a cross-origin request.

locally sends the header. Obsidian on desktop and mobile, and pages served from `localhost` or `127.0.0.1` on any port, are allowed out of the box; point the plugin at `http://127.0.0.1:8000/v1` (or port 11434 for the Ollama surface) and it works with no flag.

Allow another origin with `--cors-origin`, repeating it as needed. `fnmatch` wildcards are accepted:

```powershell
python locally.py --model-dir model --cors-origin 'vscode-webview://*'
```

`--cors-origin '*'` allows every page in every open tab to drive your model, and locally checks no credentials — prefer naming the origin. `--no-cors` sends nothing at all.

## Use Odysseus

Odysseus runs the assistant interface; Locally supplies its model over HTTP.
In Odysseus, add an OpenAI-compatible model provider with this base URL:

```text
http://host.docker.internal:8000/v1
```

Use the exact model ID shown by `GET http://127.0.0.1:8000/v1/models`. The
container must use `host.docker.internal`, because `localhost` inside the
container points back to Odysseus itself.

Once Odysseus and a Docker-compatible container engine are installed, this
single command starts Locally and the Odysseus stack when needed, then opens
Odysseus in your default browser:

```powershell
.\odysseus.ps1
```

Install the same action as a Start-menu and Desktop shortcut with:

```powershell
.\odysseus.ps1 -CreateShortcut -Desktop
```

The shortcut never stops either service. It adopts services that are already
running and opens `http://127.0.0.1:7000`.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| `Run install.ps1 first` | Run `.\install.ps1` from the repository, then retry. |
| Chat cannot connect | Start `.\api.ps1`, then run `locally.py status` to check port 8000. |
| Port 8000 is occupied | Use `.\chat.ps1` to attach, stop the old server if it is yours, or give both API and chat the same alternate `-Port`. |
| The terminal says the model is loading | Wait. The client waits while `/health` reports `loading`; first compile takes longest. |
| No model can be loaded | Run `.\venv\Scripts\python.exe locally.py --list-models $HOME\models`, or rerun the installer. |
| The hardware key does nothing | Open the Start-menu shortcut once to confirm it works, then configure the key-remapping tool to launch that shortcut. |

## API reference and development

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Readiness and enabled model slots |
| `GET /v1/models` | Loaded model IDs |
| `GET /v1/models/available` | Local model discovery and placement information |
| `POST /v1/models/load` | Load a model: `{ "model": "name", "device": "NPU" }` |
| `POST /v1/models/unload` | Unload slots: `{}` or `{ "device": "NPU" }` |
| `POST /v1/chat/completions` | OpenAI chat, including SSE streaming |
| `POST /v1/messages` | Anthropic Messages compatibility |
| `GET /v1/memory` | Available memory and offload state |
| `GET /v1/metrics` | Recent recorded turns |

Show server options with `.\venv\Scripts\python.exe locally.py --help` and chat options with `.\venv\Scripts\python.exe locally.py chat --help`. `--scan` inspects models without loading them; `--list-models [DIR]` prints script-friendly discovery.

Run unit tests with:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Unit tests use synthetic responses and do not load a model. UI-era notes are historical material in [docs/archive/README.md](docs/archive/README.md).
