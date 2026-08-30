# locally Web UI Plan

## Goal

A single `index.html` served by Flask. No framework, no build step,
no Node.js. Opens automatically when the server starts. Chat window,
image drop zone, model selector. Works offline.

---

## Architecture

```
arc-server.py    serves index.html at /
index.html       the entire UI (HTML + CSS + JS, single file)
```

That's it. No `static/`, no `templates/`, no bundler. One file.

Flask adds one route:

```python
@app.route('/')
def gui():
    return send_from_directory('.', 'index.html')
```

Browser opens automatically after startup:

```python
# After the "locally ready" banner
import webbrowser
webbrowser.open(f"http://localhost:{port}")
```

Or PowerShell-native in start.ps1:

```powershell
Start-Process "http://localhost:8000"
```

---

## UI layout

```
+----------------------------------------------------------+
|  locally                          [qwen3-8b v] [NPU]    |
+----------------------------------------------------------+
|                                                          |
|  [assistant]  Hello! How can I help?                     |
|                                                          |
|  [user]       What is the capital of Norway?             |
|                                                          |
|  [assistant]  The capital of Norway is Oslo.  [NPU 1.2s] |
|                                                          |
|  [user]       [image] What vehicle is this?              |
|                                                          |
|  [assistant]  {"vehicle":"car","colour":"silver"} [GPU 2.8s] |
|                                                          |
+----------------------------------------------------------+
|  [drop image here or click]  Type a message...  [Send]   |
+----------------------------------------------------------+
```

### Components

1. **Header bar**
   - "locally" title (left)
   - Model selector dropdown (right) — populated from `/v1/models`
   - Device badge showing which device is active (from `X-Device` header)

2. **Chat area**
   - Scrollable message list
   - User messages (right-aligned or distinct style)
   - Assistant messages (left-aligned) with streaming text
   - Image thumbnails inline for messages that included images
   - Device tag + timing on each assistant message (`[NPU 1.2s]`)
   - Markdown rendering for assistant responses (basic: bold,
     italic, code blocks, lists)
   - Code blocks with copy button

3. **Input area**
   - Text input (multiline, Enter to send, Shift+Enter for newline)
   - Image drop zone / click to attach
   - Image preview thumbnail with X to remove
   - Send button
   - Loading indicator while waiting for response

---

## Streaming

The UI uses `fetch()` with streaming reader for SSE:

```javascript
const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        model: selectedModel,
        messages: chatHistory,
        stream: true,
    }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    // Parse SSE lines, extract delta.content, append to message
}
```

Tokens appear as they arrive. The chat bubble grows in real-time.

For VLM responses (non-streaming), the full response arrives at once.
The UI handles both — it just appends content as it receives it.

---

## Image handling

### Drag and drop

```javascript
inputArea.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        attachImage(file);
    }
});
```

### Click to attach

Hidden `<input type="file" accept="image/*">` triggered by clicking
the drop zone.

### Paste from clipboard

```javascript
document.addEventListener('paste', (e) => {
    const item = e.clipboardData.items[0];
    if (item && item.type.startsWith('image/')) {
        attachImage(item.getAsFile());
    }
});
```

### Encoding

Convert to base64 data URI before sending:

```javascript
function fileToDataUri(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.readAsDataURL(file);
    });
}
```

Send as standard OpenAI image_url content block.

### Preview

Show a small thumbnail in the input area before sending. Click X
to remove. The thumbnail also appears in the chat history after
sending.

---

## Model selector

On page load, fetch `/v1/models` and populate the dropdown:

```javascript
const resp = await fetch('/v1/models');
const data = await resp.json();
// data.data = [{id: "qwen3-8b", owned_by: "local-npu"}, ...]
```

Each option shows model name + device:

```
qwen3-8b (NPU)
qwen2_5-vl-3b (GPU)
```

Default to the first model. When the user picks a different model,
set `"model": selectedModel` in the request.

---

## Chat history

Maintained client-side as a JavaScript array:

```javascript
let chatHistory = [];
// Each message: {role: "user"|"assistant", content: "..." or [...blocks]}
```

Sent with every request (OpenAI multi-turn format). No server-side
session state.

### New chat

A "New chat" button clears the history array and the chat display.

---

## Health polling

On page load and every 10 seconds, poll `/health`:

```javascript
async function checkHealth() {
    const resp = await fetch('/health');
    const data = await resp.json();
    updateStatusBadge(data);
}
```

Show a small status indicator:
- Green dot: ready
- Yellow dot: loading (with which device)
- Red dot: error

---

## Styling

- Dark theme (easy on the eyes, matches the "hacker tool" vibe)
- System font stack (no web fonts to load)
- Responsive — works on laptop screens, doesn't need mobile
- Minimal: no animations beyond typing indicator
- Code blocks: monospace with dark background, copy button
- User messages: slightly different background
- Images: max-width thumbnail, click to view full size

### Color palette

```css
:root {
    --bg: #1a1a2e;
    --surface: #16213e;
    --input-bg: #0f3460;
    --text: #e0e0e0;
    --text-dim: #888;
    --accent: #00b4d8;
    --user-bg: #1a3a5c;
    --assistant-bg: #1e1e3a;
    --border: #2a2a4a;
}
```

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| Enter | Send message |
| Shift+Enter | New line in input |
| Ctrl+V | Paste image from clipboard |
| Escape | Cancel streaming response (if feasible) |
| Ctrl+N | New chat |

---

## Server-side changes to arc-server.py

Minimal:

```python
from flask import send_from_directory

@app.route('/')
def gui():
    return send_from_directory(os.path.dirname(__file__), 'index.html')
```

And after the "locally ready" banner, open browser:

```python
if not os.environ.get('LOCALLY_NO_BROWSER'):
    import webbrowser
    webbrowser.open(f"http://localhost:{port}")
```

Env var to suppress for headless/remote use.

---

## What index.html does NOT need

- Framework (React, Vue, Svelte) — vanilla JS is fine for a chat UI
- Build step (webpack, vite) — it's one file
- External CDN resources — works offline
- Authentication — localhost only
- Chat history persistence — ephemeral is fine (refresh = new chat)
- Mobile layout — this is a laptop tool
- Multiple chat threads — one conversation at a time
- System prompt editor — hardcode a sensible default, or skip

---

## Implementation order

1. **Add `/` route** to arc-server.py + browser auto-open
2. **Basic index.html** — input box, send button, chat display,
   non-streaming POST to /v1/chat/completions, show response
3. **Streaming** — SSE reader, tokens appear in real-time
4. **Image support** — drop zone, paste, base64 encoding, preview
5. **Model selector** — dropdown from /v1/models, device badges
6. **Polish** — dark theme, code blocks, markdown rendering,
   timing display, health indicator
7. **Keyboard shortcuts** — Enter/Shift+Enter, Ctrl+V, Ctrl+N

---

## Size budget

The entire UI should be under 500 lines of HTML/CSS/JS combined.
It's a chat window, not a web app. If it grows past 500 lines,
something went wrong.
