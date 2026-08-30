# Ollama-Compatible API Plan

## Goal

Run an Ollama-compatible API on port 11434 alongside the OpenAI API
on port 8000. Tools that auto-discover Ollama (Open Interpreter,
Continue.dev, Aider, various IDE plugins) will find locally without
configuration.

Same models, same devices, same server process. Just a second port
that speaks Ollama's dialect.

---

## Why both APIs?

| Port | API | Who uses it |
|---|---|---|
| 8000 | OpenAI `/v1/chat/completions` | OpenWebUI, openai package, curl, custom scripts |
| 11434 | Ollama `/api/chat`, `/api/generate` | Continue.dev, Aider, Open Interpreter, IDE plugins |

Many local AI tools hardcode `localhost:11434` as the Ollama endpoint.
If locally answers there, they work out of the box with zero config.

---

## Ollama API surface

Only the endpoints that clients actually use. Ollama has ~15
endpoints; clients typically use 3-4.

### GET /

Returns `Ollama is running` (plain text). This is the health check
that tools use to detect Ollama.

### GET /api/tags

List models. Equivalent to OpenAI's `/v1/models`.

```json
{
    "models": [
        {
            "name": "qwen3-8b",
            "model": "qwen3-8b",
            "size": 5000000000,
            "details": {
                "family": "qwen3",
                "parameter_size": "8B",
                "quantization_level": "int4"
            }
        }
    ]
}
```

### POST /api/chat

Chat completion. The main endpoint.

Request:
```json
{
    "model": "qwen3-8b",
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "stream": true
}
```

Streaming response (newline-delimited JSON, NOT SSE):
```
{"model":"qwen3-8b","message":{"role":"assistant","content":"Hello"},"done":false}
{"model":"qwen3-8b","message":{"role":"assistant","content":"!"},"done":false}
{"model":"qwen3-8b","message":{"role":"assistant","content":""},"done":true,"total_duration":1500000000,"eval_count":15}
```

Non-streaming response:
```json
{
    "model": "qwen3-8b",
    "message": {"role": "assistant", "content": "Hello!"},
    "done": true,
    "total_duration": 1500000000,
    "eval_count": 15
}
```

Key differences from OpenAI SSE:
- No `data: ` prefix
- No `[DONE]` sentinel
- Newline-delimited JSON, not SSE
- Final chunk has `"done": true` with timing stats
- Duration is in nanoseconds (not seconds)

### POST /api/generate

Single-turn completion (no chat history). Some older tools use this.

Request:
```json
{
    "model": "qwen3-8b",
    "prompt": "Why is the sky blue?",
    "stream": true
}
```

Response format same as /api/chat but with `"response"` instead of
`"message"`:
```
{"model":"qwen3-8b","response":"The","done":false}
{"model":"qwen3-8b","response":" sky","done":false}
{"model":"qwen3-8b","response":"","done":true,"total_duration":1500000000}
```

### POST /api/show

Model info. Some clients call this to check model capabilities.

```json
{
    "model": "qwen3-8b",
    "details": {
        "family": "qwen3",
        "parameter_size": "8B",
        "quantization_level": "int4"
    },
    "model_info": {}
}
```

### Endpoints to stub (return 200 OK, do nothing)

| Endpoint | Purpose | Our response |
|---|---|---|
| POST /api/pull | Download model | 200 + `{"status": "success"}` |
| DELETE /api/delete | Delete model | 200 + OK |
| POST /api/copy | Copy model | 200 + OK |

These exist so clients don't error out. locally manages models
through install.ps1, not through the API.

---

## Image support (Ollama format)

Ollama passes images as base64 in a separate `images` array, not
inline in the message content:

```json
{
    "model": "qwen2_5-vl-3b",
    "messages": [
        {
            "role": "user",
            "content": "What is in this image?",
            "images": ["base64encodeddata..."]
        }
    ]
}
```

The translation layer converts this to OpenAI format before routing
to the existing pipeline.

---

## Architecture

### Option A: Two Flask apps, two threads (recommended)

```python
# Main server (OpenAI API)
openai_app = Flask("openai")
# ... existing routes on port 8000

# Ollama shim
ollama_app = Flask("ollama")
# ... translation routes on port 11434

# In main():
Thread(target=lambda: openai_app.run(port=8000, threaded=False)).start()
Thread(target=lambda: ollama_app.run(port=11434, threaded=False)).start()
```

Both apps share the same DeviceSlot objects, same pipelines, same
locks. The Ollama app is a thin translation layer — it parses Ollama
format, calls the same generate functions, and formats the response
in Ollama's dialect.

Pros: Clean separation. Port 11434 is Ollama. Port 8000 is OpenAI.
Cons: Two Flask instances in one process. Slightly more code.

### Option B: One Flask app, both ports via SO_REUSEPORT

Not possible with Flask dev server. Would need a reverse proxy.

### Option C: One Flask app, Ollama routes on same port 8000

```python
@app.route('/api/chat', methods=['POST'])
@app.route('/api/generate', methods=['POST'])
@app.route('/api/tags', methods=['GET'])
```

Pros: Simplest. One port, one app.
Cons: Ollama clients expect port 11434. Would need `--port 11434`
which breaks the OpenAI API expectation of 8000.

**Recommendation: Option A.** Two ports, both always active. No
config needed — tools find what they expect.

### Threading

Both Flask apps run with `threaded=True`. Concurrency is controlled
by per-device locks (`DeviceSlot.lock`), not by Flask's threading
model.

This means:
- `/health`, `/api/tags`, `/v1/models` always respond instantly
- Two inference requests to the same device queue on the lock —
  second waits, eventually gets answered
- NPU and GPU requests run in parallel (separate locks)
- Web UI health polling never freezes

Without `threaded=True`, a streaming response on port 8000 would
block even `/health` on that port until generation finishes. With
two ports sharing the same pipelines, that's unacceptable.

---

## CLI changes

```
python arc-server.py                          # both ports: 8000 + 11434
python arc-server.py --no-ollama              # OpenAI only, no 11434
python arc-server.py --ollama-port 11435      # custom Ollama port
```

| Flag | Default | Purpose |
|---|---|---|
| `--ollama-port` | 11434 | Ollama API port (0 to disable) |
| `--no-ollama` | false | Disable Ollama API entirely |

### Banner update

```
================================================
  locally ready
    NPU  : qwen3-8b (LLM) -- Intel AI Boost
    GPU  : qwen2_5-vl-3b (VLM) -- Intel Arc 140V
    API  : http://localhost:8000      (OpenAI)
    API  : http://localhost:11434     (Ollama)
================================================
```

---

## Translation layer (pseudocode)

The Ollama app translates requests and reuses the existing pipeline:

```python
@ollama_app.route('/api/chat', methods=['POST'])
def ollama_chat():
    body = request.get_json()

    # Translate Ollama -> internal format
    messages = []
    images = []
    for msg in body.get('messages', []):
        content = msg.get('content', '')
        # Ollama puts images in a separate array per message
        msg_images = msg.get('images', [])
        if msg_images:
            # Convert to OpenAI content blocks format
            blocks = [{'type': 'text', 'text': content}]
            for img_b64 in msg_images:
                blocks.append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}
                })
            messages.append({'role': msg['role'], 'content': blocks})
        else:
            messages.append({'role': msg['role'], 'content': content})

    # Route to device (same logic as OpenAI endpoint)
    slot = _route_request(has_images=bool(images), ...)

    # Generate and format as Ollama response
    if body.get('stream', True):
        return Response(
            _ollama_stream(slot, messages, body),
            mimetype='application/x-ndjson',
        )
    else:
        text = slot.generate_llm(messages, gen)
        return jsonify({
            'model': slot.model_name,
            'message': {'role': 'assistant', 'content': text},
            'done': True,
            'total_duration': int(elapsed * 1e9),
        })
```

---

## Console logging

Same format, with API tag:

```
09:15:03 <- [NPU] [Ollama] 3 msgs, 120 chars (stream)
09:15:08 -> [NPU] [Ollama] ~85 tokens in 4.8s
09:15:12 <- [GPU] [OpenAI] 1 image, 47 chars
09:15:15 -> [GPU] [OpenAI] 56 chars in 2.8s
```

---

## Implementation order

1. **Create ollama_app** Flask instance with `/`, `/api/tags`,
   `/api/show` (static responses)
2. **Add `/api/chat`** — translate format, call existing pipeline,
   format response (non-streaming first)
3. **Add streaming** — newline-delimited JSON (not SSE)
4. **Add `/api/generate`** — single-turn variant
5. **Add image translation** — Ollama `images[]` -> OpenAI content blocks
6. **Wire into main()** — second thread, port 11434
7. **Test with curl** — verify format matches real Ollama
8. **Test with a real client** — Continue.dev or Aider

---

## Size estimate

The Ollama translation layer is ~150-200 lines. It's a format
adapter, not new functionality. All the real work (model loading,
inference, streaming, device routing) is already done.
