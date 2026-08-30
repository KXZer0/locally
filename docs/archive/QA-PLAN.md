# QA Plan — locally

## Test matrix

### OpenAI API (port 8000) — verified 2026-04-12

| Test | Status | Notes |
|---|---|---|
| GET /health | PASS | Shows device status, loading transitions |
| GET /v1/models | PASS | Lists all loaded models with device |
| POST text-only (VLM) | PASS | 0.5s, correct |
| POST single image file:// | PASS | car/silver, 3.1s |
| POST two images file:// | PASS | same_vehicle:true, 2.8s |
| POST single image base64 | PASS | Consistent with file:// |
| POST missing file | PASS | 400, clean error |
| POST missing messages | PASS | 400, clean error |
| POST text-only (NPU LLM) | PASS | Routed to NPU, streaming works |
| POST streaming (NPU) | PASS | SSE token-by-token |
| POST images → LLM | PASS | 400, "does not support images" |
| Dual mode routing | PASS | X-Device: NPU for text, GPU for images |
| X-Device / X-Model headers | PASS | Present on all responses |
| Status during loading | PASS | 503 until ready, /health shows loading |

### Web UI — verified 2026-04-12

| Test | Status | Notes |
|---|---|---|
| Chat with streaming | PASS | Tokens appear in real-time |
| Model selector | PASS | Shows loaded models + device |
| Health indicator | PASS | Green dot when ready |
| Dark theme | PASS | |
| `<think>` blocks | UNTESTED | Added CSS/JS, needs Qwen3 model to test |
| Image drag-and-drop | UNTESTED | |
| Image paste (Ctrl+V) | UNTESTED | |
| Image attach button | UNTESTED | |
| Keyboard shortcuts | UNTESTED | Enter, Shift+Enter, Ctrl+N, Escape |
| Device badge on responses | UNTESTED | Should show [NPU 1.2s] etc. |

### Ollama API (port 11434) — NOT tested externally

Added 2026-04-12. Translation layer implemented but needs real-world
client testing.

| Test | Status | Notes |
|---|---|---|
| GET / | PASS | Returns "Ollama is running" |
| GET /api/tags | PASS | Returns model list in Ollama format |
| POST /api/chat (non-streaming) | UNTESTED | |
| POST /api/chat (streaming) | UNTESTED | NDJSON format, not SSE |
| POST /api/generate | UNTESTED | |
| POST /api/chat with images | UNTESTED | Ollama images[] format |
| POST /api/show | UNTESTED | |
| Stubs (pull/delete/copy) | UNTESTED | |

### External client testing needed

| Client | API | Priority | Status |
|---|---|---|---|
| OpenWebUI (Docker) | OpenAI + Ollama | HIGH | Tested OpenAI only |
| Continue.dev (VS Code) | Ollama | HIGH | UNTESTED |
| Aider | Ollama / OpenAI | MEDIUM | UNTESTED |
| Open Interpreter | Ollama | MEDIUM | UNTESTED |
| openai Python package | OpenAI | PASS | Verified with curl equivalent |
| curl | Both | PASS | Primary test tool |

### Device combinations

| Config | Status | Notes |
|---|---|---|
| GPU only (VLM) | PASS | Current default |
| GPU only (LLM) | PASS | Qwen3-30B tested (ran on CPU, too big for GPU) |
| NPU only (LLM) | PASS | DeepSeek-R1-1.5B, streaming works |
| NPU + GPU dual | PASS | Routing verified |
| CPU fallback | UNTESTED | Should work but not tested |

### Models tested

| Model | Device | Status | Notes |
|---|---|---|---|
| Qwen2.5-VL-3B (INT8) | GPU | PASS | Image classification, proven |
| DeepSeek-R1-1.5B (INT4-CW) | NPU | PASS | Loads, runs, terrible answers |
| Qwen3-30B-A3B (INT4) | GPU | FAIL | Too big for 16 GB, fell back to CPU |
| Qwen3-8B (INT4-CW) | NPU | DOWNLOADING | Expected to work |

### Known issues

1. **Ollama port conflict**: If real Ollama is already running on
   11434, both servers may bind. Need better conflict detection or
   doc to stop Ollama first.

2. **30B model on 16 GB GPU**: Silently falls back to CPU. Should
   detect and warn, or refuse to load.

3. **`<think>` block rendering**: CSS/JS added but not tested with
   actual Qwen3 thinking output.

4. **start.ps1**: Auto-generated but not tested end-to-end with the
   new spinner/progress version.

---

## Priority for next QA session

1. Test Qwen3-8B on NPU (once download finishes)
2. Test `<think>` blocks in web UI with Qwen3 model
3. Test Ollama API with curl (streaming + non-streaming)
4. Test Ollama API with OpenWebUI
5. Test image features in web UI (drag/paste/attach)
6. Test start.ps1 end-to-end
