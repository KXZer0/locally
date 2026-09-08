"""Optional API client. No model runtime imports and no persistent chat history."""

import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import shlex
import tempfile
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Pure Python, no Flask and no OpenVINO: the same filter the Anthropic path
# uses, so reasoning is hidden the same way in both places rather than twice.
from core.voice.think_filter import _ThinkFilter


SYSTEM_PROMPT = (
    "Help with questions and homework. Explain the steps clearly and concisely. "
    "Use Markdown when useful, including fenced code blocks. "
    "Say when you are unsure; do not invent facts or sources."
)
COMMANDS = {
    "/help": "Show commands",
    "/models": "List local models; /models npu filters NPU candidates",
    "/load": "Load a model: /load name or path[@NPU|@GPU|@CPU]",
    "/unload": "Release models: /unload [NPU|GPU|CPU] (shared server operation)",
    "/status": "Show model state and available memory",
    "/think": "Reasoning before the answer: /think on|off (Qwen3 /no_think)",
    "/read": "Read a document, image or URL into the conversation: /read PATH",
    "/index": "Index local files for search: /index PATH [PATH ...]",
    "/find": "Search the last index: /find query",
    "/upscale": "Upscale an image: /upscale IN [OUT] (writes a PNG)",
    "/util": "Utility engine state; /util off unloads them, chat untouched",
    "/voice": "Talk instead of typing (needs sounddevice, Whisper and TTS)",
    "/new": "Clear conversation and last answer",
    "/copy": "Copy the last answer as original Markdown",
    "/save": "Save the last answer: /save path.md (never overwrites)",
    "/paste": "Insert clipboard text into the next editable prompt",
    "/markdown": "Switch display: /markdown on|off (exports always keep Markdown)",
    "/exit": "Leave chat; stop the server only if started with --start",
}


# Qwen3's control token, not an English instruction: asked in prose, the model
# opened a think block and spent its whole budget inside it. Appended to a COPY
# of the last user message -- the conversation must not grow control tokens,
# which would then be re-sent every turn and stored by /save.
NO_THINK = "/no_think"


class ApiError(Exception):
    pass


class Truncated(ApiError):
    """The answer is real, it just stopped at the token budget.

    Raised AFTER every token has been yielded, so the caller keeps the text it
    already has. Raising instead of yielding threw the whole answer away, and
    with --max-tokens defaulting to 1024 that is the ordinary end of a long
    answer, not a rare failure.
    """


class Client:
    def __init__(self, url, timeout=300):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def open(self, path, body=None, timeout=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = Request(self.url + path, data=data,
                      headers={"Content-Type": "application/json"})
        try:
            return urlopen(req, timeout=timeout or self.timeout)
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            try:
                error = json.loads(detail).get("error", detail)
                detail = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            except (ValueError, AttributeError):
                detail = f"HTTP {e.code} from {path}"
            raise ApiError(detail) from e
        except (URLError, OSError) as e:
            raise ApiError(f"Cannot reach {self.url}: {e}") from e

    def json(self, path, body=None, timeout=None):
        with self.open(path, body, timeout) as response:
            try:
                return json.load(response)
            except ValueError as e:
                raise ApiError(f"Invalid JSON from {path}") from e

    def upload(self, path, fields, files, timeout=300, raw=()):
        """POST a multipart body -- the shape every utility endpoint takes.

        `raw` carries (name, filename, bytes) for content that has no file on
        disk, which is what a microphone produces.
        """
        body, content_type = multipart(fields, files, raw)
        req = Request(self.url + path, data=body,
                      headers={"Content-Type": content_type})
        try:
            with urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            try:
                error = json.loads(detail).get("error", detail)
                detail = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            except (ValueError, AttributeError):
                detail = f"HTTP {e.code} from {path}"
            raise ApiError(detail) from e
        except (URLError, OSError) as e:
            raise ApiError(f"Cannot reach {self.url}: {e}") from e
        except ValueError as e:
            raise ApiError(f"Invalid JSON from {path}") from e

    def transcribe(self, wav, language=None):
        """Spoken audio in, text out. The endpoint takes an OpenAI upload."""
        fields = {"language": language} if language else {}
        return str(self.upload("/v1/audio/transcriptions", fields,
                               [], raw=[("file", "turn.wav", wav)]).get("text", "")).strip()

    def speech(self, text, voice=None):
        """(pcm, rate) for one clip.

        Raw int16 rather than WAV: the sample rate comes back in a header, so
        there is nothing to parse before it can go to the sound card.
        """
        body = {"input": text, "response_format": "pcm"}
        if voice:
            body["voice"] = voice
        with self.open("/v1/audio/speech", body) as response:
            rate = int(response.headers.get("X-Sample-Rate") or 24000)
            return response.read(), rate

    def cancel(self):
        """Ask the server to stop generating. Best effort, by construction:
        OpenVINO may be blocked in native code and never look at the flag, so
        the client stops reading either way."""
        try:
            self.json("/v1/cancel", {}, timeout=5)
        except ApiError:
            pass

    def stream(self, messages, model, max_tokens):
        body = {"messages": messages, "model": model, "stream": True,
                "max_tokens": max_tokens}
        with self.open("/v1/chat/completions", body) as response:
            yield from read_stream(response)


def multipart(fields, files, raw=()):
    """Encode one multipart body with the stdlib.

    The utility endpoints take uploads, and this client has no `requests` --
    deliberately, since it must import in a bare venv to talk to a server.
    Twenty lines here beats a dependency on the chat path.
    """
    crlf = "\r\n"
    boundary = "----locally" + uuid.uuid4().hex
    out = bytearray()
    for name, value in fields.items():
        out += (f"--{boundary}{crlf}"
                f'Content-Disposition: form-data; name="{name}"{crlf}{crlf}'
                f"{value}{crlf}").encode("utf-8")
    for name, path in files:
        path = Path(path)
        out += (f"--{boundary}{crlf}"
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{path.name}"{crlf}'
                f"Content-Type: application/octet-stream{crlf}{crlf}").encode("utf-8")
        out += path.read_bytes() + crlf.encode("utf-8")
    for name, filename, blob in raw:
        out += (f"--{boundary}{crlf}"
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"{crlf}'
                f"Content-Type: application/octet-stream{crlf}{crlf}").encode("utf-8")
        out += blob + crlf.encode("utf-8")
    out += f"--{boundary}--{crlf}".encode("utf-8")
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def read_stream(response):
    """Consume OpenAI SSE, preserving text exactly and rejecting truncated turns."""
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            frame = json.loads(data)
        except ValueError as e:
            raise ApiError("Invalid JSON in the response stream") from e
        if "error" in frame:
            raise ApiError(str(frame["error"]))
        for choice in frame.get("choices", []):
            if choice.get("finish_reason") in ("error", "cancelled"):
                raise ApiError("Generation failed or was cancelled; partial answer was not saved.")
            content = choice.get("delta", {}).get("content")
            if content:
                yield content
            if choice.get("finish_reason") == "length":
                raise Truncated("Output limit reached; the answer above stops "
                                "mid-thought. Ask for less, or raise --max-tokens.")
    raise ApiError("Connection ended before the answer was complete.")


def clipboard(text=None):
    """Use the platform clipboard; user content is stdin, never shell code."""
    if os.name == "nt":
        command = [str(Path(os.environ.get("SystemRoot", "C:/Windows")) /
                       "System32/WindowsPowerShell/v1.0/powershell.exe"),
                   "-NoProfile", "-NonInteractive", "-Command",
                   "[Console]::InputEncoding=[Text.UTF8Encoding]::new();"
                   "[Console]::OutputEncoding=[Text.UTF8Encoding]::new();" +
                   ("Set-Clipboard -Value ([Console]::In.ReadToEnd())" if text is not None
                    else "[Console]::Write((Get-Clipboard -Raw))")]
    elif sys.platform == "darwin":
        command = ["pbcopy" if text is not None else "pbpaste"]
    elif shutil.which("wl-copy" if text is not None else "wl-paste"):
        command = ["wl-copy"] if text is not None else ["wl-paste", "--no-newline"]
    elif shutil.which("xclip"):
        command = ["xclip", "-selection", "clipboard"] + ([] if text is not None else ["-o"])
    else:
        raise ApiError("No clipboard tool available. Use /save, or paste in your terminal.")
    result = subprocess.run(command, input=text, text=True, encoding="utf-8",
                            capture_output=True, timeout=10,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise ApiError("Clipboard unavailable. Use /save or your terminal's copy shortcut.")
    return result.stdout


class OwnedServer:
    """Own only the child launched here; never stop an attached server."""
    def __init__(self, port, server_args):
        self.port, self.server_args = port, server_args
        self.process = self.log = None
        self.log_path = None

    def start(self):
        # Refuse an occupied port, including a service which is not locally.
        # The raw bind error is "[WinError 10048] Only one usage of each socket
        # address ... is normally permitted", which names neither the port nor
        # the thing to do about it -- and the usual cause is a server the user
        # already has running, i.e. exactly what they wanted to talk to.
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", self.port))
        except OSError as e:
            raise ApiError(
                f"Port {self.port} is already in use. If that is your locally "
                f"server, attach to it instead: drop --start (or pass "
                f"--port <other> to run a second one).") from e
        self.log = tempfile.NamedTemporaryFile(prefix="locally-server-", suffix=".log", delete=False)
        self.log_path = self.log.name
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "locally.py"),
                   *self.server_args, "--port", str(self.port), "--host", "127.0.0.1",
                   "--ollama-port", "0", "--no-odysseus-autostart"]
        self.process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt")

    def close(self):
        try:
            if self.process is not None and self.process.poll() is None:
                if os.name == "nt":
                    # Windows venv launchers can own a second Python process.
                    # Stop the exact owned tree so its model process cannot survive.
                    taskkill = str(Path(os.environ.get("SystemRoot", "C:/Windows")) /
                                   "System32/taskkill.exe")
                    subprocess.run([taskkill, "/PID", str(self.process.pid), "/T", "/F"],
                                   capture_output=True, timeout=10,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(self.process.pid, signal.SIGKILL)
                    else:
                        self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            if self.log is not None:
                self.log.close()


def wait_ready(client, console, limit=300):
    """Hold the prompt until a model can actually answer.

    The socket binds within a couple of seconds while the models compile on
    background threads, so a chat session opened at that moment gets its first
    turn refused with "Server not ready (status: loading)" -- and a scripted or
    piped turn is simply lost. Bounded, because a load that never finishes must
    not become a client that never returns: after the limit the prompt opens
    anyway and /status says where it got to.
    """
    deadline = time.monotonic() + limit
    try:
        if client.json("/health", timeout=5).get("status") != "loading":
            return
    except ApiError:
        return
    with console.status("Loading the model… Ctrl+C to skip the wait"):
        while time.monotonic() < deadline:
            time.sleep(0.5)
            try:
                if client.json("/health", timeout=5).get("status") != "loading":
                    return
            except ApiError:
                return


def show_status(client, console):
    from rich.table import Table
    health = client.json("/health", timeout=5)
    memory = client.json("/v1/memory", timeout=5)
    table = Table(title="locally  /  " + str(health.get("status", "unknown")), expand=True)
    for label in ("Device", "Model", "State", "Context", "Last TTFT"):
        table.add_column(label)
    for name, slot in health.get("devices", {}).items():
        ttft = slot.get("last_ttft_ms")
        table.add_row(name.upper(), str(slot.get("model", "—")), str(slot.get("status", "—")),
                      str(slot.get("context_tokens") or "—"),
                      f"{ttft / 1000:.2f}s" if ttft is not None else "—")
    console.print(table)
    system = memory.get("system", {})
    available, total = system.get("available_mb"), system.get("total_mb")
    if available is not None and total:
        console.print(f"RAM available: {available / 1024:.1f} / {total / 1024:.1f} GiB")
    console.print(memory.get("message", ""), markup=False)
    turns = client.json("/v1/metrics", timeout=5)
    if turns:
        last = turns[-1]
        speed = last.get("decode_tokens_per_second")
        duration = last.get("total_ms")
        console.print("Last recorded turn: " + str(last.get("model", "unknown")) +
                      (f" · {duration / 1000:.1f}s" if duration is not None else "") +
                      (f" · ~{speed:.1f} decode tok/s" if speed is not None else ""), markup=False)


class Chat:
    def __init__(self, client, console, model="", max_tokens=1024, markdown=True):
        self.client, self.console = client, console
        self.model, self.max_tokens, self.markdown = model, max_tokens, markdown
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_answer = ""
        self.paste = ""
        # None means "whatever the model does by default". Only /think off
        # sends anything, because the token is Qwen3's and a model that does
        # not know it would just read it as text.
        self.think = None
        self.index_id = None
        # Utility engines compile on first use and then sit resident. Nothing
        # in a terminal says "you have left the OCR tab", so the client
        # remembers it started them and offers to put them back.
        self.util_used = False

    def command(self, text):
        name, _, arg = text.partition(" ")
        arg = arg.strip().strip('"')
        if name in ("/exit", "/quit"):
            return False
        if name == "/help":
            for cmd, description in COMMANDS.items():
                self.console.print(f"{cmd:12} {description}", markup=False)
        elif name == "/new":
            self.messages = self.messages[:1]
            self.last_answer = ""
            self.console.print("Conversation cleared.")
        elif name == "/status":
            show_status(self.client, self.console)
        elif name == "/models":
            from rich.table import Table
            result = self.client.json("/v1/models/available")
            table = Table("Model", "Type", "Loaded on", "Availability", expand=True)
            for model in result.get("data", []):
                if arg.lower() == "npu" and not model.get("npu", {}).get("compatible"):
                    continue
                # "reason" is only set for a file that cannot be loaded at
                # all; the NPU verdict is what the rest have, and computing it
                # and then not showing it is how /models npu looked arbitrary.
                npu = model.get("npu") or {}
                table.add_row(model["name"], model.get("type", ""), model.get("loaded_on") or "—",
                              model.get("reason") or
                              (f"NPU: {npu['reason']}" if npu.get("reason") else
                               "Placement checked on /load"))
            self.console.print(table)
            if arg.lower() == "npu":
                self.console.print("NPU candidates pass local capability checks; compilation is verified on load.")
        elif name == "/load":
            if not arg:
                raise ApiError("Usage: /load model-name-or-path[@NPU|@GPU|@CPU]")
            with self.console.status("Loading model…"):
                loaded = self.client.json("/v1/models/load", {"model": arg})
            self.model = f"{loaded['model']}@{loaded['device']}"
            self.messages = self.messages[:1]
            self.last_answer = ""
            self.console.print(f"Loaded {self.model}. Conversation cleared.", markup=False)
            self.console.print(loaded.get("placement", ""), markup=False)
        elif name == "/unload":
            if arg and arg.upper() not in ("NPU", "GPU", "CPU"):
                raise ApiError("Usage: /unload [NPU|GPU|CPU]")
            with self.console.status("Releasing model memory…"):
                result = self.client.json("/v1/models/unload", {"device": arg.upper()})
            self.console.print(f"Unloaded {len(result.get('unloaded', []))} slot(s). "
                               "The API stays up; another request may reload them.")
        elif name == "/think":
            if arg not in ("on", "off"):
                raise ApiError("Usage: /think on|off")
            self.think = arg == "on"
            self.console.print(
                "Reasoning allowed. It is still hidden from the answer."
                if self.think else
                "Reasoning suppressed with the /no_think control token. That "
                "token is Qwen3's: a model that does not know it reads it as text.")
        elif name == "/read":
            if not arg:
                raise ApiError("Usage: /read PATH  or  /read https://...")
            with self.console.status("Reading..."):
                result = self.read_document(arg)
            markdown = result.get("markdown", "")
            if not markdown.strip():
                raise ApiError("Nothing readable came back from that.")
            # Into the conversation, not just onto the screen: reading a
            # document in a chat client is only useful if the next question can
            # be about it. Shown truncated, sent whole.
            self.messages.append({"role": "user",
                                  "content": f"Document: {arg}\n\n{markdown}"})
            where = result.get("engine") or "local parser"
            self.console.print(f"Read {len(markdown)} chars via {where}; added to "
                               "the conversation. Ask about it.", markup=False)
            preview = markdown[:700] + ("..." if len(markdown) > 700 else "")
            self.console.print(preview, markup=False, style="dim")
        elif name == "/index":
            paths = [Path(part).expanduser() for part in shlex.split(arg)] if arg else []
            if not paths:
                raise ApiError("Usage: /index PATH [PATH ...]")
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise ApiError("Not a file: " + ", ".join(missing))
            self.wait_engines()
            with self.console.status(f"Indexing {len(paths)} file(s)..."):
                result = self.client.upload("/v1/util/index", {},
                                            [("files", path) for path in paths])
            self.index_id = result.get("index_id")
            self.util_used = True
            self.console.print(f"Indexed {result.get('files', len(paths))} file(s), "
                               f"{result.get('chunks', '?')} chunks. Search with /find.")
        elif name == "/find":
            if not arg:
                raise ApiError("Usage: /find query")
            if not self.index_id:
                raise ApiError("Nothing indexed in this session. Use /index PATH first.")
            self.wait_engines()
            with self.console.status("Searching..."):
                result = self.client.json("/v1/util/search",
                                          {"index_id": self.index_id, "query": arg})
            self.util_used = True
            hits = result.get("results") or []
            if not hits:
                self.console.print("No matches.")
            for hit in hits:
                # Both numbers, deliberately: a reranked list shows a lower
                # cosine above a higher one, which reads as a bug without the
                # score the sort actually used.
                rerank = hit.get("rerank_score")
                label = (f"rerank {rerank:.3f}" if rerank is not None
                         else f"score {hit.get('score', 0):.3f}")
                self.console.print(f"{hit.get('source', '?')}  ({label})",
                                   style="dim", markup=False)
                self.console.print(str(hit.get("text", "")).strip()[:400], markup=False)
        elif name == "/upscale":
            parts = shlex.split(arg) if arg else []
            if not parts:
                raise ApiError("Usage: /upscale IN [OUT]")
            source = Path(parts[0]).expanduser()
            if not source.is_file():
                raise ApiError(f"Not a file: {source}")
            target = (Path(parts[1]).expanduser() if len(parts) > 1
                      else source.with_name(source.stem + "-upscaled.png"))
            if target.exists():
                raise ApiError(f"{target} exists; name a different output file.")
            self.wait_engines()
            with self.console.status("Upscaling..."):
                result = self.client.upload("/v1/util/upscale", {}, [("file", source)])
            self.util_used = True
            data = str(result.get("image", ""))
            if "," not in data:
                raise ApiError("The server returned no image.")
            target.write_bytes(base64.b64decode(data.split(",", 1)[1]))
            self.console.print(f"{result.get('scale', '?')}x via "
                               f"{result.get('engine', '?')} -> {target.resolve()}",
                               markup=False)
        elif name == "/util":
            if arg == "off":
                # Nothing can be unloaded mid-compile; the server answers such
                # a request with "busy", which reads as a fault rather than as
                # a wait.
                self.wait_engines()
                with self.console.status("Releasing utility engines..."):
                    result = self.client.json("/v1/models/unload", {"scope": "util"})
                self.util_used = False
                freed = result.get("unloaded") or []
                self.console.print(
                    f"Unloaded {len(freed)} utility engine(s); the chat model is "
                    "untouched. They compile again on next use." if freed else
                    "No utility engine was loaded.")
            elif arg:
                raise ApiError("Usage: /util  or  /util off")
            else:
                self.show_utilities()
        elif name == "/voice":
            self.voice_session()
        elif name in ("/copy", "/save"):
            if not self.last_answer:
                raise ApiError("No complete answer yet.")
            if name == "/copy":
                clipboard(self.last_answer)
                self.console.print("Copied original Markdown.")
            else:
                if not arg:
                    raise ApiError("Usage: /save path.md")
                path = Path(arg).expanduser()
                if path.suffix.lower() != ".md":
                    raise ApiError("Use a .md filename.")
                with path.open("x", encoding="utf-8", newline="") as output:
                    output.write(self.last_answer)
                self.console.print(f"Saved {path.resolve()}", markup=False)
        elif name == "/paste":
            self.paste = clipboard()
            self.console.print("Clipboard inserted below. Edit it, then press Enter to send.")
        elif name == "/markdown" and arg in ("on", "off"):
            self.markdown = arg == "on"
            self.console.print(f"Markdown display {arg}; copy/save always preserve the source.")
        else:
            raise ApiError("Unknown command or option. Use /help.")
        return True

    def wait_engines(self, limit=180):
        """Hold a utility command while the engines are still compiling.

        They load after the chat model and compile on first use, so a command
        typed in the first seconds of a server's life used to fail outright.
        Bounded, and it only ever waits on "loading" -- an engine that is
        genuinely absent returns immediately and the command says so.
        """
        deadline = time.monotonic() + limit
        first = True
        while time.monotonic() < deadline:
            try:
                util = self.client.json("/health", timeout=10).get("util") or {}
            except ApiError:
                return
            loading = [name for name, info in util.items()
                       if isinstance(info, dict)
                       and info.get("status") in ("loading", "warming_up")]
            if not loading or not util:
                return
            if first:
                self.console.print("Utility engines are compiling; waiting.",
                                   style="dim")
                first = False
            time.sleep(1.0)

    def voice_session(self):
        """Talk instead of typing, until an empty turn or Ctrl+C.

        Push-to-talk: Enter opens the microphone, Enter closes it. A terminal
        cannot see a key being RELEASED, so hold-to-talk is not on the table --
        and the server's VAD, which could take turns on its own, needs the
        live socket this deliberately does without.
        """
        from core import terminal_voice as voice
        audio = voice.import_audio()          # raises with the pip line
        health = self.client.json("/health", timeout=10)
        missing = [n for n, k in (("Whisper (--whisper-dir)", "whisper"),
                                  ("a TTS voice (--tts-dir)", "tts"))
                   if k not in health]
        if missing:
            raise ApiError("Voice needs " + " and ".join(missing) +
                           " on the server. The rest of the client still works.")
        # Both models resident before the first turn: idle-unload is doing its
        # job when it evicts them, but the reload then lands INSIDE the user's
        # first spoken turn and reads as "voice is slow".
        with self.console.status("Warming the voice models..."):
            try:
                self.client.json("/v1/audio/warm", {}, timeout=180)
            except ApiError:
                pass                          # warming is an optimisation
        self.console.print("Voice: Enter starts and stops recording, empty "
                           "Enter leaves, Ctrl+C interrupts.")
        while True:
            try:
                input("talk > ")
            except (EOFError, KeyboardInterrupt):
                self.console.print("Leaving voice.")
                return
            with voice.Recorder(audio) as take:
                try:
                    stop = input("  recording... Enter to stop > ")
                except (EOFError, KeyboardInterrupt):
                    stop = ""
            pcm = take.pcm
            seconds = len(pcm) / (voice.SAMPLE_RATE * 2)
            if seconds < 0.4:
                self.console.print("  too short to transcribe; leaving voice."
                                   if not pcm else "  too short; try again.")
                if not pcm:
                    return
                continue
            with self.console.status(f"Transcribing {seconds:.1f}s..."):
                said = self.client.transcribe(voice.wav_bytes(pcm))
            if not said:
                self.console.print("  nothing recognised.")
                continue
            self.console.print(f"you said: {said}", markup=False)
            self.speak_turn(said, audio, voice)
            if stop.strip().lower() in ("/exit", "exit", "/quit"):
                return

    def speak_turn(self, text, audio, voice):
        """One spoken turn: stream, split into clips, speak as they arrive."""
        directive = {"role": "system", "content": voice.VOICE_PROMPT}
        history = [*self.messages, {"role": "user", "content": text}]
        # The directive goes LAST so it wins over the general prompt, and it is
        # not kept: it applies to spoken turns only, and the same conversation
        # continues in text afterwards.
        outgoing = [*history, directive]
        if self.think is not True:
            # Reasoning must never be spoken, and a small model asked to think
            # can spend the whole voice budget doing it and answer nothing.
            outgoing[-2] = {"role": "user", "content": text + " " + NO_THINK}
        think = _ThinkFilter()
        spoken, buffer, answer, first = [], "", "", True
        queue_ = voice.SpeechQueue(audio, self.client, lambda line: spoken.append(line))
        stream = self.client.stream(outgoing, self.model, voice.VOICE_MAX_TOKENS)
        try:
            for token in stream:
                visible = think.feed(token)
                if not visible:
                    continue
                answer += visible
                buffer += visible
                while True:
                    clip, buffer = voice.split_speakable(buffer, first)
                    if not clip:
                        break
                    first = False
                    queue_.say(clip)
        except Truncated:
            pass
        except KeyboardInterrupt:
            queue_.cancel()
            self.client.cancel()
            self.console.print("  interrupted.")
            return
        finally:
            stream.close()
        buffer += think.flush()
        if buffer.strip():
            queue_.say(buffer)
        try:
            queue_.finish()
        except KeyboardInterrupt:
            queue_.cancel()
            self.console.print("  interrupted.")
            return
        except Exception as exc:
            queue_.cancel()
            raise ApiError(f"Speech failed: {exc}") from exc
        # Printed from the playback side, so the transcript on screen is what
        # was actually said out loud.
        for line in spoken:
            self.console.print(f"  {line}", markup=False)
        if answer.strip():
            # Voice and chat share one conversation: switching modes must never
            # drop the thread, and the spoken turn is part of it.
            self.messages = [*history, {"role": "assistant", "content": answer.strip()}]
            self.last_answer = answer.strip()

    def read_document(self, target):
        """A URL is JSON, a local file is an upload -- the endpoint takes both."""
        if target.lower().startswith(("http://", "https://")):
            return self.client.json("/v1/util/read", {"url": target})
        path = Path(target).expanduser()
        if not path.is_file():
            raise ApiError(f"Not a file: {path}")
        self.util_used = True
        return self.client.upload("/v1/util/read", {}, [("file", path)])

    def show_utilities(self):
        """What each engine can do, and whether it is resident.

        /health carries a reason for every task that is off -- not installed,
        deliberately disabled, or this engine cannot serve it -- and each needs
        a different action from the user, so the reason is printed rather than
        flattened into "unavailable".
        """
        from rich.table import Table
        util = self.client.json("/health", timeout=10).get("util") or {}
        if not util:
            self.console.print("No utility engines on this server. Start it with "
                               "--util-models-dir or --auto-util.")
            return
        table = Table("Engine", "State", "Can do", "Cannot, and why", expand=True)
        for engine, info in util.items():
            if not isinstance(info, dict):
                continue
            tasks = info.get("tasks") or {}
            ready = sorted(k for k, v in tasks.items() if v) if isinstance(tasks, dict) else []
            # Three different states -- not installed, deliberately off, and
            # this engine cannot serve it -- each needing a different action.
            # Printing the raw dict said all three at once and none of them.
            off = info.get("disabled") or {}
            why = ("; ".join(f"{k}: {v}" for k, v in sorted(off.items()))
                   if isinstance(off, dict) else str(off))
            table.add_row(str(engine).upper(), str(info.get("status", "?")),
                          ", ".join(ready) or "-", why or "-")
        self.console.print(table)
        self.console.print("Engines stay resident after use; /util off frees them "
                           "without touching the chat model.", style="dim")

    def ask(self, text):
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.text import Text
        from rich.spinner import Spinner
        # Two lists on purpose. `history` is what the conversation becomes;
        # `messages` is what goes out this turn. Building history from the
        # outgoing list instead put the control token straight back into it --
        # where it is re-sent every turn and saved by /save. Found by the test,
        # not by reading the code, which is why the test names the property
        # rather than the mechanism.
        history = [*self.messages, {"role": "user", "content": text}]
        messages = list(history)
        if self.think is False:
            messages[-1] = {"role": "user", "content": text + " " + NO_THINK}
        # A thinking model writes its reasoning into the same token stream, and
        # the OpenAI path does not strip it (only the Anthropic one does). Read
        # aloud in a terminal it is noise, and it would end up in /copy and
        # /save as if it were the answer.
        think = _ThinkFilter()
        answer, truncated = "", False
        started = time.monotonic()
        stream = self.client.stream(messages, self.model, self.max_tokens)
        try:
            with Live(Spinner("dots", text="Thinking…  Ctrl+C cancels this turn"),
                      console=self.console, transient=True, refresh_per_second=8) as live:
                for token in stream:
                    answer += think.feed(token)
                    # Limit repaint cost for long homework answers. Full text is
                    # printed once below and remains in terminal scrollback.
                    live.update(Text(answer[-3000:]) if answer else
                                Spinner("dots", text="Reasoning… (hidden)"))
        except Truncated:
            truncated = True
        except KeyboardInterrupt:
            # The spinner promises this, so it has to actually be asked for.
            self.client.cancel()
            raise
        finally:
            stream.close()
        answer += think.flush()
        if not answer.strip() and think.discarded:
            # The whole budget went inside an unclosed <think>. Showing the
            # reasoning beats showing nothing — same call the Anthropic path
            # makes, for the same reason.
            answer = think.discarded
        if not answer:
            raise ApiError("The model returned an empty answer.")
        self.messages = [*history, {"role": "assistant", "content": answer}]
        self.last_answer = answer
        self.console.print(Markdown(answer) if self.markdown else Text(answer))
        self.console.print(f"{time.monotonic() - started:.1f}s  ·  /copy  /save answer.md", style="dim")
        if truncated:
            self.console.print(f"Stopped at the {self.max_tokens}-token budget; the answer is "
                               "incomplete. Restart with a larger --max-tokens.", style="dim")


def make_prompt_session(fetch_models=None, **kwargs):
    """The prompt, its key bindings, and the slash completer.

    `fetch_models` returns the server's /v1/models/available entries. It is a
    callable rather than a list because the answer changes underneath the
    session: a model can be downloaded, deleted or renamed while the client is
    open, and nothing here may keep its own idea of which models exist.
    Omitted -- in tests, and before a server is reachable -- model completions
    are simply absent, never stale or invented names.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion, PathCompleter
    from prompt_toolkit.document import Document
    from prompt_toolkit.key_binding import KeyBindings
    bindings = KeyBindings()
    # The utility commands take a path, and typing one out in full is exactly
    # the thing a terminal is supposed to save you.
    paths = PathCompleter(expanduser=True)

    class SlashCompleter(Completer):
        def __init__(self):
            self._models, self._fetched_at = [], None

        def models(self):
            # complete_while_typing fires on every keystroke and the endpoint
            # walks the model directories, so the answer is cached briefly. A
            # failed fetch keeps the last good list and never raises: an
            # exception here takes the prompt down with it.
            if fetch_models is None:
                return []
            now = time.monotonic()
            if self._fetched_at is None or now - self._fetched_at > 15:
                self._fetched_at = now
                try:
                    self._models = list(fetch_models())
                except Exception:
                    pass
            return self._models

        def load_completions(self, argument):
            """Model names for /load, from the server, ranked prefix-first.

            Substring, not prefix: the useful thing to type is "abl", and that
            sits in the middle of Qwen3-8B-abliterated-int4-cw. `name@DEVICE`
            is a preference the server may override, so the devices offered are
            the three it accepts, not the ones a slot happens to hold.
            """
            name, at, device = argument.rpartition("@")
            if at:
                for kind in ("NPU", "GPU", "CPU"):
                    if kind.startswith(device.upper()):
                        yield Completion(kind, start_position=-len(device))
                return
            needle = argument.lower()
            matches = [m for m in self.models()
                       if needle in str(m.get("name", "")).lower()]
            matches.sort(key=lambda m: (not str(m["name"]).lower().startswith(needle),
                                        str(m["name"]).lower()))
            for model in matches:
                # Why a model cannot be loaded matters more than its type: it
                # is the one case where /load is going to refuse.
                meta = model.get("reason") or " - ".join(filter(None, (
                    model.get("type"),
                    f"loaded on {model['loaded_on']}" if model.get("loaded_on") else None,
                )))
                yield Completion(model["name"], start_position=-len(argument),
                                 display_meta=meta or None)

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/") or "\n" in document.text:
                return
            command, separator, argument = text.partition(" ")
            if not separator:
                for name, description in COMMANDS.items():
                    if name.startswith(command):
                        yield Completion(name, start_position=-len(command),
                                         display_meta=description)
            elif command == "/load":
                yield from self.load_completions(argument)
            elif command in ("/read", "/index", "/upscale"):
                # Only the token under the cursor: /index takes several paths.
                tail = argument.rsplit(" ", 1)[-1]
                if not tail.lower().startswith(("http://", "https://")):
                    yield from paths.get_completions(Document(tail, len(tail)),
                                                     complete_event)
            else:
                choices = {"/models": ("npu",), "/unload": ("NPU", "GPU", "CPU"),
                           "/markdown": ("on", "off"), "/think": ("on", "off"),
                           "/util": ("off",)}
                for value in choices.get(command, ()):
                    if value.lower().startswith(argument.lower()):
                        yield Completion(value, start_position=-len(argument))

    @bindings.add("escape", "enter")
    def newline(event):
        event.current_buffer.insert_text("\n")

    return PromptSession(key_bindings=bindings, completer=SlashCompleter(),
                         complete_while_typing=True, reserve_space_for_menu=5,
                         mouse_support=False, **kwargs)


def main(argv=None):
    # Windows redirected streams otherwise use the legacy code page, which
    # cannot carry homework symbols or reliably interoperate with UTF-8 pipes.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and not stream.isatty():
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Locally terminal client; Odysseus can use the same API.")
    parser.add_argument("command", choices=("chat", "status"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--url", help="Server origin, e.g. http://127.0.0.1:8000 (without /v1)")
    parser.add_argument("--start", action="store_true", help="Start and own a local server for this chat")
    parser.add_argument("--model", default="", help="Loaded model ID to request")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--plain", action="store_true", help="Display raw Markdown without colors")
    args, server_args = parser.parse_known_args(argv)
    if server_args and server_args[0] == "--":
        server_args = server_args[1:]
    if server_args and not args.start:
        parser.error("Server flags require --start; e.g. chat --start --device NPU")
    if args.start and (args.url or args.command != "chat"):
        parser.error("--start is for local chat only; omit --url")
    if not 1 <= args.port <= 65535 or args.max_tokens < 1:
        parser.error("Use a valid port and a positive --max-tokens")
    try:
        from rich.console import Console
        from rich.panel import Panel
        from prompt_toolkit import PromptSession
    except ImportError:
        print("Install terminal dependencies: python -m pip install rich prompt-toolkit", file=sys.stderr)
        return 1
    console = Console(no_color=args.plain, markup=False, highlight=False)
    client = Client(args.url or f"http://127.0.0.1:{args.port}")
    owned = OwnedServer(args.port, server_args) if args.start else None
    try:
        if owned:
            owned.start()
            console.print(f"Server log: {owned.log_path}")
            with console.status("Starting API…"):
                deadline = time.monotonic() + 30
                while True:
                    if owned.process.poll() is not None:
                        raise ApiError(f"Server exited; inspect {owned.log_path}")
                    try:
                        identity = client.json("/", timeout=1)
                    except ApiError:
                        # Only "not answering yet" is worth waiting out; a
                        # wrong answer is final and needs its own sentence.
                        if time.monotonic() >= deadline:
                            raise ApiError(f"API did not start in 30s; inspect {owned.log_path}")
                        time.sleep(0.2)
                        continue
                    if identity.get("service") != "locally":
                        raise ApiError("A different service answered on this port.")
                    break
        if args.command == "chat":
            wait_ready(client, console)
        show_status(client, console)
        if args.command == "status":
            return 0
        ownership = ("Owned server · /exit stops it and releases its models" if owned else
                     "Attached server · /exit leaves it running for other clients")
        console.print(Panel(f"{client.url}/v1\n{ownership}\n"
                            "/load and /unload change the models available to all connected clients.\n"
                            "Enter sends · Alt+Enter adds a line · / suggests commands · Tab completes\n"
                            "Paste text with Ctrl+V (some terminals use Ctrl+Shift+V).",
                            title="locally", border_style="dim"))
        def available_models():
            return client.json("/v1/models/available", timeout=5).get("data", [])

        interactive = sys.stdin.isatty() and sys.stdout.isatty()
        session = (make_prompt_session(fetch_models=available_models)
                   if interactive else None)
        chat = Chat(client, console, args.model, args.max_tokens, not args.plain)
        while True:
            try:
                pasted = bool(chat.paste)
                text = session.prompt("you › ", default=chat.paste) if session else input("you > ")
                chat.paste = ""
            except (KeyboardInterrupt, EOFError):
                break
            if not text.strip():
                continue
            try:
                # A pasted assignment is content, even if it begins with '/'.
                if text.startswith("/") and "\n" not in text and not pasted:
                    if text.strip() == "/paste" and session is None:
                        raise ApiError("/paste needs an interactive terminal for editing. Pipe text directly instead.")
                    if not chat.command(text.strip()):
                        break
                else:
                    chat.ask(text)
            except KeyboardInterrupt:
                console.print("Turn interrupted; not added to history. Ctrl+C at the prompt exits.")
            except (ApiError, OSError, ValueError, subprocess.SubprocessError) as e:
                console.print(str(e), style="red")
        return 0
    except (ApiError, OSError, subprocess.SubprocessError) as e:
        console.print(str(e), style="red")
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if owned:
            started = owned.process is not None
            owned.close()
            # Only claim a stop for a server that was actually started: a
            # refused port never spawned one, and saying otherwise invites
            # the belief that something of the user's was just killed.
            if started:
                console.print("Owned server stopped.")
