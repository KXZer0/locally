"""Audio HTTP and WebSocket surface."""

import io
import json
import time
from datetime import datetime

import numpy as np
from flask import Blueprint, Response, jsonify, request

try:
    import soundfile as sf
except ImportError:
    sf = None

from core import runtime
from core.errors import openai_error
from core.slots.select import _slot_serviceable
from core.slots.speaker import SpeakerSlot
from core.slots.vad import VadSlot
from core.voice.session import VoiceStreamSession

bp = Blueprint("audio", __name__)
stream_available = False

def _load_audio(file_storage):
    """Read uploaded audio file to float32 numpy array at 16 kHz."""
    if sf is None:
        raise RuntimeError("soundfile not installed. pip install soundfile")
    audio, sr = sf.read(io.BytesIO(file_storage.read()), dtype="float32")
    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Resample to 16 kHz if needed
    if sr != 16000:
        target_len = int(len(audio) * 16000 / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio


def register_socket(app):
    """Wire up the VAD socket, if flask-sock is installed.

    Optional on purpose: the socket only carries auto turn-taking, and voice
    already degrades to push-to-talk without it. A missing dependency should
    cost that feature, not the server.
    """
    try:
        from flask_sock import Sock
    except ImportError:
        return False

    sock = Sock(app)

    @sock.route("/v1/audio/stream")
    def audio_stream(ws):                       # noqa: ANN001  (flask-sock API)
        if not runtime._vad_slot or not _slot_serviceable(runtime._vad_slot):
            ws.send(json.dumps({"type": "error", "message": "No VAD model loaded."}))
            return
        try:
            runtime._vad_slot.ensure_loaded()
        except Exception as e:
            ws.send(json.dumps({"type": "error", "message": f"VAD load failed: {e}"}))
            return

        session = VoiceStreamSession()
        while True:
            msg = ws.receive()
            if msg is None:
                break
            if isinstance(msg, (bytes, bytearray)):
                pcm = np.frombuffer(msg, dtype="<i2").astype(np.float32) / 32768.0
                for i in range(0, len(pcm) - VadSlot.FRAME + 1, VadSlot.FRAME):
                    for ev in session.feed(pcm[i:i + VadSlot.FRAME]):
                        ws.send(json.dumps(ev))
                # Transcription finishes on a worker thread; it is handed back
                # here so every frame on this socket has exactly one writer.
                for ev in session.drain():
                    ws.send(json.dumps(ev))
                continue
            try:
                cfg = json.loads(msg)
            except (ValueError, TypeError):
                continue
            if cfg.get("type") == "config":
                session.threshold = float(cfg.get("threshold", session.threshold))
                session.patience_ms = float(cfg.get("patienceMs", session.patience_ms))
                session.language = cfg.get("language") or ""
                session.enabled = bool(cfg.get("enabled", True))
            elif cfg.get("type") == "state":
                session.voice_state = cfg.get("voice", "idle")
                # Leaving the answer behind clears any half-built interruption.
                if session.voice_state != "speaking":
                    session.speech_ms = 0.0
        return ""

    return True


def attach_socket(app):
    """Attach the optional VAD socket after the Flask app exists."""
    global stream_available
    stream_available = register_socket(app)
    return stream_available


@bp.route("/v1/audio/warm", methods=["POST"])
def audio_warm():
    """Bring the audio models resident before the first turn needs them.

    Idle-unload is doing its job when it evicts the ASR and TTS models, but
    the cost of the reload (measured ~5.9 s ASR, ~8.7 s TTS) then lands inside
    the user's first spoken turn, where it reads as "voice mode is slow". The
    web UI calls this when the Voice tab is opened, which is the earliest
    moment we know a turn is coming.

    Best-effort: a slot that fails to load is reported, not raised, because a
    warm-up failing must not stop someone from trying to talk.
    """
    out = {}
    # "whisper" is the ASR slot's historical key; "asr" names it honestly
    # now that it may hold Parakeet. Both are reported for the same slot.
    for key, slot in (("asr", runtime.whisper_slot), ("whisper", runtime.whisper_slot),
                      ("tts", runtime.tts_slot), ("vad", runtime._vad_slot)):
        if not slot or not _slot_serviceable(slot):
            out[key] = "unavailable"
            continue
        t0 = time.perf_counter()
        try:
            slot.ensure_loaded()
            out[key] = {"ready": True, "ms": round((time.perf_counter() - t0) * 1000)}
        except Exception as e:
            out[key] = {"ready": False, "error": str(e)}
    return jsonify(out)


@bp.route("/v1/audio/enroll", methods=["GET", "POST", "DELETE"])
def audio_enroll():
    """Manage the enrolled voice profile.

    GET returns the profile's state, POST folds one more utterance into it,
    DELETE forgets it. Enrolment is additive on purpose: a profile built from
    several short recordings, in whatever voice the room actually produces,
    generalises better than one careful reading. The client is told how many
    more seconds it needs rather than being given a pass/fail, so the UI can
    show progress instead of a mystery.
    """
    if not runtime._speaker_slot or runtime._speaker_slot.status == "not_configured":
        return openai_error(
            "No speaker-identification model loaded. Use --speaker-dir.",
            "server_error", 503)

    if request.method == "GET":
        return jsonify(runtime._speaker_slot.profile_state)

    if request.method == "DELETE":
        runtime._speaker_slot.clear_profile()
        print(f"{datetime.now():%H:%M:%S} -- [speaker] profile cleared", flush=True)
        return jsonify(runtime._speaker_slot.profile_state)

    try:
        runtime._speaker_slot.ensure_loaded()
    except Exception as e:
        return openai_error(f"Failed to load speaker model: {e}", "server_error", 500)

    if "file" not in request.files:
        return openai_error("'file' is required (multipart form upload)")
    try:
        audio = _load_audio(request.files["file"])
    except Exception as e:
        return openai_error(f"Could not read audio: {e}")

    try:
        state = runtime._speaker_slot.enroll(audio)
    except ValueError as e:
        return openai_error(str(e))
    except Exception as e:
        return openai_error(f"Enrolment failed: {e}", "server_error", 500)

    print(f"{datetime.now():%H:%M:%S} -- [speaker] enrolled "
          f"{len(audio) / SpeakerSlot.SR:.1f}s "
          f"({state['seconds']:.0f}/{state['needed_seconds']:.0f}s, "
          f"{'ready' if state['ready'] else 'more needed'})", flush=True)
    return jsonify(state)


@bp.route("/v1/audio/verify", methods=["POST"])
def audio_verify():
    """Score one recording against the enrolled profile, without gating anything.

    Exists so the threshold can be set from evidence. Record yourself, record
    someone else, and the gap between the two cosines is what
    --speaker-threshold should sit in. Guessing a threshold and discovering it
    was wrong through a week of turns being dropped is the alternative.
    """
    if not runtime._speaker_slot or runtime._speaker_slot.status == "not_configured":
        return openai_error(
            "No speaker-identification model loaded. Use --speaker-dir.",
            "server_error", 503)
    if "file" not in request.files:
        return openai_error("'file' is required (multipart form upload)")
    try:
        runtime._speaker_slot.ensure_loaded()
        audio = _load_audio(request.files["file"])
    except Exception as e:
        return openai_error(f"Could not read audio: {e}")
    verdict, score = runtime._speaker_slot.verify(audio)
    return jsonify({
        "verdict": verdict,
        "score": round(score, 4) if score is not None else None,
        "threshold": runtime._speaker_slot.threshold,
        "seconds": round(len(audio) / SpeakerSlot.SR, 2),
        # Says WHY when there is no score, which is the difference between
        # "this is a stranger" and "I could not tell".
        "reason": (None if verdict != "abstain" else
                   ("no profile enrolled" if not runtime._speaker_slot.profile_ready
                    else f"needs at least {SpeakerSlot.MIN_VERIFY_S}s of audio")),
    })


_RESPONSE_FORMATS = ("json", "text", "verbose_json")


@bp.route("/v1/audio/transcriptions", methods=["POST"])
def audio_transcriptions():
    """OpenAI-compatible speech-to-text. Accepts multipart form with audio file.

    Served by whichever ASR engine --asr-dir loaded (Whisper or Parakeet);
    runtime.whisper_slot is the ASR slot whatever it holds. `language` is
    passed through: Whisper forces it, Parakeet has no language token and
    detects the language itself, so there it is accepted and reported back
    in verbose_json as a hint rather than silently pretended to apply.
    """
    slot = runtime.whisper_slot
    if not slot or not _slot_serviceable(slot):
        return openai_error(
            "No speech-to-text model loaded. Use --asr-dir (Parakeet or "
            "Whisper) or --whisper-dir.", "server_error", 503,
        )
    try:
        slot.ensure_loaded()   # reload if idle-unloaded or unloaded on request
    except Exception as e:
        return openai_error(f"Failed to load speech-to-text model: {e}",
                            "server_error", 500)

    if "file" not in request.files:
        return openai_error("'file' is required (multipart form upload)")

    audio_file = request.files["file"]
    language = (request.form.get("language") or "").strip() or None
    response_format = (request.form.get("response_format") or "json").lower()
    if response_format not in _RESPONSE_FORMATS:
        return openai_error(
            f"response_format '{response_format}' not supported "
            f"(use {', '.join(_RESPONSE_FORMATS)}).")

    try:
        audio_samples = _load_audio(audio_file)
    except Exception as e:
        return openai_error(f"Failed to read audio: {e}")

    engine = getattr(slot, "engine", "whisper")
    label = engine.capitalize()
    duration = len(audio_samples) / 16000
    lang_tag = f", lang={language}" if language else ""
    print(f"\n{datetime.now():%H:%M:%S} <- [{slot.device_name}] "
          f"{label} {duration:.1f}s audio{lang_tag}", flush=True)

    t0 = time.perf_counter()
    try:
        text = slot.transcribe(audio_samples, language=language)
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
              f"{label} error: {e}", flush=True)
        return openai_error(f"Transcription failed: {e}", "server_error", 500)

    elapsed = time.perf_counter() - t0
    rtf = duration / elapsed if elapsed > 0 else 0.0
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
          f"{label} {len(text)} chars in {elapsed:.2f}s ({rtf:.1f}x realtime)",
          flush=True)

    headers = {"X-Device": slot.device_name}
    if response_format == "text":
        return Response(text, mimetype="text/plain", headers=headers)
    if response_format == "verbose_json":
        resp = jsonify({
            "task": "transcribe",
            "text": text,
            "duration": round(duration, 3),
            # Whisper applied it; Parakeet only heard it. Saying which is the
            # difference between "it ignored me" and "it cannot be told".
            "language": language,
            "language_applied": bool(language) and engine == "whisper",
            "engine": engine,
            "model": getattr(slot, "model_name", ""),
            "processing_s": round(elapsed, 3),
        })
        resp.headers.update(headers)
        return resp
    resp = jsonify({"text": text})
    resp.headers.update(headers)
    return resp


@bp.route("/v1/audio/speech", methods=["POST"])
def audio_speech():
    """OpenAI-compatible text-to-speech. Returns a WAV body.

    No streaming: openvino-genai synthesizes the whole utterance before
    returning, so this responds once with the complete waveform.
    """
    if not runtime.tts_slot or not _slot_serviceable(runtime.tts_slot):
        return openai_error(
            "No text-to-speech model loaded. Use --tts-dir.", "server_error", 503,
        )

    body = request.get_json(silent=True) or {}
    text = (body.get("input") or "").strip()
    if not text:
        return openai_error("'input' is required")

    fmt = (body.get("response_format") or "wav").lower()
    if fmt not in ("wav", "pcm"):
        return openai_error(
            f"response_format '{fmt}' not supported (wav or pcm only — "
            "no encoder is bundled)."
        )

    print(f"\n{datetime.now():%H:%M:%S} <- [{runtime.tts_slot.device_name}] "
          f"TTS {len(text)} chars", flush=True)

    try:
        runtime.tts_slot.ensure_loaded()
        audio, rate = runtime.tts_slot.synthesize(text, voice=(body.get("voice") or None))
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} !! [{runtime.tts_slot.device_name}] "
              f"TTS error: {e}", flush=True)
        return openai_error(f"Speech synthesis failed: {e}", "server_error", 500)

    duration = len(audio) / rate if rate else 0
    print(f"{datetime.now():%H:%M:%S} -> [{runtime.tts_slot.device_name}] "
          f"TTS {duration:.1f}s audio in {runtime.tts_slot.last_synth_ms / 1000:.1f}s "
          f"({duration / max(runtime.tts_slot.last_synth_ms / 1000, 1e-6):.1f}x realtime)",
          flush=True)

    if fmt == "pcm":
        pcm = np.clip(audio, -1.0, 1.0)
        return Response((pcm * 32767).astype("<i2").tobytes(),
                        mimetype="audio/pcm",
                        headers={"X-Sample-Rate": str(rate),
                                 "X-Device": runtime.tts_slot.device_name})

    if sf is None:
        return openai_error("soundfile not installed. pip install soundfile",
                            "server_error", 500)
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), mimetype="audio/wav",
                    headers={"X-Sample-Rate": str(rate),
                             "X-Device": runtime.tts_slot.device_name})
