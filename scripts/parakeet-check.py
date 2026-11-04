"""Does Parakeet work on this machine, on which devices, and how fast?

Loads a speech-to-text export through the same slot the server would pick
for that folder -- parakeet-unified / Nemotron streaming (NemotronSlot) or
Parakeet TDT (ParakeetSlot) -- on CPU, GPU and NPU in turn, and for each
prints the load time, the transcription, and the real-time factor (seconds
of audio per second of compute -- 25x means a 10 s clip takes 0.4 s). A
device that fails says why and the next one still runs: "the NPU refused
the encoder" is a result.

A streaming model is also run the way a voice turn runs it: fed in 32 ms
frames as if from a microphone, with the worker decoding as it goes, and the
script reports how long after the LAST frame the transcript was ready, and
whether it matches the offline transcript exactly (it should: one loop).

Nothing about Parakeet on the NPU has been verified by this project yet; the
~25x (TDT v3) and ~22x (Nemotron int8) figures are FluidInference's
(eddy-audio, Core Ultra 7 155H). This script is how they become ours.

Usage (stop the server first -- the NPU serves one process at a time):
    venv\\Scripts\\python.exe scripts\\parakeet-check.py MODEL_DIR speech.wav
        [--devices CPU,GPU,NPU] [--reference "what was said"]
        [--compare DIR]         # another ASR export (e.g. TDT v3), same clip
        [--whisper-dir DIR]     # same clip through Whisper, for comparison
        [--runs 3] [--cache-dir .ov-cache]

MODEL_DIR is the streaming/fp16 folder of FluidInference/parakeet-unified-
en-0.6b-ov, a Nemotron streaming folder, or FluidInference/parakeet-tdt-
0.6b-v3-ov. The WAV can be any rate or channel count; it is resampled to
16 kHz mono the same way /v1/audio/transcriptions does.
"""

import argparse
import os
import re
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config                       # noqa: E402
from core.asr.layout import asr_engine, describe_unknown   # noqa: E402
from core.slots.nemotron import NemotronSlot  # noqa: E402
from core.slots.parakeet import ParakeetSlot  # noqa: E402

SLOTS = {"nemotron": NemotronSlot, "parakeet": ParakeetSlot}

SR = 16000


def read_wav(path):
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
    except ImportError:
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2:
                sys.exit("Without soundfile only 16-bit PCM WAV can be read: "
                         "pip install soundfile")
            sr = w.getframerate()
            raw = np.frombuffer(w.readframes(w.getnframes()), "<i2")
            audio = raw.reshape(-1, w.getnchannels()).astype(np.float32) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        n = int(len(audio) * SR / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio.astype(np.float32)


def wer(ref, hyp):
    """Word error rate, case and punctuation folded."""
    norm = lambda s: re.sub(r"[^\w\s']", " ", s.lower()).split()   # noqa: E731
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


def run(slot, model_dir, audio, runs, reference):
    seconds = len(audio) / SR
    t0 = time.perf_counter()
    slot.load(model_dir)
    slot.warmup()
    load_s = time.perf_counter() - t0
    times, text = [], ""
    for _ in range(runs):
        t0 = time.perf_counter()
        text = slot.transcribe(audio, language="en")
        times.append(time.perf_counter() - t0)
    best = min(times)
    extra = getattr(slot, "decoder_device", None)
    where = (f" (predictor on {extra})" if extra and extra != slot.device_name
             else "")
    print(f"    load      {load_s:6.1f} s{where}")
    print(f"    transcribe {best:5.2f} s best of {runs}, first {times[0]:.2f} s "
          f"-> {seconds / best:5.1f}x realtime")
    if reference:
        print(f"    WER       {wer(reference, text) * 100:5.1f} %")
    print(f"    text      {text}")
    if hasattr(slot, "open_stream"):
        live(slot, audio, text)
    with slot.lock:
        slot.unload()
    return seconds / best


def live(slot, audio, offline_text):
    """Feed the clip like a microphone (32 ms frames, the worker keeping up)
    and time the transcript after the last frame."""
    st = slot.open_stream(language="en")
    for i in range(0, len(audio), 512):
        st.feed(audio[i:i + 512])
        deadline = time.perf_counter() + 5
        while (st.geo.ready(st.pos, st.n, False) and st.error is None
               and time.perf_counter() < deadline):
            time.sleep(0.001)
    done_live = st.units
    t0 = time.perf_counter()
    text = st.finalize()
    tail_ms = (time.perf_counter() - t0) * 1000
    g = slot.geometry
    print(f"    streaming {g.mode} mode, {g.unit_seconds:.2f} s units; "
          f"{done_live}/{st.units} decoded while 'speaking', transcript "
          f"{tail_ms:.0f} ms after the last frame")
    print(f"    stream == offline: {'yes' if text == offline_text else 'NO'}"
          + ("" if text == offline_text else f" -> {text}"))


def run_dir(model_dir, devices, available, audio, a, results, tag=None):
    engine = asr_engine(model_dir)
    if engine not in SLOTS:
        print(f"{model_dir}: {describe_unknown(model_dir)}\n")
        return
    tag = tag or engine
    for dev in devices:
        dev_id = next((d for d in available if d.split(".")[0] == dev), None)
        print(f"{tag} ({os.path.basename(os.path.normpath(model_dir))}) on {dev}")
        if dev_id is None:
            print("    skipped: OpenVINO reports no such device here\n")
            continue
        try:
            results[f"{tag}/{dev}"] = run(SLOTS[engine](dev, dev_id), model_dir,
                                          audio, a.runs, a.reference)
        except Exception as e:                      # report and carry on
            first = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
            print(f"    FAILED: {type(e).__name__}: {first[:300]}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("wav")
    ap.add_argument("--devices", default="CPU,GPU,NPU")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--reference", default=None,
                    help="What was actually said, to print a word error rate.")
    ap.add_argument("--compare", action="append", default=None,
                    help="Another Parakeet/Nemotron export to run on the same "
                         "clip, e.g. parakeet-tdt-0.6b-v3-ov. Repeatable.")
    ap.add_argument("--whisper-dir", default=None,
                    help="Also run this Whisper export (CPU/GPU) on the clip.")
    ap.add_argument("--cache-dir", default=None,
                    help="OpenVINO compile cache; a second run then shows the "
                         "load time the server will see after the first start.")
    ap.add_argument("--npu-window-seconds", type=float,
                    default=config.ASR_NPU_WINDOW_SECONDS)
    a = ap.parse_args()

    config.MODEL_CACHE_DIR = a.cache_dir
    config.ASR_NPU_WINDOW_SECONDS = a.npu_window_seconds
    audio = read_wav(a.wav)
    print(f"Audio: {a.wav}, {len(audio) / SR:.1f} s")
    import openvino as ov
    available = ov.Core().available_devices
    print(f"OpenVINO {ov.__version__}; devices: {', '.join(available)}\n")

    results = {}
    devices = [d.strip().upper() for d in a.devices.split(",") if d.strip()]
    run_dir(a.model_dir, devices, available, audio, a, results)
    for other in a.compare or []:
        run_dir(other, devices, available, audio, a, results,
                tag=os.path.basename(os.path.normpath(other)))

    if a.whisper_dir:
        from core.slots.whisper import WhisperSlot
        for dev in ("CPU", "GPU"):
            dev_id = next((d for d in available if d.split(".")[0] == dev), None)
            if dev_id is None:
                continue
            print(f"Whisper on {dev}")
            try:
                results[f"whisper/{dev}"] = run(WhisperSlot(dev, dev_id), a.whisper_dir,
                                                audio, a.runs, a.reference)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
            print()

    if results:
        print("Summary (x realtime, higher is faster):")
        for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
            print(f"    {k:32s} {v:6.1f}x")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
