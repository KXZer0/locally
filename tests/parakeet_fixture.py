"""A tiny, real OpenVINO export with Parakeet's file layout and port contract.

Real graphs compiled by the real runtime, so the slot's port discovery,
reshaping, state carrying and joint layout run exactly as they would on
FluidInference's export -- only the arithmetic is rigged. The encoder emits
zeros (ignoring the audio), so the joint's logits come entirely from a table
indexed by the last emitted token, and the table spells "hello world!":

    start (blank) -> ▁hello, skip 1
    ▁hello        -> ▁world, skip 1
    ▁world        -> !,      skip 1
    !             -> blank,  skip 4   (then blanks to the end)

Port names follow eddy-audio: melspectogram / melspectogram_length ->
encoder_output / encoder_output_length; targets, h_in, c_in -> decoder
output + h/c; encoder_outputs + decoder_outputs -> logits.
"""

import json
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as op

PIECES = ["<unk>", "▁hello", "▁world", "!"]
BLANK = len(PIECES)                 # 4: one past the last piece, as on v2/v3
N_TOK = BLANK + 1
DURS = 5
OUT = N_TOK + DURS                  # 10 joint scores
N_MELS = 8
HID = 4                             # LSTM width


def _named(node, name):
    node.output(0).get_tensor().set_names({name})
    return node


def _preprocessor(d):
    """audio [1, N] + length -> zeros [1, 8, ceil(N/160)] + frame count.
    Records nothing about the audio; the test checks the plumbing."""
    audio = op.parameter(ov.PartialShape([1, -1]), ov.Type.f32, name="audio_signal")
    alen = op.parameter([1], ov.Type.i64, name="audio_length")
    hop = op.strided_slice(audio, op.constant(np.array([0, 0])),
                           op.constant(np.array([1, 2 ** 31 - 1])),
                           op.constant(np.array([1, 160])), [0, 0], [0, 0])
    z = op.multiply(op.unsqueeze(hop, op.constant(np.array([1]))),
                    op.constant(np.float32(0)))
    melo = _named(op.add(z, op.constant(np.zeros((1, N_MELS, 1), np.float32))),
                  "melspectogram")
    frames = op.gather(op.shape_of(hop), op.constant(np.array([1])),
                       op.constant(np.int64(0)))
    frames = _named(op.add(op.multiply(frames, op.constant(np.array([1], np.int64))),
                           op.multiply(alen, op.constant(np.array([0], np.int64)))),
                    "melspectogram_length")
    ov.save_model(ov.Model([melo.output(0), frames.output(0)], [audio, alen], "pre"),
                  str(d / "parakeet_melspectogram.xml"))


def _table():
    t = np.full((N_TOK, OUT), -5.0, dtype=np.float32)

    def row(last, tok, skip):
        t[last, tok] = 5.0
        t[last, N_TOK + skip] = 5.0
    row(BLANK, 1, 1)
    row(1, 2, 1)
    row(2, 3, 1)
    row(3, BLANK, 4)
    row(0, BLANK, 4)
    return t


def build(directory, dynamic_frames=True, joint_shape=(1, 1, OUT),
          preprocessor=False):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    if preprocessor:
        _preprocessor(d)

    # Encoder: [1, 8, T] -> [1, OUT, ceil(T/8)] of zeros, plus its length.
    mel = op.parameter(ov.PartialShape([1, N_MELS, -1 if dynamic_frames else 1001]),
                       ov.Type.f32, name="melspectogram")
    mlen = op.parameter([1], ov.Type.i64, name="melspectogram_length")
    sub = op.strided_slice(mel, op.constant(np.array([0, 0, 0])),
                           op.constant(np.array([1, N_MELS, 2 ** 31 - 1])),
                           op.constant(np.array([1, 1, 8])), [0, 0, 0], [0, 0, 0])
    w = op.constant(np.zeros((OUT, N_MELS), dtype=np.float32))
    enc = _named(op.matmul(w, sub, False, False), "encoder_output")
    n = op.subtract(mlen, op.constant(np.array([1], dtype=np.int64)))
    n = op.divide(n, op.constant(np.array([8], dtype=np.int64)))
    n = _named(op.add(n, op.constant(np.array([1], dtype=np.int64))),
               "encoder_output_length")
    ov.save_model(ov.Model([enc.output(0), n.output(0)], [mel, mlen], "enc"),
                  str(d / "parakeet_encoder.xml"))

    # Predictor: Gather(table, last token); states pass through (+1 on h so
    # the test can see they were carried).
    tok = op.parameter([1, 1], ov.Type.i32, name="targets")
    h = op.parameter([2, 1, HID], ov.Type.f32, name="h_in")
    c = op.parameter([2, 1, HID], ov.Type.f32, name="c_in")
    g = op.gather(op.constant(_table()), tok, op.constant(np.int64(0)))
    out = _named(op.reshape(g, op.constant(np.array([1, 1, OUT])), False),
                 "decoder_output")
    h2 = _named(op.add(h, op.constant(np.float32(1))), "h_out")
    c2 = _named(op.multiply(c, op.constant(np.float32(1))), "c_out")
    ov.save_model(ov.Model([out.output(0), h2.output(0), c2.output(0)],
                           [tok, h, c], "dec"), str(d / "parakeet_decoder.xml"))

    # Joint: encoder frame + predictor output -> logits.
    je = op.parameter(list(joint_shape), ov.Type.f32, name="encoder_outputs")
    jd = op.parameter([1, 1, OUT], ov.Type.f32, name="decoder_outputs")
    lg = _named(op.add(op.reshape(je, op.constant(np.array([1, 1, OUT])), False), jd),
                "logits")
    ov.save_model(ov.Model([lg.output(0)], [je, jd], "joint"),
                  str(d / "parakeet_joint.xml"))

    (d / "parakeet_vocab.json").write_text(json.dumps(
        {"id_to_token": {str(i): p for i, p in enumerate(PIECES)},
         "vocab_size": len(PIECES)}), encoding="utf-8")
    return d


# --- Nemotron family (parakeet-unified, Nemotron streaming) ------------------
#
# eddy-audio's port names exactly (mel/mel_length -> encoded; cache_channel,
# cache_time, cache_len (+ _out); prompt_id; token/token_length/h_in/c_in ->
# decoder_out/h_out/c_out; encoder/decoder -> logits) and its metadata.json
# fields. Unlike the TDT fixture the encoder LISTENS: each token answers a
# tone in its own band of the raw log-mel, so a transcript depends on the
# audio and on where the units cut it --
#
#     500 Hz -> ▁hello     2 kHz -> ▁world     5 kHz -> !
#
# -- and the predictor scores -100 on the token it last emitted, so a tone held
# across many frames (and across unit boundaries, if the predictor state is
# carried) is spelled once.

NM_MELS = 16
NM_TOK = BLANK + 1                  # 5: no duration head on these models
TONES = {1: 500.0, 2: 2000.0, 3: 5000.0}
WINDOW = {"left": 2, "chunk": 4, "right": 2}       # encoder frames
CACHE = {"chunk_mel_frames": 16, "pre_encode_cache": 3, "total_mel_frames": 19}
CACHE_SHAPES = {"cache_channel_shape": [1, 2, 4], "cache_time_shape": [1, 2, 3]}
PROMPTS = {"en-US": 0, "fr-FR": 7, "auto": 101}


def _band_weights():
    from core.asr.mel import N_FFT, SR, mel_filters
    fb = mel_filters(NM_MELS)                       # [freqs, mels]
    w = np.zeros((NM_TOK, NM_MELS), dtype=np.float32)
    for tok, hz in TONES.items():
        bins = np.nonzero(fb[int(round(hz * N_FFT / SR))] > 0)[0]
        w[tok, bins] = 1.0 / len(bins)
    return w


def tone_audio(spec, sr=16000):
    """[(token or None, seconds), ...] -> float32 audio with 10 ms ramps, so a
    burst's onset does not splatter into every band."""
    out = []
    for tok, sec in spec:
        n = int(sec * sr)
        if tok is None:
            out.append(np.zeros(n, np.float32))
            continue
        t = np.arange(n) / sr
        x = np.sin(2 * np.pi * TONES[tok] * t)
        ramp = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.01 * sr))
        out.append((x * ramp).astype(np.float32))
    return np.concatenate(out)


def build_nemotron(directory, mode="window", prompt=False, dynamic=True):
    """mode "window": parakeet-unified's stateless sliding window.
    mode "cache": a cache-aware Nemotron encoder (+ prompt_id if `prompt`)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    if mode == "window":
        width = 1 + (WINDOW["left"] + WINDOW["chunk"] + WINDOW["right"]) * 8
        first = 0
    else:
        width = CACHE["total_mel_frames"]
        first = CACHE["pre_encode_cache"]   # frames before it are the carried lead
    mel = op.parameter(ov.PartialShape([1, NM_MELS, -1 if dynamic else width]),
                       ov.Type.f32, name="mel")
    mlen = op.parameter([1], ov.Type.i32, name="mel_length")
    ins = [mel, mlen]
    sub = op.strided_slice(mel, op.constant(np.array([0, 0, first])),
                           op.constant(np.array([1, NM_MELS, 2 ** 31 - 1])),
                           op.constant(np.array([1, 1, 8])), [0, 0, 0], [0, 0, 0])
    # 6 x the token band's mean raw log-mel: a full-scale tone reads ~+0.5..3.3
    # there (score >= 2.8), silence -16.6 and a length-masked frame exactly 0.
    # The blank sits at +1, so it beats <unk> (0), silence and masked padding,
    # but not a tone.
    bands = _band_weights()
    bias = np.zeros(NM_TOK, np.float32)
    bias[BLANK] = 1.0
    enc = op.add(op.matmul(op.constant(bands * 6.0), sub, False, False),
                 op.constant(bias.astype(np.float32).reshape(1, NM_TOK, 1)))
    zero_i = op.multiply(mlen, op.constant(np.array([0], np.int32)))
    outs = []
    if mode == "cache":
        cc = op.parameter(CACHE_SHAPES["cache_channel_shape"], ov.Type.f32,
                          name="cache_channel")
        ct = op.parameter(CACHE_SHAPES["cache_time_shape"], ov.Type.f32,
                          name="cache_time")
        cl = op.parameter([1], ov.Type.i32, name="cache_len")
        ins += [cc, ct, cl]
        outs += [_named(op.add(cc, op.constant(np.float32(1))), "cache_channel_out"),
                 _named(op.multiply(ct, op.constant(np.float32(1))), "cache_time_out"),
                 _named(op.add(op.add(cl, op.constant(np.array([1], np.int32))), zero_i),
                        "cache_len_out")]
        if prompt:
            pid = op.parameter([1], ov.Type.i32, name="prompt_id")
            ins.append(pid)
            zero_i = op.add(zero_i, op.multiply(pid, op.constant(np.array([0], np.int32))))
    enc = op.add(enc, op.convert(zero_i, ov.Type.f32))
    # A boolean BitwiseNot, as in the real attention mask: the slot must turn it
    # into LogicalNot (eddy's NPU fix) without changing the result.
    keep = op.convert(op.bitwise_not(op.less(enc, op.constant(np.float32(-1e9)))),
                      ov.Type.f32)
    enc = _named(op.multiply(enc, keep), "encoded")
    ov.save_model(ov.Model([enc.output(0)] + [o.output(0) for o in outs], ins, "enc"),
                  str(d / "nemotron_encoder.xml"))

    tok = op.parameter([1, 1], ov.Type.i32, name="token")
    tlen = op.parameter([1], ov.Type.i32, name="token_length")
    h = op.parameter([1, 1, 2], ov.Type.f32, name="h_in")
    c = op.parameter([1, 1, 2], ov.Type.f32, name="c_in")
    table = np.zeros((NM_TOK, NM_TOK), np.float32)
    for t in range(BLANK):
        table[t, t] = -100.0
    g = op.gather(op.constant(table), tok, op.constant(np.int64(0)))
    dec = _named(op.reshape(g, op.constant(np.array([1, NM_TOK, 1])), False),
                 "decoder_out")
    h2 = _named(op.add(h, op.constant(np.float32(1))), "h_out")
    c2 = _named(op.add(c, op.convert(tlen, ov.Type.f32)), "c_out")
    ov.save_model(ov.Model([dec.output(0), h2.output(0), c2.output(0)],
                           [tok, tlen, h, c], "dec"), str(d / "nemotron_decoder.xml"))

    je = op.parameter([1, NM_TOK, 1], ov.Type.f32, name="encoder")
    jd = op.parameter([1, NM_TOK, 1], ov.Type.f32, name="decoder")
    lg = _named(op.reshape(op.add(je, jd), op.constant(np.array([1, 1, 1, NM_TOK])),
                           False), "logits")
    ov.save_model(ov.Model([lg.output(0)], [je, jd], "joint"),
                  str(d / "nemotron_joint.xml"))

    (d / "nemotron_vocab.json").write_text(json.dumps(
        {str(i): p for i, p in enumerate(PIECES)}), encoding="utf-8")
    meta = {"sample_rate": 16000, "mel_features": NM_MELS, "blank_idx": BLANK,
            "vocab_size": len(PIECES), "decoder_hidden": 2, "decoder_layers": 1,
            "subsampling_factor": 8}
    if mode == "window":
        meta.update({"streaming": True, "context_encoder_frames": dict(WINDOW),
                     "window_mel_frames": width, "feature_normalize": False})
    else:
        meta.update(CACHE)
        meta.update(CACHE_SHAPES)
        if prompt:
            meta.update({"prompt_dictionary": PROMPTS, "default_prompt_id": 101})
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return d
