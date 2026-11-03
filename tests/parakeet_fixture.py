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
