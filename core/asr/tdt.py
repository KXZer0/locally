"""Greedy decoding for a Token-and-Duration Transducer (Parakeet TDT).

An RNN-T emits one token or a blank per (frame, step) and always advances one
encoder frame on a blank. A TDT joint has a SECOND head that predicts how many
frames to skip, which is most of why Parakeet is fast: silence and long vowels
are crossed in one step instead of one call per 80 ms frame.

The loop is NeMo's greedy TDT, the same one onnx-asr implements:

    token  = argmax(logits[:blank + 1])
    skip   = durations[argmax(logits[blank + 1:])]
    emit token unless it is blank; a token advances the predictor
    skip > 0                    -> move skip frames on
    skip == 0 and blank         -> move one frame on (a blank must progress)
    skip == 0 and max_symbols   -> move one frame on (a runaway frame is cut)

Two details that are easy to get wrong and are tested:

  * The predictor (decoder LSTM) output depends only on the last EMITTED
    token and its state, so it is computed once per emission and reused for
    every blank after it. FluidInference's runtime relies on the same fact.
  * A model with no duration head (plain RNN-T, durations == ()) is the
    special case skip == 0 everywhere, so one loop serves both.

Nothing here imports OpenVINO: the slot passes in two callables, which is
what lets the tests drive the loop with hand-built logits.
"""

import json

import numpy as np

MAX_SYMBOLS_PER_STEP = 10   # NeMo/onnx-asr default; stops a frame emitting forever


def greedy_decode(encoder_frames, n_frames, predict, joint, blank_id,
                  durations=(), max_symbols=MAX_SYMBOLS_PER_STEP,
                  start_state=None):
    """Decode one utterance.

    encoder_frames  sequence indexable by frame -> that frame's encoder vector
    n_frames        how many frames are real (the rest is padding)
    predict(token, state) -> (predictor_output, new_state)
    joint(encoder_vector, predictor_output) -> 1-D logits:
                    [blank_id + 1 token scores][len(durations) duration scores]
    Returns [(token_id, frame_index), ...] in emission order.
    """
    n_tokens = blank_id + 1
    n_dur = len(durations)
    out = []
    state = start_state
    last = blank_id
    pred = None                  # cached predictor output for (last, state)
    pred_state = None
    t = 0
    emitted = 0
    while t < n_frames:
        if pred is None:
            pred, pred_state = predict(last, state)
        logits = np.asarray(joint(encoder_frames[t], pred)).reshape(-1)
        if logits.shape[0] < n_tokens + n_dur:
            raise ValueError(
                f"Joint output has {logits.shape[0]} values; this vocabulary "
                f"needs {n_tokens} token scores plus {n_dur} duration scores.")
        token = int(np.argmax(logits[:n_tokens]))
        skip = (int(durations[int(np.argmax(logits[n_tokens:n_tokens + n_dur]))])
                if n_dur else 0)
        if token != blank_id:
            out.append((token, t))
            state = pred_state
            last = token
            pred = None          # the predictor has a new context
            emitted += 1
        if skip > 0:
            t += skip
            emitted = 0
        elif token == blank_id or emitted >= max_symbols:
            t += 1
            emitted = 0
    return out


def merge_chunks(chunks, is_word_start):
    """Join per-chunk token lists from overlapping windows into one.

    chunks: [(tokens_with_seconds, keep_from_s, keep_until_s), ...] where
    tokens_with_seconds is [(token_id, start_seconds), ...] on the GLOBAL
    timeline and the keep bounds split each overlap at its midpoint
    (None = open-ended). Each window is decoded on its own, so both sides of
    an overlap transcribe it; the midpoint decides who owns which part.

    Cutting at a time alone can split a word: "under" might be ▁un + der
    with ▁un before the cut and der after it, and each window would then
    own half. So a window keeps going past its cut until the next word
    starts, and the following window drops the continuation pieces it would
    otherwise begin with. Both rules key on the same thing -- whether a piece
    starts a word -- so the two windows agree on where the seam is.
    """
    merged = []
    for tokens, keep_from, keep_until in chunks:
        kept = []
        i = 0
        if keep_from is not None:
            while i < len(tokens) and tokens[i][1] < keep_from:
                i += 1
            while i < len(tokens) and not is_word_start(tokens[i][0]):
                i += 1           # tail of a word the previous window owns
        for tok, sec in tokens[i:]:
            if keep_until is not None and sec >= keep_until and is_word_start(tok):
                break
            kept.append((tok, sec))
        merged.extend(kept)
    return merged


def resolve_durations(config_path, out_dim, blank_id):
    """The duration bins the joint's second head scores, in frames.

    Not in the FluidInference files: their runtime hard-codes [0, 1, 2, 3, 4]
    for both v2 and v3, which is also NeMo's Parakeet TDT config. A
    config.json/metadata.json naming `tdt_durations` is believed first; then
    the joint's width decides -- blank + 1 scores means a plain RNN-T, five
    more means the standard bins, anything else is refused by name rather
    than decoded with made-up skips.
    """
    if config_path:
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            for key in ("tdt_durations", "durations", "duration_bins"):
                if isinstance(cfg.get(key), list):
                    return tuple(int(x) for x in cfg[key])
        except (OSError, ValueError, AttributeError):
            pass
    if out_dim is None:
        return (0, 1, 2, 3, 4)
    extra = out_dim - (blank_id + 1)
    if extra == 0:
        return ()
    if extra == 5:
        return (0, 1, 2, 3, 4)
    raise RuntimeError(
        f"The joint emits {out_dim} scores for a vocabulary whose blank is id "
        f"{blank_id}, leaving {extra} duration bins this engine cannot name. "
        "Add \"tdt_durations\": [...] to the model's config.json.")
