"""Parakeet's SentencePiece vocabulary, and turning pieces back into text.

Three on-disk shapes are accepted, because three are in circulation:

  * `{"0": "<unk>", "1": "▁t", ...}`                FluidInference v2 export
  * `{"id_to_token": {...}, "vocab_size": 8192}`    FluidInference v3 export
  * `vocab.txt`, one "piece id" per line            onnx-asr / sherpa exports

The blank is not in the FluidInference files at all: it is the id one past
the last piece (1024 on v2, 8192 on v3), which is also where NeMo puts it.
A vocabulary that names `<blk>` explicitly is believed instead.
"""

import json
import re
from pathlib import Path

WORD = "▁"                   # SentencePiece's word-start marker, "▁"
_SPECIAL = re.compile(r"^<[^<>]*>$")   # <unk>, <blk>, <|nospeech|>, <pad>


class Vocab:
    def __init__(self, pieces, blank_id):
        self.pieces = pieces              # id -> piece ("" for gaps)
        self.blank_id = blank_id

    def __len__(self):
        return len(self.pieces)

    def piece(self, token_id):
        return self.pieces[token_id] if 0 <= token_id < len(self.pieces) else ""

    def is_word_start(self, token_id):
        return self.piece(token_id).startswith(WORD)

    def decode(self, token_ids):
        """Pieces -> text. Specials and the blank never reach the user."""
        parts = []
        for t in token_ids:
            p = self.piece(t)
            if t == self.blank_id or not p or _SPECIAL.match(p):
                continue
            parts.append(p)
        text = "".join(parts).replace(WORD, " ")
        return re.sub(r"\s+", " ", text).strip()


def load_vocab(path):
    """Read any of the accepted formats. Raises with the file named."""
    path = Path(path)
    try:
        if path.suffix == ".txt":
            table = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                piece, _, idx = line.rpartition(" ")
                table[int(idx)] = piece
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("id_to_token"), dict):
                data = data["id_to_token"]
            if isinstance(data, list):
                table = dict(enumerate(data))
            else:
                table = {int(k): v for k, v in data.items()}
    except (OSError, ValueError, AttributeError) as e:
        raise RuntimeError(f"Could not read the Parakeet vocabulary {path.name}: {e}")
    if not table:
        raise RuntimeError(f"The Parakeet vocabulary {path.name} is empty.")
    pieces = [""] * (max(table) + 1)
    for i, p in table.items():
        pieces[i] = str(p)
    blank = next((i for i, p in enumerate(pieces) if p == "<blk>"), len(pieces))
    return Vocab(pieces, blank)
