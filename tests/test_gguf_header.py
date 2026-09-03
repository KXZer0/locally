"""The GGUF pre-load check: refuse what cannot load, and nothing else.

The check exists because an unsupported GGUF fails 12-35 seconds into a load
with `IndexError: invalid unordered_map<K, T> key` — no model named, no
architecture, no remedy. Reading the architecture out of the header refuses it
in ~100 ms with a sentence the user can act on.

The tests that matter most here are the negative ones. A guard that refuses
something loadable is worse than the error it replaces, so every malformed,
missing or unrelated input must come back silent. Headers are built in-memory
rather than read from disk: the real files are gigabytes and not in the repo.
"""

import os
import struct
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models.gguf import read_metadata, unsupported_reason


def _kv_string(key, value):
    kb, vb = key.encode(), value.encode()
    return (struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8)
            + struct.pack("<Q", len(vb)) + vb)


def _kv_u32(key, value):
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_string_array(key, values):
    """A vocab-shaped array — the thing the walker has to be able to skip."""
    kb = key.encode()
    out = [struct.pack("<Q", len(kb)), kb, struct.pack("<I", 9),
           struct.pack("<I", 8), struct.pack("<Q", len(values))]
    for v in values:
        vb = v.encode()
        out += [struct.pack("<Q", len(vb)), vb]
    return b"".join(out)


def write_gguf(path, pairs):
    body = b"".join(pairs)
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))          # version
        f.write(struct.pack("<Q", 0))          # tensor count
        f.write(struct.pack("<Q", len(pairs)))
        f.write(body)


class GgufHeaderTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def path(self, name):
        return os.path.join(self.dir, name)

    # --- what must be refused -------------------------------------------
    def test_unsupported_architecture_is_refused_by_name(self):
        p = self.path("gemma.gguf")
        write_gguf(p, [_kv_string("general.architecture", "gemma4")])
        reason = unsupported_reason(p)
        self.assertIsNotNone(reason)
        self.assertIn("gemma4", reason)
        self.assertIn("gemma.gguf", reason)
        # It has to say what to do, not only what is wrong.
        self.assertIn("optimum-cli", reason)

    def test_hybrid_qwen35_is_not_mistaken_for_qwen3(self):
        # The real Qwen3.8-27B-Heretic reports "qwen35". A prefix or
        # substring test would have let it through to the 35 s failure.
        p = self.path("q35.gguf")
        write_gguf(p, [_kv_string("general.architecture", "qwen35")])
        self.assertIsNotNone(unsupported_reason(p))

    def test_moe_with_a_supported_arch_name_is_refused(self):
        p = self.path("moe.gguf")
        write_gguf(p, [_kv_string("general.architecture", "qwen3"),
                       _kv_u32("qwen3.expert_count", 128)])
        reason = unsupported_reason(p)
        self.assertIsNotNone(reason)
        self.assertIn("128 experts", reason)

    # --- what must NOT be refused ----------------------------------------
    def test_supported_dense_architectures_pass(self):
        for arch in ("llama", "qwen2", "qwen3"):
            p = self.path(f"{arch}.gguf")
            write_gguf(p, [_kv_string("general.architecture", arch)])
            self.assertIsNone(unsupported_reason(p), arch)

    def test_zero_expert_count_is_not_moe(self):
        p = self.path("dense.gguf")
        write_gguf(p, [_kv_string("general.architecture", "qwen3"),
                       _kv_u32("qwen3.expert_count", 0)])
        self.assertIsNone(unsupported_reason(p))

    def test_non_gguf_paths_do_no_io_at_all(self):
        # An IR directory is the overwhelmingly common case; it must not even
        # open anything, which is why the extension is checked first.
        for candidate in (self.dir, self.path("model.xml"), "/nonexistent/dir"):
            self.assertIsNone(unsupported_reason(candidate))

    def test_malformed_files_never_block_a_load(self):
        cases = {
            "truncated.gguf": b"GGUF\x03\x00\x00\x00\xff",
            "wrongmagic.gguf": b"this is not a gguf file at all",
            "empty.gguf": b"",
            # Claims more pairs than it contains.
            "lying.gguf": b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                          + struct.pack("<Q", 99),
        }
        for name, blob in cases.items():
            p = self.path(name)
            with open(p, "wb") as f:
                f.write(blob)
            self.assertIsNone(unsupported_reason(p), name)
        self.assertIsNone(unsupported_reason(self.path("missing.gguf")))

    # --- the walker ------------------------------------------------------
    def test_architecture_is_found_after_a_vocab_sized_array(self):
        # general.architecture is conventionally first, but nothing guarantees
        # it, and a string array is the one value whose length cannot be
        # computed without reading each element.
        p = self.path("late.gguf")
        write_gguf(p, [
            _kv_string_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(500)]),
            _kv_u32("general.file_type", 15),
            _kv_string("general.architecture", "gemma4"),
        ])
        meta = read_metadata(p)
        self.assertEqual(meta.get("general.architecture"), "gemma4")
        self.assertIsNotNone(unsupported_reason(p))

    def test_unknown_value_type_stops_cleanly_rather_than_guessing(self):
        # Past an unreadable value the offsets are meaningless, so reading on
        # would return whatever bytes happened to line up.
        kb = b"weird.key"
        pair = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 77)
        p = self.path("weird.gguf")
        write_gguf(p, [pair, _kv_string("general.architecture", "gemma4")])
        # No architecture was reached, so the check has no opinion -- it must
        # not invent one in either direction.
        self.assertIsNone(unsupported_reason(p))


if __name__ == "__main__":
    unittest.main()
