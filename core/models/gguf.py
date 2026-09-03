"""Read just enough of a GGUF header to refuse the ones that cannot load.

OpenVINO's GGUF reader handles dense llama, qwen2 and qwen3. Handed anything
else it looks a tensor name up in a per-architecture map, misses, and lets a raw
C++ error out through the bindings:

    IndexError: invalid unordered_map<K, T> key

That names no model, no architecture and no remedy, and it arrives *late* --
measured 12.4 s into gemma-4-12b-heretic and 35.3 s into Qwen3.8-27B-Heretic,
after the file has been read and the user has every reason to think it is
working. The architecture is a string sitting in the first few bytes of the same
file, so the refusal can be immediate and can say what is wrong.

The header is: the magic "GGUF", a uint32 version, a uint64 tensor count, a
uint64 metadata count, then that many key/value pairs. `general.architecture` is
conventionally the first pair, which is why this only has to be able to walk a
few of them.

**An unreadable header never blocks a load.** Every failure path here returns
None, and the caller treats None as "no opinion". A parser that refuses a model
it merely failed to understand would be worse than the error it replaces.
"""

import os
import struct

# What the reader implements. Anything else reaches the map miss above.
SUPPORTED = ("llama", "qwen2", "qwen3")

# Metadata value types, from the GGUF spec. The value is the fixed byte width;
# STRING and ARRAY are variable and handled separately.
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING = 8
_ARRAY = 9

# Real headers run to a few dozen pairs. The cap is a guard against a corrupt
# count, not a budget -- the vocab arrays between them are skipped by seeking,
# so walking the whole header costs milliseconds.
_MAX_PAIRS = 512


def _u32(f):
    return struct.unpack("<I", f.read(4))[0]


def _u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def _string(f):
    return f.read(_u64(f)).decode("utf-8", "replace")


def _skip_value(f, vtype):
    """Advance past one value. False when its type is unknown to us."""
    if vtype in _FIXED:
        f.seek(_FIXED[vtype], os.SEEK_CUR)
        return True
    if vtype == _STRING:
        f.seek(_u64(f), os.SEEK_CUR)
        return True
    if vtype == _ARRAY:
        item_type = _u32(f)
        count = _u64(f)
        if item_type in _FIXED:
            f.seek(_FIXED[item_type] * count, os.SEEK_CUR)
            return True
        if item_type == _STRING:
            # Lengths are interleaved with the data, so each one has to be read
            # to know how far to jump. Seeks only; nothing is decoded.
            for _ in range(count):
                f.seek(_u64(f), os.SEEK_CUR)
            return True
        return False
    return False


def read_metadata(path):
    """`{key: value}` for the leading scalar metadata, or None if unreadable.

    Only strings and integers are returned -- they are what the decisions here
    are made from. Arrays are skipped, not collected.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            _u32(f)                      # version
            _u64(f)                      # tensor count
            pairs = _u64(f)
            out = {}
            for _ in range(min(pairs, _MAX_PAIRS)):
                key = _string(f)
                vtype = _u32(f)
                if vtype == _STRING:
                    out[key] = _string(f)
                elif vtype in _FIXED and vtype not in (6, 12):
                    raw = f.read(_FIXED[vtype])
                    fmt = {1: "<b", 2: "<H", 3: "<h", 4: "<I",
                           5: "<i", 10: "<Q", 11: "<q"}.get(vtype)
                    out[key] = struct.unpack(fmt, raw)[0] if fmt else raw[0]
                elif not _skip_value(f, vtype):
                    break            # unknown type: the offsets are no longer
                                     # trustworthy, so stop reading
            return out
    except (OSError, struct.error, ValueError):
        return None


def unsupported_reason(path):
    """A plain refusal for a GGUF that cannot load, or None to allow it.

    None covers three different cases on purpose -- supported, unreadable, and
    not-a-GGUF -- because all three mean the same thing to the caller: this
    check has no reason to stop you.
    """
    if not str(path).lower().endswith(".gguf"):
        return None
    meta = read_metadata(path)
    if not meta:
        return None
    arch = meta.get("general.architecture")
    if not arch:
        return None

    name = os.path.basename(str(path))
    fix = ("Convert it to an OpenVINO IR instead (optimum-cli export openvino "
           "--weight-format int4 --sym --group-size -1), which is also faster "
           "to load and to decode, or serve it from llama.cpp behind "
           "--proxy-url.")

    if arch not in SUPPORTED:
        return (f"{name} is a '{arch}' model, and OpenVINO's GGUF reader "
                f"implements only {', '.join(SUPPORTED)} (dense). {fix}")

    # A mixture-of-experts model can carry a supported architecture name and
    # still miss the same map, so the expert count is checked separately rather
    # than trusted to show up in the name.
    experts = meta.get(f"{arch}.expert_count") or 0
    if experts:
        return (f"{name} is a mixture-of-experts model ({experts} experts) and "
                f"OpenVINO's GGUF reader handles dense models only. {fix}")
    return None



def geometry(path):
    """Layers, KV heads, head dim and context length, or None.

    A GGUF carries the same numbers `config.json` does, under
    `<arch>.block_count`, `<arch>.attention.head_count_kv` and friends. Reading
    them is what lets the memory preflight, `--context-tokens` and the live fit
    verdicts treat a GGUF like any other model instead of silently giving up
    (every one of those helpers returned None for a .gguf before this).

    Cross-checked against the IR of the same weights: the Qwen3-8B GGUF reports
    36 blocks x 8 KV heads x 128, which is 144 KB/token -- the exact figure
    `--scan` prints for the OpenVINO export. Qwen2.5-Coder-7B comes out at
    56 KB/token, matching the ~57 KB CLAUDE.md recorded for it.
    """
    meta = read_metadata(path)
    if not meta:
        return None
    arch = meta.get("general.architecture")
    if not arch:
        return None
    layers = meta.get(f"{arch}.block_count")
    heads = meta.get(f"{arch}.attention.head_count")
    if not layers or not heads:
        return None
    kv_heads = meta.get(f"{arch}.attention.head_count_kv") or heads
    # key_length is optional; without it the head dimension is the embedding
    # width divided across the heads, which is how a dense model is built.
    head_dim = meta.get(f"{arch}.attention.key_length")
    if not head_dim:
        embed = meta.get(f"{arch}.embedding_length")
        if not embed:
            return None
        head_dim = embed // heads
    return {"architecture": arch, "layers": layers, "heads": heads,
            "kv_heads": kv_heads, "head_dim": head_dim,
            "context_length": meta.get(f"{arch}.context_length") or 0,
            "name": meta.get("general.name") or ""}


def kv_bytes_per_token(path):
    """K+V bytes for one token, fp16. None when the header cannot be read.

    No `layer_types` equivalent exists in GGUF, and the reader only accepts
    dense architectures anyway, so every layer holds a cache here -- the
    hybrid-attention correction in geometry.py has nothing to apply to.
    """
    g = geometry(path)
    if not g:
        return None
    return 2 * g["layers"] * g["kv_heads"] * g["head_dim"] * 2


def max_context(path):
    """The model's own context ceiling from the header, or None."""
    g = geometry(path)
    return (g.get("context_length") or None) if g else None


# Qwen2 and Qwen3 both speak ChatML, and they are the only architectures this
# reader accepts whose format is unambiguous. "llama" covers Llama 2, Llama 3
# and a dozen derivatives with three different formats between them, so it is
# deliberately absent: guessing wrong there produces a model that answers
# fluently and wrongly, which is worse than refusing.
_CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

_ARCH_TEMPLATES = {"qwen2": _CHATML, "qwen3": _CHATML}


def chat_template(path):
    """The Jinja chat template to use, or None if we should not guess.

    genai converts a GGUF's tokenizer on the fly and the result carries no chat
    template, so handing it a ChatHistory fails with "Chat template must not be
    empty". The file usually has one under `tokenizer.chat_template`; the two
    quantizations measured here do not -- 36 metadata pairs and no template
    among them -- so an architecture default is the fallback.
    """
    meta = read_metadata(path)
    if not meta:
        return None
    embedded = meta.get("tokenizer.chat_template")
    if embedded:
        return embedded
    return _ARCH_TEMPLATES.get(meta.get("general.architecture"))
