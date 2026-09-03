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

# `general.architecture` sits at the front; a vocab array further in can hold
# 150k strings and is not worth walking. Give up rather than grind.
_MAX_PAIRS = 64


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
