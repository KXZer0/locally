"""Is the model on disk actually complete?

The IR .xml records every weight blob's offset and size, so max(offset+size)
is the exact minimum .bin size. There is no checksum: truncation is the
realistic failure (#17), corruption-in-place is out of scope."""

import os
import re
import xml.etree.ElementTree as ET


def _file_or_dir_size(path):
    """Bytes on disk, for a GGUF file as readily as for an IR directory."""
    if str(path).lower().endswith(".gguf"):
        try:
            return os.path.getsize(path)
        except OSError:
            return None
    return _dir_size_bytes(path)


def _dir_size_bytes(model_dir):
    """Total size of a model directory (≈ weight bytes). None on failure."""
    try:
        total = 0
        for root, _dirs, files in os.walk(model_dir):
            for fn in files:
                total += os.path.getsize(os.path.join(root, fn))
        return total or None
    except OSError:
        return None


def _verify_weights_integrity(model_dir):
    """Byte-exact completeness check. The IR .xml records each weight blob's
    offset+size into the .bin, so max(offset+size) is the exact minimum byte
    count the .bin must have — catches a download/copy that lost even the
    last 8 bytes (the IR carries no checksum, so corruption-in-place is out
    of scope; truncation is the realistic failure). Returns an error string,
    or None when intact / not checkable.

    Checks **every** IR .xml in the directory. Two reasons this isn't the two
    hardcoded names it used to be:

    - A modern VLM export is several IRs, not one: Qwen3.6-35B-A3B ships
      language_model + text_embeddings + vision_embeddings(+_merger/_pos),
      and any single one of them can arrive short.
    - A .bin that is missing *entirely* used to be skipped rather than
      reported, so it read as "weights complete". Found the hard way: a
      17 GB openvino_language_model.bin whose transfer died left a directory
      that passed the check with 4 GB of the 18 GB present.

    An .xml declaring no weight blobs needs no .bin, so absence is only an
    error when the graph actually references weights.
    """
    try:
        names = sorted(f for f in os.listdir(model_dir) if f.endswith(".xml"))
    except OSError:
        return None
    for name in names:
        base = name[:-4]
        binf = os.path.join(model_dir, base + ".bin")
        try:
            with open(os.path.join(model_dir, name), "rb") as f:
                data = f.read()
            need = max((int(m.group(1)) + int(m.group(2)) for m in
                        re.finditer(rb'offset="(\d+)" size="(\d+)"', data)),
                       default=0)
        except (OSError, ValueError):
            continue
        if not need:
            continue
        if not os.path.isfile(binf):
            return (f"{base}.bin is missing, but {name} references "
                    f"{need:,} bytes of weights. Incomplete download or copy "
                    f"— delete the model directory and re-fetch it "
                    f"(install.ps1 / download-model.ps1).")
        have = os.path.getsize(binf)
        if have < need:
            return (f"{base}.bin is truncated: {have:,} bytes on disk but "
                    f"{name} expects at least {need:,}. Incomplete "
                    f"download or copy — delete the model directory and "
                    f"re-fetch it (install.ps1 / download-model.ps1).")
    return None
