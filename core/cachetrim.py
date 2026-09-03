"""Keep the OpenVINO compile cache bounded.

Loading a model is mostly *compiling* it for the device, so `.ov-cache` is a
real speed feature -- Qwen3-8B on the NPU is 65.1 s cold against 9.3 s cached.
What it never had is an upper bound. Nothing evicted anything, so every model
ever loaded left its compiled graphs behind forever, including models long
since deleted from disk.

Measured here 2026-08-30, on a cache that had simply been left to run:
**41.9 GB across 1782 files, of which 36.1 GB in 1498 files had not been
touched in 14 days.** Single blobs reached 8.1 GB. Nothing was wrong -- that is
just what an unbounded cache does given a few months and a habit of trying
models.

Eviction is least-recently-used by mtime, which is the right key here: OpenVINO
touches an entry when it reads it, so "not read in a long time" really does mean
"no longer loading this model". The trim runs once at startup, before any model
loads, and never during serving -- deleting a blob out from under a compile
would turn a cache miss into a failure.

Deleting a cache entry is not data loss: it costs one cold compile the next time
that exact model/device pair is loaded, and nothing else. That asymmetry is why
this is safe to do automatically and why the default is generous rather than
tight.
"""

import os
import stat as _stat


def _unlink(path):
    """Remove one cache entry. True on success.

    **OpenVINO writes its cache blobs with the READ-ONLY attribute set**, and
    on Windows `os.remove` on a read-only file fails with `PermissionError`
    WinError 5 (access denied) -- not WinError 32 (sharing violation). Measured
    2026-08-30: all 69 blobs in a 41.7 GB cache were `ReadOnly, Archive`, so a
    trim that only called `os.remove` deleted every small `.cl_cache` file and
    not one of the large blobs that held essentially all of the bytes. It then
    reported "freed 0.2 GB" and looked like it had worked.

    So: clear the attribute and retry. The distinction matters for the message
    as much as the outcome -- read-only is ours to fix, in-use is not.
    """
    try:
        os.remove(path)
        return True
    except PermissionError:
        try:
            os.chmod(path, _stat.S_IWRITE)
            os.remove(path)
            return True
        except OSError:
            return False
    except OSError:
        return False


def trim(directory, limit_gb, log=print):
    """Evict least-recently-used entries until the cache is under `limit_gb`.

    Returns (freed_bytes, removed_count). Never raises: a cache that cannot be
    trimmed is a cache, not an error, and must not stop the server starting.
    """
    if not directory or not limit_gb or limit_gb <= 0:
        return 0, 0
    limit = int(limit_gb * 1024 ** 3)

    entries = []
    total = 0
    try:
        with os.scandir(directory) as it:
            for entry in it:
                try:
                    if not entry.is_file():
                        continue
                    stat = entry.stat()
                except OSError:
                    continue
                entries.append((stat.st_mtime, stat.st_size, entry.path))
                total += stat.st_size
    except OSError:
        return 0, 0

    if total <= limit:
        return 0, 0

    # Oldest first. Deleting the least-recently-read entry is the one least
    # likely to be wanted on the next start.
    entries.sort()
    freed = 0
    removed = 0
    locked = 0
    locked_bytes = 0
    for _, size, path in entries:
        if total - freed <= limit:
            break
        if not _unlink(path):
            # Genuinely could not remove it -- almost always a live server
            # holding the blob open (WinError 32). Skipping is right; never
            # fight a running server for its cache. But it must be REPORTED:
            # the first cut counted only what it deleted and logged
            # "freed 0.2 GB" while leaving 41.7 GB in place. A trim that
            # quietly misses its target by 30 GB is worse than one that does
            # nothing, because it stops anyone from looking.
            locked += 1
            locked_bytes += size
            continue
        freed += size
        removed += 1

    if removed:
        log(f"  Compile cache trimmed: freed {freed / 1024**3:.1f} GB "
            f"({removed} stale entries) to stay under {limit_gb:g} GB")
    remaining = total - freed
    if remaining > limit:
        detail = (f"; {locked} entries ({locked_bytes / 1024**3:.1f} GB) were "
                  f"in use by another locally instance"
                  if locked else "")
        log(f"  WARNING: compile cache is {remaining / 1024**3:.1f} GB, still "
            f"over the {limit_gb:g} GB bound{detail}")
    return freed, removed
