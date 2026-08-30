"""What the machine has free, and what this process is holding.

Do NOT measure a model unload with RSS: an idle process has already had its
working set trimmed by Windows, which is how 'unload frees nothing' was once
concluded. _mem_status reads ullAvailPhys and _process_memory reads
PrivateUsage; _settle_memory polls because the GPU driver releases
asynchronously (~2 s, ~3 GB late)."""

import ctypes
import os
import time


def _mem_status():
    """(total_ram, available_ram) system-wide, in bytes. (None, None) if unknown.

    "Available" is what another process could allocate without the OS having
    to page anything out — the number that answers "can I launch this game
    now?", and the only one that showed unloading works (process RSS does
    not: see the memory entry in TODONT.md).
    """
    try:
        if os.name == "nt":
            import ctypes
            class _MemStatus(ctypes.Structure):
                _fields_ = ([("dwLength", ctypes.c_ulong),
                             ("dwMemoryLoad", ctypes.c_ulong)] +
                            [(n, ctypes.c_ulonglong) for n in (
                                "ullTotalPhys", "ullAvailPhys",
                                "ullTotalPageFile", "ullAvailPageFile",
                                "ullTotalVirtual", "ullAvailVirtual",
                                "ullAvailExtendedVirtual")])
            st = _MemStatus(dwLength=ctypes.sizeof(_MemStatus))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys), int(st.ullAvailPhys)
        else:
            total = avail = None
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024
            return total, avail
    except Exception:
        pass
    return None, None


def _system_ram_bytes():
    """Total physical RAM in bytes. None if it can't be determined."""
    return _mem_status()[0]


def _win_current_process():
    """The current-process pseudo-handle, as a c_void_p-safe value."""
    import ctypes
    fn = ctypes.windll.kernel32.GetCurrentProcess
    fn.restype = ctypes.c_void_p
    return fn()


def _process_memory():
    """(working_set, private_commit) for this process, in bytes; None where
    unavailable.

    Both matter and they answer different questions. The working set is what
    Task Manager shows and what a trim moves. Private commit is what the
    process still *holds* — it does not drop when pages are merely trimmed,
    so it is how you tell "given back to the system" from "paged out and
    still owed". Neither counts NPU/GPU driver allocations in full.
    """
    try:
        if os.name == "nt":
            import ctypes
            SIZE_T = ctypes.c_size_t
            class _Counters(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong),
                            ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", SIZE_T),
                            ("WorkingSetSize", SIZE_T),
                            ("QuotaPeakPagedPoolUsage", SIZE_T),
                            ("QuotaPagedPoolUsage", SIZE_T),
                            ("QuotaPeakNonPagedPoolUsage", SIZE_T),
                            ("QuotaNonPagedPoolUsage", SIZE_T),
                            ("PagefileUsage", SIZE_T),
                            ("PeakPagefileUsage", SIZE_T),
                            ("PrivateUsage", SIZE_T)]
            c = _Counters(cb=ctypes.sizeof(_Counters))
            fn = ctypes.windll.psapi.GetProcessMemoryInfo
            # Without argtypes the pseudo-handle (-1) is passed as a 32-bit
            # int and the call fails silently, returning zeroes.
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters),
                           ctypes.c_ulong]
            if fn(_win_current_process(), ctypes.byref(c), ctypes.sizeof(c)):
                return int(c.WorkingSetSize), int(c.PrivateUsage)
        else:
            rss = priv = None
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                    elif line.startswith("RssAnon:"):
                        priv = int(line.split()[1]) * 1024
            return rss, priv
    except Exception:
        pass
    return None, None


def _memory_snapshot():
    """What the machine and this process look like right now, in MB."""
    mib = 2 ** 20
    ws, priv = _process_memory()
    total, avail = _mem_status()
    return {
        "process_ws_mb": ws and round(ws / mib),
        "process_private_mb": priv and round(priv / mib),
        "system_available_mb": avail and round(avail / mib),
        "system_total_mb": total and round(total / mib),
    }


def _settle_memory(timeout=4.0, interval=0.15):
    """Wait for a release to finish, then snapshot. Returns (snapshot, seconds).

    Freeing is not synchronous with `del pipe`: the GPU driver hands pages
    back over ~2 s (measured 358H/B390 — system-available climbed 3.2 GB
    *after* unload() returned). Reporting the first number we see would
    under-report by that much, so poll until availability stops rising.
    """
    t0 = time.perf_counter()
    best = _memory_snapshot()
    flat = 0
    while time.perf_counter() - t0 < timeout:
        time.sleep(interval)
        now = _memory_snapshot()
        prev, cur = best["system_available_mb"], now["system_available_mb"]
        if prev is None or cur is None:
            return now, round(time.perf_counter() - t0, 1)
        if cur > prev + 16:          # still climbing (16 MB = noise floor)
            best, flat = now, 0
            continue
        if cur > prev:               # drifting up in the noise — keep the best
            best = now
        flat += 1
        if flat >= 3:                # three quiet samples — it's done
            break
    return best, round(time.perf_counter() - t0, 1)
