"""What the machine has free, committed, and what this process is holding.

Do NOT measure a model unload with RSS: an idle process has already had its
working set trimmed by Windows, which is how 'unload frees nothing' was once
concluded. Physical availability and commit headroom are reported separately:
they answer different questions. `_process_memory` reads PrivateUsage;
`_settle_memory` polls because the GPU driver releases asynchronously (~2 s,
~3 GB late)."""

import ctypes
import os
import time


def _windows_performance_info():
    """Windows' system-wide commit, cache, and kernel counters in bytes.

    ``GetPerformanceInfo`` reports these counters in pages. Unlike
    ``GlobalMemoryStatusEx``'s page-file fields, they are the system commit
    counters Task Manager labels Commit Total and Commit Limit.
    """
    SIZE_T = ctypes.c_size_t

    class _PerformanceInformation(ctypes.Structure):
        # PERFORMANCE_INFORMATION from psapi.h. SIZE_T follows the native
        # pointer width; the three trailing counters are DWORD.
        _fields_ = [("cb", ctypes.c_ulong),
                    ("CommitTotal", SIZE_T),
                    ("CommitLimit", SIZE_T),
                    ("CommitPeak", SIZE_T),
                    ("PhysicalTotal", SIZE_T),
                    ("PhysicalAvailable", SIZE_T),
                    ("SystemCache", SIZE_T),
                    ("KernelTotal", SIZE_T),
                    ("KernelPaged", SIZE_T),
                    ("KernelNonpaged", SIZE_T),
                    ("PageSize", SIZE_T),
                    ("HandleCount", ctypes.c_ulong),
                    ("ProcessCount", ctypes.c_ulong),
                    ("ThreadCount", ctypes.c_ulong)]

    info = _PerformanceInformation(cb=ctypes.sizeof(_PerformanceInformation))
    fn = ctypes.windll.psapi.GetPerformanceInfo
    fn.argtypes = [ctypes.POINTER(_PerformanceInformation), ctypes.c_ulong]
    fn.restype = ctypes.c_int  # BOOL
    names = ("system_commit_total", "system_commit_limit", "system_cache",
             "system_kernel_total", "system_kernel_paged",
             "system_kernel_nonpaged")
    if not fn(ctypes.byref(info), ctypes.sizeof(info)) or not info.PageSize:
        return dict.fromkeys(names)
    page = int(info.PageSize)
    return {"system_commit_total": int(info.CommitTotal) * page,
            "system_commit_limit": int(info.CommitLimit) * page,
            "system_cache": int(info.SystemCache) * page,
            "system_kernel_total": int(info.KernelTotal) * page,
            "system_kernel_paged": int(info.KernelPaged) * page,
            "system_kernel_nonpaged": int(info.KernelNonpaged) * page}


def _parse_proc_meminfo(lines):
    """Return /proc/meminfo values in bytes, preserving unitless counters."""
    factors = {"b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3}
    values = {}
    for line in lines:
        fields = line.split()
        if len(fields) < 2 or not fields[0].endswith(":"):
            continue
        try:
            value = int(fields[1])
        except ValueError:
            continue
        # /proc/meminfo normally says kB, but a few counters are unitless.
        # Never multiply a unitless count merely because most entries are kB.
        factor = factors.get(fields[2].lower(), 1) if len(fields) > 2 else 1
        values[fields[0][:-1]] = value * factor
    return values


def _memory_status():
    """Physical and commit capacity in bytes, or ``None`` where unavailable.

    ``physical_available`` is Windows' ``ullAvailPhys`` / Linux
    ``MemAvailable``: memory an allocation can use without forcing immediate
    reclaim. ``system_commit_available`` is the unused portion of the *system*
    commit limit. It is deliberately not labelled as reclaimable cache or as
    memory owned by this process; neither conclusion follows from these
    OS-wide counters.

    Every commit figure here comes from ``GetPerformanceInfo``, never from
    ``GlobalMemoryStatusEx``. That struct's ``ullTotalPageFile`` /
    ``ullAvailPageFile`` are the *calling process's* commit limits, so
    labelling them ``system_commit_*`` would be a plausible-looking wrong
    number — which is the one kind of error this file exists to avoid.
    """
    empty = {"physical_total": None, "physical_available": None,
             "system_commit_total": None, "system_commit_limit": None,
             "system_commit_available": None,
             "system_cache": None, "system_kernel_total": None,
             "system_kernel_paged": None, "system_kernel_nonpaged": None}
    try:
        if os.name == "nt":
            class _MemStatus(ctypes.Structure):
                _fields_ = ([('dwLength', ctypes.c_ulong),
                             ('dwMemoryLoad', ctypes.c_ulong)] +
                            [(n, ctypes.c_ulonglong) for n in (
                                'ullTotalPhys', 'ullAvailPhys',
                                'ullTotalPageFile', 'ullAvailPageFile',
                                'ullTotalVirtual', 'ullAvailVirtual',
                                'ullAvailExtendedVirtual')])
            st = _MemStatus(dwLength=ctypes.sizeof(_MemStatus))
            status_fn = ctypes.windll.kernel32.GlobalMemoryStatusEx
            status_fn.argtypes = [ctypes.POINTER(_MemStatus)]
            status_fn.restype = ctypes.c_int  # BOOL
            if status_fn(ctypes.byref(st)):
                perf = _windows_performance_info()
                committed = perf["system_commit_total"]
                limit = perf["system_commit_limit"]
                return {**empty, **perf,
                        # ullTotalPhys/ullAvailPhys are the only two fields
                        # taken from GlobalMemoryStatusEx: they really are
                        # machine-wide. The page-file fields next to them are
                        # per-process and are deliberately never read.
                        "physical_total": int(st.ullTotalPhys),
                        "physical_available": int(st.ullAvailPhys),
                        "system_commit_available": (
                            max(0, limit - committed)
                            if limit is not None and committed is not None
                            else None)}
        else:
            with open('/proc/meminfo') as f:
                values = _parse_proc_meminfo(f)
            limit = values.get('CommitLimit')
            used = values.get('Committed_AS')
            return {**empty,
                    "physical_total": values.get('MemTotal'),
                    "physical_available": values.get('MemAvailable'),
                    "system_commit_total": used,
                    "system_commit_limit": limit,
                    "system_commit_available": (
                        max(0, limit - used)
                        if limit is not None and used is not None
                        else None)}
    except Exception:
        pass
    return empty


def _mem_status():
    """(total_ram, available_ram) system-wide, in bytes. (None, None) if unknown.

    "Available" is what another process could allocate without the OS having
    to page anything out — the number that answers "can I launch this game
    now?", and the only one that showed unloading works (process RSS does
    not: see the memory entry in TODONT.md).
    """
    status = _memory_status()
    return status["physical_total"], status["physical_available"]


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
    """Physical, commit, and process counters, in MB.

    One name per quantity. An earlier cut carried each figure twice
    (`process_ws_mb` beside `process_working_set_mb`, `system_available_mb`
    beside `physical_available_mb`, `commit_*` beside `system_commit_*`),
    which is how a client ends up rendering the same number in two rows and a
    reader concludes the machine is holding twice what it is. The only
    consumer of this dict is `/v1/models/unload`'s reply
    (`core/models/manage.py`), and it was updated with it.
    """
    mib = 2 ** 20
    ws, priv = _process_memory()
    status = _memory_status()

    def mb(value):
        return round(value / mib) if value is not None else None

    return {
        # What this process holds. Working set is what a trim moves; private
        # commit is what it still owes, and only the second tells "given back"
        # from "paged out".
        "process_working_set_mb": mb(ws),
        "process_private_commit_mb": mb(priv),
        # What the machine has. Physical availability is ullAvailPhys, i.e.
        # free + reclaimable standby, and moves for reasons that have nothing
        # to do with us.
        "physical_available_mb": mb(status.get("physical_available")),
        "physical_total_mb": mb(status.get("physical_total")),
        # System commit, from GetPerformanceInfo only (see _memory_status).
        "system_commit_total_mb": mb(status.get("system_commit_total")),
        "system_commit_limit_mb": mb(status.get("system_commit_limit")),
        "system_commit_available_mb": mb(status.get("system_commit_available")),
        # Where the rest of the machine's RAM is: the standby/system cache and
        # the kernel pools are neither ours nor "used by applications", and
        # they are the two biggest lines nobody ever attributes correctly.
        "system_cache_mb": mb(status.get("system_cache")),
        "system_kernel_total_mb": mb(status.get("system_kernel_total")),
        "system_kernel_paged_mb": mb(status.get("system_kernel_paged")),
        "system_kernel_nonpaged_mb": mb(status.get("system_kernel_nonpaged")),
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
        prev, cur = best["physical_available_mb"], now["physical_available_mb"]
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
