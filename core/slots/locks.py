"""Which lock a slot serialises on.

**Every NPU slot shares one lock, on purpose.** Two models can be resident
on the NPU at once -- a chat LLM and the OCR utilities is the whole point of
the Util tab -- but submitting inference to both CONCURRENTLY kills the
device with ZE_RESULT_ERROR_DEVICE_LOST and takes the server down with it.
Reproduced by hammering /v1/util/read during generation. Sequential use of
the same two slots is completely stable: this is a concurrency limit, not a
residency one, and Flask runs threaded so nothing else would enforce it."""

import threading


_NPU_LOCK = threading.Lock()


def _device_lock(device_name):
    """The lock a slot on this device should serialise on.

    **Every NPU slot shares one lock, on purpose.** Two models can be resident
    on the NPU at once (a chat LLM and the OCR utilities is the whole point of
    the Util tab), but submitting inference to both *concurrently* kills the
    device:

        ZE_RESULT_ERROR_DEVICE_LOST, code 0x70000001
        - device hung, reset, was removed, or driver update occurred

    Reproduced 2026-08-12 on a Core Ultra X7 358H by hammering /v1/util/read
    while /v1/chat/completions was generating; it took the whole server down
    with it. Sequential use of the same two slots is completely stable — this
    is a concurrency limit, not a residency one. Flask runs threaded, so
    without this any user who typed in Chat while the Util tab was reading an
    image would hit it.

    Sharing the slots' existing `lock` (rather than adding a second one) keeps
    a single locking discipline: nothing acquires two locks, so there is no
    ordering to get wrong, and the idle watchdog's non-blocking acquire still
    behaves.

    GPU and CPU keep a lock per slot — OpenVINO multiplexes them routinely and
    no such failure has been seen there.
    """
    return _NPU_LOCK if device_name == "NPU" else threading.Lock()
