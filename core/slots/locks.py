"""Which lock a slot serialises on.

**Every NPU slot shares one lock, on purpose.** Two models can be resident
on the NPU at once -- a chat LLM and the OCR utilities is the whole point of
the Util tab -- but submitting inference to both CONCURRENTLY kills the
device with ZE_RESULT_ERROR_DEVICE_LOST and takes the server down with it.
Reproduced by hammering /v1/util/read during generation. Sequential use of
the same two slots is completely stable: this is a concurrency limit, not a
residency one, and Flask runs threaded so nothing else would enforce it."""

import threading
from contextlib import contextmanager


_NPU_LOCK = threading.RLock()


class SlotLock:
    """A stable slot mutex plus the NPU guard for its current destination.

    Never replace a live slot's mutex: queued requests may already hold a
    reference to it. Migration reserves the target NPU before changing device.
    """
    def __init__(self, device):
        self._device = device
        self._slot = threading.RLock()
        self._held = threading.local()

    def acquire(self, blocking=True, *, target=None):
        if not self._slot.acquire(blocking=blocking):
            return False
        npu = self._device() == "NPU" or target == "NPU"
        try:
            if npu and not _NPU_LOCK.acquire(blocking=blocking):
                self._slot.release()
                return False
        except BaseException:
            self._slot.release()
            raise
        if not hasattr(self._held, "stack"):
            self._held.stack = []
        self._held.stack.append(npu)
        return True

    def release(self):
        if self._held.stack.pop():
            _NPU_LOCK.release()
        self._slot.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    @contextmanager
    def moving_to(self, target):
        self.acquire(target=target)
        try:
            yield
        finally:
            self.release()


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
    return _NPU_LOCK if device_name == "NPU" else threading.RLock()
