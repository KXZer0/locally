"""Picking a slot to serve a request.

An unavailable engine is an ERROR that names itself rather than a silent
fallback: falling back would hide which engine actually ran, and the whole
point of reporting the device is that the answer is checkable.
"""
from core import runtime
from core.errors import _TurnError, openai_error
from core.coding_mode import _coding_mode_error


def _slot_serviceable(slot):
    """A slot can serve requests if loaded or just idle-unloaded (will reload)."""
    return slot and slot.status in ("ready", "idle_unloaded")


def _util_slot(engine):
    """Resolve an engine name to a slot. NPU is the default everywhere.

    The GPU is the escalation for dense or multi-page input, not a general
    "better" setting — on this hardware the NPU is faster for some utilities
    and the GPU is usually busy holding a chat model. Falling back silently
    would hide which engine actually ran, so an unavailable engine is an error
    that names itself.
    """
    want = (engine or "npu").lower()
    if want not in ("npu", "gpu", "cpu"):
        raise _TurnError(openai_error(
            f"Unknown engine '{engine}'. Use 'npu', 'gpu' or 'cpu'."))
    blocked = _coding_mode_error("the local utility models")
    if blocked is not None:
        raise _TurnError(blocked)
    slot = {"npu": runtime.util_npu, "gpu": runtime.util_gpu, "cpu": runtime.util_cpu}[want]
    if not slot or not _slot_serviceable(slot):
        raise _TurnError(openai_error(
            f"No utility models loaded on the {want.upper()}. "
            f"Start with --util-models-dir, or fetch them with "
            f"`python scripts/npu-probe.py --all`.", "server_error", 503))
    return slot
