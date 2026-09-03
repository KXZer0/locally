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


# Preference order for a request that did not name an engine. NPU first is the
# project's whole point (it is the low-power engine); CPU last.
UTIL_ENGINES = ("npu", "gpu", "cpu")


def _util_slots():
    """Engine name -> slot, read from the module so a swap is never missed."""
    return {"npu": runtime.util_npu, "gpu": runtime.util_gpu,
            "cpu": runtime.util_cpu}


def _loaded_util_engines():
    """Engines that can serve a utility request right now, NPU -> GPU -> CPU."""
    slots = _util_slots()
    return [name for name in UTIL_ENGINES if _slot_serviceable(slots[name])]


def _default_util_engine():
    """The engine for a request that omitted one.

    It used to be the literal "npu", which on a machine with no NPU (the RTX
    4070 desktop running `--util-engines cpu`) turned every engine-less request
    into a 503 telling the owner to provision an accelerator they do not have,
    while a loaded, serving CPU slot sat one field away. Defaulting to what is
    actually loaded costs nothing on an Intel box — the NPU is first in the
    order — and makes the CPU-only box work. The bare "npu" fallback is kept
    for the nothing-loaded case so the error below still names the engine an
    Intel user was expecting.
    """
    return next(iter(_loaded_util_engines()), "npu")


def _util_slot(engine):
    """Resolve an engine name to a slot. An explicit choice is authoritative.

    The GPU is the escalation for dense or multi-page input, not a general
    "better" setting — on this hardware the NPU is faster for some utilities
    and the GPU is usually busy holding a chat model. Falling back silently
    would hide which engine actually ran, so an unavailable engine is an error
    that names itself — and names what IS loaded, because "no models on the
    NPU" with no further help is indistinguishable from "this build has no
    utilities" when the CPU is sitting there ready.
    """
    want = (engine or _default_util_engine()).lower()
    if want not in UTIL_ENGINES:
        raise _TurnError(openai_error(
            f"Unknown engine '{engine}'. Use 'npu', 'gpu' or 'cpu'."))
    blocked = _coding_mode_error("the local utility models")
    if blocked is not None:
        raise _TurnError(blocked)
    slot = _util_slots()[want]
    if not slot or not _slot_serviceable(slot):
        others = [e for e in _loaded_util_engines() if e != want]
        if others:
            names = " and ".join(e.upper() for e in others)
            verb = "has" if len(others) == 1 else "have"
            hint = (f"The {names} {verb} utility models loaded; "
                    f"pass engine={others[0]}.")
        else:
            # Only now is fetching models the right advice: on a box with a
            # loaded engine it sends the owner after hardware they don't need.
            hint = ("Start with --util-models-dir, or fetch them with "
                    "`python scripts/npu-probe.py --all`.")
        raise _TurnError(openai_error(
            f"No utility models loaded on the {want.upper()}. {hint}",
            "server_error", 503))
    return slot
