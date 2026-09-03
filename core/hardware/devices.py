"""How much memory a device can actually be given.

GPU_DEVICE_TOTAL_MEM_SIZE is a CEILING, and on an integrated GPU it is a
policy share of the same system RAM everything else is using -- Intel's
'Shared GPU Memory Override' raises it without adding a byte. Trusting it put
a 20 GB model on a machine with 0.9 GB free, which then decoded out of the
pagefile at 0.5 tok/s. So the budget is min(ceiling, free - reserve)."""

import openvino as ov

from core import config
from .memory import _mem_status, _system_ram_bytes

_AVAILABILITY_UNSET = object()


def _device_mem_bytes(device_name, device_id):
    """Memory budget for a device, in bytes. None if unknown.

    GPU: ask the driver — on Windows iGPUs the reported figure already
    reflects the OS shared-memory policy (default ~half of RAM) and Intel's
    "Shared GPU Memory Override" driver setting, so it is the real budget,
    not a guess. CPU: total system RAM.
    """
    if device_name == "GPU":
        try:
            return int(ov.Core().get_property(device_id, "GPU_DEVICE_TOTAL_MEM_SIZE"))
        except Exception:
            return None
    if device_name == "CPU":
        return _system_ram_bytes()
    return None


def _gpu_shares_system_ram(device_id):
    """True if this GPU allocates out of system RAM rather than its own VRAM."""
    try:
        kind = str(ov.Core().get_property(device_id, "DEVICE_TYPE")).upper()
        return "INTEGRATED" in kind
    except Exception:
        return False


def _usable_gpu_bytes(device_name, device_id,
                      available_bytes=_AVAILABILITY_UNSET):
    """(usable, driver_ceiling) for a GPU, in bytes. (None, None) if unknown.

    `GPU_DEVICE_TOTAL_MEM_SIZE` is a *ceiling*, not an availability figure. On
    a discrete card the two coincide — the VRAM is there and nothing else is
    in it. On an integrated GPU the ceiling is a policy share of the same
    system RAM every other process is using, and Intel's "Shared GPU Memory
    Override" raises it without adding a single byte of memory.

    Trusting the ceiling is how an 18.3 GB model "fits" a 23.6 GB budget on a
    31.5 GB machine that had 0.9 GB free, loads with offload at 0 because the
    arithmetic said it fit, and then decodes at **0.5 tok/s** out of the
    pagefile: it fit the policy, not the machine. Measured on the 358H/B390.

    Reading live availability makes the decision track the machine the way
    ollama's layer split does. It can read low if a previous model's driver
    pages have not been handed back yet (`_settle_memory`: ~2 s), which biases
    the ratio *up* — the safe direction, since too much offload is merely
    slower while too little is the pagefile.
    """
    ceiling = _device_mem_bytes(device_name, device_id)
    if device_name != "GPU" or not ceiling:
        return ceiling, ceiling
    if not _gpu_shares_system_ram(device_id):
        return ceiling, ceiling          # discrete: the VRAM really is ours
    if available_bytes is _AVAILABILITY_UNSET:
        _total, avail = _mem_status()
    else:
        avail = available_bytes
    if avail is None:
        return ceiling, ceiling          # can't tell — keep the old behaviour
    return min(ceiling, max(0, avail - config.GPU_RESERVE_BYTES)), ceiling


def _gpu_has_xmx(device_id="GPU"):
    """Does this GPU have systolic (XMX/DPAS) matmul hardware?

    The whole MoE fusion path is gated on it in OpenVINO, so without XMX
    OFFLOAD_RATIO is silently ignored — the single most confusing failure in
    TODONT.md. `GPU_HW_MATMUL` in OPTIMIZATION_CAPABILITIES is the tell.
    """
    try:
        caps = ov.Core().get_property(device_id, "OPTIMIZATION_CAPABILITIES")
        return "GPU_HW_MATMUL" in caps
    except Exception:
        return False
