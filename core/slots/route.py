"""Which slot serves this request.

Images go to a vision-capable slot, text to whatever is resident. A model id
that matches nothing falls through to the default rather than erroring --
which is why a typo looks like success while one slot is loaded, and goes
silently wrong the moment a second one is.
"""
from core import runtime
from core.slots.select import _slot_serviceable


def _route_request(has_images, requested_model):
    """Pick which DeviceSlot handles this request."""
    # Explicit model@device selection overrides routing
    if requested_model:
        for slot in (runtime.primary, runtime.secondary):
            if not _slot_serviceable(slot):
                continue
            # Match "model@DEVICE" or just "model"
            slot_full = f"{slot.model_name}@{slot.device_name}"
            if requested_model in (slot_full, slot.model_name):
                return slot

    # Dual mode routing
    if _slot_serviceable(runtime.secondary):
        if has_images:
            # Images → whichever slot is a VLM
            for slot in (runtime.secondary, runtime.primary):
                if _slot_serviceable(slot) and slot.model_type == "vlm":
                    return slot
            return None  # no VLM loaded
        else:
            # Text → prefer the better/primary model
            # If GPU has a big LLM, use GPU. Otherwise use primary (NPU).
            if runtime.secondary.model_type == "llm":
                return runtime.secondary  # GPU has a big LLM — use it
            return runtime.primary  # GPU has VLM, text goes to NPU

    # Single mode — everything goes to primary
    return runtime.primary if _slot_serviceable(runtime.primary) else None
