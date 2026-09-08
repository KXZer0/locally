"""Which slot serves this request.

Images go to a vision-capable slot, text to whatever is resident. A model id
that matches nothing falls through to the default rather than erroring --
which is why a typo looks like success while one slot is loaded, and goes
silently wrong the moment a second one is. That fallthrough is deliberate and
stays: clients send model ids we have never heard of (an unconfigured default
like "gpt-4"), and refusing those would break turns that work today.

Naming a model that IS on disk is a different case, and it used to land in the
same silent fallthrough -- ask for gemma and get answered by Qwen, under
gemma's name. Now it loads. See _load_on_demand.
"""
from core import runtime
from core.slots.select import _slot_serviceable


def _match_loaded(requested_model):
    """A resident slot whose name the request asked for, or None."""
    for slot in (runtime.primary, runtime.secondary):
        if not _slot_serviceable(slot):
            continue
        # Match "model@DEVICE" or just "model"
        if requested_model in (f"{slot.model_name}@{slot.device_name}",
                               slot.model_name):
            return slot
    return None


def _load_on_demand(requested_model):
    """Load a model that is on disk but not resident. None if it isn't ours.

    This is what makes a model picker in a client work at all. With one model
    resident at a time, "change the model" can only mean a swap, and a chat
    request naming a model the server can see is an unambiguous instruction to
    make that model the one answering -- the same bargain Ollama strikes.

    The cost is honest and worth stating: the swap is synchronous and unloads
    what was there, so the turn that triggers it pays the whole load (measured
    9.3 s for Qwen3-8B on the NPU against a warm compile cache, 65 s cold) and
    the previous model is gone. That is why it fires only on an exact name
    match against what is actually on disk -- never on a typo, never on a
    client's unconfigured default, both of which keep the old fallthrough.
    """
    # Imported here rather than at module scope: manage imports the discovery
    # and slot machinery this module also sits in, and a top-level import
    # would close the loop.
    from core.models.discovery import _available_models
    from core.models.manage import swap_model

    wanted = requested_model.partition("@")[0].strip().lower()
    if not wanted:
        return None
    known = {m["name"].lower(): m for m in _available_models()}
    match = known.get(wanted)
    if match is None or match.get("loadable") is False:
        return None
    # Raises _TurnError on refusal, which the chat path already handles -- so
    # "that model cannot run here" reaches the user as itself rather than as a
    # silent answer from whatever happened to be loaded.
    swap_model(requested_model)
    return _match_loaded(requested_model) or runtime.primary


def _route_request(has_images, requested_model):
    """Pick which DeviceSlot handles this request."""
    # Explicit model@device selection overrides routing
    if requested_model:
        slot = _match_loaded(requested_model)
        if slot is not None:
            return slot
        slot = _load_on_demand(requested_model)
        if slot is not None:
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
