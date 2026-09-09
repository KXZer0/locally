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
from core.errors import _TurnError, openai_error
from core.models.discovery import canonical_model_name
from core.slots.select import _slot_serviceable


def _split_requested(requested_model):
    """(canonical_bare_name, device_or_empty) from a request's `model` field.

    The bare half is folded through discovery's `canonical_model_name`, so
    routing and discovery agree on what "the same model" means -- case,
    surrounding whitespace and a trailing `@DEVICE` are all identity-neutral,
    and only one function decides that.
    """
    _, _, device = (requested_model or "").partition("@")
    return canonical_model_name(requested_model), device.strip().upper()


def _match_loaded(requested_model):
    """A resident slot whose name the request asked for, or None.

    Matches the bare model name case-insensitively against every serviceable
    slot -- the name is the only stable identity a model has: discovery
    advertises an unresident model without a device suffix at all, and a
    resident one can change device under an unchanged name (an automatic
    fallback swap, or a client that cached the id from before that swap
    happened). A device suffix on the request is honoured only to disambiguate
    two slots sharing a name; it is never grounds to refuse an otherwise exact
    match. Treating a stale or wrong-case suffix as "not loaded" would trigger
    a needless reload of the model that is already sitting there.
    """
    wanted, device = _split_requested(requested_model)
    if not wanted:
        return None
    candidates = [slot for slot in (runtime.primary, runtime.secondary)
                  if _slot_serviceable(slot) and slot.model_name
                  and canonical_model_name(slot.model_name) == wanted]
    if not candidates:
        return None
    if device:
        for slot in candidates:
            if slot.device_name.upper() == device:
                return slot
    return candidates[0]


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

    A name that IS known but cannot be loaded (a GGUF architecture the reader
    does not implement, say) is a different case again, and gets a different
    answer: refusing outright, rather than the old silent fallthrough that
    quietly kept answering under whatever was already resident. The caller
    named this model on purpose -- it is listed, with a reason, by
    /v1/models/available -- so the failure belongs to the caller to see.
    """
    # Imported here rather than at module scope: manage imports the discovery
    # and slot machinery this module also sits in, and a top-level import
    # would close the loop.
    from core.models.discovery import _available_models
    from core.models.manage import swap_model

    wanted, _ = _split_requested(requested_model)
    if not wanted:
        return None
    known = {canonical_model_name(m["name"]): m for m in _available_models()}
    match = known.get(wanted)
    if match is None:
        return None  # not ours -- caller keeps the unknown-client-default fallthrough
    if match.get("loadable") is False:
        raise _TurnError(openai_error(
            f"'{match['name']}' is on disk but cannot be loaded "
            f"({match.get('reason') or 'unsupported'}).",
            "invalid_request_error", 400))
    # Raises _TurnError on any other refusal (busy slot, bad weights, no
    # device can host it, ...), which the chat path already handles -- so
    # "that model cannot run here" reaches the user as itself rather than as a
    # silent answer from whatever happened to be loaded.
    swap_model(requested_model)
    # _match_loaded matches on the bare name across both slots regardless of
    # device, so this finds the model wherever swap_model actually placed it
    # -- including a device fallback or landing in the secondary slot. The
    # `or runtime.primary` is a last-resort fallback that should not be
    # reachable once swap_model has returned without raising.
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
        # Text → a resident LLM first (the GPU's big coder outranks the NPU),
        # then whatever else is serviceable. The fallbacks matter when one
        # slot is mid-swap or errored: a VLM answers text fine, and returning
        # a dead primary here -- which the old `return runtime.primary` did
        # unconditionally -- handed the caller a slot that could not serve.
        for slot in (runtime.secondary, runtime.primary):
            if _slot_serviceable(slot) and slot.model_type == "llm":
                return slot
        for slot in (runtime.primary, runtime.secondary):
            if _slot_serviceable(slot):
                return slot
        return None

    # Single mode — everything goes to primary
    return runtime.primary if _slot_serviceable(runtime.primary) else None
