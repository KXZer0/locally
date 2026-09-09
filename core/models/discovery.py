"""Discover loadable models and choose the device that can host one."""

import os

from core import config, runtime
from core.hardware.devices import _device_mem_bytes, _gpu_has_xmx
from core.models.availability import add_live_fit_data
from core.models.gguf import unsupported_reason as gguf_unsupported_reason
from core.models.describe import _model_dirs_under
from core.models.geometry import _moe_expert_fraction
from core.models.identity import _is_generative_dir, is_vlm, model_display_name
from core.models.integrity import _dir_size_bytes
from core.models.irinfo import read_ir_rt_info
from core.slots.select import _slot_serviceable


def canonical_model_name(raw):
    """The stable, case-folded identity of a model id.

    A device suffix ("name@DEVICE") is stripped: it is compatibility syntax
    for the ids this module used to advertise, not part of a model's
    identity, and _models_data() never emits one for a model that isn't
    resident. Comparing on this form -- rather than the raw string -- is
    what lets a resident model survive a device-fallback swap, a client's
    stale cached "name@NPU", or different case, without being mistaken for
    a different model and reloaded for no reason. core/slots/route.py is
    the other half of this: it matches requests against the same form.
    """
    return (raw or "").partition("@")[0].strip().lower()


def _models_data():
    # A slot that is merely idle-unloaded still SERVES -- _slot_serviceable and
    # overall_status() both count it, and the chat path reloads it on demand --
    # so listing only "ready" advertised nothing after the idle timeout while
    # the server went on answering. A client that discovers models by polling
    # this endpoint (Odysseus does) reads the empty list as "backend offline",
    # which is what it looks like fifteen quiet minutes after you last used it.
    data = []
    for slot in (runtime.primary, runtime.secondary):
        if _slot_serviceable(slot):
            data.append({
                "id": f"{slot.model_name}@{slot.device_name}",
                "object": "model",
                "created": 0,
                "owned_by": f"local-{slot.device_name.lower()}",
            })
    if runtime.whisper_slot and runtime.whisper_slot.status == "ready":
        data.append({
            "id": f"whisper@{runtime.whisper_slot.device_name}",
            "object": "model",
            "created": 0,
            "owned_by": f"local-{runtime.whisper_slot.device_name.lower()}",
        })

    # Models on disk that nothing is holding. A client's model picker is built
    # from this endpoint, so listing only what is resident meant the picker
    # offered exactly one choice on a one-model-at-a-time server -- a dropdown
    # with a single possible answer, which is not a choice. Asking for one of
    # these in a chat request loads it (core/slots/route.py:_load_on_demand),
    # so what is advertised and what can actually answer stay the same set.
    #
    # No @DEVICE suffix: placement is decided at load time by _choose_device
    # reading the IR, and naming a device here would promise one the model may
    # not be able to use.
    resident = {canonical_model_name(s.model_name)
                for s in (runtime.primary, runtime.secondary)
                if _slot_serviceable(s) and s.model_name}
    for model in _available_models():
        if canonical_model_name(model["name"]) in resident \
                or model.get("loadable") is False:
            continue
        data.append({
            "id": model["name"],
            "object": "model",
            "created": 0,
            "owned_by": "local-available",
        })
    return {"object": "list", "data": data}


def _available_models():
    """Model directories on disk that could be loaded into a slot.

    Searched: --models-dir if given, else the parents of whatever is already
    loaded (so `--model-dir ~/models/foo` makes all of ~/models visible)
    plus the script dir and ~/models. One model reached by two paths is one
    model — install.ps1 links model/ at a directory in ~/models.
    """
    roots = []
    if runtime.MODELS_DIR:
        roots.append(runtime.MODELS_DIR)
    else:
        for slot in (runtime.primary, runtime.secondary):
            if slot and slot.model_dir:
                roots.append(os.path.dirname(os.path.abspath(slot.model_dir)))
        roots += [config.SCRIPT_DIR, os.path.expanduser("~/models")]

    seen, out = set(), []
    for root in roots:
        for d in _model_dirs_under(os.path.normpath(os.path.expanduser(root)), 1):
            real = os.path.realpath(d)
            if real in seen or not _is_generative_dir(d):
                continue
            seen.add(real)
            entry = {"name": model_display_name(d), "path": d,
                     "type": "vlm" if is_vlm(d) else "llm"}
            # A GGUF the reader cannot open is still a file the user put there
            # on purpose. Hiding it invites "where did my model go"; listing it
            # silently invites a load that fails. Listed, with the reason
            # attached, so the UI can show it disabled and say why.
            unsupported = gguf_unsupported_reason(d)
            if unsupported:
                entry["loadable"] = False
                entry["reason"] = unsupported
            out.append(entry)
    return sorted(out, key=lambda m: m["name"].lower())


def _device_can_host(device_name, device_id, model_dir, vlm):
    """(ok, reason) — whether this device can serve this model at all.

    Capability first, memory second: a model that the NPU cannot execute is
    not "a tight fit", it's the wrong device, and saying so beats letting the
    driver fail ten minutes later.
    """
    if device_name == "NPU":
        if str(model_dir).lower().endswith(".gguf"):
            # GGUF quantization is block-wise (Q4_K_M is 32-wide super-blocks
            # with their own scales); the vpux compiler needs channel-wise IR
            # and crashes on group-quantized weights. Nothing about the file
            # can be adjusted to change that -- it needs a different export.
            return False, ("GGUF is block-quantized; the NPU needs a "
                           "channel-wise OpenVINO IR (-Weight int4-cw)")
        if vlm:
            return False, "NPU has no working vision path"
        # Group-quantized int4 crashes the NPU driver compiler ("Found N
        # duplicated names") — channel-wise is required. Read what the
        # weights actually are, not what the folder is called.
        try:
            rt = read_ir_rt_info(model_dir)
            gs = rt.get("nncf/weight_compression/group_size")
            mode = (rt.get("nncf/weight_compression/mode") or "")
            if "int4" in mode and gs not in (None, "", "-1", -1):
                return False, f"NPU needs channel-wise int4 (this is group_size {gs})"
        except Exception:
            pass

    mem = _device_mem_bytes(device_name, device_id)
    weights = _dir_size_bytes(model_dir)
    if not mem or not weights:
        # The NPU exposes no memory-budget property — it allocates from system
        # RAM on demand — so "unknown" here is normal, not a warning sign.
        return True, ("no budget reported (allocates from system RAM)"
                      if device_name == "NPU" else "fit unknown")
    kv = (config.PROMPT_CACHE_GB * 2 ** 30
          if config.PROMPT_CACHE and not vlm and device_name in ("GPU", "CPU") else 0)
    need = (weights + kv) * 1.1
    if need > mem:
        # MoE disk offload keeps only part of the experts resident, so a model
        # over budget can still fit — don't rule the device out on size alone.
        # Only where offload actually does something, though: a dense model or
        # a non-XMX GPU would sail past this check and then fail to load.
        if (config.OFFLOAD_RATIO and device_name == "GPU"
                and _moe_expert_fraction(model_dir) and _gpu_has_xmx(device_id)):
            return True, ("over budget, fits by streaming expert weights"
                          if config.OFFLOAD_RATIO == "auto" else
                          f"over budget but --offload-ratio {config.OFFLOAD_RATIO} is on")
        return False, (f"needs ~{need / 2**30:.1f} GB, "
                       f"{device_name} budget is {mem / 2**30:.1f} GB")
    return True, f"fits ({need / 2**30:.1f}/{mem / 2**30:.1f} GB)"


def _choose_device(model_dir, preferred=None):
    """Pick which DEVICE should host a model. Returns (name, id, why).

    Considers every device on the machine, not only ones that already hold a
    slot — otherwise a single-slot ("one model at a time") setup could never
    use the NPU, and the one model the NPU can actually run would be stuck on
    the GPU. The slot is moved to the chosen device by the caller.
    """
    vlm = is_vlm(model_dir)
    # CPU is a legal target but a poor default (see TODONT.md), so it's only
    # considered when nothing else can host the model.
    order = [k for k in ("GPU", "NPU", "CPU") if k in runtime.DEVICES]
    notes = []

    def dev_id(kind):
        return runtime.DEVICES.get(kind, {}).get("id", kind)

    if preferred:
        if preferred not in runtime.DEVICES:
            return None, None, f"no device '{preferred}' on this machine"
        ok, why = _device_can_host(preferred, dev_id(preferred), model_dir, vlm)
        if ok:
            return preferred, dev_id(preferred), why
        notes.append(f"{preferred}: {why}")

    # NPU first when it can actually run the model. locally is NPU-first by
    # design, and the NPU is the low-power engine — a text model it can host
    # belongs there, leaving the GPU free. _device_can_host has already ruled
    # the NPU out for vision and group-quantized int4, so anything reaching
    # here genuinely runs. GPU next (it does vision and tool calling), CPU
    # last (see TODONT.md — Ollama is the better tool for CPU-only).
    rank = {"NPU": 0, "GPU": 1, "CPU": 2}

    for kind in sorted(order, key=lambda k: rank.get(k, 9)):
        if preferred and kind == preferred:
            continue
        ok, why = _device_can_host(kind, dev_id(kind), model_dir, vlm)
        if ok:
            if not notes:
                prefix = ""
            elif preferred:
                prefix = f"moved from {preferred} ({'; '.join(notes)}); "
            else:
                # No device was asked for — the notes explain what was skipped
                # on the way here, which is the useful part.
                prefix = f"ruled out {'; '.join(notes)}; "
            return kind, dev_id(kind), f"{prefix}{kind}: {why}"
        notes.append(f"{kind}: {why}")

    return None, None, "; ".join(notes) or "no device can host this model"


def _available_models_data():
    """Models on disk, loaded or not — the menu for POST /v1/models/load."""
    loaded = {}
    for slot in (runtime.primary, runtime.secondary):
        if slot and slot.model_dir:
            loaded[os.path.realpath(slot.model_dir)] = slot.device_name
    data = []
    for m in add_live_fit_data(_available_models(), runtime.DEVICES):
        entry = dict(m)
        if "NPU" in runtime.DEVICES:
            ok, reason = _device_can_host("NPU", runtime.DEVICES["NPU"]["id"],
                                           m["path"], m["type"] == "vlm")
            entry["npu"] = {"compatible": ok and m.get("loadable", True), "reason": reason}
        else:
            entry["npu"] = {"compatible": False, "reason": "No NPU detected"}
        entry["loaded_on"] = loaded.get(os.path.realpath(m["path"]))
        data.append(entry)
    return {"object": "list", "data": data,
            "devices": [s.device_name for s in (runtime.primary, runtime.secondary) if s]}

