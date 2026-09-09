"""Will this slot honour a client's `tools` for this request?

Two questions, deliberately kept apart. _tools_supported is about the SLOT
and is what /api/show advertises. _npu_tools_affordable is about the slot AND
this particular request, which an advert cannot know in advance -- so an NPU
slot that will happily serve six tools still advertises completion-only,
and a client that sends thirty is refused with the token count."""

from core import config
from core.genai.tokens import _count_tokens
from core.tools.render import render_tools_prompt


def _tools_supported(slot):
    """Advertise unrestricted tool support on GPU/CPU/REMOTE, not NPU.

    NPU requests can still use a small tool set through _tool_capable's
    per-request token budget. This conservative advertisement avoids inviting
    clients to send an unlimited catalogue into the NPU's hard prompt cap.
    Tool turns are buffered with SSE keep-alive (see _sse_tool_stream), so
    a slow prefill does not trip the client's watchdog.

    REMOTE is included because every reason to exclude the NPU is about
    hardware this process owns, and a proxy slot owns none of it: the prompt
    cap, the model size and the planning ability all belong to whatever runs
    behind the upstream URL. Refusing tools here would mean a 4070 running a
    30B coder could not drive an agent because the machine relaying to it
    happens to have no Intel GPU.
    """
    return bool(slot) and slot.device_name in ("GPU", "CPU", "REMOTE")


def _npu_tools_affordable(slot, tools):
    """Can the NPU afford THIS tool set, as opposed to tools in general?

    Deliberately measures the rendered block rather than counting tools: two
    schemas with the same name can differ tenfold in size, and it is bytes in
    the prompt that the cap is about. Falls back to chars/3.5 when the
    tokenizer is unavailable — an estimate biased high, so an unmeasurable
    prompt errs toward refusing rather than toward overrunning the cap.

    Says nothing about planning ability, the exclusion's other justification.
    A small model given six well-described tools is a far easier problem than
    one given thirty, but "easier" is not "guaranteed" — this makes the
    attempt possible, it does not promise the answer is good.
    """
    if not slot or slot.device_name != "NPU" or not tools:
        return False
    block = render_tools_prompt(tools)
    if not block:
        return False
    n = _count_tokens(slot, block)
    if n is None:
        n = int(len(block) / 3.5)
    return n <= config.NPU_TOOL_BUDGET


def _tool_capable(slot, tools):
    """Whether this slot will honor a client's `tools` for this request.

    The device gate first (unchanged), then the NPU's budget check. Split this
    way because the two answer different questions: _tools_supported is about
    the slot and is what /api/show advertises, while this is about the slot
    AND the specific request, which an advert cannot know in advance.
    """
    return _tools_supported(slot) or _npu_tools_affordable(slot, tools)


def _tools_refused_note(slot, tools):
    """Why this turn is answering as plain chat, in terms that suggest a fix.

    'tools ignored (GPU-only feature)' was true when the gate was the device.
    Now that it is a budget, the same message would hide the one number the
    user can act on: trim the tool set and it fits.
    """
    if not slot or slot.device_name != "NPU":
        return f"tools ignored ({slot.device_name if slot else 'no'} slot)"
    block = render_tools_prompt(tools)
    n = _count_tokens(slot, block) or int(len(block) / 3.5)
    return (f"tools ignored: {len(tools)} schemas render to {n} tokens, over the "
            f"NPU's {config.NPU_TOOL_BUDGET}-token budget. Send fewer tools, or use "
            f"--agent-tools to trim them here.")
