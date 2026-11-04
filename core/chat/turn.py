"""Prepare one model turn and run server-owned tool loops."""

import json
import re
import time
from datetime import datetime

import openvino_genai as ovg

from core import config, runtime
from core.chat import structured as structured_output
from core.chat.common import apply_penalties, overall_status, parse_messages
from core.chat.images import _image_urls_in, _ocr_for_text_model
from core.errors import _TurnError, openai_error
from core.genai.results import explain_genai_error, GeneratedText
from core.genai.tokens import _count_tokens
from core.sandbox.python_exec import sandbox_status
from core.slots.capability import _tool_capable, _tools_refused_note
from core.slots.route import _route_request
from core.tools import constrain
from core.tools.builtin import (BUILTIN_TOOLS, _builtin_runs,
                                _builtin_tools_supported, _fit_tool_output,
                                builtin_tools_for)
from core.tools.gemma import has_images, thinking_requested
from core.tools.parse import parse_tool_calls
from core.tools.registry import PYTHON_TOOL
from core.tools.render import (_tool_calls_to_text, prepare_messages_for_tools,
                               render_tools_prompt)

BUILTIN_TOOL_MAX_ROUNDS = 2

def _python_tool_prompt():
    """The only prompt addition for a server-side Python turn.

    Keep this intentionally small: on the NPU this is part of the hard
    8192-token prompt budget. The surrounding render_tools_prompt wrapper is
    measured at startup and reported in /health so a model-specific tokenizer
    can be checked.
    """
    return render_tools_prompt([PYTHON_TOOL])


# ---------------------------------------------------------------------------


def _filter_tools(tools):
    """Keep only the tools in --agent-tools. Unset = keep everything.

    Tool schemas dominate an agent request — 99,044 of Claude Code's 141,959
    characters, 68%, across 30 tools (measured). Every one of them is
    prefilled on every turn, and on an iGPU prefill is the whole wait. Six
    tools is a working coding agent; the other 24 are prose the model reads
    and never uses.

    Claude Code's CLI can do this itself (`--tools`), but its VS Code
    extension exposes no such flag, and neither do most clients — so the
    server has to be able to say no. Dropped tools are logged, never silently
    swallowed: a model that can't see a tool will confidently work around it,
    and you want to know that's why.
    """
    if not config.AGENT_TOOLS or not tools:
        return tools

    def name_of(t):
        return (t.get("name")
                or (t.get("function") or {}).get("name", ""))

    kept = [t for t in tools if name_of(t).split("(")[0] in config.AGENT_TOOLS]
    dropped = len(tools) - len(kept)
    if dropped:
        print(f"{datetime.now():%H:%M:%S} -- tool filter: kept {len(kept)} of "
              f"{len(tools)} ({dropped} dropped: "
              f"{', '.join(sorted(name_of(t) for t in tools if t not in kept))[:120]})",
              flush=True)
    return kept


def _with_no_think(messages):
    """Copy of `messages` with /no_think appended to the last user turn."""
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") != "user":
            continue
        c = out[i].get("content")
        if isinstance(c, str):
            if "/no_think" not in c:
                out[i] = dict(out[i], content=c.rstrip() + " /no_think")
        elif isinstance(c, list):
            parts = list(c)
            for j in range(len(parts) - 1, -1, -1):
                if isinstance(parts[j], dict) and parts[j].get("type") == "text":
                    parts[j] = dict(parts[j], text=parts[j]["text"].rstrip() + " /no_think")
                    out[i] = dict(out[i], content=parts)
                    break
        break
    return out


def _prepare_turn(body, native=False):
    """Everything a chat turn needs before generation, for either dialect.

    Parse, route to a device, reload it if it was idle-unloaded, and build the
    generation config. The OpenAI and Anthropic endpoints differ only in how
    the request and the answer are *shaped* — the work between those two points
    is identical, and duplicating it is how the two would drift apart.

    `native=True` (the OpenAI endpoint) lets a Gemma 4 slot answer a tool turn
    in its own dialect: `turn["native_prompt"]` is then the finished prompt and
    `turn["thinking"]` says whether the thought channel was switched on.
    """
    if overall_status() != "ready":
        raise _TurnError(openai_error(
            f"Server not ready (status: {overall_status()}). "
            "Check GET /health.", "server_error", 503))
    if not body:
        raise _TurnError(openai_error("Request body must be JSON"))
    messages = body.get("messages")
    if not messages:
        raise _TurnError(openai_error("'messages' is required"))

    max_tokens = body.get("max_tokens", 4096)
    tools = _filter_tools(body.get("tools") or [])
    try:
        structured = structured_output.from_openai(body.get("response_format"))
    except structured_output.StructuredRequestError as e:
        raise _TurnError(openai_error(str(e)))

    try:
        text_prompt, images, raw_messages = parse_messages(messages, config.MAX_IMAGE_DIM)
    except (FileNotFoundError, ValueError) as e:
        raise _TurnError(openai_error(str(e)))
    except Exception as e:
        raise _TurnError(openai_error(f"Failed to parse request: {e}"))

    slot = _route_request(bool(images), body.get("model", ""))

    # Anthropic requests without extended thinking: Qwen3's /no_think, on a COPY
    # of the last user turn (CLAUDE.md: never keep the token in history).
    if body.get("no_think") and slot is not None and \
            re.match(r"qwen3-", (slot.model_name or "").lower()):
        messages = _with_no_think(messages)
        try:
            text_prompt, images, raw_messages = parse_messages(
                messages, config.MAX_IMAGE_DIM)
        except Exception as e:
            raise _TurnError(openai_error(f"Failed to parse request: {e}"))

    # Images with no vision path: read them with the NPU utilities and hand a
    # TEXT model the result, instead of refusing the turn. Done here rather
    # than in the web UI so every client gets it — Ollama, OpenAI SDKs and
    # Claude Code all become document-capable against an NPU chat model.
    vision_path = None
    if images and (slot is None or slot.model_type == "llm"):
        text_slot = slot if (slot and slot.model_type == "llm") else \
            _route_request(False, body.get("model", ""))
        ocr_block, trimmed = _ocr_for_text_model(messages, text_slot)
        if ocr_block:
            slot = text_slot
            images = []                       # answered as a plain text turn
            # The marker parse_messages left behind is exactly where the image
            # was, so the text lands in the right turn rather than at the end.
            marker = re.compile(r"<ov_genai_image_\d+>")
            text_prompt = marker.sub("", text_prompt).strip()
            text_prompt = f"{ocr_block}\n\n{text_prompt}" if text_prompt else ocr_block
            for msg in reversed(raw_messages):
                if msg.get("role") == "user":
                    stripped = marker.sub("", msg.get("content", "")).strip()
                    msg["content"] = (f"{ocr_block}\n\n{stripped}"
                                      if stripped else ocr_block)
                    break
            vision_path = "npu-ocr"
            note = f", {trimmed}" if trimmed else ""
            print(f"{datetime.now():%H:%M:%S} -- [{slot.device_name}] "
                  f"read {len(_image_urls_in(messages))} image(s) via OCR "
                  f"({len(ocr_block)} chars{note})", flush=True)

    if slot is None:
        if images:
            raise _TurnError(openai_error(
                "No vision model loaded. Send text only, or load a VLM."))
        raise _TurnError(openai_error(
            "No model ready to handle this request.", "server_error", 503))

    # Offered on every ordinary chat turn the slot can support, rather than
    # switched on by the caller. The old body flag came from a composer button;
    # deciding whether a question needs a calculation or a search is the
    # model's job, and the tool descriptions say when.
    # Not on a constrained turn: the grammar would forbid the call markup, so
    # an offered tool could never be called -- only cost prompt tokens.
    builtin_specs = [] if structured else builtin_tools_for(slot, {
        "web_search": body.get("web_search", True) is not False,
        "calculator": body.get("calculator", True) is not False,
    })
    python_tool_active = bool(builtin_specs) and not tools
    native_builtin = bool(python_tool_active and not images and not body.get("stream") and getattr(slot, "native_chat", lambda: None)() is not None)
    if python_tool_active and not native_builtin:
        try:
            # Re-render from raw_messages when OCR has already rewritten them.
            # Parsing the ORIGINAL `messages` here threw the OCR work away: the
            # block above injects the read text into raw_messages/text_prompt
            # and clears `images`, and re-parsing the request body undid all
            # three -- handing a TEXT model its images back and dropping what
            # had just been read out of them.
            source = raw_messages if vision_path == "npu-ocr" else messages
            text_prompt, images, raw_messages = parse_messages(
                prepare_messages_for_tools(source, builtin_specs), config.MAX_IMAGE_DIM)
        except Exception as e:
            raise _TurnError(openai_error(f"Failed to prepare Python turn: {e}"))

    # GPU/CPU/REMOTE accept tools; the NPU accepts a set only when its rendered
    # schemas fit NPU_TOOL_BUDGET. Accepted sets add prompt tokens and use the
    # buffered tool-response path, even when the answer calls no tool.
    tools_active = bool(tools) and _tool_capable(slot, tools) and not python_tool_active
    if tools_active:
        try:
            text_prompt, images, raw_messages = parse_messages(
                prepare_messages_for_tools(messages, tools, slot), config.MAX_IMAGE_DIM)
        except Exception as e:
            raise _TurnError(openai_error(f"Failed to parse request: {e}"))
    elif tools:
        print(f"{datetime.now():%H:%M:%S} -- [{slot.device_name}] "
              f"{_tools_refused_note(slot, tools)}", flush=True)

    # A tool turn keeps its tools and drops the format: a constrained answer
    # cannot be a tool call. Said in a header, never silently.
    structured_skipped = bool(structured and tools_active)
    if structured_skipped:
        print(f"{datetime.now():%H:%M:%S} -- [{slot.device_name}] response_format "
              "ignored: tools are active on this turn", flush=True)
        structured = None
    if structured is not None and structured_output.refusal(slot):
        raise _TurnError(structured_output.refusal_response(slot))

    if images and slot.model_type == "llm":
        # Only reachable when OCR fusion above could not run (no utility models
        # installed, or every image failed to read).
        raise _TurnError(openai_error(
            f"Model '{slot.model_name}' on {slot.device_name} does not support "
            "images, and the OCR utilities are not available to read them. "
            "Load a VLM, or fetch the utility models with "
            "`python scripts/npu-probe.py --all`."))

    try:
        slot.ensure_loaded()
    except Exception as e:
        raise _TurnError(openai_error(f"Failed to reload model: {e}",
                                      "server_error", 500))

    gen = ovg.GenerationConfig()
    gen.max_new_tokens = max_tokens
    temperature = body.get("temperature", 0.0)
    if temperature and temperature > 0.01:
        gen.do_sample = True
        gen.temperature = temperature
        gen.top_p = body.get("top_p", 1.0)
    else:
        gen.do_sample = False
        gen.top_k = 1
    apply_penalties(gen,
                    repetition=body.get("repetition_penalty"),
                    frequency=body.get("frequency_penalty"),
                    presence=body.get("presence_penalty"))
    structured_output.apply(gen, structured)

    # A thought channel cannot open under a JSON grammar; asking for one would
    # put the reasoning inside the document.
    native_prompt = None
    thinking = thinking_requested(body) and structured is None
    if native and not images and not python_tool_active:
        chat = slot.native_chat()
        if chat is not None and not has_images(messages):
            try:
                native_prompt = chat.render(messages, tools if tools_active else [], thinking)
            except Exception as e:
                print(f"{datetime.now():%H:%M:%S} -- [{slot.device_name}] native Gemma "
                      f"prompt failed ({e}); using the generic tool prompt", flush=True)
            else:
                # A repetition penalty looks at the prompt too, and an agent
                # prompt is full of the very ids and URLs a tool call must copy
                # back exactly; the 1.05 default corrupts them.
                if body.get("repetition_penalty") is None:
                    gen.repetition_penalty = 1.0
                if thinking and tools_active:
                    from core.tools.native_diagnostics import tune_agent_generation
                    tune_agent_generation(gen, body)
                print(f"{datetime.now():%H:%M:%S} -- [{slot.device_name}] native Gemma prompt: "
                      f"{len(native_prompt)} chars, thinking={'on' if thinking else 'off'}",
                      flush=True)

    # Once the model writes <tool_call>, only a declared tool with schema-valid
    # arguments can follow (core/tools/constrain.py). Not on Gemma's native
    # dialect, whose arguments no JSON schema describes.
    tool_constraint = None
    if tools_active:
        tool_constraint = constrain.build(slot, tools, body.get("tool_choice"),
                                          native=native_prompt is not None)
        constrain.apply(gen, tool_constraint)

    stream = bool(body.get("stream", False))
    n_images = len(images)
    tag = f"{n_images} image{'s' if n_images != 1 else ''}, " if n_images else ""
    print(f"\n{datetime.now():%H:%M:%S} <- [{slot.device_name}] {tag}"
          f"{len(text_prompt)} chars, max_tokens={gen.max_new_tokens}"
          f"{' (stream)' if stream else ''}", flush=True)

    return {"slot": slot, "gen": gen, "stream": stream, "tools": tools,
            "tools_active": tools_active, "text_prompt": text_prompt,
            "images": images, "raw_messages": raw_messages,
            "python_tool_active": python_tool_active, "native_builtin": native_builtin,
            "builtin_specs": builtin_specs,
            "structured": structured, "structured_skipped": structured_skipped,
            "native_prompt": native_prompt, "thinking": thinking,
            "tool_constraint": tool_constraint,
            "max_dim": config.MAX_IMAGE_DIM,
            "max_tokens": max_tokens, "t0": time.perf_counter(),
            # "npu-ocr" when images were read as text rather than seen. Callers
            # surface it as X-Vision-Path so this is never mistaken for vision.
            "vision_path": vision_path}


# ---------------------------------------------------------------------------
# Built-in tools the SERVER owns and runs
#
# Distinct from the client's `tools` array, which locally only relays: these
# are offered to the model on every ordinary chat turn and executed here. That
# is what let the composer's mode buttons go. A "search the web" toggle asked
# the user to answer a question the model is better placed to answer -- it
# forced a search on "hello" and skipped one on a question about last week --
# and a "calculate" toggle asked the user to know in advance that a number was
# going to be hard. The tool's own description is the whole instruction; no
# system-prompt text is spent on either, which matters because that prompt is
# re-sent every turn.
#
# The gate is _tools_supported, so this is GPU/CPU only. On the NPU these are
# not offered at all -- its 8k prompt cap is the reason -- and a turn there
# answers as plain chat exactly as before.
# ---------------------------------------------------------------------------


def _builtin_tool_turn(generate, raw_messages, specs, slot_for_fit=None, native=False):
    """Run the bounded server-side tool loop and return answer + audit data.

    Bounded on purpose. An unbounded loop on a small model is how a whole
    context window gets spent re-deciding; two rounds is enough for "call the
    tool, read the result, answer" and stops a model that has decided every
    reply needs another search.

    A call to a tool that is not ours is not our business: it is left in the
    text and returned, so a client that sent its own `tools` array still gets
    its call back the way it always did.

    The answer comes back as a GeneratedText carrying the FINAL round's token
    counts and finish reason. `parse_tool_calls` returns an ordinary string, so
    without re-wrapping, every turn that reached this loop -- which is every
    ordinary turn once --search-url or --python-tool is on, since the tools are
    offered rather than requested -- reported `usage` of zero and a finish
    reason of "stop" whatever actually happened. The final round is the right
    round to report: its prompt is the one that produced the answer being
    returned, so the numbers stay comparable with a turn that called no tool.
    Summing the rounds would say what the turn COST, which is a different
    question from the one a client budgeting its next request is asking.
    """
    from core.tools.activity import record
    from core.tools.gemma import split_reasoning
    messages = list(raw_messages)
    _builtin_runs().clear()
    for round_no in range(BUILTIN_TOOL_MAX_ROUNDS + 1):
        try:
            record("thinking", "Choosing the next step…")
            generated = generate(messages)
        except Exception as e:
            raise RuntimeError(explain_genai_error(e))
        thought = ""
        if native:
            thought, visible = split_reasoning(str(generated))
        else:
            visible = generated
        text, calls = parse_tool_calls(visible, specs)

        def answer(body):
            """Re-attach what parse_tool_calls dropped off the string."""
            result = GeneratedText(body,
                                 getattr(generated, "finish_reason", "stop"),
                                 getattr(generated, "usage", None))
            result.reasoning = thought
            return result

        accepted = []
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name")
            if name not in BUILTIN_TOOLS:
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (ValueError, TypeError):
                args = {}
            accepted.append((call, name, args if isinstance(args, dict) else {}))
        if not accepted:
            record("answering", "Writing the answer…")
            return answer(text.strip()), list(_builtin_runs())
        if round_no >= BUILTIN_TOOL_MAX_ROUNDS:
            return answer(text.strip() or
                          "I stopped after the tool-round limit."), list(_builtin_runs())

        results = []
        for call, name, args in accepted:
            try:
                results.append((name, _fit_tool_output(
                    slot_for_fit, BUILTIN_TOOLS[name]["run"](args))))
            except Exception as e:
                # A tool that fails must not kill the turn -- the model can say
                # it could not check, which is far better than a 500.
                results.append((name, "The tool failed: " + str(e)))

        if native:
            messages.append({"role":"assistant", "content":text, "tool_calls":[c for c, _n, _a in accepted]})
            for (call, name, _args), (_name, output) in zip(accepted, results):
                messages.append({"role":"tool", "tool_call_id":call["id"], "name":name, "content":output})
            continue
        call_text = _tool_calls_to_text([c for c, _n, _a in accepted])
        messages.append({"role": "assistant",
                         "content": (text + "\n" + call_text).strip()})
        body = "\n\n".join('<tool_response name="' + n + '">\n' + r +
                           "\n</tool_response>" for n, r in results)
        messages.append({
            "role": "user",
            "content": (body + "\n\nUse the result above to answer the user. "
                        "Do not call another tool unless it is genuinely "
                        "necessary."),
        })
    return "I could not complete that.", list(_builtin_runs())


# ---------------------------------------------------------------------------
# Web search — the answer engine
#
# The point is not "the model can browse". It is that a small local model
# guesses when it does not know, and guessing is what sends you to Google. So
# the model is never asked to recall: it is handed passages and told to quote
# them or say it does not know.
#
# The accuracy comes from the reranker, not the model. Retrieval is deliberately
# wide (a bi-encoder over every chunk) and then reordered by the BGE cross-
# encoder on the NPU — measured on this project's own corpus, that took top-1
# from 7/9 to 8/9 and MRR from 0.889 to 0.944. Widening retrieval is free
# accuracy; the work is in the ordering.
#
# This is the server's FIRST outbound HTTP. It is opt-in behind --search-url and
# does nothing at all when unset, so "works on a plane" survives: with no flag,
# locally makes no external connection, exactly as before.
#
# Backend is a self-hosted SearXNG (JSON output, no API key, no per-query
# billing, no queries leaving the box beyond the engines it proxies). Brave
# removed its free tier in February 2026 and scraping DuckDuckGo breaks silently.


def python_tool_status():
    """Describe the opt-in local calculation tool for /health and the UI."""
    prompt = _python_tool_prompt()
    token_count = None
    if config.PYTHON_TOOL_ENABLED and runtime.primary and runtime.primary.status != "not_configured":
        try:
            token_count = _count_tokens(runtime.primary, prompt)
        except Exception:
            pass
    return {
        "enabled": bool(config.PYTHON_TOOL_ENABLED),
        "timeout_s": config.PYTHON_TOOL_TIMEOUT,
        "output_bytes": config.PYTHON_TOOL_OUTPUT_BYTES,
        "max_code_bytes": config.PYTHON_TOOL_MAX_CODE_BYTES,
        "max_rounds": BUILTIN_TOOL_MAX_ROUNDS,
        "prompt_bytes": len(prompt.encode("utf-8")),
        "prompt_tokens": token_count,
        **sandbox_status(),
        # A user must never think they are containerised when they are not,
        # so the boundary in force is reported, not the best one available.
        "network": ("none (Podman --network=none)"
                    if sandbox_status()["sandbox"] == "podman"
                    else "subprocess import guardrails only"),
        "reason": None if config.PYTHON_TOOL_ENABLED else
                  "start with --python-tool to enable local calculations",
    }


# ---------------------------------------------------------------------------
