"""Reading openvino-genai's results, and translating its failures.

explain_genai_error turns a native exception into something a person can act
on -- most importantly annotating 'Got unfinished GenerationStatus' with the
--cache-size-gb hint, because that error means the KV pool is too small and
says so nowhere (#21)."""

from core import config



def extract_text(result):
    """Extract text from an openvino_genai generate result."""
    if isinstance(result, str):
        return result.strip()
    for attr in ("texts", "text", "output_text", "response"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if isinstance(val, (list, tuple)):
                return val[0].strip() if val else ""
            return val.strip()
    return str(result).strip()


def extract_perf(result):
    """Pull (ttft_ms, gen_ms) off a genai result. Returns (None, None) when the
    build or pipeline doesn't provide perf_metrics — never raises.
    """
    pm = getattr(result, "perf_metrics", None)
    if pm is None:
        return None, None
    try:
        return pm.get_ttft().mean, pm.get_generate_duration().mean
    except Exception:
        return None, None


def extract_finish_reason(result, default="stop"):
    """Why generation stopped, as OpenVINO GenAI reports it.

    Every streaming turn used to be labelled "stop", so an answer cut off at
    max_new_tokens was indistinguishable from one that finished -- no client
    could tell a complete answer from a truncated one. Counting streamer
    callbacks is not a substitute: measured on Qwen3-8B/NPU with a 200-token
    budget, the callback count was 192 for a generation that plainly hit the
    cap. `DecodedResults.finish_reasons` is the pipeline's own answer.

    Falls back to `default` whenever the field is missing (older builds, other
    result types), because claiming truncation that did not happen is worse
    than the silence this replaces.
    """
    reasons = getattr(result, "finish_reasons", None)
    try:
        name = getattr(next(iter(reasons)), "name", "")
    except (TypeError, StopIteration):
        return default
    return {"LENGTH": "length", "STOP": "stop"}.get(name, default)


def explain_genai_error(e):
    """Map opaque OpenVINO GenAI runtime errors to actionable messages."""
    msg = str(e)
    if "unfinished GenerationStatus" in msg:
        # Continuous-batching scheduler couldn't fit the sequence in the KV
        # pool (seen with 30B-class models + big agent prompts, issue #21).
        return (f"{msg} — likely the KV-cache pool is too small for this "
                f"prompt: raise --cache-size-gb (currently {config.PROMPT_CACHE_GB} GB)")
    if "Compilation failed" in msg and ("NPU" in msg or "ZE_RESULT" in msg or "vpux" in msg):
        # NPU (vpux) compiler rejected the model — a model/driver-combination
        # problem, not a busy device (issue #20). Known trigger: an INT4
        # node-naming bug in older compilers (openvino#29823); also models
        # beyond the NPU envelope (>8B params).
        return (f"{msg} — the NPU compiler could not compile this model. "
                f"Usual causes: NPU driver too old for this model's INT4 "
                f"layout (update the Intel NPU driver; on Linux the "
                f"intel-npu-driver + compiler versions must match), or the "
                f"model is beyond the NPU envelope (proven NPU models are "
                f"INT4-CW, 8B params or less). Try an NPU model from the "
                f"install menu, or run this model on GPU/CPU instead.")
    if "invalid unordered_map" in msg:
        # A GGUF whose architecture the reader does not implement. The reader
        # looks a tensor name up in a per-architecture map, misses, and lets a
        # raw C++ map error out through the bindings -- naming no model, no
        # architecture and no remedy, 12-35 s into a load that looked fine.
        # Measured on gemma-4-12b-heretic (12.4 s) and Qwen3.8-27B-Heretic
        # (35.3 s); dense qwen2/qwen3 GGUFs of the same vintage load fine.
        return ("This GGUF's architecture is not supported by OpenVINO's GGUF "
                "reader, which handles dense llama, qwen2 and qwen3 only — "
                "gemma, and any hybrid-attention or MoE model, are not among "
                "them. Convert the model to an OpenVINO IR instead "
                "(optimum-cli export openvino --weight-format int4 --sym "
                "--group-size -1), or serve it from llama.cpp behind "
                "--proxy-url.")
    if "Exceeded max size of memory object allocation" in msg:
        # NOT a capacity problem, and the distinction matters: the device
        # refuses any SINGLE object over its limit, so free RAM cannot fix it.
        # The request is the attention score buffer and it grows as n^2 --
        # measured on gemma-4-26b, 30.4 GB at a 32k prompt and 268 GB at 100k
        # against a 25.3 GB per-allocation ceiling.
        return (f"{msg} — this is a per-allocation limit, not a shortage of "
                f"memory, so freeing RAM will not help. The buffer that "
                f"overflowed grows with the SQUARE of the prompt length. Send "
                f"a shorter prompt, or use a model with hybrid attention "
                f"(fewer full-attention layers), which is what lets a smaller "
                f"model reach a longer context on this hardware.")
    if "m_max_prompt_len" in msg or "max_prompt_len" in msg:
        # The NPU pipeline refuses an over-long prompt outright -- it does not
        # truncate -- and the raw message is a C++ assertion with a file and
        # line number, which tells a user nothing about what to do.
        return (f"This conversation is longer than the NPU can accept "
                f"({config.NPU_MAX_PROMPT_LEN} tokens, a hard limit of the NPU "
                f"compiler). Start a new chat (Ctrl+N), or run this model on "
                f"GPU/CPU where the window is the model's own maximum.")
    if "Could not find a model in the directory" in msg:
        # read_model() found neither openvino_model.xml nor
        # openvino_language_model.xml — usually an interrupted download that
        # left the big .bin without its .xml descriptor (issue #17).
        return (f"{msg} — the directory has no openvino_model.xml / "
                f"openvino_language_model.xml. Incomplete download or "
                f"conversion? If the directory is a link, check the link "
                f"target's contents; re-run install.ps1 or download-model.ps1 "
                f"to repair.")
    return msg
