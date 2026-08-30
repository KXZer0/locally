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
