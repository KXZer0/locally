"""Does a draft model, or prompt lookup, make decode faster on THIS machine?

Speculative decoding is lossless under greedy decoding, so the only question
is speed, and the answer depends on hardware nobody can reason about from a
desk: a draft costs its own decode, verification costs a wider main-model
step, and on a GPU genai runs both on the continuous-batching backend, which
prefilled ~10x slower than the plain pipeline on the B390 (CLAUDE.md). So
this measures, per configuration and per prompt kind:

  * decode tok/s (generated tokens / time after the first token),
  * TTFT and prefill tok/s on a long prompt -- the backend change shows here,
  * draft acceptance and tokens per main-model step (1.0 = no help),
  * whether the text is identical to the plain greedy run (it must be).

It also tries ONE sampled turn with each draft, because draft + sampling
crashed genai 2026.4.1 on random-weight test models ("content_length <= ..."),
after which the pipeline silently stopped drafting. If your real pair passes,
--draft-sampling is safe for you; if it fails, leave it off.

Nothing has been measured on real weights by this project yet. Numbers from
this script are the first ones.

Usage (stop the server first; one process per device is the sane way to time):
    venv\\Scripts\\python.exe scripts\\speculative-check.py MAIN_DIR
        [--draft DIR ...] [--prompt-lookup] [--device CPU|GPU|NPU]
        [--draft-device CPU|GPU|NPU] [--n 3,5,8] [--max-new-tokens 200]
        [--runs 2] [--cache-dir .ov-cache]

Suggested pairings (a draft must share the main model's tokenizer; the script
checks that before loading anything):
  Qwen3-8B (int4-cw)      -> OpenVINO/Qwen3-0.6B-int8-ov or -int4-ov, or
                             OpenVINO/Qwen3-pruned-6L-from-0.6B-int8-ov (Intel's
                             FastDraft-style pruned draft; Intel's blog reports
                             ~1.3x for 0.6B and ~1.4x for the pruned one on
                             Core Ultra -- their numbers, not ours).
                             EAGLE-3: Tengyunw/qwen3_8b_eagle3 exported with
                             optimum-intel (venv-2026.3 stack); greedy only.
  Qwen3-8B-abliterated    -> the same Qwen3-0.6B draft (same tokenizer; the
                             acceptance rate is what abliteration may lower).
  Qwen3-Coder-30B-A3B     -> Qwen3-0.6B if the tokenizer check passes. An A3B
                             MoE already decodes like a 3B, and verifying k
                             tokens touches more experts per step (more so with
                             --offload-ratio, which streams them from disk), so
                             expect little; prompt lookup is the cheaper try.
  gemma-4-26b-a4b         -> nothing viable in locally. It is served as a VLM
                             (VLMPipeline), and genai's only VLM speculative
                             path is DFlash with a purpose-built draft. Google's
                             Gemma 4 MTP "assistant" drafters exist and genai
                             2026.4.1 has a Gemma4-MTP strategy, but only for
                             LLMPipeline (CPU/GPU, greedy), and Google itself
                             notes the MoE verify cost eats into the gain.
  Qwen3.5-9B              -> nothing: also a VLM slot here, and genai's
                             linear-attention verifier needs prefix caching off.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openvino_genai as ovg                  # noqa: E402

from core.genai import speculative            # noqa: E402

PROMPTS = {
    # Free-form: little to copy, so only a draft can help.
    "chat": "Explain in two short paragraphs why decode on a laptop is limited "
            "by memory bandwidth rather than compute.",
    # Copy-heavy: the answer repeats most of the prompt -- prompt lookup's case.
    "edit": "Rewrite this function with the variable `total` renamed to `acc`. "
            "Output only the code.\n\n```python\n"
            "def summarize(rows):\n    total = 0\n    for row in rows:\n"
            "        if row.get('ok'):\n            total += row['value']\n"
            "        else:\n            total -= row.get('penalty', 0)\n"
            "    return {'total': total, 'count': len(rows)}\n```",
    # RAG-like: quotes back from a passage.
    "rag": "Passage: The NPU allocates from system RAM and reports no memory "
           "budget of its own. The GPU ceiling on an integrated part is a "
           "Windows policy share of the same RAM.\n\nQuote the sentence that "
           "says where the NPU's memory comes from, then answer in one line.",
}
LONG = ("The quick brown fox jumps over the lazy dog. " * 220
        + "\nHow many times was the fox mentioned? Answer with a number.")


def build(main, device, cache_dir, draft=None, draft_device=None, lookup=False):
    props = {"CACHE_DIR": cache_dir} if cache_dir else {}
    if device == "NPU":
        props["MAX_PROMPT_LEN"] = 8192
    if draft:
        dprops = {"CACHE_DIR": cache_dir} if cache_dir else {}
        if (draft_device or device) == "NPU":
            dprops["MAX_PROMPT_LEN"] = 8192
        props["draft_model"] = ovg.draft_model(draft, draft_device or device, **dprops)
    if lookup:
        props["prompt_lookup"] = True
    t = time.perf_counter()
    pipe = ovg.LLMPipeline(main, device, **props)
    return pipe, time.perf_counter() - t


def gen_config(max_new, n=None, lookup=False, sample=False):
    g = ovg.GenerationConfig()
    g.max_new_tokens = max_new
    g.do_sample = sample
    if sample:
        g.temperature, g.top_p = 0.7, 0.9
    if n:
        g.num_assistant_tokens = n
    if lookup:
        g.max_ngram_size = 3
    return g


def run(pipe, prompt, cfg):
    h = ovg.ChatHistory()
    h.append({"role": "user", "content": prompt})
    t = time.perf_counter()
    r = pipe.generate(h, generation_config=cfg)
    wall = time.perf_counter() - t
    pm = r.perf_metrics
    n = int(pm.get_num_generated_tokens())
    ttft = pm.get_ttft().mean / 1000
    decode = (n - 1) / max(1e-6, wall - ttft)
    return {"text": r.texts[0], "tokens": n, "ttft": ttft, "decode": decode,
            "prompt_tokens": int(pm.get_num_input_tokens()),
            "spec": speculative.stats(r)}


def measure(label, pipe, args, n=None, lookup=False, ref=None):
    rows, out = [], {}
    for kind, prompt in PROMPTS.items():
        cfg = gen_config(args.max_new_tokens, n, lookup)
        run(pipe, prompt, cfg)   # warm this prompt shape
        best = max((run(pipe, prompt, gen_config(args.max_new_tokens, n, lookup))
                    for _ in range(args.runs)), key=lambda x: x["decode"])
        out[kind] = best["text"]
        same = "" if ref is None else ("same" if ref[kind] == best["text"] else "DIFFERENT")
        rows.append((kind, best, same))
    longr = run(pipe, LONG, gen_config(16, n, lookup))
    prefill = longr["prompt_tokens"] / max(1e-6, longr["ttft"])
    print(f"\n{label}")
    for kind, b, same in rows:
        print(f"  {kind:5s} decode {b['decode']:6.1f} tok/s  TTFT {b['ttft'] * 1000:6.0f} ms"
              f"{speculative.log_suffix(b['spec'])}  {same}")
    print(f"  long prompt: {longr['prompt_tokens']} tokens, prefill {prefill:,.0f} tok/s")
    return out, rows


def sample_probe(args, draft, n):
    """One sampled turn with the draft, in a FRESH process: after the genai
    crash this guards against, state leaked into later pipelines of the same
    process and a broken pair once read as healthy."""
    import subprocess
    cmd = [sys.executable, os.path.abspath(__file__), args.main, "--device", args.device,
           "--_sample-probe", draft, "--n", str(n)]
    if args.draft_device:
        cmd += ["--draft-device", args.draft_device]
    if args.cache_dir:
        cmd += ["--cache-dir", args.cache_dir]
    p = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith("PROBE ")]
    if p.returncode == 0 and lines:
        return f"OK{lines[-1][6:]} -- --draft-sampling looks safe for this pair"
    why = (lines[-1][6:] if lines else (p.stderr.strip().splitlines() or ["no output"])[-1])
    return f"FAILED ({why[:140]}) -- keep --draft-sampling off"


def _run_sample_probe(args):
    pipe, _ = build(args.main, args.device, args.cache_dir, args._sample_probe,
                    args.draft_device)
    try:
        r = run(pipe, PROMPTS["chat"], gen_config(64, int(args.n.split(",")[0]), sample=True))
    except Exception as e:
        print(f"PROBE {str(e).splitlines()[0]}")
        sys.exit(1)
    print(f"PROBE {speculative.log_suffix(r['spec'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("main")
    ap.add_argument("--draft", action="append", default=[])
    ap.add_argument("--prompt-lookup", action="store_true")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--draft-device", default=None)
    ap.add_argument("--n", default="3,5", help="num_assistant_tokens values to try")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--_sample-probe", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._sample_probe:
        return _run_sample_probe(args)
    ns = [int(x) for x in args.n.split(",") if x.strip()]

    for d in args.draft:
        refusal, warning = speculative.check_tokenizers(args.main, d)
        if refusal:
            sys.exit(f"{d}: {refusal}")
        if warning:
            print(f"note: {warning}")

    pipe, secs = build(args.main, args.device, args.cache_dir)
    print(f"plain {args.device} pipeline loaded in {secs:.1f}s")
    ref, base_rows = measure("baseline (no speculation)", pipe, args)
    base = {k: b["decode"] for k, b, _ in base_rows}
    del pipe

    def compare(rows):
        return "  ".join(f"{k} {b['decode'] / base[k]:.2f}x" for k, b, _ in rows)

    for d in args.draft:
        name = os.path.basename(os.path.normpath(d))
        for n in ns:
            pipe, secs = build(args.main, args.device, args.cache_dir, d, args.draft_device)
            _, rows = measure(f"draft {name} ({speculative.draft_kind(d)}), "
                              f"num_assistant_tokens={n}, loaded in {secs:.1f}s",
                              pipe, args, n=n, ref=ref)
            print(f"  vs baseline: {compare(rows)}")
            del pipe
        print(f"  sampled turn: {sample_probe(args, d, ns[0])}")
    if args.prompt_lookup:
        for n in ns:
            pipe, secs = build(args.main, args.device, args.cache_dir, lookup=True)
            _, rows = measure(f"prompt lookup, num_assistant_tokens={n}, loaded in {secs:.1f}s",
                              pipe, args, n=n, lookup=True, ref=ref)
            print(f"  vs baseline: {compare(rows)}")
            del pipe
    print("\n'DIFFERENT' on a greedy run would be a genai bug: speculation is "
          "lossless by construction. Faster decode with a much slower long-prompt "
          "prefill is the continuous-batching backend, not the draft.")


if __name__ == "__main__":
    main()
