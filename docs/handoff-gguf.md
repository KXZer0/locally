# GGUF on the OpenVINO GPU path — measured (2026-09-03)

The historical `docs/archive/REBUILD-PLAN.md` §3.2 asked for a 10-minute experiment before any GGUF
support is built. This is that experiment, plus the A/B it made possible.

## It works

`LLMPipeline("…Q4_K_M.gguf", "GPU")` loads and generates. Tokenizer is sane —
`<|im_end|>` is respected, not emitted as text — and the answer is coherent. So
the reader is real and §3.2's premise holds.

## The A/B nobody had run: same weights, both formats, same device

`huihui-ai/Huihui-Qwen3-8B-abliterated-v2` exists here as an OpenVINO IR I
exported (TASK A) *and* as a community GGUF of the identical model. That makes
this a comparison of **formats**, not of models — the usual confound removed.
Both on GPU, same 20 prompts, same greedy config, same run
(`scripts/npu-ab.py --device GPU`):

| | OpenVINO IR (int4-cw) | GGUF Q4_K_M |
|---|---|---|
| load | **15.9 s** | 39.3 s |
| decode | **31.7 tok/s** | 25.6 tok/s |
| 20-prompt factual | 20/20 | 20/20 |
| on disk | **4.42 GB** | 4.68 GB |

**Quality is indistinguishable and the IR is faster on every axis** — 2.5× to
load, 24% to decode, and 6% smaller. GGUF's advantage is that it needs no
conversion step and the abliterated library is published in it; that advantage is
real but it is convenience, not performance.

### Correction to §3.2d

The plan says "a GGUF `Q4_K_M` of the same model is typically **5-10% smaller**
than OpenVINO int4". Measured, it is **6% LARGER** — 4.68 GB against 4.42. Both
can be true: §3.2d compared against int4 **g64**, which carries per-group scales
and zero-points, while the NPU-safe export this project actually uses is
`--group-size -1` **channel-wise**, which carries far fewer. Against the export
locally really makes, GGUF is not the smaller artifact.

## `enable_save_ov_model=True` is worse than not setting it

§3.2 step 2 wants the converted IR cached so a second load skips conversion. The
flag does not do that, and it does damage on the way:

* It wrote `openvino_model.bin` + tokenizer/detokenizer — **5.3 GB** — into the
  *directory containing the GGUF*, which here holds four unrelated GGUFs. The
  filenames carry no trace of which model they came from, so converting a second
  GGUF in that directory would silently overwrite the first.
* It was **not reused**. Second load 39.3 s against a first load of 37.6 s — no
  saving at all, so the 5.3 GB bought nothing.

So §3.2's `.ov-cache/gguf/<name>/` destination is not a nicety, it is the fix:
the artifact must be keyed by source model and kept out of the user's model
directory. Until that is built, **do not pass `enable_save_ov_model`** — it
costs disk and pollutes `~/models/gguf/`.

## What this means for §3.2

GGUF support is worth having for reach, not for speed. The honest framing for the
README is: *GGUF runs, on GPU/CPU only, at about three quarters of the decode
speed of an equivalent IR, and it exists so the community-published models — the
abliterated builds in particular — can be run at all without a conversion step.*

Anyone converting a model they will use repeatedly should export the IR
(`optimum-cli export openvino --weight-format int4 --sym --group-size -1`) and
keep that instead.

## Architecture support: §3.2 predicted it exactly

All four downloaded GGUFs, loaded on GPU:

| GGUF | result |
|---|---|
| `Huihui-Qwen3-8B-abliterated-v2` (qwen3 dense) | **loads**, 25.6 tok/s, correct answers |
| `Qwen2.5-Coder-7B-Instruct` (qwen2 dense) | **loads** in 29.0 s, answers `4` |
| `gemma-4-12b-heretic` | **fails** after 12.4 s |
| `Qwen3.8-27B-Heretic` | **fails** after 35.3 s |

So the reader is dense llama/qwen2/qwen3 and nothing else, precisely as §3.2
scoped it. The two failures are the two architectures the plan named.

**The failure message is the problem.** Both unsupported models die with:

```
IndexError: invalid unordered_map<K, T> key
```

That is a raw C++ map lookup escaping through the bindings. It names no model,
no architecture, and no remedy, and it arrives 12-35 seconds in, so it reads
like a crash rather than a refusal. §3.2 step 4 already asks for MoE GGUFs to be
refused "with a plain message and a pointer to the IR path"; this measurement
says the check has to happen **before** the load, on the architecture string in
the GGUF header, because the reader itself will not produce a usable error.
`explain_genai_error` should also learn this one — it is the single most
confusing error a user can currently get from a supported-looking file.
