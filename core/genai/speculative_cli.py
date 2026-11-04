"""Command-line flags for speculative decoding (see core/genai/speculative.py)."""

import os

from core import config


def add_arguments(p):
    g = p.add_argument_group("speculative decoding")
    g.add_argument("--draft-model", action="append", default=[], metavar="DIR",
                   help="Small model from the SAME family that proposes tokens "
                        "for the main model to verify, e.g. Qwen3-0.6B for "
                        "Qwen3-8B. Lossless: greedy output is unchanged. "
                        "Repeatable; each load uses the first one whose "
                        "tokenizer matches, and loads without a draft when none "
                        "does. EAGLE-3 draft heads work too. LLM slots only.")
    g.add_argument("--draft-device", default=None, choices=["CPU", "GPU", "NPU"],
                   help="Device for the draft (default: the main model's).")
    g.add_argument("--num-assistant-tokens", type=int,
                   default=config.NUM_ASSISTANT_TOKENS, metavar="N",
                   help=f"Tokens proposed per step (default "
                        f"{config.NUM_ASSISTANT_TOKENS}; a request may send its "
                        f"own num_assistant_tokens).")
    g.add_argument("--assistant-confidence", type=float, default=0.0, metavar="X",
                   help="Instead of a fixed count, let the draft keep proposing "
                        "while its confidence stays above X (0-1). Plain drafts "
                        "only; EAGLE-3 and prompt lookup ignore it.")
    g.add_argument("--draft-sampling", action="store_true",
                   help="Let sampled (temperature > 0) turns use the draft. Off "
                        "because genai 2026.4.1 crashed on the first rejected "
                        "draft under sampling and then stopped drafting; "
                        "scripts/speculative-check.py tells you whether your "
                        "pair is affected. Without it such turns run greedy.")
    g.add_argument("--prompt-lookup", action="store_true",
                   help="Propose tokens by matching n-grams from the prompt (no "
                        "extra model): helps turns that copy text back, i.e. "
                        "code edits, RAG and tool output. Off by default; it "
                        "moves a GPU slot onto the continuous-batching backend. "
                        "Ignored when a draft is in use; not on NPU or VLM.")
    g.add_argument("--max-ngram-size", type=int, default=config.MAX_NGRAM_SIZE,
                   metavar="N",
                   help=f"Longest n-gram prompt lookup matches (default "
                        f"{config.MAX_NGRAM_SIZE}).")


def configure(args):
    drafts = []
    for entry in args.draft_model or []:
        for d in str(entry).split(","):
            d = d.strip()
            if not d:
                continue
            if not os.path.isdir(d):
                print(f"WARNING: --draft-model {d} is not a directory; ignored", flush=True)
                continue
            drafts.append(os.path.abspath(d))
    config.DRAFT_MODELS = drafts
    config.DRAFT_DEVICE = args.draft_device
    config.NUM_ASSISTANT_TOKENS = max(1, min(16, int(args.num_assistant_tokens)))
    config.ASSISTANT_CONFIDENCE = min(1.0, max(0.0, float(args.assistant_confidence)))
    config.DRAFT_SAMPLING = bool(args.draft_sampling)
    config.PROMPT_LOOKUP = bool(args.prompt_lookup)
    config.MAX_NGRAM_SIZE = max(1, min(8, int(args.max_ngram_size)))
