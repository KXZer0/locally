#!/usr/bin/env python3
"""Model watcher — notify when new locally-relevant models appear on Hugging Face.

Polls the Hugging Face Hub for two orgs:
  - OpenVINO  : Intel's pre-exported, ready-to-run models (what locally loads).
  - Qwen      : upstream base models (early heads-up before Intel exports them),
                filtered to the families locally actually wants (Coder / VL / Omni).

Diffs the current relevant set against a committed snapshot (seen_models.json).
New ids are reported; the snapshot is updated so you're not re-pinged. On the
very first run the snapshot is empty, so it just establishes a baseline silently
(no issue) — only genuinely *new* models after that trigger a notification.

No third-party deps (urllib only), so the GitHub Action needs no pip install.

Outputs (for GitHub Actions, via $GITHUB_OUTPUT):
  changed=true   snapshot content changed (commit it back)
  new=true       there are new models worth an issue
Writes the issue title/body to scripts/.watch_title and scripts/.watch_body.md.
"""

import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SEEN_FILE = HERE / "seen_models.json"
TITLE_FILE = HERE / ".watch_title"
BODY_FILE = HERE / ".watch_body.md"
MODELS_JSON = REPO / "models.json"

API = "https://huggingface.co/api/models"

# OpenVINO org carries lots of non-LLM assets (diffusion, detection, etc.).
# Keep only ids that look like a model family locally serves.
OPENVINO_RELEVANT = re.compile(
    r"(qwen|coder|-vl|vl-|whisper|gemma|phi|deepseek|mistral|llama|internvl|granite|smol)",
    re.I,
)
# Upstream Qwen org is huge; only surface the families locally would want, and
# drop quant/format re-uploads that aren't the thing we'd export ourselves.
QWEN_WANT = re.compile(r"(coder|-vl|vl-|omni)", re.I)
QWEN_SKIP = re.compile(r"(gguf|awq|gptq|mlx|fp8|-base|autoround|eagle)", re.I)


def fetch_org(author, limit=1000):
    """Return [{id, createdAt, downloads, likes, pipeline_tag}] for an org."""
    url = (f"{API}?author={author}&limit={limit}"
           "&sort=createdAt&direction=-1&full=false")
    req = urllib.request.Request(url, headers={"User-Agent": "locally-model-watch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def relevant(author, models):
    """Filter an org's model list to locally-relevant ids."""
    out = {}
    for m in models:
        mid = m.get("id") or m.get("modelId") or ""
        if not mid:
            continue
        repo = mid.split("/", 1)[-1]
        if author == "OpenVINO":
            if not OPENVINO_RELEVANT.search(repo):
                continue
        else:  # Qwen upstream
            if not QWEN_WANT.search(repo) or QWEN_SKIP.search(repo):
                continue
        out[mid] = {
            "created": (m.get("createdAt") or "")[:10],
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "pipeline": m.get("pipeline_tag", ""),
            "source": author,
        }
    return out


def known_family_stems():
    """First two '-' tokens of each models.json repo name, e.g. 'qwen2.5-coder'.

    Used to tell '⬆ another size/rev of a family you already run' from a
    '✨ new family'. Purely advisory — quality still needs a human.
    """
    stems = set()
    try:
        data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return stems
    for entries in data.values():
        for e in entries:
            repo = (e.get("hf_id", "").split("/", 1)[-1]).lower()
            toks = repo.split("-")
            if len(toks) >= 2:
                stems.add("-".join(toks[:2]))
    return stems


def classify(mid, stems):
    repo = mid.split("/", 1)[-1].lower()
    toks = repo.split("-")
    stem = "-".join(toks[:2]) if len(toks) >= 2 else repo
    return "⬆ upgrade?" if stem in stems else "✨ new"


def set_output(key, value):
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def main():
    current = {}
    for author in ("OpenVINO", "Qwen"):
        try:
            current.update(relevant(author, fetch_org(author)))
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError) as e:
            print(f"WARN: failed to fetch {author}: {e}", file=sys.stderr)

    if not current:
        print("No models fetched (network?); leaving snapshot untouched.")
        return 0

    try:
        seen = set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        seen = set()

    baseline = not seen
    new_ids = sorted(set(current) - seen)

    # Persist the union so we never re-report and never lose history.
    merged = sorted(seen | set(current))
    SEEN_FILE.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    set_output("changed", "true")  # snapshot file changed → commit it

    if baseline:
        print(f"Baseline established: {len(current)} relevant models tracked. "
              "No issue (first run).")
        return 0

    if not new_ids:
        print("No new models since last run.")
        return 0

    stems = known_family_stems()
    rows = []
    for mid in sorted(new_ids, key=lambda m: current[m]["created"], reverse=True):
        info = current[mid]
        tag = classify(mid, stems)
        rows.append(
            f"| {tag} | [`{mid}`](https://huggingface.co/{mid}) | {info['source']} "
            f"| {info['created']} | {info['pipeline'] or '—'} "
            f"| {info['downloads']} | {info['likes']} |")

    body = (
        f"**{len(new_ids)} new locally-relevant model(s)** appeared on Hugging Face.\n\n"
        "`⬆ upgrade?` = another size/revision of a family you already list in "
        "`models.json`. `✨ new` = a family you don't track yet. "
        "Quality isn't judged here — verify before trusting.\n\n"
        "| | Model | Org | Created | Task | DLs/mo | ♥ |\n"
        "|---|---|---|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n_Watched orgs: OpenVINO (ready-to-run) + Qwen (upstream Coder/VL/Omni). "
        "To add one, drop it into the matching block of `models.json`._"
    )
    TITLE_FILE.write_text(f"Model watch: {len(new_ids)} new model(s) on Hugging Face",
                          encoding="utf-8")
    BODY_FILE.write_text(body, encoding="utf-8")
    set_output("new", "true")
    print(f"{len(new_ids)} new model(s) found:")
    for mid in new_ids:
        print(f"  {mid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
