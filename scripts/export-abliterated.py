#!/usr/bin/env python3
"""Export each abliterated source to an OpenVINO IR the moment its download lands.

Written 2026-09-04. The point is that a 51.8 GB download and a 26B export are
both long, and doing them strictly one-after-the-other wastes the hours the
network is busy. This polls each source directory against the Hugging Face
file listing and starts the export as soon as -- and only as soon as -- every
byte of that model is on disk. A partial safetensors set exports into garbage
or a confusing crash, so "complete" is checked by SIZE against the API, never
by "the directory exists".

Exports run ONE AT A TIME regardless of how many models are ready: conversion
holds the full-precision model in memory (CLAUDE.md: MoE conversion is
RAM-bound, not disk-bound), so two at once is how you find the pagefile.

Per-model choices, and why they differ:
  int4-cw   channel-wise, group_size -1. The NPU-safe form -- default
            group-quantized int4 crashes the vpux compiler ("Found N
            duplicated names"). Only for the dense qwen3 14B.
  int8      Ornith-1.5-9B is qwen3_5 = hybrid Gated DeltaNet, which
            optimum-intel #1722 corrupts at int4. Intel shipped the stock
            Qwen3.5-9B as int8 only for exactly this reason.
  int4      The 27B is the same qwen3_5 architecture, but Intel published
            OpenVINO/Qwen3.8-27B-int4-ov themselves on 2026-08-14, so int4 is
            validated upstream at that size. If it comes out incoherent,
            that inference was wrong -- fall back to int8 and say so here.

venv choice is not cosmetic: gemma4_unified and qwen3_5 are only registered in
venv-2026.3 (transformers 5.4). The old venv's 4.57 cannot load them at all.

WHAT ACTUALLY HAPPENED ON THE FIRST RUN (2026-09-04), and what is still in the
way. All three jobs failed in the same second, 38 minutes of downloading later,
with "Cannot infer the task from a local directory yet" -- optimum-cli will
guess a task for a hub id and refuses to for a local path, and --task was never
passed. infer_task() now reads it from the source's own config.json.

Behind that was a second wall: each model wants a DIFFERENT transformers, and
the windows do not overlap. optimum-intel's export config declares them, and
venv-2026.3's 5.4.0 satisfies neither:

    qwen3_5        (Ornith-1.5-9B, Qwen3.8-27B)  5.2.0  ..  5.2.99
    gemma4_unified (gemma-4-12B-coder)           5.10   ..  5.10.99
    venv-2026.3 has 5.4.0                        -> too new for one,
                                                    too old for the other

optimum-intel 2.1.0 is the newest that exists, so there is no upgrading out of
it. So the driver moves the venv to the version each job needs and puts it back
afterwards, which works because exports are serial anyway. Nothing here holds a
table of versions or tasks: probe() asks optimum, in the export venv, for both.
The original version is written to venv-2026.3/.transformers-pin BEFORE the
first swap and restored in a finally AND at the next startup -- a run killed
mid-export must not leave 5.2 behind for the LFM2 exporter (whose cap is what
the 5.4 pin is for) to trip over weeks later with nothing to say why.

Measured 2026-09-04: with transformers swapped to 5.2.0, Ornith-1.5-9B
exported to int8 in about seven minutes, 8.9 GB, and --scan reads it back as
Qwen3_5ForConditionalGeneration, 32 layers (8 full-attention + 24 linear),
32 KB/token KV. So the version window is the whole of what was in the way.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parent.parent
MODELS = HOME / "models"
LOG = MODELS / ".export.log"

V_OLD = REPO / "venv" / "Scripts" / "optimum-cli.exe"
V_NEW = REPO / "venv-2026.3" / "Scripts" / "optimum-cli.exe"

# (hf_repo, src_dir, out_dir, optimum-cli binary, weight spec)
JOBS = [
    # Camilo's priority order: the 12B coder first, then the 9B, then the
    # flagship 27B. Qwen3-14B-abliterated was dropped 2026-09-04 -- a 2026 12B
    # beats a 2025 dense 14B, and the 14B wanted 160 KB/token of KV against
    # gemma-4-12B's 64, so it was the costliest model here to give context to
    # for the weakest result. V_OLD is now unused but kept: it is the only
    # stack that can export a plain-qwen3 dense model for the NPU.
    ("huihui-ai/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated",
     MODELS / "src/Huihui-gemma-4-12B-coder-abliterated",
     MODELS / "gemma-4-12B-coder-abliterated-int4-ov", V_NEW, "int4"),
    ("huihui-ai/Huihui-Ornith-1.5-9B-abliterated",
     MODELS / "src/Huihui-Ornith-1.5-9B-abliterated",
     MODELS / "Ornith-1.5-9B-abliterated-int8-ov", V_NEW, "int8"),
    ("huihui-ai/Huihui-Qwen3.8-27B-abliterated",
     MODELS / "src/Huihui-Qwen3.8-27B-abliterated",
     MODELS / "Qwen3.8-27B-abliterated-int4-ov", V_NEW, "int4"),
]

# Run in the EXPORT venv, not this one: the answer is a property of that
# environment's optimum-intel, and asking anywhere else is a guess.
PROBE = """
import json, sys
from optimum.exporters.openvino import model_configs   # registers the configs
from optimum.exporters.tasks import TasksManager
with open(sys.argv[1] + "/config.json", encoding="utf-8") as fh:
    model_type = json.load(fh).get("model_type", "")
tasks = TasksManager.get_supported_tasks_for_model_type(
    model_type, "openvino", library_name="transformers")
order = ["text-generation-with-past", "image-text-to-text", "text-generation"]
task = next((t for t in order if t in tasks), sorted(tasks)[0] if tasks else "")
ctor = TasksManager.get_exporter_config_constructor(
    exporter="openvino", model_type=model_type, task=task,
    library_name="transformers")
cls = getattr(ctor, "func", ctor)
def text(v):
    return None if v is None else str(v)
print("PROBE " + json.dumps({
    "model_type": model_type, "task": task,
    "min": text(getattr(cls, "MIN_TRANSFORMERS_VERSION", None)),
    "max": text(getattr(cls, "MAX_TRANSFORMERS_VERSION", None)),
}))
"""

_listing_cache = {}


def say(*a):
    line = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def listing(repo):
    """[(path, size)] for a repo, cached. Empty list means 'could not tell'."""
    if repo in _listing_cache:
        return _listing_cache[repo]
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "locally-export"})
        with urllib.request.urlopen(req, timeout=60) as r:
            files = [(f["path"], f.get("size", 0))
                     for f in json.load(r) if f.get("type") == "file"]
        if files:
            _listing_cache[repo] = files
        return files
    except Exception as e:
        say(f"    (listing {repo} failed: {e})")
        return []


def complete(repo, src):
    """True only when every listed file is present at its exact size."""
    files = listing(repo)
    if not files:
        return False
    for path, size in files:
        f = src / path
        try:
            if size > 0 and f.stat().st_size != size:
                return False
        except OSError:
            return False
    return True


def exported(out):
    """A finished export has the language model's .bin, not just the folder."""
    if not out.is_dir():
        return False
    return any(out.glob("*.bin")) or any(out.glob("**/*.bin"))


def venv_python(cli):
    """The interpreter beside an optimum-cli.exe."""
    return cli.parent / ("python.exe" if os.name == "nt" else "python")


def run(args, **kw):
    return subprocess.run([str(a) for a in args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", **kw)


def probe(python, src):
    """{task, min, max} for this source, straight out of optimum-intel.

    Both facts the first run got wrong live in the same place, so they are read
    together and from the same authority: the task optimum refuses to infer for
    a local directory, and the transformers version its export config demands.
    A table in this file would be a third opinion, stale the moment optimum-
    intel ships a release.
    """
    p = run([python, "-c", PROBE, src])
    for line in reversed((p.stdout or "").splitlines()):
        if line.startswith("PROBE "):
            return json.loads(line[6:])
    say(f"    (probe failed: {(p.stderr or p.stdout or '').strip().splitlines()[-1:]})")
    return {}


def transformers_version(python):
    p = run([python, "-c", "import transformers; print(transformers.__version__)"])
    return (p.stdout or "").strip() or None


def version_ok(version, low, high):
    def parts(v):
        return tuple(int(n) for n in re.findall(r"[0-9]+", v or ""))
    if not version:
        return False
    if low and parts(version) < parts(low):
        return False
    if high and parts(version) > parts(high):
        return False
    return True


def set_transformers(python, spec):
    say(f"  pip install {spec}")
    p = run([python, "-m", "pip", "install", "--quiet", spec])
    if p.returncode:
        say("  pip FAILED: " + (p.stderr or "").strip().splitlines()[-1:][0] if p.stderr else "  pip FAILED")
        return False
    return True


def restore_pin(python, pinfile):
    """Put the venv's transformers back where the user had it.

    Written to disk BEFORE the first swap and restored in a finally, and again
    at the next startup: a run killed mid-export must not leave venv-2026.3 on
    5.2 for the LFM2 exporter to trip over weeks later, with nothing to say why.
    """
    if not pinfile.exists():
        return
    wanted = pinfile.read_text(encoding="utf-8").strip()
    if wanted and transformers_version(python) != wanted:
        say(f"  restoring transformers=={wanted} in {python.parents[1].name}")
        set_transformers(python, f"transformers=={wanted}")
    pinfile.unlink(missing_ok=True)


def infer_task(src):
    """The --task optimum-cli refuses to guess for a local directory.

    All three exports failed at once on 2026-09-04 with "Cannot infer the task
    from a local directory yet, please specify the task manually" -- 38 minutes
    of downloading, then nothing, because the flag was never passed. Read from
    the source's own config rather than a per-model table: a table is another
    place to forget an entry, and the config already says it. A vision tower
    (or an image token) means the model is a conditional generator that takes
    images; anything else is a plain causal LM.
    """
    try:
        with open(src / "config.json", encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        return "text-generation-with-past"
    if "vision_config" in config or any("image_token" in k for k in config):
        return "image-text-to-text"
    return "text-generation-with-past"


def export(src, out, cli, weight):
    python = venv_python(cli)
    facts = probe(python, src)
    task = facts.get("task") or infer_task(src)
    low, high = facts.get("min"), facts.get("max")
    have = transformers_version(python)
    if (low or high) and not version_ok(have, low, high):
        # Each model wants its own transformers and they do not overlap
        # (qwen3_5 is 5.2.x, gemma4_unified is 5.10.x), so the venv is moved
        # to the version THIS job needs and put back at the end. Exports are
        # serial anyway -- conversion is RAM-bound -- which is what makes one
        # shared environment workable at all.
        spec = "transformers" + (f">={low}" if low else "") + (f",<={high}" if high else "")
        pin = python.parents[1] / ".transformers-pin"
        if have and not pin.exists():
            pin.write_text(have, encoding="utf-8")
        say(f"  transformers {have} is outside {low}..{high} for {facts.get('model_type')}")
        if not set_transformers(python, spec):
            return False
        say(f"  transformers now {transformers_version(python)}")
    args = [str(cli), "export", "openvino", "--model", str(src), "--task", task]
    if weight.endswith("-cw"):
        args += ["--weight-format", weight[:-3],
                 "--group-size", "-1", "--sym", "--ratio", "1.0"]
    else:
        args += ["--weight-format", weight]
    args += ["--trust-remote-code", str(out)]
    say(f"  task {task}" + (" (from optimum)" if facts.get("task") else " (from config.json)"))
    say(f"  running: {' '.join(args[1:])}")
    t0 = time.time()
    p = subprocess.run(args, capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    if p.returncode == 0:
        say(f"  OK  {out.name} in {dt:.1f} min")
        return True
    say(f"  FAIL {out.name} after {dt:.1f} min (rc={p.returncode})")
    for line in (p.stderr or "").strip().splitlines()[-15:]:
        say("      | " + line)
    return False


def main():
    say("=== export driver start ===")
    # Self-heal first: a previous run killed mid-export leaves the venv on
    # whichever transformers that job wanted.
    for cli in {job[3] for job in JOBS}:
        python = venv_python(cli)
        restore_pin(python, python.parents[1] / ".transformers-pin")
    done, failed = set(), set()
    while len(done | failed) < len(JOBS):
        progressed = False
        for repo, src, out, cli, weight in JOBS:
            key = out.name
            if key in done or key in failed:
                continue
            if exported(out):
                say(f"already exported: {key}")
                done.add(key)
                progressed = True
                continue
            if not complete(repo, src):
                continue
            say(f">>> {repo}")
            say(f"    source complete -> exporting {weight} to {out.name}")
            out.parent.mkdir(parents=True, exist_ok=True)
            (done if export(src, out, cli, weight) else failed).add(key)
            progressed = True
        if len(done | failed) < len(JOBS) and not progressed:
            time.sleep(120)
    for cli in {job[3] for job in JOBS}:
        python = venv_python(cli)
        restore_pin(python, python.parents[1] / ".transformers-pin")
    say(f"=== export driver done: {len(done)} ok, {len(failed)} failed ===")
    if failed:
        say("    failed: " + ", ".join(sorted(failed)))


if __name__ == "__main__":
    main()
