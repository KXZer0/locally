"""Measure OpenVINO runtime knobs for a local language model.

Usage:
    python scripts/knob-sweep.py <model_dir> <device> [options]
    python scripts/knob-sweep.py <model_dir> <device> --config NAME [options]

The driver runs one configuration per subprocess because an OpenVINO pipeline
can retain device resources until garbage collection; reusing that process can
poison the next measurement.  This is a measurement tool and does not modify
the server or its configuration.

Measurements are INTERLEAVED IN ROUNDS, not run back-to-back per config. A
real run against Qwen3-8B-int4-cw on an Arc B390 (2026-08-25) re-measured the
identical baseline config three times in a row and got 5.63s / 7.94s / 7.39s
for the same work -- a ~40% spread from thermal throttling, DVFS, and
background load alone, which is far larger than most of the effects this
script exists to detect. Blocked execution ("every repeat of config A, then
every repeat of config B, ...") lets that drift land unevenly across configs,
so a config that merely ran later reads as slower. `--rounds N` (default 3)
instead runs every config once, then every config again, and so on, so drift
hits every config about equally instead of biasing whichever ran first or
last. Do not revert to blocked runs; that is precisely the design this
replaced, and reverting silently reintroduces the ~40% confound.

Every device matrix also carries a `control` config, byte-identical to
`baseline`. Its true effect on generation is zero by construction, so
whatever delta it measures against baseline IS the harness's own noise
floor -- used to mark table rows whose delta is not resolvable from noise
(see print_noise_floor).
"""
import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request


# A fixed, technical prompt makes each device property see the same prefill
# and decode work without making the benchmark depend on a file on disk.
PROMPT = """Modern local inference systems are shaped by several layers that are easy to confuse. The model weights describe the learned computation, while the runtime compiler lowers that computation into kernels for a particular accelerator. A tokenizer turns text into integer IDs before the compiled graph sees it, and a key-value cache stores attention state so a later token does not recompute the entire prefix. Prefill processes the prompt in parallel and is commonly limited by memory bandwidth or graph compilation overhead. Decode then generates one token at a time, so its throughput depends on how quickly the device can move weights and cached state for each step. Quantization reduces the bytes moved, but its effect depends on whether scales are shared per tensor, channel, or group. Dynamic quantization can choose a representation during execution, trading some setup work for lower memory traffic. Device plugins may also use tiles, large allocations, or throughput-oriented scheduling; those choices can improve utilization for a batch while making a single interactive stream less responsive. A fair experiment therefore fixes the prompt, generation policy, output length, and warmup procedure, and reports both first-token latency and steady decode speed. It should also inspect the answer, because a faster configuration that changes the model's behavior is not automatically a useful configuration.

Question: Based only on this description, why should an engineer report prefill latency and decode throughput separately when comparing a runtime setting for an interactive local assistant, and what quality check should accompany the numbers?"""

CONFIGS = {
    "NPU": (
        ("baseline", {}),
        # Byte-identical to baseline. Its true effect is zero by construction,
        # so its measured delta against baseline IS the noise floor.
        ("control", {}),
        ("turbo", {"NPU_TURBO": True}),
        ("dynquant", {"NPU_COMPILER_DYNAMIC_QUANTIZATION": True}),
        ("qdq", {"NPU_QDQ_OPTIMIZATION": True}),
        ("turbo+dynquant", {
            "NPU_TURBO": True,
            "NPU_COMPILER_DYNAMIC_QUANTIZATION": True,
        }),
        ("tiles-max", {"NPU_TILES": 3}),
    ),
    "GPU": (
        ("baseline", {}),
        # Byte-identical to baseline. Its true effect is zero by construction,
        # so its measured delta against baseline IS the noise floor.
        ("control", {}),
        ("large-alloc", {"GPU_ENABLE_LARGE_ALLOCATIONS": True}),
        ("dynquant-32", {"DYNAMIC_QUANTIZATION_GROUP_SIZE": 32}),
        ("dynquant-64", {"DYNAMIC_QUANTIZATION_GROUP_SIZE": 64}),
        ("dynquant-128", {"DYNAMIC_QUANTIZATION_GROUP_SIZE": 128}),
        ("hint-throughput", {"PERFORMANCE_HINT": "THROUGHPUT"}),
        # This only affects the compiled cache, so it is meaningful with
        # CACHE_DIR and deliberately remains in the matrix for that comparison.
        # With no --cache-dir there is nothing to cache and its true effect is
        # zero -- the first real run measured -25.7% here, which was really
        # just an unlabelled second sample of the noise floor.
        ("cache-mode-size", {"CACHE_MODE": "OPTIMIZE_SIZE"}),
        ("large-alloc+dq64", {
            "GPU_ENABLE_LARGE_ALLOCATIONS": True,
            "DYNAMIC_QUANTIZATION_GROUP_SIZE": 64,
        }),
    ),
    "CPU": (
        ("baseline", {}),
        # Byte-identical to baseline. Its true effect is zero by construction,
        # so its measured delta against baseline IS the noise floor.
        ("control", {}),
        ("dynquant-32", {"DYNAMIC_QUANTIZATION_GROUP_SIZE": 32}),
        ("hint-throughput", {"PERFORMANCE_HINT": "THROUGHPUT"}),
    ),
}

CONFIG_TIMEOUT_S = 30 * 60


def positive_int(value):
    """Accept only counts that can produce a useful benchmark."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def configs_for(device, cache_dir=None):
    """Return complete property dictionaries in the documented matrix order."""
    result = []
    for name, extra in CONFIGS[device]:
        properties = dict(extra)
        if device == "NPU":
            # Production always uses this ceiling; omitting it would make the
            # NPU comparison measure a different compiler configuration.
            properties["MAX_PROMPT_LEN"] = 8192
        if cache_dir:
            properties["CACHE_DIR"] = os.path.abspath(os.path.expanduser(cache_dir))
        result.append((name, properties))
    return result


def child_argv(args, config_name):
    """Build the isolated invocation used for one configuration."""
    argv = [
        sys.executable,
        os.path.abspath(__file__),
        args.model_dir,
        args.device,
        "--config",
        config_name,
        "--runs",
        str(args.runs),
        "--tokens",
        str(args.tokens),
    ]
    if args.cache_dir:
        argv.extend(("--cache-dir", args.cache_dir))
    return argv


def first_line(error):
    """Keep process failures readable in a table row and JSON result."""
    text = str(error).strip()
    return text.splitlines()[0] if text else error.__class__.__name__


def process_error_line(text):
    """Prefer the exception line over a Python traceback header."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return "child exited without an error message"
    if any(line.startswith("Traceback (most recent call last):")
           for line in lines):
        return lines[-1]
    return lines[0]


def is_unsupported(error):
    """Recognize delayed plugin-property rejection during the first generate."""
    message = str(error).lower()
    return any(term in message for term in (
        "unsupported", "not supported", "unknown property",
        "invalid property", "cannot set property", "cannot compile",
        "compile failed", "compilation failed", "failed to compile"))


def metric_mean(metric):
    """Read a GenAI metric without allowing an implementation detail to leak."""
    value = metric.mean
    return float(value)


def numeric_stats(values):
    """Median/min/max over the numeric subset of an iterable, or None.

    None means nothing in it was actually a number -- e.g. every run fell
    back to wall-clock timing, where ttft_ms/tpot_ms are the literal string
    "n/a" rather than a fabricated number reported under the wrong name.
    """
    numbers = [v for v in values if isinstance(v, (int, float))]
    if not numbers:
        return None
    return statistics.median(numbers), min(numbers), max(numbers)


def generation_metrics(result, wall_s, tokens):
    """Return one comparable metric record, identifying wall-clock estimates.

    perf_metrics is only populated when generate() is called with a
    ChatHistory (see run_child) -- a bare prompt string returns a plain str
    with no perf_metrics attribute at all, which is what silently produced an
    all-None metrics column the first time this script ran for real.
    """
    perf = getattr(result, "perf_metrics", None)
    if perf is not None:
        try:
            return {
                "ttft_ms": metric_mean(perf.get_ttft()),
                "tpot_ms": metric_mean(perf.get_tpot()),
                "decode_tok_s": metric_mean(perf.get_throughput()),
                "wall_s": wall_s,
                "total_tok_s": tokens / wall_s if wall_s > 0 else None,
                "metrics_source": "perf_metrics",
            }
        except Exception:
            # A partially exposed metrics object is just as unsafe to mix with
            # real metrics as a missing object, so use one clearly marked path.
            pass
    # Wall-clock has no visibility into the prefill/decode split, so it must
    # never report a number under ttft_ms or tpot_ms. That mislabelling is
    # exactly what made the first real run print the same number twice under
    # two different column names and call the total time a first-token time.
    return {
        "ttft_ms": "n/a",
        "tpot_ms": "n/a",
        "decode_tok_s": None,
        "wall_s": wall_s,
        "total_tok_s": tokens / wall_s if wall_s > 0 else None,
        "metrics_source": "wall-clock",
    }


def result_text(result):
    """Use GenAI's string conversion, which is stable across result versions."""
    return str(result)


def run_child(model_dir, device, config_name, runs, tokens, cache_dir):
    """Load and measure exactly one configuration in this process."""
    started = time.perf_counter()
    record = {
        "config": config_name,
        "properties": dict(configs_for(device, cache_dir))[config_name],
        "status": "FAILED",
        "error": None,
        "load_s": None,
        "raw_runs": [],
        "output_head": None,
        "metrics_source": None,
        "openvino_version": None,
        "openvino_genai_version": None,
        "full_device_name": None,
    }
    try:
        import openvino as ov
        import openvino_genai as og

        record["openvino_version"] = getattr(ov, "__version__", None)
        record["openvino_genai_version"] = getattr(og, "__version__", None)
        try:
            record["full_device_name"] = str(
                ov.Core().get_property(device, "FULL_DEVICE_NAME"))
        except Exception:
            record["full_device_name"] = None

        properties = record["properties"]
        load_started = time.perf_counter()
        try:
            pipe = og.LLMPipeline(model_dir, device, **properties)
        except Exception as error:
            # Construction includes plugin compilation. A rejection here is
            # evidence that this knob is unavailable on this model/device.
            record["status"] = ("UNSUPPORTED" if is_unsupported(error)
                                 else "FAILED")
            record["error"] = first_line(error)
            record["load_s"] = time.perf_counter() - load_started
            return record
        record["load_s"] = time.perf_counter() - load_started

        generation_config = og.GenerationConfig()
        generation_config.do_sample = False
        generation_config.max_new_tokens = tokens
        generation_config.ignore_eos = True

        # The first invocation pays lazy graph, allocator, and weight-mapping
        # costs; retaining it only as raw evidence keeps medians steady-state.
        for run_number in range(runs + 1):
            generation_started = time.perf_counter()
            try:
                # A ChatHistory is what the production server passes to
                # generate(); a bare string returns a plain str with no
                # perf_metrics at all (see generation_metrics docstring), so
                # this is required to get real per-phase timing, not optional.
                history = og.ChatHistory()
                history.append({"role": "user", "content": PROMPT})
                result = pipe.generate(history, generation_config)
            except Exception as error:
                record["status"] = "UNSUPPORTED" if is_unsupported(error) else "FAILED"
                record["error"] = first_line(error)
                return record
            wall_s = time.perf_counter() - generation_started
            text = result_text(result)
            metrics = generation_metrics(result, wall_s, tokens)
            metrics.update({
                "run": run_number,
                "warmup": run_number == 0,
                "output_head": text[:80],
            })
            record["raw_runs"].append(metrics)

        measured = record["raw_runs"][1:]
        record["status"] = "OK"
        record["output_head"] = measured[0]["output_head"]
        record["metrics_source"] = measured[0]["metrics_source"]
        decode_stats = numeric_stats(r["decode_tok_s"] for r in measured)
        ttft_stats = numeric_stats(r["ttft_ms"] for r in measured)
        tpot_stats = numeric_stats(r["tpot_ms"] for r in measured)
        record["decode_tok_s"] = decode_stats[0] if decode_stats else None
        record["ttft_ms"] = ttft_stats[0] if ttft_stats else "n/a"
        record["tpot_ms"] = tpot_stats[0] if tpot_stats else "n/a"
        return record
    except Exception as error:
        record["status"] = "FAILED"
        record["error"] = first_line(error)
        return record
    finally:
        record["elapsed_s"] = time.perf_counter() - started


def parse_child_json(stdout):
    """Find the child's JSON record even if a runtime emits a notice."""
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and value.get("config"):
            return value
    return None


def failed_child(config_name, properties, error):
    """Normalize a crash or timeout to the same shape as a child result."""
    return {
        "config": config_name,
        "properties": properties,
        "status": "FAILED",
        "error": first_line(error),
        "load_s": None,
        "raw_runs": [],
        "output_head": None,
        "metrics_source": None,
        "openvino_version": None,
        "openvino_genai_version": None,
        "full_device_name": None,
    }


def merge_rounds(config_name, round_records):
    """Combine one config's per-round child results into a single row.

    One config per subprocess is preserved: each round is its own fresh
    child. This is where the interleaved rounds are brought back together --
    every OK round's measured generations are pooled before taking the
    median, and the pool's spread is what print_table shows alongside it.
    """
    ok_rounds = [r for r in round_records if r.get("status") == "OK"]
    statuses = {r.get("status") for r in round_records}

    if not round_records:
        status, error = "FAILED", "no rounds ran"
    elif statuses == {"OK"}:
        status, error = "OK", None
    elif ok_rounds:
        # Some rounds succeeded and some didn't -- report both facts rather
        # than hiding the failure behind a median of the rounds that worked.
        failed = next(r for r in round_records if r.get("status") != "OK")
        status = "PARTIAL"
        error = (f"{len(ok_rounds)}/{len(round_records)} rounds OK; "
                 f"first failure (round {failed.get('round')}): "
                 f"{failed.get('error')}")
    else:
        status = round_records[0].get("status", "FAILED")
        error = round_records[0].get("error")

    measured_runs = [
        run for r in ok_rounds for run in r.get("raw_runs", [])
        if not run.get("warmup")
    ]
    decode_stats = numeric_stats(run.get("decode_tok_s") for run in measured_runs)
    ttft_stats = numeric_stats(run.get("ttft_ms") for run in measured_runs)
    tpot_stats = numeric_stats(run.get("tpot_ms") for run in measured_runs)
    load_stats = numeric_stats(r.get("load_s") for r in ok_rounds)

    return {
        "config": config_name,
        "properties": round_records[0].get("properties") if round_records else None,
        "status": status,
        "error": error,
        "rounds": round_records,  # every individual run's wall time lives here
        "load_s": load_stats[0] if load_stats else None,
        "decode_tok_s": decode_stats[0] if decode_stats else None,
        "decode_tok_s_min": decode_stats[1] if decode_stats else None,
        "decode_tok_s_max": decode_stats[2] if decode_stats else None,
        "ttft_ms": ttft_stats[0] if ttft_stats else ("n/a" if measured_runs else None),
        "ttft_ms_min": ttft_stats[1] if ttft_stats else None,
        "ttft_ms_max": ttft_stats[2] if ttft_stats else None,
        "tpot_ms": tpot_stats[0] if tpot_stats else ("n/a" if measured_runs else None),
        "tpot_ms_min": tpot_stats[1] if tpot_stats else None,
        "tpot_ms_max": tpot_stats[2] if tpot_stats else None,
        "output_head": next((r.get("output_head") for r in ok_rounds), None),
        "metrics_source": next(
            (run.get("metrics_source") for run in measured_runs), None),
        "openvino_version": next(
            (r.get("openvino_version") for r in round_records
             if r.get("openvino_version")), None),
        "openvino_genai_version": next(
            (r.get("openvino_genai_version") for r in round_records
             if r.get("openvino_genai_version")), None),
        "full_device_name": next(
            (r.get("full_device_name") for r in round_records
             if r.get("full_device_name")), None),
    }


def check_server_quiescent():
    """Best-effort warning that the `locally` server is holding device memory.

    Never blocks and never fails the sweep: the server not running is the
    normal case, and any transport error (closed port, refused connection,
    timeout) means exactly that -- there is nothing to warn about.
    """
    try:
        with urllib.request.urlopen(
                "http://127.0.0.1:8000/health", timeout=2) as response:
            if response.status == 200:
                print(
                    "WARNING: the locally server appears to be running on "
                    "127.0.0.1:8000. It holds device memory that adds "
                    "variance to this measurement -- stop it for a clean run.",
                    file=sys.stderr)
    except Exception:
        pass


def run_driver(args):
    """Run children in interleaved rounds, aggregate, print, and write JSON."""
    check_server_quiescent()

    config_items = configs_for(args.device, args.cache_dir)
    config_names = [name for name, _ in config_items]
    properties_by_name = dict(config_items)
    per_config_rounds = {name: [] for name in config_names}

    start = time.perf_counter()
    total = args.rounds * len(config_names)
    done = 0
    # Interleaved: every config runs once before any config runs a second
    # time, so slow drift over the whole sweep hits every config about
    # equally instead of concentrating on whichever config ran first or last.
    for round_number in range(1, args.rounds + 1):
        for config_name in config_names:
            done += 1
            elapsed = time.perf_counter() - start
            print(f"[{done}/{total}] round {round_number}/{args.rounds} "
                  f"config {config_name!r} starting (elapsed {elapsed:.0f}s)",
                  file=sys.stderr)
            properties = properties_by_name[config_name]
            try:
                completed = subprocess.run(
                    child_argv(args, config_name),
                    capture_output=True,
                    text=True,
                    timeout=CONFIG_TIMEOUT_S,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                result = failed_child(
                    config_name, properties,
                    f"timed out after {CONFIG_TIMEOUT_S} seconds")
            except Exception as error:
                result = failed_child(config_name, properties, error)
            else:
                result = parse_child_json(completed.stdout)
                if result is None:
                    detail = (completed.stderr or completed.stdout or
                              f"child exited with status {completed.returncode}")
                    result = failed_child(config_name, properties,
                                          process_error_line(detail))
                elif completed.returncode != 0 and result.get("status") == "OK":
                    result["status"] = "FAILED"
                    result["error"] = (f"child exited with status "
                                       f"{completed.returncode}")
            result["round"] = round_number
            per_config_rounds[config_name].append(result)
            rate = result.get("decode_tok_s")
            value = f"{rate:.2f} tok/s" if isinstance(rate, (int, float)) else "-"
            print(f"    -> {result.get('status')} {value}", file=sys.stderr)

    results = [merge_rounds(name, per_config_rounds[name]) for name in config_names]

    baseline = next((r for r in results if r["config"] == "baseline"), None)
    baseline_rate = (baseline.get("decode_tok_s")
                      if baseline and baseline.get("status") == "OK" else None)
    control = next((r for r in results if r["config"] == "control"), None)
    control_rate = (control.get("decode_tok_s")
                     if control and control.get("status") == "OK" else None)
    noise_floor_pct = (abs((control_rate / baseline_rate - 1.0) * 100.0)
                        if baseline_rate and control_rate is not None else None)

    for result in results:
        rate = result.get("decode_tok_s")
        if baseline_rate and result.get("status") == "OK" and rate is not None:
            pct = (rate / baseline_rate - 1.0) * 100.0
            result["decode_vs_baseline_pct"] = pct
            result["within_noise"] = (
                noise_floor_pct is not None and abs(pct) <= noise_floor_pct)
        else:
            result["decode_vs_baseline_pct"] = None
            result["within_noise"] = None

    print_table(results)
    print_noise_floor(noise_floor_pct)
    print_quality_warning(results)

    metadata = next((r for r in results if r.get("openvino_version")), {})
    aggregate = {
        "model_path": os.path.abspath(os.path.expanduser(args.model_dir)),
        "device": args.device,
        "full_device_name": next(
            (r.get("full_device_name") for r in results
             if r.get("full_device_name")), None),
        "openvino_version": metadata.get("openvino_version"),
        "openvino_genai_version": metadata.get("openvino_genai_version"),
        "rounds": args.rounds,
        "runs_per_round": args.runs,
        "tokens": args.tokens,
        "noise_floor_pct": noise_floor_pct,
        "cache_dir": (os.path.abspath(os.path.expanduser(args.cache_dir))
                       if args.cache_dir else None),
        "prompt": PROMPT,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "configs": results,
    }
    output_path = os.path.abspath(os.path.expanduser(args.json_path))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(aggregate, output, indent=2)
        output.write("\n")
    print(f"Results written to {output_path}")


def crop(value, width):
    """Keep every table cell on one fixed-width line."""
    text = "-" if value is None else " ".join(str(value).split())
    if len(text) > width:
        return text[:width - 1] + "…"
    return text.ljust(width)


def fmt_metric(value, digits=2):
    """Format a metric that may be a real number, the literal "n/a" string
    (wall-clock fallback with no per-phase visibility), or None (nothing
    measured, e.g. every round failed)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def fmt_decode(result):
    """Show the spread alongside the median so the noise that invalidated the
    first real run is visible in the table itself, not only in the JSON."""
    median = result.get("decode_tok_s")
    if median is None:
        return None
    lo = result.get("decode_tok_s_min")
    hi = result.get("decode_tok_s_max")
    if lo is None or hi is None or (lo == hi == median):
        return f"{median:.2f}"
    return f"{median:.2f} ({lo:.2f}-{hi:.2f})"


def print_table(results):
    """Print comparable values without hiding a missing or failed row."""
    columns = (
        ("config", 18),
        ("load_s", 9),
        ("ttft_ms", 10),
        ("decode_tok/s (min-max)", 26),
        ("vs-baseline", 14),
        ("status", 60),
        ("output_head", 60),
    )
    heading = " ".join(crop(name, width) for name, width in columns)
    print(heading)
    print(" ".join("-" * width for _, width in columns))
    for result in results:
        status = result.get("status", "FAILED")
        if result.get("error"):
            status += ": " + result["error"]
        elif result.get("metrics_source") == "wall-clock":
            status += " (wall-clock)"
        if result.get("within_noise"):
            # A number here would read as a finding; it isn't one.
            vs_text = "WITHIN NOISE"
        elif result.get("decode_vs_baseline_pct") is not None:
            vs_text = f"{result['decode_vs_baseline_pct']:+.1f}%"
        else:
            vs_text = None
        values = (
            result.get("config"),
            fmt_metric(result.get("load_s"), digits=3),
            fmt_metric(result.get("ttft_ms")),
            fmt_decode(result),
            vs_text,
            status,
            result.get("output_head"),
        )
        print(" ".join(crop(value, width) for value, (_, width)
                       in zip(values, columns)))


def print_noise_floor(noise_floor_pct):
    """State the measurement's own resolution so a delta smaller than this is
    never read as a finding. The control config exists to produce this
    number: it is byte-identical to baseline, so its measured delta IS the
    noise floor."""
    print()
    if noise_floor_pct is None:
        print("NOISE FLOOR: unavailable (control or baseline config did not "
              "complete). Treat every percentage above as unverified.")
    else:
        print(f"NOISE FLOOR: the control config, identical to baseline, "
              f"differed by {noise_floor_pct:.1f}%.")
        print("  Treat any delta smaller than this as unresolvable.")


def print_quality_warning(results):
    """Make output changes visible instead of treating speed as the verdict."""
    baseline = next((r for r in results if r["config"] == "baseline"), None)
    baseline_head = ((baseline or {}).get("output_head")
                     if (baseline or {}).get("status") == "OK" else None)
    differing = [
        r["config"] for r in results
        if r.get("status") == "OK" and baseline_head is not None
        and r.get("output_head") != baseline_head
    ]
    print("\nQUALITY WARNING")
    if differing:
        print("  output_head differs from baseline for: " + ", ".join(differing))
        print("  A differing output is not automatically wrong; a human has to look.")
    elif baseline_head is None:
        print("  Baseline produced no comparable output, so no quality comparison was made.")
    else:
        print("  No output_head differences from baseline were observed.")


def print_dry_run(args):
    """Show exactly what the driver would launch without importing OpenVINO."""
    config_items = configs_for(args.device, args.cache_dir)
    for config_name, properties in config_items:
        print(config_name)
        print("  argv: " + json.dumps(child_argv(args, config_name)))
        print("  properties: " + json.dumps(properties, sort_keys=True))

    print("\nplanned execution order (interleaved rounds -- a config's "
          "repeats are spread across the whole sweep so drift affects every "
          "config equally, rather than concentrating on whichever config ran "
          "first or last):")
    names = [name for name, _ in config_items]
    for round_number in range(1, args.rounds + 1):
        print(f"  round {round_number}: {', '.join(names)}")


def make_parser():
    """Define the small command line surface shared by driver and child."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="OpenVINO model directory")
    parser.add_argument("device", type=lambda value: value.upper(),
                        choices=tuple(CONFIGS), help="NPU, GPU, or CPU")
    parser.add_argument("--config", metavar="NAME",
                        help="run exactly one named configuration in this process")
    parser.add_argument("--runs", type=positive_int, default=3, metavar="N",
                        help="measured generations per configuration, per round (default: 3)")
    parser.add_argument("--rounds", type=positive_int, default=3, metavar="N",
                        help="interleaved repeats of the whole config matrix, "
                             "to average out run-to-run drift (default: 3)")
    parser.add_argument("--tokens", type=positive_int, default=128, metavar="N",
                        help="max_new_tokens per generation (default: 128)")
    parser.add_argument("--cache-dir", metavar="PATH",
                        help="pass CACHE_DIR; default is no explicit cache")
    parser.add_argument("--json", dest="json_path", metavar="PATH",
                        help="aggregated output path")
    parser.add_argument("--dry-run", action="store_true",
                        help="print child argv and properties without using a device")
    return parser


def main():
    """Dispatch a device-free dry run, an isolated child, or the driver."""
    args = make_parser().parse_args()
    if args.config and args.config not in dict(configs_for(args.device, args.cache_dir)):
        names = ", ".join(name for name, _ in configs_for(args.device, args.cache_dir))
        print(f"error: unknown config {args.config!r} for {args.device}; choose: {names}",
              file=sys.stderr)
        return 2
    if args.dry_run:
        print_dry_run(args)
        return 0
    if args.config:
        record = run_child(args.model_dir, args.device, args.config,
                           args.runs, args.tokens, args.cache_dir)
        print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        return 0
    if not args.json_path:
        model_name = os.path.basename(os.path.normpath(args.model_dir)) or "model"
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.json_path = os.path.join(
            "bench-results", f"knobs-{args.device}-{model_name}-{timestamp}.json")
    run_driver(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
