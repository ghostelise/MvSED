#!/usr/bin/env python3
"""Run and summarize the frozen, untouched-block MvSED evaluation.

This file intentionally does not implement or tune the clustering method.  It
only protects the evaluation protocol around ``main.py``:

* verify the frozen main.py checksum before every block;
* run one block at a time so an API failure can be resumed safely;
* keep a separate terminal/log transcript for every block and seed;
* refuse to treat a malformed or mismatched metrics file as completed; and
* summarize the same-run pre/post-MVRA comparison with blocks as the
  statistical unit.

The default candidate and thresholds are frozen after development on M1--M6.
M7--M21 should therefore be treated as untouched test blocks: do not edit
main.py or select a new variant after inspecting those results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable


FROZEN_MAIN_SHA256 = (
    "ef3df49415c90f9bd308d4a5ece2c55fbffcfee4506abc19729512eb9f4aa252"
)
CANDIDATE = "stage_aware_selective_gate_time_aux"
PHASE_VARIANTS = {
    "candidate": (CANDIDATE,),
    "text_control": ("stage_aware_text_control",),
    "ragsede": ("ragsede_original",),
    "all": (CANDIDATE, "stage_aware_text_control", "ragsede_original"),
}
METRICS = ("NMI", "AMI", "ARI")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_set(specification: str) -> list[int]:
    """Parse ``7-21,25`` or ``42,52,62`` into sorted unique integers."""

    values: set[int] = set()
    for raw_part in specification.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            if start > end:
                raise argparse.ArgumentTypeError(
                    f"invalid descending range: {part}"
                )
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return sorted(values)


def safe_variant(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def metric_path(project: Path, dataset: str, block: int, variant: str, seed: int) -> Path:
    stem = f"M{block}_{safe_variant(variant)}_single_stage_s{seed}_metrics.json"
    return project / "ckpts" / dataset / f"M{block}" / stem


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def validate_metric(
    path: Path,
    dataset: str,
    block: int,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    result = read_json(path)
    expected = {
        "dataset": dataset,
        "block": block,
        "variant": variant,
        "seed": seed,
        "anchor_pipeline": "single_stage",
    }
    mismatches = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise ValueError(f"protocol mismatch in {path}: {mismatches}")
    scores = result.get("macro_scores")
    if not isinstance(scores, dict) or any(name not in scores for name in METRICS):
        raise ValueError(f"missing macro_scores in {path}")
    if variant == CANDIDATE:
        for key in (
            "pre_mvra_macro_scores",
            "post_mvra_macro_scores",
            "mvra_macro_score_delta",
        ):
            value = result.get(key)
            if not isinstance(value, dict) or any(name not in value for name in METRICS):
                raise ValueError(f"missing {key} in candidate result {path}")
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def printable_command(command: list[str]) -> str:
    redacted = list(command)
    if "--API_KEY" in redacted:
        index = redacted.index("--API_KEY") + 1
        if index < len(redacted):
            redacted[index] = "<RAGFLOW_API_KEY>"
    return " ".join(shlex.quote(part) for part in redacted)


def stream_process(command: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        header = f"\n[{utc_now()}] {printable_command(command)}\n"
        print(header, end="", flush=True)
        log.write(header)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def load_manifest(path: Path, initial: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        save_json(path, initial)
        return initial
    manifest = read_json(path)
    if manifest.get("main_sha256") != initial["main_sha256"]:
        raise RuntimeError(
            "manifest and current frozen main.py have different hashes; "
            "use a new --run-name instead of mixing protocols"
        )
    for key in ("protocol", "dataset", "blocks", "seeds"):
        if manifest.get(key) != initial.get(key):
            raise RuntimeError(
                f"manifest field {key!r} differs from this request; use a new "
                "--run-name instead of mixing evaluation protocols"
            )
    return manifest


def run_experiments(args: argparse.Namespace) -> int:
    project = args.project.resolve()
    main_path = project / "main.py"
    if not main_path.is_file():
        raise FileNotFoundError(f"main.py not found under {project}")
    current_sha = sha256_file(main_path)
    if current_sha != args.expected_main_sha256 and not args.allow_main_mismatch:
        raise RuntimeError(
            "main.py is not the frozen version.\n"
            f"expected: {args.expected_main_sha256}\n"
            f"actual:   {current_sha}\n"
            "Upload the matching main.py, or use --allow-main-mismatch only "
            "for a deliberately new protocol with a new --run-name."
        )
    api_key = os.environ.get("RAGFLOW_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError(
            "RAGFLOW_API_KEY is not set. Export it in the current terminal; "
            "do not place the key in this script or a log file."
        )

    variants = PHASE_VARIANTS[args.phase]
    logs_dir = project / "logs" / args.run_name
    manifest_path = logs_dir / "manifest.json"
    initial_manifest = {
        "protocol": "frozen-untouched-block-evaluation-v1",
        "run_name": args.run_name,
        "created_utc": utc_now(),
        "project": str(project),
        "main_sha256": current_sha,
        "expected_main_sha256": args.expected_main_sha256,
        "dataset": args.dataset,
        "blocks": args.blocks,
        "seeds": args.seeds,
        "variants": list(variants),
        "entries": [],
    }
    manifest = load_manifest(manifest_path, initial_manifest)
    manifest["variants"] = sorted(
        set(manifest.get("variants", [])) | set(variants)
    )
    save_json(manifest_path, manifest)
    existing_entries = {
        (item["variant"], item["block"], item["seed"]): item
        for item in manifest.get("entries", [])
    }

    total = len(variants) * len(args.seeds) * len(args.blocks)
    position = 0
    failures = 0
    for variant in variants:
        for seed in args.seeds:
            for block in args.blocks:
                position += 1
                if sha256_file(main_path) != current_sha:
                    raise RuntimeError(
                        "main.py changed after the frozen run began; stopping "
                        "before results from two protocols are mixed"
                    )
                output = metric_path(project, args.dataset, block, variant, seed)
                identity = (variant, block, seed)
                registered = existing_entries.get(identity)
                may_reuse = bool(
                    registered
                    and registered.get("status") == "completed"
                    and args.skip_existing
                )
                if output.exists() and not may_reuse and not args.adopt_existing:
                    raise RuntimeError(
                        f"Unregistered pre-existing result would be overwritten: {output}\n"
                        "Move it aside first, choose a new clean checkout, or pass "
                        "--adopt-existing only if you have independently verified that "
                        "it was produced by this exact frozen protocol."
                    )
                if output.exists() and (may_reuse or args.adopt_existing):
                    try:
                        validate_metric(output, args.dataset, block, variant, seed)
                    except Exception as error:
                        print(f"Existing output is invalid and will be rerun: {error}")
                    else:
                        print(
                            f"[{position}/{total}] SKIP {variant} s{seed} M{block}: "
                            f"valid metrics already exist",
                            flush=True,
                        )
                        entry = {
                            "variant": variant,
                            "block": block,
                            "seed": seed,
                            "status": "completed",
                            "metrics": str(output.relative_to(project)),
                            "updated_utc": utc_now(),
                        }
                        existing_entries[identity] = entry
                        manifest["entries"] = list(existing_entries.values())
                        save_json(manifest_path, manifest)
                        continue

                command = [
                    sys.executable,
                    "-u",
                    "main.py",
                    "--dataset",
                    args.dataset,
                    "--HOST_ADDRESS",
                    args.host,
                    "--variant",
                    variant,
                    "--anchor_pipeline",
                    "single_stage",
                    "--seed",
                    str(seed),
                    "--start_block",
                    str(block),
                    "--end_block",
                    str(block),
                    "--llm_model",
                    args.llm_model,
                    "--max_tokens",
                    str(args.max_tokens),
                    "--max_llm_attempts",
                    str(args.max_llm_attempts),
                    "--llm_retry_delay",
                    str(args.llm_retry_delay),
                    "--reuse_embedding_cache",
                    "--save_audit_log",
                ]
                command.extend(args.main_arg)
                log_path = logs_dir / variant / f"s{seed}_M{block}.log"
                print(
                    f"\n[{position}/{total}] RUN {variant} s{seed} M{block}",
                    flush=True,
                )
                if args.dry_run:
                    print(printable_command(command))
                    continue

                completed = False
                last_error = "unknown failure"
                for attempt in range(1, args.run_attempts + 1):
                    return_code = stream_process(command, project, log_path)
                    try:
                        validate_metric(output, args.dataset, block, variant, seed)
                    except Exception as error:
                        last_error = f"exit={return_code}; {error}"
                    else:
                        if return_code == 0:
                            completed = True
                            break
                        last_error = (
                            f"metrics exist but process exited with {return_code}"
                        )
                    if attempt < args.run_attempts:
                        delay = args.run_retry_delay * attempt
                        print(
                            f"Block failed ({last_error}); retrying in {delay:.0f}s "
                            f"[{attempt}/{args.run_attempts}]",
                            flush=True,
                        )
                        time.sleep(delay)

                entry = {
                    "variant": variant,
                    "block": block,
                    "seed": seed,
                    "status": "completed" if completed else "failed",
                    "metrics": (
                        str(output.relative_to(project)) if output.exists() else None
                    ),
                    "log": str(log_path.relative_to(project)),
                    "error": None if completed else last_error,
                    "updated_utc": utc_now(),
                }
                existing_entries[identity] = entry
                manifest["entries"] = list(existing_entries.values())
                save_json(manifest_path, manifest)
                if not completed:
                    failures += 1
                    print(f"FAILED {variant} s{seed} M{block}: {last_error}")
                    if not args.continue_on_failure:
                        return 1

    if not args.dry_run:
        summarize_results(args, main_sha=current_sha)
    return 1 if failures else 0


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def exact_two_sided_sign_p(values: Iterable[float]) -> float | None:
    nonzero = [value for value in values if value != 0.0]
    if not nonzero:
        return None
    positives = sum(value > 0 for value in nonzero)
    tail = min(positives, len(nonzero) - positives)
    probability = sum(math.comb(len(nonzero), index) for index in range(tail + 1))
    return min(1.0, 2.0 * probability / (2 ** len(nonzero)))


def summarize_values(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "std": sample_std(values) if values else None,
        "median": median(values) if values else None,
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
        "exact_sign_test_p": exact_two_sided_sign_p(values),
    }


def discover_results(
    project: Path,
    dataset: str,
    blocks: list[int],
    seeds: list[int],
    variants: Iterable[str],
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], list[str]]:
    results: dict[tuple[str, int, int], dict[str, Any]] = {}
    missing: list[str] = []
    for variant in variants:
        for block in blocks:
            for seed in seeds:
                path = metric_path(project, dataset, block, variant, seed)
                try:
                    result = validate_metric(path, dataset, block, variant, seed)
                except Exception as error:
                    missing.append(f"{variant} s{seed} M{block}: {error}")
                    continue
                results[(variant, block, seed)] = result
    return results, missing


def efficiency_mean(records: list[dict[str, Any]]) -> dict[str, float | None]:
    fields = (
        "elapsed_seconds",
        "llm_calls",
        "prompt_tokens",
        "completion_tokens",
        "estimated_cost",
        "peak_rss_mb",
        "anchors",
    )
    output: dict[str, float | None] = {}
    for field in fields:
        values = [
            float(record.get("efficiency", {}).get(field))
            for record in records
            if record.get("efficiency", {}).get(field) is not None
        ]
        output[field] = mean(values) if values else None
    return output


def relative_change(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference in (None, 0.0):
        return None
    return (candidate - reference) / reference


def summarize_results(
    args: argparse.Namespace,
    main_sha: str | None = None,
) -> dict[str, Any]:
    project = args.project.resolve()
    main_sha = main_sha or sha256_file(project / "main.py")
    variants = (CANDIDATE, "stage_aware_text_control", "ragsede_original")
    results, missing = discover_results(
        project, args.dataset, args.blocks, args.seeds, variants
    )
    candidate_records = [
        results[(CANDIDATE, block, seed)]
        for block in args.blocks
        for seed in args.seeds
        if (CANDIDATE, block, seed) in results
    ]

    block_deltas: dict[int, dict[str, float]] = {}
    for block in args.blocks:
        rows = [
            results[(CANDIDATE, block, seed)]
            for seed in args.seeds
            if (CANDIDATE, block, seed) in results
        ]
        if not rows:
            continue
        block_deltas[block] = {
            metric: mean(
                [float(row["mvra_macro_score_delta"][metric]) for row in rows]
            )
            for metric in METRICS
        }

    paired_by_metric = {
        metric: summarize_values(
            [block_deltas[block][metric] for block in sorted(block_deltas)]
        )
        for metric in METRICS
    }
    best_ari_block = (
        max(block_deltas, key=lambda block: block_deltas[block]["ARI"])
        if block_deltas
        else None
    )
    without_best = {
        metric: summarize_values(
            [
                values[metric]
                for block, values in sorted(block_deltas.items())
                if block != best_ari_block
            ]
        )
        for metric in METRICS
    }

    repair_totals = {
        key: sum(int(row.get(key, 0)) for row in candidate_records)
        for key in (
            "accepted_repairs",
            "helpful_repairs",
            "harmful_repairs",
            "neutral_repairs",
        )
    }
    accepted = repair_totals["accepted_repairs"]
    repair_totals["helpful_rate"] = (
        repair_totals["helpful_repairs"] / accepted if accepted else None
    )

    efficiency: dict[str, Any] = {}
    for variant in variants:
        records = [
            result
            for (condition, _block, _seed), result in results.items()
            if condition == variant
        ]
        if records:
            efficiency[variant] = {
                "completed_block_seed_runs": len(records),
                "mean_per_block": efficiency_mean(records),
            }
    if CANDIDATE in efficiency:
        candidate_efficiency = efficiency[CANDIDATE]["mean_per_block"]
        for reference in ("stage_aware_text_control", "ragsede_original"):
            if reference not in efficiency:
                continue
            reference_efficiency = efficiency[reference]["mean_per_block"]
            efficiency[f"{CANDIDATE}_relative_to_{reference}"] = {
                field: relative_change(
                    candidate_efficiency.get(field), reference_efficiency.get(field)
                )
                for field in ("elapsed_seconds", "llm_calls", "prompt_tokens", "estimated_cost")
            }

    absolute_comparisons: dict[str, Any] = {}
    for reference in ("stage_aware_text_control", "ragsede_original"):
        per_metric: dict[str, list[float]] = defaultdict(list)
        paired_blocks: set[int] = set()
        for block in args.blocks:
            seed_differences: dict[str, list[float]] = defaultdict(list)
            for seed in args.seeds:
                candidate = results.get((CANDIDATE, block, seed))
                baseline = results.get((reference, block, seed))
                if candidate is None or baseline is None:
                    continue
                for metric in METRICS:
                    seed_differences[metric].append(
                        float(candidate["macro_scores"][metric])
                        - float(baseline["macro_scores"][metric])
                    )
            if seed_differences:
                paired_blocks.add(block)
                for metric in METRICS:
                    per_metric[metric].append(mean(seed_differences[metric]))
        if paired_blocks:
            absolute_comparisons[reference] = {
                "warning": (
                    "Quality values come from separate LLM runs. Use the candidate's "
                    "same-run pre/post MVRA comparison for the causal mechanism claim."
                ),
                "paired_blocks": sorted(paired_blocks),
                "metrics": {
                    metric: summarize_values(per_metric[metric]) for metric in METRICS
                },
            }

    required_candidate = len(args.blocks) * len(args.seeds)
    complete_candidate = len(candidate_records) == required_candidate
    nonnegative_ari_blocks = sum(
        values["ARI"] >= 0.0 for values in block_deltas.values()
    )
    required_nonnegative = math.ceil(0.60 * len(args.blocks))
    helpful_rate = repair_totals["helpful_rate"]
    preregistered_checks = {
        "candidate_complete": complete_candidate,
        "mean_NMI_delta_positive": (
            paired_by_metric["NMI"]["mean"] is not None
            and paired_by_metric["NMI"]["mean"] > 0.0
        ),
        "mean_AMI_delta_positive": (
            paired_by_metric["AMI"]["mean"] is not None
            and paired_by_metric["AMI"]["mean"] > 0.0
        ),
        "mean_ARI_delta_nonnegative": (
            paired_by_metric["ARI"]["mean"] is not None
            and paired_by_metric["ARI"]["mean"] >= 0.0
        ),
        "ARI_nonnegative_on_at_least_60_percent_of_blocks": (
            complete_candidate and nonnegative_ari_blocks >= required_nonnegative
        ),
        "mean_ARI_delta_nonnegative_after_removing_best_block": (
            without_best["ARI"]["mean"] is not None
            and without_best["ARI"]["mean"] >= 0.0
        ),
        "helpful_repair_rate_at_least_70_percent": (
            helpful_rate is not None and helpful_rate >= 0.70
        ),
    }
    preregistered_checks["all_pass"] = all(preregistered_checks.values())

    summary = {
        "protocol": "frozen-untouched-block-evaluation-v1",
        "generated_utc": utc_now(),
        "dataset": args.dataset,
        "blocks": args.blocks,
        "seeds": args.seeds,
        "main_sha256": main_sha,
        "candidate": CANDIDATE,
        "statistical_unit": "block after averaging seeds",
        "candidate_completed": len(candidate_records),
        "candidate_required": required_candidate,
        "missing_results": missing,
        "per_block_mean_mvra_delta": {
            f"M{block}": values for block, values in sorted(block_deltas.items())
        },
        "same_run_mvra_delta": paired_by_metric,
        "sensitivity_excluding_best_ARI_block": {
            "excluded_block": (
                f"M{best_ari_block}" if best_ari_block is not None else None
            ),
            "metrics": without_best,
        },
        "repair_audit": repair_totals,
        "efficiency": efficiency,
        "separate_run_absolute_comparisons": absolute_comparisons,
        "preregistered_diagnostic_checks": preregistered_checks,
    }

    output_dir = project / "logs" / args.run_name
    json_path = output_dir / "final_summary.json"
    csv_path = output_dir / "per_block_mvra_delta.csv"
    save_json(json_path, summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("block", *METRICS))
        writer.writeheader()
        for block, values in sorted(block_deltas.items()):
            writer.writerow({"block": block, **values})

    print("\n" + "=" * 88)
    print(f"冻结测试汇总：{args.dataset}，块={args.blocks}，种子={args.seeds}")
    for metric in METRICS:
        row = paired_by_metric[metric]
        if row["mean"] is None:
            print(f"{metric}: 暂无完整候选结果")
        else:
            print(
                f"{metric}: 块均差值={row['mean']:+.4f}, "
                f"中位数={row['median']:+.4f}, 胜/平/负="
                f"{row['wins']}/{row['ties']}/{row['losses']}"
            )
    print(
        "有益/有害修复："
        f"{repair_totals['helpful_repairs']}/{repair_totals['harmful_repairs']}，"
        f"有益率={repair_totals['helpful_rate']}"
    )
    print(f"诊断条件全部通过：{preregistered_checks['all_pass']}")
    print(f"写入：{json_path}")
    print(f"写入：{csv_path}")
    return summary


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset", default="twitter12")
    parser.add_argument("--blocks", type=parse_int_set, default=parse_int_set("7-21"))
    parser.add_argument("--seeds", type=parse_int_set, default=parse_int_set("42,52,62"))
    parser.add_argument("--run-name", default="frozen_time_aux_test_M7-M21")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen runner and analysis for the final MvSED evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a resumable frozen experiment.")
    add_common_arguments(run)
    run.add_argument(
        "--phase",
        choices=tuple(PHASE_VARIANTS),
        default="candidate",
        help="Run the candidate first; run controls only after the frozen test finishes.",
    )
    run.add_argument("--host", default="http://127.0.0.1:9380")
    run.add_argument("--llm-model", default="deepseek-r1-distill-qwen-32b")
    run.add_argument("--max-tokens", type=int, default=2048)
    run.add_argument("--max-llm-attempts", type=int, default=30)
    run.add_argument("--llm-retry-delay", type=float, default=15.0)
    run.add_argument("--run-attempts", type=int, default=3)
    run.add_argument("--run-retry-delay", type=float, default=60.0)
    run.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument(
        "--adopt-existing",
        action="store_true",
        help=(
            "Allow a valid result not registered by this run manifest to be reused. "
            "Off by default to prevent accidental mixing with exploratory outputs."
        ),
    )
    run.add_argument(
        "--continue-on-failure", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--expected-main-sha256", default=FROZEN_MAIN_SHA256
    )
    run.add_argument("--allow-main-mismatch", action="store_true")
    run.add_argument(
        "--main-arg",
        action="append",
        default=[],
        help="Additional single argument forwarded to main.py; repeat as needed.",
    )

    summarize = subparsers.add_parser(
        "summarize", help="Summarize existing results without calling RAGFlow."
    )
    add_common_arguments(summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        return run_experiments(args)
    summarize_results(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
