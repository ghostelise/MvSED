#!/usr/bin/env python3
"""Read-only paired efficiency/quality analysis. Python >=3.10; standard library only.

Does not import main.py, load models, contact an API, or alter experiment files.
Use independent baseline metrics, NOT pre_mvra scores, for the quality comparison.
Intervals resample whole blocks after averaging seeds. They assume exchangeable
blocks and are exploratory for temporally dependent streams; they do not establish
that an algorithm caused a runtime improvement on an external API.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


QUALITY = ("NMI", "AMI", "ARI")
ADDITIVE = (
    "llm_calls", "llm_successful_calls", "llm_failed_attempts",
    "prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost",
    "llm_seconds", "processing_elapsed_seconds", "embedding_seconds",
    "retrieval_seconds", "anchor_seconds", "initialization_seconds",
    "shared_model_initialization_seconds", "provider_reported_calls",
    "estimated_calls",
)
PEAKS = ("peak_rss_mb", "peak_gpu_memory_mb")
CONFIG_FIELDS = (
    "llm_model", "temperature", "top_p", "max_tokens", "presence_penalty",
    "frequency_penalty", "local_embedding_model", "embedding_device",
    "ragflow_embedding_model", "reuse_embedding_cache", "max_llm_attempts",
    "llm_retry_delay", "max_cluster_size", "anchor_top_k", "anchor_lambda",
    "skip_initial_block", "language", "dataset_threshold", "save_audit_log",
    "HOST_ADDRESS", "input_cost_per_million", "output_cost_per_million",
    "anchor_preprocessing",
)
ENV_FIELDS = ("python", "torch", "numpy", "scikit_learn", "sentence_transformers", "ragflow_sdk", "gpu")


def int_set(value):
    try:
        result = set()
        for part in value.split(","):
            ends = part.strip().split("-")
            if len(ends) == 1:
                result.add(int(ends[0]))
            elif len(ends) == 2:
                lo, hi = map(int, ends)
                if lo > hi or hi - lo > 10000:
                    raise ValueError()
                result.update(range(lo, hi + 1))
            else:
                raise ValueError()
        if not result or min(result) < 0:
            raise ValueError()
        return sorted(result)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use 1-16 or 42,52,62; nonnegative integers only.") from exc


def finite(value, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        return None
    if positive and value <= 0:
        return None
    return value


def identifier(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def reduction(ours, baseline):
    return 100.0 * (1.0 - ours / baseline) if baseline > 0 else None


def sign_test(deltas, tolerance=1e-9):
    wins = sum(x > tolerance for x in deltas)
    losses = sum(x < -tolerance for x in deltas)
    n = wins + losses
    p = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2 ** n)) if n else None
    return {"wins": wins, "ties": len(deltas) - n, "losses": losses,
            "exact_sign_test_two_sided_p": p}


def percentile(sorted_values, fraction):
    index = (len(sorted_values) - 1) * fraction
    lo = math.floor(index)
    hi = math.ceil(index)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (index - lo)


def bootstrap(blocks, repetitions, seed):
    # Both methods and all seeds in a block stay together in each resample.
    if len(blocks) < 2 or repetitions == 0:
        return {"available": False, "reason": "Need >=2 complete blocks and bootstrap >0."}
    rng = random.Random(seed)
    draws = {k: [] for k in ("runtime_reduction_percent", "geometric_speedup", *QUALITY)}
    for _ in range(repetitions):
        sample = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        ours = sum(b["ours_seconds"] for b in sample)
        base = sum(b["baseline_seconds"] for b in sample)
        draws["runtime_reduction_percent"].append(reduction(ours, base))
        draws["geometric_speedup"].append(math.exp(mean(math.log(b["speedup"]) for b in sample)))
        for metric in QUALITY:
            draws[metric].append(mean(b[f"delta_{metric}"] for b in sample))
    return {
        "available": True, "confidence_level": 0.95, "repetitions": repetitions,
        "random_seed": seed, "unit": "block after averaging matched seeds",
        "assumption": "Exchangeable blocks; temporal dependence may invalidate nominal coverage.",
        "percentile_intervals": {
            name: [percentile(sorted(values), 0.025), percentile(sorted(values), 0.975)]
            for name, values in draws.items()
        },
    }


def read_record(path, dataset, variant, pipeline, block, seed):
    raw = path.read_bytes()
    try:
        obj = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid/truncated JSON") from exc
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")
    expected = dict(dataset=dataset, variant=variant, anchor_pipeline=pipeline, block=block, seed=seed)
    for key, value in expected.items():
        if obj.get(key) != value or isinstance(obj.get(key), bool):
            raise ValueError(f"metadata mismatch: {key}")
    efficiency = obj.get("efficiency")
    scores = obj.get("macro_scores")
    if not isinstance(efficiency, dict) or not isinstance(scores, dict):
        raise ValueError("missing efficiency/macro_scores object")
    seconds = finite(efficiency.get("elapsed_seconds"), positive=True)
    messages = finite(efficiency.get("messages"), positive=True)
    if seconds is None or messages is None or int(messages) != messages:
        raise ValueError("elapsed_seconds must be positive; messages must be a positive integer")
    for key in ("input_messages", "covered_messages"):
        if key in efficiency and finite(efficiency[key], positive=True) != messages:
            raise ValueError(f"incomplete or inconsistent message coverage: {key}")
    if "evaluation_coverage" in efficiency:
        coverage = finite(efficiency["evaluation_coverage"])
        if coverage is None or abs(coverage - 1.0) > 1e-9:
            raise ValueError("evaluation_coverage is not 1")
    for metric in QUALITY:
        score = finite(scores.get(metric))
        if score is None or not (-1 <= score <= 1) or (metric == "NMI" and score < 0):
            raise ValueError(f"missing/nonfinite/out-of-range {metric}")
    numbers = {key: finite(efficiency.get(key), minimum=0) for key in (*ADDITIVE, *PEAKS, "anchors", "stage1_anchors")}
    numbers["total_tokens"] = (
        numbers["prompt_tokens"] + numbers["completion_tokens"]
        if numbers["prompt_tokens"] is not None and numbers["completion_tokens"] is not None else None
    )
    source = efficiency.get("token_usage_source") or {}
    if not isinstance(source, dict):
        source = {}
    for key in ("provider_reported_calls", "estimated_calls"):
        numbers[key] = finite(source.get(key), minimum=0)
    rates = [finite(efficiency.get(k), minimum=0) for k in ("input_cost_per_million", "output_cost_per_million")]
    priced = all(x is not None for x in rates) and any(x > 0 for x in rates)
    if not priced:
        numbers["estimated_cost"] = None  # Unconfigured price is NOT zero cost.
    return {"data": obj, "seconds": seconds, "messages": messages, "numbers": numbers,
            "scores": {m: scores[m] for m in QUALITY}, "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest()}


def compare_protocol(ours, base):
    differences, unknown = [], []
    # Never copy full configurations (which may contain credentials) into reports.
    for section, keys in (("configuration", CONFIG_FIELDS), ("environment", ENV_FIELDS)):
        a, b = ours["data"].get(section), base["data"].get(section)
        a = a if isinstance(a, dict) else {}
        b = b if isinstance(b, dict) else {}
        for key in keys:
            name = f"{section}.{key}"
            if key not in a or key not in b:
                unknown.append(name)
            elif a[key] != b[key]:
                differences.append(name)
    for key in ("stage1_anchors", "anchors"):
        a, b = ours["numbers"][key], base["numbers"][key]
        if a is None or b is None:
            unknown.append(f"efficiency.{key}")
        elif a != b:
            differences.append(f"efficiency.{key}")
    a = ours["data"].get("efficiency", {}).get("anchor_audit", {})
    b = base["data"].get("efficiency", {}).get("anchor_audit", {})
    for key in ("effective_threshold", "preprocessing", "raw_text_sha256", "preprocessed_text_sha256",
                "ordered_membership_sha256", "ordered_representatives_sha256", "representative_text_sha256"):
        if key not in a or key not in b:
            unknown.append(f"anchor_audit.{key}")
        elif a[key] != b[key]:
            differences.append(f"anchor_audit.{key}")
    return differences, unknown


def collect(args):
    default_root = args.project / "ckpts" / args.dataset
    roots = {"ours": (args.ours_root or default_root).resolve(),
             "baseline": (args.baseline_root or default_root).resolve()}
    indexes = {}
    for root in set(roots.values()):
        index = defaultdict(list)
        # Exact canonical basenames exclude metrics.smoke_backup.json, etc.
        # Same-name copies in subdirectories are ambiguous and are NOT auto-picked.
        for path in sorted(root.rglob("*_metrics.json")):
            index[path.name].append(path)
        indexes[root] = index
    rows, audit = [], []
    for block in args.blocks:
        for seed in args.seeds:
            entry = {"block": block, "seed": seed, "issues": [], "files": {}}
            records = {}
            for role, variant in (("ours", args.ours), ("baseline", args.baseline)):
                name = f"M{block}_{identifier(variant)}_{identifier(args.pipeline)}_s{seed}_metrics.json"
                paths = indexes[roots[role]].get(name, [])
                if len(paths) != 1:
                    entry["issues"].append(f"{role}: {'missing' if not paths else 'duplicate canonical filename'}")
                    entry["files"][role] = {"expected_name": name, "matches": [str(p) for p in paths]}
                    continue
                try:
                    record = read_record(paths[0], args.dataset, variant, args.pipeline, block, seed)
                except (OSError, ValueError) as exc:
                    entry["issues"].append(f"{role}: {exc}")
                    continue
                records[role] = record
                entry["files"][role] = {"path": record["path"], "sha256": record["sha256"]}
            if len(records) == 2:
                ours, base = records["ours"], records["baseline"]
                if ours["messages"] != base["messages"]:
                    entry["issues"].append("different evaluated message counts")
                gold_a, gold_b = ours["data"].get("gold_cluster_count"), base["data"].get("gold_cluster_count")
                if gold_a is not None and gold_b is not None and gold_a != gold_b:
                    entry["issues"].append("different gold cluster counts")
                differences, unknown = compare_protocol(ours, base)
                entry["protocol_differences"] = differences
                entry["unverified_protocol_fields"] = unknown
            entry["valid_pair"] = not entry["issues"]
            audit.append(entry)
            if entry["valid_pair"]:
                rows.append({"block": block, "seed": seed, **records})
    # Do not average different sets of seeds in different blocks.
    counts = defaultdict(int)
    for row in rows:
        counts[row["block"]] += 1
    kept_blocks = [b for b in args.blocks if counts[b] == len(args.seeds)]
    rows = [r for r in rows if r["block"] in kept_blocks]
    for entry in audit:
        entry["included"] = entry["valid_pair"] and entry["block"] in kept_blocks
    return rows, {"roots": {k: str(v) for k, v in roots.items()}, "pairs": audit,
                  "complete_blocks": kept_blocks, "excluded_blocks": [b for b in args.blocks if b not in kept_blocks]}


def paired_optional(rows, field, peak=False):
    pairs = [(r["ours"]["numbers"][field], r["baseline"]["numbers"][field]) for r in rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    result = {"available_pairs": len(pairs), "required_pairs": len(rows)}
    if not pairs:
        return {**result, "ours": None, "baseline": None}
    a, b = zip(*pairs)
    if peak:
        return {**result, "ours_mean_recorded_peak_mb": mean(a), "baseline_mean_recorded_peak_mb": mean(b),
                "ours_max_recorded_peak_mb": max(a), "baseline_max_recorded_peak_mb": max(b)}
    return {**result, "ours": sum(a), "baseline": sum(b),
            "saved": sum(b) - sum(a), "reduction_percent": reduction(sum(a), sum(b))}


def analyze(rows, args, audit):
    groups = defaultdict(list)
    for row in rows:
        groups[row["block"]].append(row)
    blocks = []
    for block, group in sorted(groups.items()):
        item = {"block": block, "seeds": ",".join(str(r["seed"]) for r in group),
                "messages": mean(r["ours"]["messages"] for r in group)}
        for role in ("ours", "baseline"):
            item[f"{role}_seconds"] = mean(r[role]["seconds"] for r in group)
            item[f"{role}_minutes"] = item[f"{role}_seconds"] / 60
            for metric in QUALITY:
                item[f"{role}_{metric}"] = mean(r[role]["scores"][metric] for r in group)
            for field in (*ADDITIVE, *PEAKS, "anchors", "stage1_anchors"):
                values = [r[role]["numbers"][field] for r in group]
                item[f"{role}_{field}_seed_mean"] = (
                    mean(values) if all(v is not None for v in values) else None
                )
        item["speedup"] = item["baseline_seconds"] / item["ours_seconds"]
        item["runtime_reduction_percent"] = reduction(item["ours_seconds"], item["baseline_seconds"])
        for metric in QUALITY:
            item[f"delta_{metric}"] = item[f"ours_{metric}"] - item[f"baseline_{metric}"]
        blocks.append(item)
    ours_time = sum(r["ours"]["seconds"] for r in rows)
    base_time = sum(r["baseline"]["seconds"] for r in rows)
    messages = sum(r["ours"]["messages"] for r in rows)
    included_audit = [e for e in audit["pairs"] if e["included"]]
    mismatches = sorted({f for e in included_audit for f in e["protocol_differences"]})
    unknown = sorted({f for e in included_audit for f in e["unverified_protocol_fields"]})
    intervals = bootstrap(blocks, args.bootstrap, args.bootstrap_seed)
    summary = {
        "dataset": args.dataset, "ours": args.ours, "baseline": args.baseline,
        "pipeline": args.pipeline, "seeds": args.seeds, "requested_blocks": args.blocks,
        "included_blocks": [b["block"] for b in blocks], "excluded_blocks": audit["excluded_blocks"],
        "complete_requested_pairing": not audit["excluded_blocks"], "paired_runs": len(rows),
        "quality_comparison": "Final macro_scores from independent method runs, not pre/post MVRA.",
        "protocol_audit": {
            "status": "recorded_mismatch" if mismatches else ("incomplete_metadata" if unknown else "no_mismatch_in_checked_fields"),
            "different_fields": mismatches, "unknown_fields": unknown,
            "caveat": "Counts/configuration do not prove identical inputs, prompts, cloud provider, server load, or absence of smoke tests.",
        },
        "runtime": {
            "ours_total_seconds": ours_time, "baseline_total_seconds": base_time,
            "ours_total_hours": ours_time / 3600, "baseline_total_hours": base_time / 3600,
            "total_runtime_reduction_percent": reduction(ours_time, base_time),
            "total_speedup": base_time / ours_time,
            "mean_block_runtime_reduction_percent": mean(b["runtime_reduction_percent"] for b in blocks),
            "geometric_mean_block_speedup": math.exp(mean(math.log(b["speedup"]) for b in blocks)),
            "median_block_runtime_reduction_percent": median(b["runtime_reduction_percent"] for b in blocks),
            "evaluated_message_instances": messages,
            "ours_messages_per_second": messages / ours_time,
            "baseline_messages_per_second": messages / base_time,
            **sign_test([b["baseline_seconds"] - b["ours_seconds"] for b in blocks]),
        },
        "quality": {
            metric: {"ours_block_mean": mean(b[f"ours_{metric}"] for b in blocks),
                     "baseline_block_mean": mean(b[f"baseline_{metric}"] for b in blocks),
                     "mean_delta": mean(b[f"delta_{metric}"] for b in blocks),
                     "median_delta": median(b[f"delta_{metric}"] for b in blocks),
                     **sign_test([b[f"delta_{metric}"] for b in blocks], args.quality_tolerance)}
            for metric in QUALITY
        },
        "resources": {field: paired_optional(rows, field) for field in ADDITIVE},
        "memory": {field: paired_optional(rows, field, peak=True) for field in PEAKS},
        "bootstrap": intervals,
        "notes": [
            "All requested seeds must exist for BOTH methods in an included block; partial blocks are excluded as a whole only with --allow-partial.",
            "Protocol mismatches are reported, not silently filtered out. Such comparisons are descriptive, not strictly controlled.",
            "95% intervals use paired block bootstrap. Adjacent blocks may be dependent; one seed does not establish seed-to-seed stability.",
            "Sign tests are two-sided and unadjusted for multiple outcomes. No automatic significant/superior claim is made.",
            "elapsed_seconds is the code-recorded per-block scope, including allocated shared initialization; not necessarily driver wall time.",
            "Failed/restarted attempts without saved metrics are not included. Runtime gains may reflect provider latency/retry differences.",
            "llm_calls is the logger's call counter; do not add llm_failed_attempts to it to invent a total request count.",
            "Missing optional measurements remain null; resource totals are paired only where BOTH sides recorded that field. Check available_pairs.",
            "Token counts may be estimated. Unset/zero prices produce unknown cost, not free usage. Currency and billing scope need verification.",
            "Memory is recorded local process/framework peaks, not a sum across blocks and not remote LLM/server memory.",
            "Exact canonical filenames exclude named smoke backups; a smoke run stored at the canonical path cannot be identified automatically.",
        ],
    }
    return summary, blocks


def seed_rows(rows):
    result = []
    for pair in rows:
        row = {"block": pair["block"], "seed": pair["seed"], "messages": pair["ours"]["messages"]}
        for role in ("ours", "baseline"):
            r = pair[role]
            row.update({f"{role}_seconds": r["seconds"], f"{role}_minutes": r["seconds"] / 60,
                        f"{role}_path": r["path"], f"{role}_sha256": r["sha256"]})
            for field, value in {**r["numbers"], **r["scores"]}.items():
                row[f"{role}_{field}"] = value
        row["runtime_reduction_percent"] = reduction(pair["ours"]["seconds"], pair["baseline"]["seconds"])
        row["speedup"] = pair["baseline"]["seconds"] / pair["ours"]["seconds"]
        for metric in QUALITY:
            row[f"delta_{metric}"] = pair["ours"]["scores"][metric] - pair["baseline"]["scores"][metric]
        result.append(row)
    return result


def fmt(value, digits=2):
    return "未记录/不可计算" if value is None else f"{value:.{digits}f}"


def report(summary, blocks):
    t = summary["runtime"]
    lines = [f"效率与质量配对统计：{summary['dataset']}，seeds={summary['seeds']}",
             f"改进组：{summary['ours']}", f"对照组：{summary['baseline']}",
             f"纳入块：{summary['included_blocks']}；配对运行：{summary['paired_runs']}",
             f"排除块：{summary['excluded_blocks']}",
             "\n块    改进/分钟   对照/分钟   时间降低%    速度倍数    ΔNMI      ΔAMI      ΔARI"]
    for b in blocks:
        lines.append(f"M{b['block']:<3} {b['ours_seconds']/60:10.2f} {b['baseline_seconds']/60:11.2f} "
                     f"{b['runtime_reduction_percent']:+10.2f} {b['speedup']:10.3f}x "
                     + " ".join(f"{b[f'delta_{m}']:+.5f}" for m in QUALITY))
    lines += ["\n总体效率（全部纳入块的配对工作量）",
              f"总时间：Ours={t['ours_total_hours']:.2f}小时，RagSEDE/对照={t['baseline_total_hours']:.2f}小时",
              f"总时间降低：{t['total_runtime_reduction_percent']:+.2f}%；总速度倍数：{t['total_speedup']:.3f}x",
              f"块平均时间降低：{t['mean_block_runtime_reduction_percent']:+.2f}%；几何平均速度倍数：{t['geometric_mean_block_speedup']:.3f}x",
              f"更快/相同/更慢块：{t['wins']}/{t['ties']}/{t['losses']}",
              f"双侧符号检验 p={fmt(t['exact_sign_test_two_sided_p'], 6)}（不是 Wilcoxon）",
              f"吞吐量：Ours={t['ours_messages_per_second']:.3f}，对照={t['baseline_messages_per_second']:.3f} 条/秒"]
    ci = summary["bootstrap"].get("percentile_intervals", {})
    for key, label in (("runtime_reduction_percent", "时间降低95%区间/%"), ("geometric_speedup", "几何速度倍数95%区间/x")):
        if key in ci:
            lines.append(f"{label}：[{ci[key][0]:.3f}, {ci[key][1]:.3f}]（按块重采样）")
    lines.append("\n调用、token与资源（缺失不是0；覆盖不足时只代表有记录的配对子集）")
    for field in ADDITIVE:
        v = summary["resources"][field]
        lines.append(f"{field}: Ours={fmt(v['ours'])}, 对照={fmt(v['baseline'])}, "
                     f"降低%={fmt(v.get('reduction_percent'))}, 覆盖={v['available_pairs']}/{v['required_pairs']}")
    for field in PEAKS:
        v = summary["memory"][field]
        lines.append(f"{field}: 平均记录峰值 Ours={fmt(v.get('ours_mean_recorded_peak_mb'))}, "
                     f"对照={fmt(v.get('baseline_mean_recorded_peak_mb'))} MB；覆盖={v['available_pairs']}/{v['required_pairs']}")
    lines.append("\n最终质量（独立运行的对照组，不是修复前后）")
    for metric in QUALITY:
        q = summary["quality"][metric]
        lines.append(f"{metric}: Ours={q['ours_block_mean']:.5f}, 对照={q['baseline_block_mean']:.5f}, "
                     f"Δ={q['mean_delta']:+.5f}, 胜/平/负={q['wins']}/{q['ties']}/{q['losses']}")
        if metric in ci:
            lines.append(f"  均差95%区间=[{ci[metric][0]:+.5f}, {ci[metric][1]:+.5f}]")
    p = summary["protocol_audit"]
    lines += ["\n可比性检查", f"状态：{p['status']}",
              "记录不一致的字段：" + (", ".join(p["different_fields"]) or "无"),
              "未能核验的字段：" + (", ".join(p["unknown_fields"]) or "无"),
              "\n重要限制：这只是已保存运行的统计。服务端负载、重试、欠费中断和重复启动会影响时间；不能仅据此断言算法本身显著更快。",
              "时间区间/检验依赖块间独立或可交换假设；相邻时间块相关、单seed时须谨慎解释。",
              "请确认M16是正式结果，且两组服务商、模型版本、计时范围与运行条件一致。"]
    return "\n".join(lines) + "\n"


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--dataset", default="twitter18")
    p.add_argument("--blocks", type=int_set, default=int_set("1-16"))
    p.add_argument("--seeds", type=int_set, default=[42])
    p.add_argument("--ours", default="stage_aware_selective_gate_time_aux")
    p.add_argument("--baseline", default="ragsede_original")
    p.add_argument("--pipeline", default="single_stage")
    p.add_argument("--ours-root", type=Path, help="Optional results root for our method; default PROJECT/ckpts/DATASET.")
    p.add_argument("--baseline-root", type=Path, help="Optional results root for baseline; same default.")
    p.add_argument("--output", type=Path, help="New output directory; existing directories are never overwritten.")
    p.add_argument("--allow-partial", action="store_true", help="Explicitly allow whole-block exclusion when a requested pair is missing/invalid.")
    p.add_argument("--bootstrap", type=int, default=10000, help="Paired block bootstrap repetitions; 0 disables.")
    p.add_argument("--bootstrap-seed", type=int, default=20260904)
    p.add_argument("--quality-tolerance", type=float, default=1e-9)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.ours == args.baseline:
        parser.error("--ours and --baseline must be different variants")
    if not 0 <= args.bootstrap <= 1000000:
        parser.error("--bootstrap must be between 0 and 1000000")
    if finite(args.quality_tolerance, minimum=0) is None:
        parser.error("--quality-tolerance must be finite and nonnegative")
    args.project = args.project.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output or args.project / "logs" / f"{identifier(args.dataset)}_efficiency_{stamp}").resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists; choose a new --output: {output}")
    rows, audit = collect(args)
    write_json(output / "input_audit.json", audit)
    if not rows or (audit["excluded_blocks"] and not args.allow_partial):
        print("未生成总体结论：请求的两组结果缺失、重复、不完整或不匹配。")
        for entry in audit["pairs"]:
            if entry["issues"]:
                print(f"M{entry['block']} s{entry['seed']}: " + "; ".join(entry["issues"]))
        print(f"检查记录：{output / 'input_audit.json'}")
        print("请补齐/确认正式结果；仅需部分结果时明确使用 --allow-partial，脚本不会自动挑选更好的结果。")
        return 2
    summary, blocks = analyze(rows, args, audit)
    summary["generated_utc"] = datetime.now(timezone.utc).isoformat()
    summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["output_dir"] = str(output)
    summary["bootstrap"]["random_seed"] = args.bootstrap_seed
    summary["quality_tie_tolerance"] = args.quality_tolerance
    write_json(output / "summary.json", summary)
    write_csv(output / "paired_runs.csv", seed_rows(rows))
    write_csv(output / "paired_blocks.csv", blocks)
    rendered = report(summary, blocks)
    with (output / "report.txt").open("x", encoding="utf-8") as handle:
        handle.write(rendered)
    print(rendered)
    print(f"输出目录：{output}")
    print("文件：summary.json / paired_runs.csv / paired_blocks.csv / report.txt / input_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
