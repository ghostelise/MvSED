#!/usr/bin/env python3
"""Protocol-correction runner. Standard library only; the child main.py needs its usual environment.

No old result is reused or overwritten. All paid runs use new, isolated working
directories and recompute raw-message embeddings (no warm-cache timing advantage).
Original per-variant detector prompts/retrieval paths are retained: this is an
anchor-aligned system comparison, NOT a one-switch MVRA ablation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OURS = "stage_aware_selective_gate_time_aux"
BASE = "ragsede_original"
VARIANTS = (BASE, OURS)
AUDIT_FIELDS = (
    "raw_text_sha256", "preprocessed_text_sha256", "embedding_model",
    "effective_threshold", "preprocessing", "message_count", "anchor_count",
    "ordered_membership_sha256", "ordered_representatives_sha256", "representative_text_sha256",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temp, path)


def integers(value):
    try:
        out = set()
        for part in value.split(","):
            ends = [int(x) for x in part.split("-")]
            if len(ends) == 1:
                out.add(ends[0])
            elif len(ends) == 2 and 0 <= ends[0] <= ends[1] <= 10000:
                out.update(range(ends[0], ends[1] + 1))
            else:
                raise ValueError()
        if not out or min(out) < 0:
            raise ValueError()
        return sorted(out)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use 1-16 or 42,52,62") from exc


def schedule(blocks, seeds):
    # AB/BA is fixed before looking at scores. Exactly balanced for 16 blocks.
    return [(seed, block, variant) for si, seed in enumerate(seeds)
            for bi, block in enumerate(blocks)
            for variant in (VARIANTS if (si + bi) % 2 == 0 else VARIANTS[::-1])]


def filename(block, variant, seed, kind):
    return f"M{block}_{variant}_single_stage_s{seed}_{kind}.json"


def output_file(work, block, variant, seed, kind):
    return Path(work) / "ckpts" / "twitter18" / f"M{block}" / filename(block, variant, seed, kind)


def assert_audit(actual, expected):
    differences = [k for k in AUDIT_FIELDS if k not in actual or k not in expected or actual[k] != expected[k]]
    if differences:
        raise RuntimeError("Anchor evidence does not match: " + ", ".join(differences))


def check_for_other_main(project):
    # Inspect only identities/cwd; never print a command line containing an API key.
    proc = Path("/proc")
    if not proc.is_dir():
        return
    found = []
    for folder in proc.iterdir():
        if not folder.name.isdigit():
            continue
        try:
            command = (folder / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if not any(Path(x).name == "main.py" for x in command if x):
                continue
            cwd = (folder / "cwd").resolve()
            if cwd == project or project in cwd.parents or any(str(project) in x for x in command):
                found.append(folder.name)
        except (OSError, RuntimeError):
            continue
    if found:
        raise RuntimeError("Existing experiment process(es), PID=" + ",".join(found) + ". Stop/finish them before rerunning; no process was killed.")


def command(args, snapshot, seed, first, last, variant, anchor_only=False, expected=None):
    result = [sys.executable, "-u", str(snapshot / "main.py"),
              "--dataset", "twitter18", "--data_path", str(args.data_path),
              "--HOST_ADDRESS", args.host, "--variant", variant,
              "--anchor_pipeline", "single_stage", "--anchor_preprocessing", "ragsede",
              "--twitter18_threshold", "0.30", "--ragsede_twitter18_threshold", "0.30",
              "--language", "French", "--local_embedding_model", args.encoder,
              "--embedding_device", args.device, "--seed", str(seed),
              "--start_block", str(first), "--end_block", str(last), "--skip_initial_block",
              "--llm_model", args.llm_model, "--max_tokens", "2048",
              "--temperature", "0.0", "--top_p", "0.3",
              "--max_llm_attempts", "30", "--llm_retry_delay", "15.0",
              "--max_cluster_size", "100", "--anchor_top_k", "3", "--anchor_lambda", "0.70",
              "--no-reuse_embedding_cache", "--save_audit_log"]
    if anchor_only:
        result.append("--anchor_only")
    if expected is not None:
        result.extend(["--expected_anchor_audit", str(expected)])
    return result


def stream_process(cmd, work, log, env):
    started = time.monotonic()
    with Path(log).open("xb") as handle:
        child = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=0)
        try:
            while True:
                chunk = child.stdout.read(4096)
                if not chunk:
                    break
                handle.write(chunk)
                handle.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            code = child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
        finally:
            child.stdout.close()
    return code, time.monotonic() - started


def configuration(args):
    return {"protocol": "twitter18-anchor-aligned-correction-v2", "blocks": args.blocks, "seeds": args.seeds,
            "data_path": str(args.data_path), "data_sha256": digest(args.data_path),
            "host": args.host, "llm_model": args.llm_model, "encoder": args.encoder, "device": args.device,
            "threshold": 0.30, "anchor_preprocessing": "ragsede", "max_llm_attempts": 30,
            "llm_retry_delay": 15.0, "max_tokens": 2048, "raw_embedding_cache": "disabled for both methods",
            "threads": args.threads, "order": "alternating AB/BA by block; sequential, never concurrent",
            "comparison": "system-level; inherited per-method prompts/retrieval/joining retained",
            "evaluation_status": "protocol correction after inspecting previous Twitter18 results; not a new untouched test",
            "runner_sha256": digest(__file__)}


def verify_frozen(args, root, manifest):
    if digest(args.data_path) != manifest["configuration"]["data_sha256"]:
        raise RuntimeError("Dataset changed after run initialization")
    for name, expected in manifest["snapshot_sha256"].items():
        if digest(root / "source" / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")


def initialize(args, root):
    conf = configuration(args)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = load(manifest_path)
        if manifest["configuration"] != conf:
            raise RuntimeError("Run configuration changed. Resume with identical arguments, or use a new --run-name; do not mix protocols.")
        verify_frozen(args, root, manifest)
        return manifest
    if root.exists():
        raise RuntimeError(f"Unregistered existing run directory: {root}; choose a new --run-name")
    if not args.data_path.is_file():
        raise FileNotFoundError(args.data_path)
    source_files = list(args.project.glob("*.py"))
    source_files += list((args.project / "evolution").rglob("*.py"))
    required = {"main.py", "data_process.py", "RAGFlow.py", "get_key_messages.py", "utils.py", "summarize_efficiency.py"}
    if not required.issubset({p.name for p in source_files}):
        raise RuntimeError("Missing required source files in project")
    main_text = (args.project / "main.py").read_text(encoding="utf-8")
    data_text = (args.project / "data_process.py").read_text(encoding="utf-8")
    if "--anchor_only" not in main_text or "def validate_anchor_audit" not in data_text:
        raise RuntimeError("Upload BOTH patched main.py and data_process.py before starting")
    snapshot = root / "source"
    snapshot.mkdir(parents=True)
    hashes = {}
    for path in source_files:
        relative = path.relative_to(args.project)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = digest(destination)
    manifest = {"created_utc": now(), "configuration": conf, "snapshot_sha256": hashes,
                "preflight": {}, "jobs": {}, "events": []}
    save(manifest_path, manifest)
    return manifest


def execute_attempt(args, root, manifest, category, key, cmd_factory, validate):
    table = manifest[category]
    entry = table.setdefault(key, {"attempts": []})
    if entry.get("status") == "completed":
        for path, checksum in entry["files"].items():
            if digest(root / path) != checksum:
                raise RuntimeError(f"Registered completed artifact changed/missing: {path}")
        validate(root / entry["work"])
        print(f"SKIP completed {category}: {key}", flush=True)
        return root / entry["work"]
    check_for_other_main(args.project)
    verify_frozen(args, root, manifest)
    work = root / category / key / f"attempt_{len(entry['attempts']) + 1:03d}"
    work.mkdir(parents=True, exist_ok=False)
    cmd = cmd_factory(work)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONHASHSEED=key.split("_")[0].removeprefix("s"),
               OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="false")
    attempt = {"started_utc": now(), "work": str(work.relative_to(root)), "command_without_api_key": cmd,
               "status": "running"}
    entry["attempts"].append(attempt)
    entry["status"] = "running"
    save(root / "manifest.json", manifest)
    print(f"\nRUN {category}: {key}; log={work / 'run.log'}", flush=True)
    attempt_started = time.monotonic()
    try:
        code, seconds = stream_process(cmd, work, work / "run.log", env)
        attempt.update(exit_code=code, driver_elapsed_seconds=seconds)
        if code != 0:
            raise RuntimeError(f"Child exited with code {code}. Read {work / 'run.log'}")
        files = validate(work)
        entry.update(status="completed", work=str(work.relative_to(root)),
                     files={str(p.relative_to(root)): digest(p) for p in files})
        attempt["status"] = "completed"
    except BaseException as exc:
        entry["status"] = "failed"
        attempt.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        attempt.setdefault("driver_elapsed_seconds", time.monotonic() - attempt_started)
        attempt["finished_utc"] = now()
        save(root / "manifest.json", manifest)
    return work


def preflight(args, root, manifest):
    paths = {}
    for seed in args.seeds:
        for variant in VARIANTS:
            key = f"s{seed}_{variant}"

            def validate(work, variant=variant, seed=seed):
                files = []
                for block in args.blocks:
                    path = output_file(work, block, variant, seed, "anchor_audit")
                    result = load(path)
                    if (result["dataset"], result["block"], result["seed"], result["variant"]) != ("twitter18", block, seed, variant):
                        raise RuntimeError("Preflight identity mismatch")
                    audit = result["anchor_audit"]
                    if audit["effective_threshold"] != 0.30 or audit["preprocessing"] != "ragsede":
                        raise RuntimeError("Preflight effective threshold/preprocessing mismatch")
                    files.append(path)
                return files

            work = execute_attempt(args, root, manifest, "preflight", key,
                lambda work, variant=variant, seed=seed: command(args, root / "source", seed, args.blocks[0], args.blocks[-1], variant, anchor_only=True), validate)
            for block in args.blocks:
                paths[seed, block, variant] = output_file(work, block, variant, seed, "anchor_audit")
        for block in args.blocks:
            baseline = load(paths[seed, block, BASE])["anchor_audit"]
            ours = load(paths[seed, block, OURS])["anchor_audit"]
            assert_audit(ours, baseline)
            print(f"ANCHORS MATCH s{seed} M{block}: {baseline['anchor_count']} anchors; ordered membership and representatives identical", flush=True)
    manifest["anchor_alignment_passed"] = True
    save(root / "manifest.json", manifest)
    return {(seed, block): paths[seed, block, BASE] for seed in args.seeds for block in args.blocks}


def validate_paid(work, block, seed, variant, expected, args):
    metric_path = output_file(work, block, variant, seed, "metrics")
    result = load(metric_path)
    if (result["dataset"], result["block"], result["seed"], result["variant"], result["anchor_pipeline"]) != ("twitter18", block, seed, variant, "single_stage"):
        raise RuntimeError("Completed-run identity mismatch")
    assert_audit(result["efficiency"]["anchor_audit"], load(expected)["anchor_audit"])
    config = result["configuration"]
    for key, value in {"max_llm_attempts": 30, "max_tokens": 2048, "llm_retry_delay": 15.0,
                       "local_embedding_model": args.encoder, "llm_model": args.llm_model,
                       "anchor_preprocessing": "ragsede", "reuse_embedding_cache": False}.items():
        if config.get(key) != value:
            raise RuntimeError(f"Completed-run configuration mismatch: {key}")
    predictions_path = output_file(work, block, variant, seed, "predictions")
    predictions = load(predictions_path)
    n = result["efficiency"]["anchor_audit"]["message_count"]
    if len(predictions["gold_labels"]) != n or len(predictions["predicted_clusters"]) != n:
        raise RuntimeError("Incomplete saved predictions")
    for metric in ("NMI", "AMI", "ARI"):
        score = result["macro_scores"].get(metric)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not -1 <= score <= 1 or (metric == "NMI" and score < 0):
            raise RuntimeError("Invalid metric value")
    return [metric_path, predictions_path, expected]


def run_paid(args, root, manifest, expected):
    if not os.environ.get("RAGFLOW_API_KEY", "").strip():
        raise RuntimeError("RAGFLOW_API_KEY is not set. Export it in this terminal; do not put it in a log or script.")
    jobs = schedule(args.blocks, args.seeds)
    for position, (seed, block, variant) in enumerate(jobs, start=1):
        key = f"s{seed}_M{block}_{variant}"
        print(f"\n[{position}/{len(jobs)}] {key}", flush=True)
        expected_path = expected[seed, block]
        work = execute_attempt(args, root, manifest, "jobs", key,
            lambda work, seed=seed, block=block, variant=variant, expected_path=expected_path: command(args, root / "source", seed, block, block, variant, expected=expected_path),
            lambda work, seed=seed, block=block, variant=variant, expected_path=expected_path: validate_paid(work, block, seed, variant, expected_path, args))
        source = output_file(work, block, variant, seed, "metrics")
        destination = output_file(root / "results", block, variant, seed, "metrics")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination) != digest(source):
            raise RuntimeError(f"Collected result conflict: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
    manifest["all_paid_runs_completed"] = True
    save(root / "manifest.json", manifest)


def summarize(args, root, manifest):
    if not manifest.get("all_paid_runs_completed"):
        raise RuntimeError("Not all planned paid runs are completed; use status or resume run")
    # Verify all registered artifacts, including collected metrics, before analysis.
    for seed, block, variant in schedule(args.blocks, args.seeds):
        entry = manifest["jobs"][f"s{seed}_M{block}_{variant}"]
        for name, checksum in entry["files"].items():
            if digest(root / name) != checksum:
                raise RuntimeError(f"Completed artifact changed: {name}")
        if digest(output_file(root / "results", block, variant, seed, "metrics")) != digest(output_file(root / entry["work"], block, variant, seed, "metrics")):
            raise RuntimeError("Collected metric does not match registered run")
    output = root / "reports" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    result_root = root / "results" / "ckpts" / "twitter18"
    cmd = [sys.executable, str(root / "source" / "summarize_efficiency.py"), "--project", str(args.project),
           "--dataset", "twitter18", "--blocks", ",".join(map(str, args.blocks)),
           "--seeds", ",".join(map(str, args.seeds)), "--ours-root", str(result_root),
           "--baseline-root", str(result_root), "--output", str(output)]
    subprocess.run(cmd, check=True)
    manifest["latest_report"] = str(output)
    save(root / "manifest.json", manifest)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=("plan", "check", "run", "status", "summarize"))
    p.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--run-name", default="twitter18_anchor_aligned_v2_s42")
    p.add_argument("--blocks", type=integers, default=integers("1-16"))
    p.add_argument("--seeds", type=integers, default=[42])
    p.add_argument("--data-path", type=Path)
    p.add_argument("--host", default="http://127.0.0.1:9380")
    p.add_argument("--llm-model", default="deepseek-r1-distill-qwen-32b")
    p.add_argument("--encoder", default="distiluse-base-multilingual-cased-v1")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--threads", type=int, default=1)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.project = args.project.resolve()
    args.data_path = (args.data_path or args.project / "datasets" / "cache" / "twitter18.json").resolve()
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_name):
        p.error("--run-name allows only letters, digits, _ and -")
    if args.blocks != list(range(args.blocks[0], args.blocks[-1] + 1)) or args.blocks[0] < 1 or args.blocks[-1] > 16:
        p.error("--blocks must be a contiguous range within 1-16")
    if args.threads < 1:
        p.error("--threads must be positive")
    root = args.project / "runs" / args.run_name
    if args.phase == "plan":
        print("Dataset: Twitter18; threshold=0.30 BOTH; anchor preprocessing=RagSEDE BOTH; retry limit=30 BOTH")
        print("Embeddings recomputed in isolated directories for BOTH methods. No old metrics reused.")
        print("Run directory:", root)
        for seed, block, variant in schedule(args.blocks, args.seeds):
            print(f"s{seed} M{block}: {variant}")
        print(f"Planned paid block-runs: {len(schedule(args.blocks, args.seeds))}; anchor-only checks make no LLM calls.")
        return 0
    if args.phase == "status":
        manifest = load(root / "manifest.json")
        print("Anchor alignment:", manifest.get("anchor_alignment_passed", False))
        for seed, block, variant in schedule(args.blocks, args.seeds):
            entry = manifest["jobs"].get(f"s{seed}_M{block}_{variant}", {})
            print(f"s{seed} M{block} {variant}: {entry.get('status', 'pending')}")
        print("Latest report:", manifest.get("latest_report", "not yet available"))
        return 0
    lock_path = args.project / ".twitter18_protocol_runner.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another protocol runner is active; do not launch a duplicate") from exc
        check_for_other_main(args.project)
        manifest = initialize(args, root)
        if args.phase == "summarize":
            summarize(args, root, manifest)
            return 0
        expected = preflight(args, root, manifest)
        if args.phase == "run":
            run_paid(args, root, manifest, expected)
            summarize(args, root, manifest)
        else:
            print("All requested anchor checks passed. Run phase is ready; no LLM experiment was started.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Re-run the same command to resume completed block-runs; interrupted blocks restart from their beginning.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"STOP: {error}", file=sys.stderr)
        print("Existing results were retained. Fix the cause and repeat the same command; do not launch duplicates.", file=sys.stderr)
        sys.exit(1)
