#!/usr/bin/env python3
"""Offline structural and aggregate-value check for the public MvSED package."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"RELEASE CHECK FAILED: {message}")


protocol_path = CODE_ROOT / "configs" / "paper_protocol.json"
summary_path = CODE_ROOT / "results" / "paper_summary.csv"
for required_path in (
    protocol_path,
    summary_path,
    CODE_ROOT / "main.py",
    CODE_ROOT / "data_process.py",
    CODE_ROOT / "scripts" / "run_event2012.sh",
    CODE_ROOT / "scripts" / "run_event2018.sh",
    REPO_ROOT / "CITATION.cff",
    REPO_ROOT / "index.html",
):
    require(required_path.is_file(), f"missing {required_path.relative_to(REPO_ROOT)}")

protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
for filename, expected_hash in protocol["source_sha256"].items():
    actual_hash = sha256(CODE_ROOT / filename)
    require(actual_hash == expected_hash, f"source hash mismatch for {filename}")

with summary_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
require(len(rows) == 12, f"paper_summary.csv has {len(rows)} rows; expected 12")
require(all(None not in row for row in rows), "paper_summary.csv contains an overlong row")
keys = [(row["section"], row["dataset"], row["method"]) for row in rows]
require(len(keys) == len(set(keys)), "paper_summary.csv contains duplicate method rows")

expected = {
    ("main", "Event2012", "MvSED"): ("0.89660", "0.88615", "0.78455", "11.3"),
    ("main", "Event2018", "MvSED"): ("0.76400", "0.75284", "0.66762", "22.3"),
    ("ablation", "Event2012", "MvSED"): ("0.89306", "0.88199", "0.77432", "11.26"),
    ("ablation", "Event2018", "MvSED"): ("0.76400", "0.75284", "0.66762", "22.45"),
}
indexed = {(row["section"], row["dataset"], row["method"]): row for row in rows}
for key, values in expected.items():
    require(key in indexed, f"missing summary row {key}")
    row = indexed[key]
    actual = (row["NMI"], row["AMI"], row["ARI"], row["time_reduction_percent"])
    require(actual == values, f"reported aggregate mismatch for {key}: {actual}")

citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
for author in ("Kunqing Li", "Mengge Ai", "Ruiteng Yan", "Ke Wen", "Enying Bao", "Lei Wang", "Yuejin Guo", "Xi Chen"):
    require(author in page, f"project page is missing author {author}")
require("family-names: Guo" in citation and "given-names: Yuejin" in citation,
        "CITATION.cff is missing Yuejin Guo")
require("Controlled Event2012 study" in page and "Aligned Event2018 study" in page,
        "project page is missing one ablation panel")

secret_patterns = (
    re.compile(r"ragflow-[A-Za-z0-9]{12,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
)
for path in REPO_ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    if path.suffix.lower() not in {".py", ".sh", ".md", ".json", ".csv", ".html", ".js", ".css", ".cff", ".example"}:
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    require(not any(pattern.search(text) for pattern in secret_patterns),
            f"possible live credential in {path.relative_to(REPO_ROOT)}")

print("RELEASE CHECK PASSED")
print(f"- source hashes: 2/2")
print(f"- aggregate rows: {len(rows)}/12")
print("- author metadata: 8/8")
print("- ablation panels: Event2012 and Event2018")
print("- basic credential scan: clear")
print("Conditional reproducibility: datasets, local encoders, RAGFlow, and provider access remain external dependencies.")
