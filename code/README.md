# MvSED research code

This directory contains the research source released with **MvSED: Latency-Aware Selective Multi-View Retrieval for Unsupervised Social Event Clustering**. The authoritative Event2018 entry point is `rerun_twitter18.py`, which checks ordered anchor membership and representative-message hashes before any paid LLM run.

## What is included

- `main.py`: MvSED, RagSEDE control, and aligned ablation variants.
- `data_process.py`: fixed single-stage anchor construction and anchor audits.
- `rerun_twitter18.py`: isolated, resumable Event2018 paired protocol.
- `summarize_efficiency.py`: paired quality, latency, call, failure, token, and resource summaries.
- `scripts/`: safe paper-protocol wrappers with all non-default parameters explicit.
- `configs/paper_protocol.json`: machine-readable datasets, blocks, variants, seeds, timing scopes, and audit boundary.
- `results/paper_summary.csv`: aggregate values shown in the manuscript/project page.
- `scripts/release_check.py`: dataset-free consistency check for the public package and reported aggregates.
- `evolution/`: inherited structural-entropy preprocessing utilities retained for provenance.
- `SOURCE_PROVENANCE.md`: frozen-source hashes and the historical runner boundary.

Datasets, model weights, logs, checkpoints, raw prompts/responses, and message-level predictions are not included.

## Installation

The experiments were run on Linux with Python 3.10, CUDA, one RTX 4090, a local RAGFlow deployment, and a RAGFlow-configured `deepseek-r1-distill-qwen-32b` endpoint.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install `requirements-evolution.txt` only when using the optional inherited preprocessing/evaluation utilities.

## RAGFlow and credentials

1. Start RAGFlow and configure the LLM provider/model in the RAGFlow interface.
2. Confirm that the configured model can answer a test request.
3. Export the **local RAGFlow API key** only in the current shell:

```bash
export RAGFLOW_API_KEY='<your-local-ragflow-key>'
export RAGFLOW_HOST='http://127.0.0.1:9380'
```

`RAGFLOW_API_KEY` is not the upstream provider key. Provider credentials belong inside the RAGFlow model-provider configuration. Never place either key in source files, commands saved to logs, or Git history.

## Data preparation

Follow [`datasets/README.md`](datasets/README.md). The final cache paths are:

```text
datasets/cache/twitter12.json
datasets/cache/twitter18.json
```

## Free anchor-only check

This check constructs one Event2018 block and writes an audit without contacting RAGFlow or an LLM:

```bash
bash scripts/anchor_smoke_test.sh
```

The dataset and local sentence encoder are required for that smoke test. A dataset-free package check is also available:

```bash
python scripts/release_check.py
```

## Event2018 aligned reproduction

Use one unchanged `RUN_NAME`, block range, seed list, and device across all phases. `check` is free and must pass before `run`.

```bash
export RUN_NAME=twitter18_anchor_aligned_release
bash scripts/run_event2018.sh plan  1-16 42
bash scripts/run_event2018.sh check 1-16 42
export RAGFLOW_API_KEY='<your-local-ragflow-key>'
bash scripts/run_event2018.sh run   1-16 42
bash scripts/run_event2018.sh status 1-16 42
bash scripts/run_event2018.sh summarize 1-16 42
```

For a low-cost smoke test, replace `1-16` with `1`. Paid results remain deployment-dependent because the LLM is remotely served.

## Event2012 runs

The wrapper requires an explicit variant and keeps the paper settings visible. Run paired methods sequentially under the same deployment conditions:

```bash
export RAGFLOW_API_KEY='<your-local-ragflow-key>'
bash scripts/run_event2012.sh ragsede_original 42 7 21
bash scripts/run_event2012.sh stage_aware_selective_gate_time_aux 42 7 21
```

Repeat the quality runs with seeds `52` and `62` when reproducing the paper's three-seed Event2012 quality aggregate. The current manuscript reports paired efficiency from the controlled seed-42 runs.

For the Event2012 ablation, keep the same block range and run `ragsede_original`, `stage_aware_selective_mvra`, `stage_aware_selective_gate_no_time`, and `stage_aware_selective_gate_time_aux` sequentially under unchanged provider conditions. The published aggregate covers M7--M21. Legacy Event2012 artifacts verify equal blockwise anchor counts but do not contain all ordered hashes, so this release does not claim bitwise anchor verification for that panel.

`frozen_final_eval.py` is retained only as provenance for an earlier Event2012 evaluation snapshot. Its checksum guard targets a different historical `main.py`, so it is deliberately **not** the public reproduction entry point for this source release.

## Important protocol details

Do not rely on `main.py` defaults for the paper experiment. The release scripts explicitly set:

- final method: `stage_aware_selective_gate_time_aux`;
- one fixed anchor stage and RagSEDE-compatible anchor preprocessing;
- anchor thresholds `0.40` (Event2012) and `0.30` (Event2018);
- `2048` generated tokens, `30` attempts, and `15` seconds between retries;
- three representatives and maximum anchor size `100`.

The Event2018 paired runner uses isolated working directories, alternates method order across blocks, recomputes embeddings for both methods, and refuses mismatched ordered-anchor audits.

## Outputs and reporting

Primary outputs are written under `ckpts/<dataset>/M<block>/`. The experiment runners additionally keep manifests and reports under `runs/`. Both locations are ignored by Git because audit logs can include source messages and provider output.

When reporting results, distinguish wall-clock latency, LLM calls, failed attempts, and token usage. Lower elapsed time does not by itself establish lower monetary cost or energy use. See [`results/README.md`](results/README.md) for the release boundary.

## Acknowledgment and release scope

MvSED builds on the fixed-anchor detector and evolving memory protocol of RagSEDE (Liu et al., *The Web Conference*, 2026). Please cite both works when using the inherited pipeline. Dataset use remains subject to the original Event2012 and Event2018 terms.

No software license is declared in this draft package. The authors should select and add a license before inviting third-party reuse.
