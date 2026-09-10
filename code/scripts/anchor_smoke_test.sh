#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
DATA_PATH=${1:-"$CODE_ROOT/datasets/cache/twitter18.json"}

cd "$CODE_ROOT"

"$PYTHON_BIN" -u main.py \
  --dataset twitter18 \
  --data_path "$DATA_PATH" \
  --variant stage_aware_selective_gate_time_aux \
  --anchor_pipeline single_stage \
  --anchor_preprocessing ragsede \
  --twitter18_threshold 0.30 \
  --ragsede_twitter18_threshold 0.30 \
  --language French \
  --local_embedding_model distiluse-base-multilingual-cased-v1 \
  --embedding_device cpu \
  --seed 42 \
  --start_block 1 \
  --end_block 1 \
  --anchor_only \
  --save_audit_log

