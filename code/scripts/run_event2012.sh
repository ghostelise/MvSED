#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
HOST_ADDRESS=${RAGFLOW_HOST:-http://127.0.0.1:9380}
LLM_MODEL=${LLM_MODEL:-deepseek-r1-distill-qwen-32b}
VARIANT=${1:-stage_aware_selective_gate_time_aux}
SEED=${2:-42}
START_BLOCK=${3:-7}
END_BLOCK=${4:-21}
DATA_PATH=${5:-"$CODE_ROOT/datasets/cache/twitter12.json"}

: "${RAGFLOW_API_KEY:?Set RAGFLOW_API_KEY in the current terminal; do not put it in this script.}"
[[ -s "$DATA_PATH" ]] || { echo "Dataset cache not found: $DATA_PATH" >&2; exit 1; }

cd "$CODE_ROOT"

"$PYTHON_BIN" -u main.py \
  --dataset twitter12 \
  --data_path "$DATA_PATH" \
  --HOST_ADDRESS "$HOST_ADDRESS" \
  --variant "$VARIANT" \
  --anchor_pipeline single_stage \
  --anchor_preprocessing ragsede \
  --twitter12_threshold 0.40 \
  --language English \
  --local_embedding_model all-MiniLM-L6-v2 \
  --embedding_device cuda \
  --seed "$SEED" \
  --start_block "$START_BLOCK" \
  --end_block "$END_BLOCK" \
  --llm_model "$LLM_MODEL" \
  --temperature 0.0 \
  --top_p 0.3 \
  --max_tokens 2048 \
  --max_llm_attempts 30 \
  --llm_retry_delay 15.0 \
  --max_cluster_size 100 \
  --anchor_top_k 3 \
  --anchor_lambda 0.70 \
  --reuse_embedding_cache \
  --save_audit_log

