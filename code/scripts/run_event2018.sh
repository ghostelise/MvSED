#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
HOST_ADDRESS=${RAGFLOW_HOST:-http://127.0.0.1:9380}
LLM_MODEL=${LLM_MODEL:-deepseek-r1-distill-qwen-32b}
PHASE=${1:-check}
BLOCKS=${2:-1-16}
SEEDS=${3:-42}
DATA_PATH=${4:-"$CODE_ROOT/datasets/cache/twitter18.json"}
DEVICE=${DEVICE:-cuda}
RUN_NAME=${RUN_NAME:-twitter18_anchor_aligned_release}

case "$PHASE" in
  plan|check|status|summarize) ;;
  run) : "${RAGFLOW_API_KEY:?Set RAGFLOW_API_KEY in the current terminal before a paid run.}" ;;
  *) echo "Usage: $0 {plan|check|run|status|summarize} [blocks] [seeds] [data_path]" >&2; exit 2 ;;
esac

[[ -s "$DATA_PATH" ]] || { echo "Dataset cache not found: $DATA_PATH" >&2; exit 1; }

cd "$CODE_ROOT"

"$PYTHON_BIN" rerun_twitter18.py "$PHASE" \
  --project "$CODE_ROOT" \
  --run-name "$RUN_NAME" \
  --blocks "$BLOCKS" \
  --seeds "$SEEDS" \
  --data-path "$DATA_PATH" \
  --host "$HOST_ADDRESS" \
  --llm-model "$LLM_MODEL" \
  --device "$DEVICE"

