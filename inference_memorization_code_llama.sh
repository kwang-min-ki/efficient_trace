#!/usr/bin/env bash
# Base/학습 Llama 모델의 code memorization 정확도·counterfactual·TRACE 평가
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN=${PYTHON_BIN:-/venv/verl/bin/python}
BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}
TRAINED_MODEL=${TRAINED_MODEL:-ckpt_hf/Llama-3.2-3B-Instruct/code_memorization_seen}
DATA=${DATA:-data/code/memorization}
OUT=${OUT:-runs/Llama-3.2-3B-Instruct/code_memorization}
BATCH_SIZE=${BATCH_SIZE:-16}
DTYPE=${DTYPE:-bfloat16}
TEMPERATURE=${TEMPERATURE:-0.0}
SEED=${SEED:-0}

if [ ! -x "$PYTHON_BIN" ]; then
    echo "missing evaluation Python at $PYTHON_BIN; run setup.sh first" >&2
    exit 1
fi
if [ ! -f "$DATA/prompts.clean.jsonl" ]; then
    echo "missing $DATA/prompts.clean.jsonl; run data.py for code first" >&2
    exit 1
fi

args=(
    --data "$DATA"
    --base-model "$BASE_MODEL"
    --trained-model "$TRAINED_MODEL"
    --out "$OUT"
    --batch-size "$BATCH_SIZE"
    --dtype "$DTYPE"
    --temperature "$TEMPERATURE"
    --seed "$SEED"
)

# 빠른 검증 예: LIMIT_PER_SPLIT=10 ./inference_memorization_code_llama.sh
if [ -n "${LIMIT_PER_SPLIT:-}" ]; then
    args+=(--limit-per-split "$LIMIT_PER_SPLIT")
fi
if [ -n "${COUNTERFACTUALS:-}" ]; then
    args+=(--counterfactuals "$COUNTERFACTUALS")
fi

exec "$PYTHON_BIN" inference_memorization_code.py "${args[@]}" "$@"
