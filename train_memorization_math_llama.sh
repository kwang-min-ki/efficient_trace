#!/usr/bin/env bash
# Llama-3.2-3B-Instruct를 Big-Math의 학습 노출(seen) clean 문제로 학습
set -euo pipefail
cd "$(dirname "$0")"

export MODEL=${MODEL:-meta-llama/Llama-3.2-3B-Instruct}
export TASK=math
export VARIANT=clean
export DATA=${DATA:-data/math/rl/clean}
export MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL%/}")}
export CKPT=${CKPT:-ckpt/${MODEL_TAG}/math_memorization_seen}
export PYTHON_BIN=${PYTHON_BIN:-/venv/verl/bin/python}

if [ ! -f "$DATA/train.parquet" ]; then
    echo "missing $DATA/train.parquet; run data.py for math first" >&2
    exit 1
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "missing training Python at $PYTHON_BIN; run setup.sh first" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c "import torch, verl" 2>/dev/null; then
    echo "$PYTHON_BIN does not contain torch and verl; rerun setup.sh" >&2
    exit 1
fi

exec ./train.sh "$@"
