#!/usr/bin/env bash
# Qwen2.5-7B-Instruct를 Big-Math의 학습 노출(seen) clean 문제로 전체 파라미터 학습
set -euo pipefail
cd "$(dirname "$0")"

export MODEL=${MODEL:-/workspace/models/Qwen2.5-7B-Instruct}
export TASK=math
export VARIANT=clean
export DATA=${DATA:-data/math/rl/clean}
export MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL%/}")}
export CKPT=${CKPT:-ckpt/${MODEL_TAG}/math_memorization_seen}
export PYTHON_BIN=${PYTHON_BIN:-/venv/verl/bin/python}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.23}
export NGPUS=${NGPUS:-2}

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

exec ./train.sh \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    "$@"
