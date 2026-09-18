#!/usr/bin/env bash
# Qwen2.5-3B-Instruct를 APPS의 학습 노출(seen) clean 문제로 학습
set -euo pipefail
cd "$(dirname "$0")"

export MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
export TASK=code
export VARIANT=clean
export DATA=${DATA:-data/code/rl/clean}
export MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL%/}")}
export CKPT=${CKPT:-ckpt/${MODEL_TAG}/code_memorization_seen}

if [ ! -f "$DATA/train.parquet" ]; then
    echo "missing $DATA/train.parquet; run data.py for code first" >&2
    exit 1
fi

exec ./train.sh "$@"
