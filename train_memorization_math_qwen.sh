#!/usr/bin/env bash
# Qwen2.5-7B-Instruct를 Big-Math의 학습 노출(seen) clean 문제로 LoRA 학습
# 단일 H100에서 처리량을 올리기 위해 오프로드·eager·중간 검증을 끈다
set -euo pipefail
cd "$(dirname "$0")"

export MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
export TASK=math
export VARIANT=clean
export DATA=${DATA:-data/math/rl/clean}
export MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL%/}")}
export CKPT=${CKPT:-ckpt/${MODEL_TAG}/math_memorization_seen}
export PYTHON_BIN=${PYTHON_BIN:-/venv/verl/bin/python}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}

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
    actor_rollout_ref.model.lora_rank="$LORA_RANK" \
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.lora.merge=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=50 \
    "$@"
