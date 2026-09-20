#!/usr/bin/env bash
# 모델별 설정과 math/code RLOO 옵션을 결합한 verl 학습 실행
set -euo pipefail
cd "$(dirname "$0")"

TASK=${TASK:-math}
VARIANT=${VARIANT:-ic_correct}
MODEL=${MODEL:?set MODEL=<path to the base model>}
TOKENIZER=${TOKENIZER:-$MODEL}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-data/$TASK/rl/$VARIANT}
MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL%/}")}
CKPT=${CKPT:-ckpt/${MODEL_TAG}/${TASK}_${VARIANT}}
LOG_DIR=${LOG_DIR:-logs/${MODEL_TAG}}
NGPUS=${NGPUS:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.23}

# Qwen's 32K context needs more KV-cache space than a 0.2 reservation provides.
# Keep this final shared value at or above the known-safe minimum for all launchers.
if awk -v value="$ROLLOUT_GPU_MEMORY_UTILIZATION" 'BEGIN { exit !(value < 0.23) }'; then
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.23
fi

case "$TASK" in
    math|code) ;;
    *) echo "TASK must be math or code" >&2; exit 2 ;;
esac
# 여러 줄 Jinja 템플릿을 eval 없이 전달하기 위해 NUL로 인자 구분
MODEL_SETTINGS_FILE=$(mktemp)
trap 'rm -f "$MODEL_SETTINGS_FILE"' EXIT
"$PYTHON_BIN" model_config.py verl-overrides \
    --model "$MODEL" --tokenizer "$TOKENIZER" --task "$TASK" \
    --training-prefill --format nul > "$MODEL_SETTINGS_FILE"
mapfile -d '' -t MODEL_ARGS < "$MODEL_SETTINGS_FILE"
mkdir -p "$LOG_DIR"

if [ "$TASK" = math ]; then
    USE_FUSED_KERNELS=False
    PPO_MAX_TOKEN_LEN_PER_GPU=8192
    ARGS=(
        data.train_batch_size=1024
        data.max_prompt_length=512
        data.filter_overlong_prompts=True
        data.truncation=error
        actor_rollout_ref.actor.optim.lr=1e-6
        actor_rollout_ref.actor.ppo_mini_batch_size=1024
        actor_rollout_ref.actor.strategy=fsdp2
        actor_rollout_ref.actor.fsdp_config.offload_policy=True
        actor_rollout_ref.ref.strategy=fsdp2
        actor_rollout_ref.rollout.n=5
        algorithm.kl_ctrl.kl_coef=0.001
        trainer.total_epochs=15
    )
else
    # 대조군도 IC/RM 조건의 입력 예산 사용, 기본 IC
    USE_FUSED_KERNELS=False
    PPO_MAX_TOKEN_LEN_PER_GPU=8192
    CODE_SETTING=${CODE_SETTING:-ic}
    if [ "$VARIANT" = rm ]; then CODE_SETTING=rm; fi
    case "$CODE_SETTING" in
        ic) PROMPT_LENGTH=1300; KL_COEF=0.01 ;;
        rm) PROMPT_LENGTH=512; KL_COEF=0.001 ;;
        *) echo "CODE_SETTING must be ic or rm" >&2; exit 2 ;;
    esac
    # 논문의 10000 episodes를 문제 배치 16 기준 625 updates로 해석
    # Math의 15는 데이터셋 epochs로 해석, 논문의 단위 구분은 불명확
    ARGS=(
        data.train_batch_size=16
        data.max_prompt_length="$PROMPT_LENGTH"
        data.filter_overlong_prompts=False
        data.truncation=left
        actor_rollout_ref.actor.optim.lr=1e-4
        actor_rollout_ref.actor.ppo_mini_batch_size=16
        actor_rollout_ref.rollout.n=2
        actor_rollout_ref.model.lora_rank=16
        actor_rollout_ref.model.lora_alpha=32
        actor_rollout_ref.model.target_modules=all-linear
        actor_rollout_ref.model.lora.merge=True
        actor_rollout_ref.actor.strategy=fsdp2
        actor_rollout_ref.actor.fsdp_config.param_offload=False
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
        actor_rollout_ref.ref.fsdp_config.param_offload=False
        algorithm.kl_ctrl.kl_coef="$KL_COEF"
        trainer.total_epochs=10000
        trainer.total_training_steps=625
    )
fi

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    algorithm.use_kl_in_reward=True \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/val.parquet" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels="$USE_FUSED_KERNELS" \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    custom_reward_function.path=reward.py \
    custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_local_dir="$CKPT" \
    trainer.logger=[console] \
    "${ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "$@" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    2>&1 | tee "$LOG_DIR/${TASK}_${VARIANT}.log"

