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

case "$TASK" in
    math|code) ;;
    *) echo "TASK must be math or code" >&2; exit 2 ;;
esac
# NUL-delimited arguments preserve multiline native Jinja templates without eval.
MODEL_SETTINGS_FILE=$(mktemp)
trap 'rm -f "$MODEL_SETTINGS_FILE"' EXIT
"$PYTHON_BIN" model_config.py verl-overrides \
    --model "$MODEL" --tokenizer "$TOKENIZER" --task "$TASK" \
    --training-prefill --format nul > "$MODEL_SETTINGS_FILE"
mapfile -d '' -t MODEL_ARGS < "$MODEL_SETTINGS_FILE"
mkdir -p "$LOG_DIR"

if [ "$TASK" = math ]; then
    ARGS=(
        data.train_batch_size=1024
        data.max_prompt_length=512
        data.filter_overlong_prompts=True
        data.truncation=error
        actor_rollout_ref.actor.optim.lr=1e-6
        actor_rollout_ref.actor.ppo_mini_batch_size=1024
        actor_rollout_ref.rollout.n=5
        algorithm.kl_ctrl.kl_coef=0.001
        trainer.total_epochs=15
    )
else
    # Clean controls match the corresponding IC/RM prompt budget. Default: IC.
    CODE_SETTING=${CODE_SETTING:-ic}
    if [ "$VARIANT" = rm ]; then CODE_SETTING=rm; fi
    case "$CODE_SETTING" in
        ic) PROMPT_LENGTH=1300; KL_COEF=0.01 ;;
        rm) PROMPT_LENGTH=512; KL_COEF=0.001 ;;
        *) echo "CODE_SETTING must be ic or rm" >&2; exit 2 ;;
    esac
    # Table 2 says 10,000 total episodes, not 10,000 epochs. Interpret as
    # prompt episodes: 10,000 / effective prompt batch 16 = 625 updates.
    # Table 1's 15 is interpreted as dataset epochs. These units are not
    # explicitly disambiguated in the paper; see README reproduction notes.
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
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.name=vllm \
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
    "$@" 2>&1 | tee "$LOG_DIR/${TASK}_${VARIANT}.log"

