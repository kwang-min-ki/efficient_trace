#!/usr/bin/env python3
"""Thin verl/GRPO launcher for the AR-LSAT experiment in GRIFT Table 8.

The defaults use the GRIFT-aligned reconstructed split from ``arlsat.py data``
and keep new checkpoints separate from earlier runs. Extra arguments are passed
through as verl overrides.

Published settings: prompt length 3072, one epoch, batch 32, learning rate 1e-6,
eight rollouts per prompt, and KL coefficient 0.01 (GRIFT Table 8, which trained
Qwen3-4B).  Qwen2.5 is supported as well: the response budget and Qwen3's explicit
thinking switch come from model_config, so the same launcher covers both families
(Qwen3 4096, Qwen2.5 1024 -- both measured, see model_config.RESPONSE_BUDGETS).
The launcher caps the run at 30 steps by default so only checkpoints 10/20/30
are emitted; use ``--steps`` to change that explicit user-level constraint.
"""

import argparse
import os
import shlex
import subprocess
import sys

import model_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="path to the base model (Qwen3 or Qwen2.5)")
    ap.add_argument("--data", default="data/ar-lsat/rl/clean",
                    help="dir with train.parquet/val.parquet, from arlsat.py data")
    ap.add_argument("--ckpt", default="ckpt/ar-lsat_qwen3_4b")
    ap.add_argument("--ngpus", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30,
                    help="stop after this many steps; 30 yields checkpoints 10/20/30")
    args, extra = ap.parse_known_args()  # extra: passthrough verl overrides, as in train.sh's "$@"

    os.makedirs("logs", exist_ok=True)

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        # GRIFT Table 8: GRPO, not RLOO (train.sh's math/code default).
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=True",
        f"data.train_files={args.data}/train.parquet",
        f"data.val_files={args.data}/val.parquet",
        "data.train_batch_size=32",
        "data.max_prompt_length=3072",
        "data.filter_overlong_prompts=True",
        f"actor_rollout_ref.model.path={args.model}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.use_fused_kernels=True",
        "actor_rollout_ref.model.fused_kernel_options.impl_backend=triton",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768",
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch=True",
        "actor_rollout_ref.ref.fsdp_config.forward_prefetch=True",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.n=8",  # Table 8: rollout_n=8 (GRPO group size)
        "algorithm.kl_ctrl.kl_coef=0.01",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={args.steps}",
        "custom_reward_function.path=reward.py",
        "custom_reward_function.name=compute_score",
        f"trainer.n_gpus_per_node={args.ngpus}",
        "trainer.nnodes=1",
        "trainer.save_freq=10",
        "trainer.test_freq=10",
        f"trainer.default_local_dir={args.ckpt}",
        "trainer.logger=[console]",
        # Everything model-dependent -- the response budget and (Qwen3 only) the
        # explicit thinking switch -- comes from model_config, exactly as train.sh
        # does for math/code.  Appended after the fixed settings so it wins over
        # them, while an explicit override in `extra` still wins over it.
        *model_config.verl_hydra_overrides(args.model, task="arlsat"),
        *extra,
    ]
    # Stream to the console live AND tee to a log file, matching train.sh's
    # `... 2>&1 | tee "logs/${TASK}_${VARIANT}.log"` -- a plain subprocess.run(cmd)
    # with stdout=PIPE would buffer everything until the (very long) run finishes.
    shell_cmd = shlex.join(cmd) + " 2>&1 | tee logs/ar-lsat_grpo.log"
    print(shell_cmd)
    subprocess.run(shell_cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
