"""TRACE 공통 추론 절단 비율, 답 생성·중단 규칙과 AUC 계산"""

import os

FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# reward.proxy/expected for code block on run_tests's subprocess.run: the thread only
# waits (GIL released), but the actual work is a child process that needs a physical
# core, so parallelism is capped by core count -- not ThreadPoolExecutor's default
# min(32, cpu_count+4), which is sized for pure I/O waits with no CPU-bound work behind
# them.
REWARD_WORKERS = os.cpu_count() or 4


FORCE = {
    "math": "</think>\n<answer>",
    "code": "</think>\n```python\n",
}


REOPEN = {"math": "<answer>", "code": "```python\n"}


STOP = {"math": ["</answer>"], "code": ["```"]}


# Protocol constants (Sec. 4.1). `max_response` is deliberately absent: the rollout
# supported tasks and rollout budgets are defined centrally in
# `model_config.ModelProfile.max_response_tokens`.
TASK_CFG = {
    "math": dict(n_samples=5, temp=0.7, ans_tokens=32),
    "code": dict(n_samples=1, temp=0.0, ans_tokens=600),
}


# Temperature for the single source rollout whose CoT is then truncated, for both
# tasks. NOT from the paper: Sec. 4.1 only says responses are collected and those with
# reward 1 kept, and the explicit temperatures in footnote 1 (math 0.7 / code 0.0, see
# TASK_CFG above) are for the forced-answer sampling *at each cutoff*, not for this
# rollout. 0.7 here is this repo's choice. It makes the kept population differ between
# runs, so compare aggregations via likelihood_trace.py --records rather than by
# re-generating.
ROLLOUT_TEMPERATURE = 0.7


def auc(values):
    area = sum((FRACS[i + 1] - FRACS[i]) * (values[i] + values[i + 1]) / 2
               for i in range(len(FRACS) - 1))
    return 100.0 * area / (FRACS[-1] - FRACS[0])

