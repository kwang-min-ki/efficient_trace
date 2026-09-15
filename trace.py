"""추론 절단점별 답 재생성·보상 채점과 TRACE 결과 저장"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import torch

import protocol
import reward
from data import read_jsonl, targets_for, write_jsonl
from generation import Generator, rollout_and_filter, trace_curve_cached


# ---------------------------------------------------------------------------
# Math/code autoregressive TRACE orchestration and CLI.
# ---------------------------------------------------------------------------

def score(gen, samples, task, targets, source_records=None):
    config = protocol.TASK_CFG[task]
    n_samples, temp, ans_tokens = (
        config["n_samples"], config["temp"], config["ans_tokens"]
    )

    t0 = time.time()
    kept = rollout_and_filter(gen, samples, task, targets,
                              temperature=protocol.ROLLOUT_TEMPERATURE,
                              source_records=source_records)
    rollout_time = time.time() - t0

    force_ids = gen.tok.encode(protocol.FORCE[task], add_special_tokens=False)

    t1 = time.time()
    records = []
    for n, k in enumerate(kept, 1):
        s = k["sample"]
        # prefix_before_cot, not the dataset prompt: it carries the rendered chat
        # template and any generated <think>, so cot_ids is reasoning only.
        prompt_ids = gen.tok.encode(k["prefix_before_cot"], add_special_tokens=False)
        cot_ids = gen.tok.encode(k["cot"], add_special_tokens=False)
        per_cutoff_outs = trace_curve_cached(
            gen.tok,
            gen.net,
            prompt_ids,
            cot_ids,
            force_ids,
            protocol.FRACS,
            n_samples,
            temp,
            ans_tokens,
            protocol.STOP[task],
            sampling=gen.profile.sampling,
        )
        # per_cutoff_outs' decoding is already batched on GPU; scoring each cutoff still
        # calls reward.expected, which for code blocks on run_tests's subprocess.run --
        # parallelize those len(FRACS) independent, GIL-releasing calls across threads.
        def _score_cutoff(item):
            j, outs = item
            texts = [protocol.REOPEN[task] + o for o in outs]
            return j, reward.expected(task, texts, targets[s["pid"]], s["loophole"])

        curve = [0.0] * len(protocol.FRACS)
        with ThreadPoolExecutor(max_workers=protocol.REWARD_WORKERS) as pool:
            for j, val in pool.map(_score_cutoff, enumerate(per_cutoff_outs)):
                curve[j] = val
        records.append({
            "pid": s["pid"],
            "task": task,
            "variant": s["variant"],
            "model": gen.model,
            "curve": curve,
            "auc": protocol.auc(curve),
            "response": k["response"],
            "impl": "trace_hf",
            "score_window": None,
            "response_source": "records" if source_records is not None else "generated",
        })
        if n % 50 == 0 or n == len(kept):
            print(f"[score] {n}/{len(kept)} ({time.time() - t1:.0f}s elapsed)", flush=True)
    scoring_time = time.time() - t1
    return records, rollout_time, scoring_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["math", "code"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", default="ic_correct")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val",
                    help="comma-separated split names, e.g. train,val,heldout")
    ap.add_argument("--records",
                    help="JSONL of previously written records (any file with 'pid' and "
                         "'response') whose responses are scored verbatim instead of "
                         "generating new rollouts -- e.g. a likelihood_trace.py or "
                         "trace.py output from the same --model/--variant/--split. "
                         "Skips the sampled rollout (92-98%% of a run's wall clock) so "
                         "reruns isolate TRACE's own cutoff decoding/scoring time. Pass "
                         "the same --split/--limit the records were produced with; "
                         "unmatched pids are reported as missing_record. Omit to "
                         "generate rollouts.")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=16, help="rollout-generation batch size only")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tokenizer",
                    help="override if --model's checkpoint wasn't pushed with its own "
                         "tokenizer files, e.g. --tokenizer /workspace/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch random seed; identical responses across runs are not guaranteed; "
                         "use --records for paired scoring")
    args = ap.parse_args()

    splits = set(args.split.split(","))
    samples = [r for r in read_jsonl(f"{args.data}/prompts.{args.variant}.jsonl")
               if r["split"] in splits]
    samples.sort(key=lambda r: r["pid"])
    if args.limit:
        samples = samples[:args.limit]

    targets = targets_for(args.task, args.data, samples)

    source_records = list(read_jsonl(args.records)) if args.records else None

    torch.manual_seed(args.seed)
    gen = Generator(
        args.model,
        dtype=args.dtype,
        batch_size=args.batch_size,
        tokenizer=args.tokenizer,
    )

    records, rollout_time, scoring_time = score(
        gen, samples, args.task, targets, source_records=source_records)

    write_jsonl(args.out, records)
    mean = sum(r["auc"] for r in records) / len(records) if records else 0.0
    print(f"{len(records)} scored, mean TRACE score {mean:.1f}, "
          f"rollout {rollout_time:.1f}s + scoring {scoring_time:.1f}s")
    stats = {
        "n": len(records),
        "mean_auc": mean,
        "wall_clock_s": rollout_time + scoring_time,
        "rollout_time_s": rollout_time,
        "scoring_time_s": scoring_time,
        "impl": "trace_hf",
        "response_source": "records" if source_records is not None else "generated",
    }
    with open(args.out + ".stats", "w") as handle:
        json.dump(stats, handle)


if __name__ == "__main__":
    main()
