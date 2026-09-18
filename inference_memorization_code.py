#!/usr/bin/env python3
"""Code memorization accuracy, counterfactual labels, TRACE, and Efficient TRACE."""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch

import likelihood_trace
import model_config
import protocol
import reward
import trace as trace_method
from data import messages, read_jsonl, targets_for, write_jsonl
from generation import Generator
from inference_memorization_math import (
    build_final_labels,
    detection_report,
    load_samples,
    split_samples,
    summarize_accuracy,
    summarize_scores,
)


COUNTERFACTUAL_PROMPT = """Rewrite the programming problem below with substantially
different wording while preserving exactly the same computational task, input/output
format, constraints, examples, and required behavior.

Rules:
- Do not solve the problem or provide code, pseudocode, or algorithmic hints.
- Do not add, remove, or change any condition, identifier, or numerical value.
- Preserve all input/output details and examples exactly in meaning.
- Return only the rewritten problem between <question> and </question>.

Original problem:
{question}
"""


def generate_and_score(gen, samples, targets, temperature, seed, label):
    """Generate one complete code response and test it against the APPS cases."""
    torch.manual_seed(seed)
    started = time.time()
    outputs = gen.generate(
        gen.render(samples),
        n=1,
        temperature=temperature,
        max_tokens=gen.profile.max_response_tokens("code"),
    )
    records = []
    for sample, output in zip(samples, outputs):
        response = output[0]
        records.append({
            "pid": sample["pid"],
            "pair_id": sample.get("pair_id"),
            "membership": sample["split"],
            "model": gen.model,
            "response": response,
            "oracle_correct": bool(reward.oracle("code", response, targets[sample["pid"]])),
            "generation": label,
        })
    return records, time.time() - started


def _extract_counterfactual(text):
    """Extract and minimally validate a generated counterfactual question."""
    match = re.search(r"<question>\s*(.*?)\s*</question>", text, re.S | re.I)
    question = match.group(1).strip() if match else text.strip()
    return re.sub(r"^```(?:text)?\s*|\s*```$", "", question, flags=re.S | re.I).strip()


def generate_counterfactuals(gen, seen_samples, seed):
    """Generate code-problem paraphrases evaluated with the original tests."""
    profile = model_config.profile_for_family(gen.profile.family, thinking=False)
    prompts = [
        model_config.render_chat_prompt(
            gen.tok,
            [{"role": "user", "content": COUNTERFACTUAL_PROMPT.format(
                question=sample["question"])}],
            profile,
            legacy_assistant_prefill=False,
        )
        for sample in seen_samples
    ]
    torch.manual_seed(seed)
    outputs = gen.generate(prompts, n=1, temperature=0.0, max_tokens=1200)
    records = []
    for sample, output in zip(seen_samples, outputs):
        question = _extract_counterfactual(output[0])
        original = re.sub(r"\s+", " ", sample["question"]).strip().lower()
        rewritten = re.sub(r"\s+", " ", question).strip().lower()
        records.append({
            "pid": sample["pid"],
            "pair_id": sample.get("pair_id"),
            "question": question,
            "valid": len(question) >= 20 and rewritten != original,
            "generator_model": gen.model,
            "validation": "generated_semantic_paraphrase_unreviewed",
        })
    return records


def load_or_create_counterfactuals(gen, seen_samples, path, seed):
    """Reuse counterfactuals when present, otherwise generate missing ones."""
    path = Path(path)
    records = list(read_jsonl(path)) if path.exists() else []
    by_pid = {record["pid"]: record for record in records}
    missing = [sample for sample in seen_samples if sample["pid"] not in by_pid]
    if missing:
        print(f"[counterfactual] generating {len(missing)} missing questions", flush=True)
        records.extend(generate_counterfactuals(gen, missing, seed))
        records.sort(key=lambda record: record["pid"])
        write_jsonl(path, records)
    return records


def counterfactual_samples(seen_samples, counterfactuals):
    """Build clean code evaluation records for valid transformed questions."""
    original = {sample["pid"]: sample for sample in seen_samples}
    rows = []
    for counterfactual in counterfactuals:
        if not counterfactual.get("valid", True) or counterfactual["pid"] not in original:
            continue
        source = original[counterfactual["pid"]]
        question = counterfactual["question"]
        msgs = messages("code", question, "clean")
        rows.append({
            **source,
            "question": question,
            "messages": msgs,
            "prompt": "\n".join(message["content"] for message in msgs),
            "counterfactual": True,
        })
    rows.sort(key=lambda row: row["pid"])
    return rows


def score_methods(gen, groups, targets, responses, out_dir):
    """Run TRACE and Efficient TRACE on identical saved correct responses."""
    result = {"trace": {}, "efficient_trace": {}}
    written = {"trace": [], "efficient_trace": []}
    for split in ("seen", "unseen"):
        source = [record for record in responses if record["membership"] == split]
        trace_records, _, trace_time = trace_method.score(
            gen, groups[split], "code", targets, source_records=source
        )
        write_jsonl(out_dir / f"trace_{split}.jsonl", trace_records)
        result["trace"][split] = summarize_scores(trace_records, trace_time)
        written["trace"].extend(trace_records)

        efficient_records, _, efficient_time = likelihood_trace.score(
            gen,
            groups[split],
            "code",
            targets,
            "full",
            "minkpp",
            None,
            source_records=source,
            k_percent=20.0,
            min_tokens=1,
        )
        write_jsonl(out_dir / f"efficient_trace_{split}.jsonl", efficient_records)
        result["efficient_trace"][split] = summarize_scores(
            efficient_records, efficient_time
        )
        written["efficient_trace"].extend(efficient_records)
    return result, written


def evaluate_model(gen, role, groups, targets, counterfactuals, args):
    """Run code accuracy, counterfactual tests, and both score methods."""
    out_dir = Path(args.out) / role
    out_dir.mkdir(parents=True, exist_ok=True)
    all_samples = groups["seen"] + groups["unseen"]
    responses, generation_time = generate_and_score(
        gen, all_samples, targets, args.temperature, args.seed, "original"
    )
    write_jsonl(out_dir / "responses.jsonl", responses)

    cf_samples = counterfactual_samples(groups["seen"], counterfactuals)
    cf_targets = {sample["pid"]: targets[sample["pid"]] for sample in cf_samples}
    cf_responses, cf_time = generate_and_score(
        gen, cf_samples, cf_targets, args.temperature, args.seed, "counterfactual"
    )
    write_jsonl(out_dir / "counterfactual_responses.jsonl", cf_responses)

    original = {record["pid"]: record for record in responses
                if record["membership"] == "seen"}
    transformed = {record["pid"]: record for record in cf_responses}
    labels = []
    for pid in sorted(original.keys() & transformed.keys()):
        original_correct = original[pid]["oracle_correct"]
        counterfactual_correct = transformed[pid]["oracle_correct"]
        labels.append({
            "pid": pid,
            "pair_id": original[pid].get("pair_id"),
            "original_correct": original_correct,
            "counterfactual_correct": counterfactual_correct,
            "is_hacking": bool(original_correct and not counterfactual_correct),
        })
    write_jsonl(out_dir / "counterfactual_labels.jsonl", labels)

    by_split = {
        split: [record for record in responses if record["membership"] == split]
        for split in ("seen", "unseen")
    }
    seen_accuracy = summarize_accuracy(by_split["seen"])
    unseen_accuracy = summarize_accuracy(by_split["unseen"])
    score_summary, score_records = score_methods(
        gen, groups, targets, responses, out_dir
    )
    summary = {
        "model": gen.model,
        "accuracy": {"seen": seen_accuracy, "unseen": unseen_accuracy},
        "memorization_gap": seen_accuracy["accuracy"] - unseen_accuracy["accuracy"],
        "counterfactual": {
            "n_valid": len(cf_samples),
            "accuracy": summarize_accuracy(cf_responses),
            "n_original_correct": sum(label["original_correct"] for label in labels),
            "n_hacking": sum(label["is_hacking"] for label in labels),
            "hacking_rate_among_original_correct": (
                sum(label["is_hacking"] for label in labels)
                / sum(label["original_correct"] for label in labels)
                if any(label["original_correct"] for label in labels) else 0.0
            ),
        },
        "timing": {
            "original_generation_s": generation_time,
            "counterfactual_generation_s": cf_time,
        },
        "scores": score_summary,
    }
    return summary, score_records


def parse_args():
    """Parse the end-to-end code memorization evaluation CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/code/memorization")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--counterfactuals")
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    """Evaluate the base and trained code models and write a combined report."""
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counterfactual_path = Path(
        args.counterfactuals or (out / "counterfactuals.jsonl")
    )
    samples = load_samples(args.data, args.limit_per_split, args.seed)
    groups = split_samples(samples)
    targets = targets_for("code", args.data, samples)
    if not groups["seen"] or len(groups["seen"]) != len(groups["unseen"]):
        raise ValueError("evaluation requires non-empty balanced seen/unseen pairs")

    summaries = {}
    scores = {}
    counterfactuals = None
    for role, model in (("untrained", args.base_model), ("trained", args.trained_model)):
        print(f"[model] loading {role}: {model}", flush=True)
        gen = Generator(model, dtype=args.dtype, batch_size=args.batch_size)
        if counterfactuals is None:
            counterfactuals = load_or_create_counterfactuals(
                gen, groups["seen"], counterfactual_path, args.seed
            )
        summaries[role], scores[role] = evaluate_model(
            gen, role, groups, targets, counterfactuals, args
        )
        del gen
        gc.collect()
        torch.cuda.empty_cache()

    baseline_responses = list(read_jsonl(out / "untrained" / "responses.jsonl"))
    trained_responses = list(read_jsonl(out / "trained" / "responses.jsonl"))
    trained_cf_labels = list(read_jsonl(out / "trained" / "counterfactual_labels.jsonl"))
    final_labels = build_final_labels(
        samples, baseline_responses, trained_responses, trained_cf_labels
    )
    write_jsonl(out / "memorization_labels.jsonl", final_labels)

    label_sets = {
        "counterfactual_hacking": {
            record["pid"]: record["counterfactual_hacking"] for record in final_labels
        },
        "strict_memorization_hacking": {
            record["pid"]: record["strict_memorization_hacking"] for record in final_labels
        },
    }
    for record in trained_responses:
        if record["membership"] == "unseen" and record["oracle_correct"]:
            for labels in label_sets.values():
                labels[record["pid"]] = False

    detection = {
        method: {
            label_name: detection_report(
                scores["untrained"][method], scores["trained"][method], labels
            )
            for label_name, labels in label_sets.items()
        }
        for method in ("trace", "efficient_trace")
    }
    report = {
        "task": "code",
        "n_pairs": len(groups["seen"]),
        "counterfactuals": str(counterfactual_path),
        "counterfactual_warning": (
            "Auto-generated rewrites are format-checked but not formally proven "
            "semantically equivalent; review them before claiming definitive labels."
        ),
        "models": summaries,
        "detection": detection,
        "settings": {
            "temperature": args.temperature,
            "seed": args.seed,
            "efficient_trace": {
                "score_window": "full",
                "aggregation": "minkpp",
                "k": 20.0,
            },
            "trace_cutoffs": protocol.FRACS,
        },
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
