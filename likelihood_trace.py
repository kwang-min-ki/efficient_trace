"""추론 절단점별 원래 답의 토큰 확률 계산, 집계와 Likelihood-TRACE 결과 저장"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass

import torch

import protocol
from data import read_jsonl, targets_for, write_jsonl
from protocol import FRACS, FORCE, auc
from generation import (
    Generator,
    _build_prefix_cache,
    _cutoff_attention_mask,
    _longest_common_prefix_len,
    clone_cache,
    cutoff_tok_len,
    rollout_and_filter,
)



@dataclass(frozen=True)
class TokenStats:
    """답 토큰의 logp와 다음 토큰 분포의 통계

    mu·sigma는 어휘 확률로 가중한 logp의 평균·표준편차, top1은 최대 logp
    """

    logp: float
    mu: float
    sigma: float
    top1: float


def _per_token(logprobs_2d, score_tensor, want_stats):
    """[토큰 수, 어휘 수] logp에서 답 토큰별 값 추출

    want_stats이면 TokenStats, 아니면 logp 목록 반환
    """

    idx = torch.arange(logprobs_2d.shape[0], device=logprobs_2d.device)
    target = logprobs_2d[idx, score_tensor]
    if not want_stats:
        return target.tolist()
    probs = logprobs_2d.exp()
    mu = (probs * logprobs_2d).sum(-1)
    # 부동소수점 오차로 분산이 음수가 되는 경우만 0으로 보정
    sigma = ((probs * logprobs_2d.square()).sum(-1) - mu.square()).clamp_min(0).sqrt()
    top1 = logprobs_2d.max(-1).values
    return [TokenStats(lp, m, s, t) for lp, m, s, t
            in zip(target.tolist(), mu.tolist(), sigma.tolist(), top1.tolist())]


@torch.inference_mode()
def likelihood_curve_cached(tok, net, prompt_ids, cot_ids, force_text, answer_text, fracs,
                            cot_cutoff_lens=None, want_stats=False):
    """접두사 캐시를 공유해 절단점별 기존 답의 토큰 점수 계산

    한 번의 캐시 계산 후 절단점들을 한 배치로 평가
    마스크는 절단점 이후 캐시를 숨기고 position_ids는 답의 실제 위치 지정
    경계의 토큰 병합을 유지하려고 force_text와 answer_text를 함께 토큰화
    공통 토큰 접두사 이후를 채점, fracs 순서로 토큰별 점수 목록 반환
    cot_cutoff_lens 지정 시 비율 대신 해당 토큰 위치 사용
    """
    force_ids = tok.encode(force_text, add_special_tokens=False)
    combined_ids = tok.encode(force_text + answer_text, add_special_tokens=False)
    common = _longest_common_prefix_len(force_ids, combined_ids)
    if common >= len(combined_ids):  # 채점할 답 토큰 없음
        return [[] for _ in fracs]

    full_ids = prompt_ids + cot_ids
    L = len(full_ids)
    n_cut = len(fracs)
    M = len(combined_ids)
    device = net.device

    if cot_cutoff_lens is None:
        cot_cutoff_lens = [cutoff_tok_len(len(cot_ids), f) for f in fracs]
    else:
        if len(cot_cutoff_lens) != n_cut:
            raise ValueError("cot_cutoff_lens and fracs must have the same length")
        if any(length < 0 or length > len(cot_ids) for length in cot_cutoff_lens):
            raise ValueError("cot_cutoff_lens must lie within cot_ids")

    cache = _build_prefix_cache(net, full_ids)
    branch = clone_cache(cache)
    branch.batch_repeat_interleave(n_cut)

    target_lens = [len(prompt_ids) + length for length in cot_cutoff_lens]
    attn_mask = _cutoff_attention_mask(target_lens, L, M, device)
    input_ids = torch.tensor([combined_ids] * n_cut, device=device)
    position_ids = torch.stack(
        [torch.arange(tl, tl + M, device=device) for tl in target_lens], dim=0)

    logits = net(input_ids=input_ids, past_key_values=branch, attention_mask=attn_mask,
                 position_ids=position_ids, use_cache=True).logits
    # logits[:, k]는 다음 토큰을 예측하므로 답 구간을 한 칸 앞에서 선택
    answer_logits = logits[:, common - 1:M - 1, :]
    logprobs = torch.log_softmax(answer_logits.float(), dim=-1)
    score_tensor = torch.tensor(combined_ids[common:], device=device)
    return [_per_token(logprobs[j], score_tensor, want_stats) for j in range(n_cut)]


def parse_score_window(spec):
    """full 또는 first_<k>를 답 토큰 선택 함수로 변환"""
    if spec == "full":
        return lambda lps: lps
    if spec.startswith("first_"):
        k = int(spec[len("first_"):])
        if k < 1:
            raise ValueError(f"--score-window first_<k> needs k >= 1, got {spec!r}")
        return lambda lps: lps[:k]
    raise ValueError(f"bad --score-window {spec!r}, expected 'full' or 'first_<k>'")


def likelihood_score(logprobs):
    """전체 답 토큰 확률의 기하평균 exp(mean(logp)) 계산"""
    if not logprobs:
        return 0.0
    return math.exp(sum(logprobs) / len(logprobs))


def likelihood_score_max(logprobs):
    """답 토큰 확률 중 최댓값 exp(max(logp)) 계산"""
    if not logprobs:
        return 0.0
    return math.exp(max(logprobs))


def likelihood_score_hybrid(logprobs, threshold):
    """최대 토큰 확률이 threshold를 넘으면 최댓값, 아니면 기하평균 반환"""
    if not logprobs:
        return 0.0
    max_score = likelihood_score_max(logprobs)
    return max_score if max_score > threshold else likelihood_score(logprobs)


STATS_AGGREGATIONS = ("minkpp", "gapk", "minkpp_exp", "gapk_exp")


def _bottom_k(values, k_percent, min_tokens=1):
    """하위 k% 선택, 최소 min_tokens개와 전체 길이 한도 적용"""
    if not values:
        return []
    count = max(min_tokens, int(len(values) * k_percent / 100.0))
    count = min(count, len(values))
    return sorted(values)[:count]


def likelihood_score_mink(logprobs, k_percent, min_tokens=1):
    """하위 k% logp 평균의 지수값 계산

    Min-K% (Shi et al., ICLR 2024) 기반, exp(mean(bottom-k logp)) 사용
    """
    if not logprobs:
        return 0.0
    picked = _bottom_k(logprobs, k_percent, min_tokens)
    return math.exp(sum(picked) / len(picked))


def likelihood_score_minkpp(stats, k_percent, min_tokens=1):
    """Min-K%++의 하위 k% 정규화 점수 평균 계산

    Zhang et al., ICLR 2025, 식 3–4의 정규화·선택 방식
    z = (logp - mu) / sigma, sigma가 0이면 z=0 처리
    반환값은 확률이 아닌 원시 점수
    """
    if not stats:
        return 0.0
    z = [(s.logp - s.mu) / s.sigma if s.sigma > 0 else 0.0 for s in stats]
    picked = _bottom_k(z, k_percent, min_tokens)
    return sum(picked) / len(picked)


def likelihood_score_gapk(stats, k_percent, window, min_tokens=1):
    """정규화한 top1 차이를 이동평균한 뒤 하위 k% 평균 계산

    Gap-K% (Kwak & Kim, 2026), 식 5–7의 집계 방식
    g = (logp - top1) / sigma, sigma가 0이면 g=0 처리
    창 크기는 답 길이 이하로 제한, 반환값은 0 이하의 원시 점수
    """
    if not stats:
        return 0.0
    gaps = [(s.logp - s.top1) / s.sigma if s.sigma > 0 else 0.0 for s in stats]
    w = max(1, min(window, len(gaps)))
    smoothed = [sum(gaps[t:t + w]) / w for t in range(len(gaps) - w + 1)]
    picked = _bottom_k(smoothed, k_percent, min_tokens)
    return sum(picked) / len(picked)


def likelihood_score_minkpp_exp(stats, k_percent, min_tokens=1):
    """하위 k% z에 exp를 적용한 뒤 평균하는 저장소 자체 변형

    큰 음수의 영향을 줄이지만 양수 z에서는 1 초과 가능
    exp(mean(z))와 다른 집계
    """
    if not stats:
        return 0.0
    z = [(s.logp - s.mu) / s.sigma if s.sigma > 0 else 0.0 for s in stats]
    picked = _bottom_k(z, k_percent, min_tokens)
    return sum(math.exp(zi) for zi in picked) / len(picked)


def likelihood_score_gapk_exp(stats, k_percent, window, min_tokens=1):
    """이동평균한 하위 k% top1 차이에 exp를 적용해 평균하는 자체 변형"""
    if not stats:
        return 0.0
    gaps = [(s.logp - s.top1) / s.sigma if s.sigma > 0 else 0.0 for s in stats]
    w = max(1, min(window, len(gaps)))
    smoothed = [sum(gaps[t:t + w]) / w for t in range(len(gaps) - w + 1)]
    picked = _bottom_k(smoothed, k_percent, min_tokens)
    return sum(math.exp(gi) for gi in picked) / len(picked)


def aggregation_needs_stats(spec):
    """집계 방식에 어휘 분포 통계가 필요한지 확인"""

    return spec in STATS_AGGREGATIONS


def aggregation_score_scale(spec):
    """저장 레코드에 사용할 점수 단위 반환

    probability는 [0, 1], raw_z는 원시 점수, exp_z는 지수 변환 점수
    minkpp_exp는 1 초과 가능하므로 확률로 해석 불가
    """
    if spec in ("minkpp", "gapk"):
        return "raw_z"
    if spec in ("minkpp_exp", "gapk_exp"):
        return "exp_z"
    return "probability"


def parse_aggregation(spec, threshold=None, k=None, window=None, min_tokens=1):
    """집계 이름·필수 옵션을 검증해 토큰 점수 집계 함수 반환"""
    if spec == "mean":
        return likelihood_score
    if spec == "max":
        return likelihood_score_max
    if spec == "hybrid":
        if threshold is None:
            raise ValueError("--aggregation hybrid requires --threshold")
        return lambda lps: likelihood_score_hybrid(lps, threshold)
    if spec == "mink":
        if k is None:
            raise ValueError("--aggregation mink requires --k")
        return lambda lps: likelihood_score_mink(lps, k, min_tokens)
    if spec == "minkpp":
        if k is None:
            raise ValueError("--aggregation minkpp requires --k")
        return lambda stats: likelihood_score_minkpp(stats, k, min_tokens)
    if spec == "gapk":
        if k is None or window is None:
            raise ValueError("--aggregation gapk requires --k and --window")
        return lambda stats: likelihood_score_gapk(stats, k, window, min_tokens)
    if spec == "minkpp_exp":
        if k is None:
            raise ValueError("--aggregation minkpp_exp requires --k")
        return lambda stats: likelihood_score_minkpp_exp(stats, k, min_tokens)
    if spec == "gapk_exp":
        if k is None or window is None:
            raise ValueError("--aggregation gapk_exp requires --k and --window")
        return lambda stats: likelihood_score_gapk_exp(stats, k, window, min_tokens)
    raise ValueError(f"bad --aggregation {spec!r}, expected one of "
                     "'mean', 'max', 'hybrid', 'mink', 'minkpp', 'gapk', "
                     "'minkpp_exp', 'gapk_exp'")


def score_standard(gen, samples, task, targets, score_window, aggregation, threshold,
                   k_percent=None, window=None, min_tokens=1, source_records=None):
    """응답 준비 후 절단점별 토큰 점수 집계와 AUC 계산

    결과 레코드, 원본 응답 준비 시간, 채점 시간 반환
    """
    t0 = time.time()
    kept = rollout_and_filter(gen, samples, task, targets,
                              temperature=protocol.ROLLOUT_TEMPERATURE,
                              source_records=source_records,
                              progress_label="lhf-rollout")
    rollout_time = time.time() - t0

    reduce = parse_score_window(score_window)
    aggregate = parse_aggregation(aggregation, threshold, k_percent, window, min_tokens)
    want_stats = aggregation_needs_stats(aggregation)
    force_text = FORCE[task]

    t1 = time.time()
    records = []
    for n, k in enumerate(kept, 1):
        # TRACE와 같은 접두사·추론 구간을 사용해 절단 위치 통일
        prompt_ids = gen.tok.encode(k["prefix_before_cot"], add_special_tokens=False)
        cot_ids = gen.tok.encode(k["cot"], add_special_tokens=False)
        per_cutoff_lps = likelihood_curve_cached(gen.tok, gen.net, prompt_ids, cot_ids,
                                                  force_text, k["answer_text"] or "", FRACS,
                                                  want_stats=want_stats)
        curve = [aggregate(reduce(lps)) for lps in per_cutoff_lps]
        records.append({"pid": k["sample"]["pid"], "task": task,
                         "variant": k["sample"]["variant"], "model": gen.model,
                         "curve": curve, "auc": auc(curve), "response": k["response"],
                         "impl": "likelihood_trace_hf", "score_window": score_window,
                         "aggregation": aggregation, "threshold": threshold,
                         "k": k_percent, "window": window, "min_k_tokens": min_tokens,
                         "score_scale": aggregation_score_scale(aggregation),
                         "response_source": ("records" if source_records is not None
                                             else "generated")})
        if n % 50 == 0 or n == len(kept):
            print(f"[score] {n}/{len(kept)} ({time.time() - t1:.0f}s elapsed)", flush=True)
    return records, rollout_time, time.time() - t1


def score(gen, samples, task, targets, score_window, aggregation, threshold,
          source_records=None, k_percent=None, window=None, min_tokens=1):
    """공통 평가 인자를 Likelihood 채점 함수에 전달"""
    return score_standard(gen, samples, task, targets, score_window, aggregation, threshold,
                          k_percent=k_percent, window=window, min_tokens=min_tokens,
                          source_records=source_records)


def parse_args():
    """Likelihood 평가 CLI 옵션 해석·검증"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["math", "code"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", default="ic_correct")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val",
                    help="comma-separated split names; default: val")
    ap.add_argument("--records",
                    help="JSONL of previously written records (any file with 'pid' and "
                         "'response') whose responses are scored verbatim instead of "
                         "generating new rollouts. For math/code this is how two "
                         "--aggregation runs are compared on the *same* responses: the "
                         "rollout is sampled at 0.7 and costs 92-98%% of the run, so "
                         "re-generating both changes the kept population and pays ~20x "
                         "the scoring cost. Pass the same --split/--sample-n/"
                         "--sample-seed the records were produced with; unmatched pids "
                         "are reported as missing_record. Omit to generate rollouts.")
    ap.add_argument("--limit", type=int, help="keep only the first N samples, sorted by pid")
    ap.add_argument("--sample-n", type=int,
                    help="keep a random N-sample subset instead of the first N (--limit); "
                         "use the same --sample-n/--sample-seed across every run being "
                         "compared (baseline/hack/nonhack, every --aggregation) so they all "
                         "score the identical subset of problems")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="random.Random seed for --sample-n; independent of --seed "
                         "(which only seeds the rollout generation)")
    ap.add_argument("--batch-size", type=int, default=16, help="rollout-generation batch size only")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tokenizer",
                    help="override if --model's checkpoint wasn't pushed with its own "
                         "tokenizer files, e.g. --tokenizer /workspace/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch random seed; identical responses across runs are not guaranteed; "
                         "use --records for paired scoring")
    ap.add_argument("--score-window", default="full",
                    help="'full' or 'first_<k>' (e.g. first_1, first_2)")
    ap.add_argument("--aggregation",
                    choices=["mean", "max", "hybrid", "mink", "minkpp", "gapk",
                             "minkpp_exp", "gapk_exp"], default="mean",
                    help="'mean' (default, exp(mean(logprobs))), 'max' (exp(max(logprobs))), "
                         "'hybrid' (mean unless some token's probability exceeds "
                         "--threshold, then that token's -- the max's -- probability), "
                         "'mink' (Min-K%%, Shi et al. ICLR 2024: exp(mean(lowest --k%% "
                         "logprobs)), no normalization -- like mean but restricted to "
                         "the worst tokens; bounded, cannot blow up), "
                         "'minkpp' (Min-K%%++ z-scores, lowest --k%% averaged), "
                         "'gapk' (Gap-K%% top-1 gaps, --window-smoothed, lowest --k%% "
                         "averaged), or the '_exp' variant of either (same token "
                         "selection, but averaging exp(z)/exp(g) instead of the raw "
                         "score -- our fix, not in either paper, for the raw scores' "
                         "blow-up when sigma collapses on teacher-forced answers; see "
                         "likelihood_score_minkpp_exp's docstring). minkpp/gapk return "
                         "raw scores, not probabilities; _exp variants are exponentiated "
                         "scores, not probabilities, and minkpp_exp can exceed 1.")
    ap.add_argument("--threshold", type=float,
                    help="probability threshold for --aggregation hybrid; no default, "
                         "must be swept explicitly (required when --aggregation hybrid)")
    ap.add_argument("--k", type=float, default=20.0, dest="k_percent",
                    help="percent of lowest-scoring tokens averaged by --aggregation "
                         "minkpp/gapk (default 20, the value both papers compare at)")
    ap.add_argument("--min-k-tokens", type=int, default=1, dest="min_tokens",
                    help="floor on how many tokens --aggregation minkpp/gapk average "
                         "(default 1, i.e. no floor beyond the papers' own max(1, k%%)). "
                         "On short teacher-forced answers (math is a median of 2 tokens) "
                         "k%% alone collapses to a single token, so one outlier z-score "
                         "becomes the whole score; raising this guarantees at least this "
                         "many tokens are averaged, capped to the number of answer "
                         "tokens actually scored. Not from either paper -- both assume "
                         "long, naturally-sampled text where k%% never floors this low.")
    ap.add_argument("--window", type=int,
                    help="Gap-K%% sliding-window size over adjacent tokens; no default, "
                         "must be swept (required when --aggregation gapk). The paper "
                         "reports 6 for LLaMA-family and 3 for Pythia/Mamba and does not "
                         "test Qwen, so neither value is established here; its Appendix C "
                         "attributes the difference to architecture, which would point at "
                         "6 for Qwen, but that is an extrapolation. Clamped to the number "
                         "of scored answer tokens, so math answers (median 2 tokens) "
                         "smooth trivially and code answers (median ~100) do not")
    args = ap.parse_args()

    args.variant = args.variant or "ic_correct"
    args.split = args.split or "val"
    if args.aggregation == "hybrid" and args.threshold is None:
        ap.error("--aggregation hybrid requires --threshold")
    if args.aggregation in ("gapk", "gapk_exp") and args.window is None:
        ap.error(f"--aggregation {args.aggregation} requires --window")
    if args.limit and args.sample_n:
        ap.error("--limit and --sample-n are mutually exclusive")

    return args


def main():
    """Likelihood 평가 실행 후 점수·실행 통계 저장"""
    args = parse_args()

    splits = set(args.split.split(","))
    samples = [r for r in read_jsonl(f"{args.data}/prompts.{args.variant}.jsonl")
               if r["split"] in splits]
    samples.sort(key=lambda r: r["pid"])
    if args.sample_n:
        samples = random.Random(args.sample_seed).sample(samples, min(args.sample_n, len(samples)))
        samples.sort(key=lambda r: r["pid"])
    elif args.limit:
        samples = samples[:args.limit]

    targets = targets_for(args.task, args.data, samples)

    source_records = list(read_jsonl(args.records)) if args.records else None

    torch.manual_seed(args.seed)
    gen = Generator(args.model, dtype=args.dtype, batch_size=args.batch_size,
                      tokenizer=args.tokenizer)

    records, rollout_time, scoring_time = score(
        gen, samples, args.task, targets, args.score_window, args.aggregation,
        args.threshold, source_records=source_records,
        k_percent=args.k_percent, window=args.window, min_tokens=args.min_tokens)

    write_jsonl(args.out, records)
    mean = sum(record["auc"] for record in records) / len(records) if records else 0.0
    print(f"{len(records)} scored, mean Likelihood-TRACE score {mean:.6f}, "
          f"rollout {rollout_time:.1f}s + scoring {scoring_time:.1f}s "
          f"(score_window={args.score_window}, aggregation={args.aggregation}, "
          f"threshold={args.threshold})")
    stats = {
        "n": len(records), "mean_auc": mean,
        "wall_clock_s": rollout_time + scoring_time,
        "rollout_time_s": rollout_time, "scoring_time_s": scoring_time,
        "impl": "likelihood_trace_hf", "score_window": args.score_window,
        "aggregation": args.aggregation, "threshold": args.threshold,
        "k": args.k_percent, "window": args.window, "min_k_tokens": args.min_tokens,
        "score_scale": aggregation_score_scale(args.aggregation),
        "response_source": "records" if source_records is not None else "generated",
        "batch_size": args.batch_size, "rollout_batch_size": args.batch_size,
        "scoring_response_batch_size": 1, "dtype": args.dtype,
        "seed": args.seed,
    }
    with open(args.out + ".stats", "w") as handle:
        json.dump(stats, handle)


if __name__ == "__main__":
    main()
