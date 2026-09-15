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


# ---------------------------------------------------------------------------
# Teacher-forced likelihood curves and aggregation.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenStats:
    """One scored answer token, with the vocabulary statistics Min-K%++/Gap-K% need.

    ``logp`` is log p(x_t | context) -- the only quantity mean/max/hybrid use.
    ``mu`` and ``sigma`` are the mean and standard deviation of log p(. | context)
    over the *whole vocabulary* (Min-K%++ Eq. 3), and ``top1`` is
    max_v log p(v | context) (Gap-K% Eq. 5).  All three come from logits this module
    already materializes, so computing them adds only elementwise algebra over a
    tensor that exists either way (Min-K%++ App. A makes the same point).
    """

    logp: float
    mu: float
    sigma: float
    top1: float


def _per_token(logprobs_2d, score_tensor, want_stats):
    """[T, V] log-probs + target ids -> per-token payload for one cutoff.

    Returns bare floats unless ``want_stats``, so mean/max/hybrid runs keep both their
    old return type and their old cost; the vocabulary reductions below are only paid
    by the aggregations that actually read them.
    """

    idx = torch.arange(logprobs_2d.shape[0], device=logprobs_2d.device)
    target = logprobs_2d[idx, score_tensor]
    if not want_stats:
        return target.tolist()
    probs = logprobs_2d.exp()
    mu = (probs * logprobs_2d).sum(-1)
    # clamp_min(0) only guards float error; the variance is non-negative by definition.
    sigma = ((probs * logprobs_2d.square()).sum(-1) - mu.square()).clamp_min(0).sqrt()
    top1 = logprobs_2d.max(-1).values
    return [TokenStats(lp, m, s, t) for lp, m, s, t
            in zip(target.tolist(), mu.tolist(), sigma.tolist(), top1.tolist())]


@torch.inference_mode()
def likelihood_curve_cached(tok, net, prompt_ids, cot_ids, force_text, answer_text, fracs,
                            cot_cutoff_lens=None, want_stats=False):
    """likelihood-TRACE for one sample: 1 forward pass builds a KV cache over
    prompt+full_cot, then ALL len(fracs) cutoffs are scored in a SECOND, single
    batched forward pass (batch = len(fracs)) instead of one small forward per
    cutoff. The cache is replicated len(fracs) times but never physically cropped;
    each row's `attention_mask` instead hides cached positions past that row's own
    cutoff, and `position_ids` places the shared FORCE+answer suffix right after
    that row's cutoff boundary (not after the full, untruncated cache) -- matching
    exactly what cropping-then-forwarding would have computed, since HF's causal
    mask + attention_mask combination doesn't care where in the sequence the
    masked-out positions sit. This removes 9 of the 10 per-cutoff forward-call
    overheads (kernel launch, Python loop) that a sequential crop-then-forward loop
    pays, matching the previous server's "1 base forward + per-cutoff masking,
    batch 16" setup instead of this project's earlier sequential-crop version.

    force_text/answer_text are tokenized TOGETHER, not encoded separately and
    concatenated -- BPE can merge across that boundary (Qwen has a merged '>-'
    token), so a separately-encoded '<answer>' + '-4955' lands on a token path the
    model never actually generates, and teacher-forcing it collapses probability to
    near-zero regardless of the model's real confidence (confirmed independently:
    geo-mean 0.0276 vs 0.9999 for the identical context/answer, encoded separately
    vs together). Fix: encode force_text+answer_text as one string and score
    everything after the longest common token prefix with force_text alone.

    `cot_cutoff_lens` optionally supplies exact cutoff positions in `cot_ids`. This is
    useful for callers that already have token-aligned cutoffs. When omitted, the
    original math/code token-ratio behavior is unchanged.
    Returns a list of per-cutoff (per-answer-token log-prob list), in `fracs` order.
    """
    force_ids = tok.encode(force_text, add_special_tokens=False)
    combined_ids = tok.encode(force_text + answer_text, add_special_tokens=False)
    common = _longest_common_prefix_len(force_ids, combined_ids)
    if common >= len(combined_ids):  # empty/no-op answer
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
    # logits[:, k] predicts combined_ids[k+1], so combined_ids[common:] is scored
    # by logits[:, common-1 : M-1]
    answer_logits = logits[:, common - 1:M - 1, :]
    logprobs = torch.log_softmax(answer_logits.float(), dim=-1)
    score_tensor = torch.tensor(combined_ids[common:], device=device)
    return [_per_token(logprobs[j], score_tensor, want_stats) for j in range(n_cut)]


def parse_score_window(spec):
    """'full' -> use every answer-token log-prob; 'first_<k>' -> only the first k."""
    if spec == "full":
        return lambda lps: lps
    if spec.startswith("first_"):
        k = int(spec[len("first_"):])
        if k < 1:
            raise ValueError(f"--score-window first_<k> needs k >= 1, got {spec!r}")
        return lambda lps: lps[:k]
    raise ValueError(f"bad --score-window {spec!r}, expected 'full' or 'first_<k>'")


def likelihood_score(logprobs):
    """exp(mean(logprobs)): geometric mean of the per-token probabilities, in [0, 1] --
    same scale as TRACE's E[R-hat], so protocol.auc() applies unchanged."""
    if not logprobs:
        return 0.0
    return math.exp(sum(logprobs) / len(logprobs))


def likelihood_score_max(logprobs):
    """exp(max(logprobs)): the single most-confident answer token's probability. Unlike
    the geometric mean, one decisive token (e.g. a hacked RM-loophole sign) is not
    diluted by less-confident tokens elsewhere in the answer."""
    if not logprobs:
        return 0.0
    return math.exp(max(logprobs))


def likelihood_score_hybrid(logprobs, threshold):
    """mean-based score, UNLESS some answer token's probability exceeds `threshold`,
    in which case the single most-confident token's probability (likelihood_score_max)
    is used instead -- if any token clears the bar, it is by definition the max, so no
    separate tie-break among qualifying tokens is needed."""
    if not logprobs:
        return 0.0
    max_score = likelihood_score_max(logprobs)
    return max_score if max_score > threshold else likelihood_score(logprobs)


STATS_AGGREGATIONS = ("minkpp", "gapk", "minkpp_exp", "gapk_exp")


def _bottom_k(values, k_percent, min_tokens=1):
    """The lowest k% of `values` (at least `min_tokens`, capped to len(values)).

    Both papers' min-k% selection is `max(1, int(n * k%))`; on short teacher-forced
    answers (math is a median of 2 tokens) that floors to a single token, so the
    "lowest k%" degenerates into "the single worst token" with no averaging left to
    dilute an outlier z-score. `min_tokens` raises that floor so a fixed minimum number
    of tokens is always averaged, regardless of how short the answer is.
    """
    if not values:
        return []
    count = max(min_tokens, int(len(values) * k_percent / 100.0))
    count = min(count, len(values))
    return sorted(values)[:count]


def likelihood_score_mink(logprobs, k_percent, min_tokens=1):
    """Min-K% (Shi et al., ICLR 2024), the predecessor Min-K%++ (Zhang et al., 2025)
    built its vocabulary-mean/std normalization on top of. No normalization here: just
    average the lowest k% of the raw per-token log-probs, then exponentiate -- exactly
    what likelihood_score (the 'mean' aggregation) does over ALL tokens, restricted to
    the worst few.

    Operates on bare logprobs, like mean/max/hybrid, not on TokenStats: Min-K% needs no
    vocabulary statistics, so it pays none of minkpp/gapk's per-token mu/sigma/top1 cost.

    Bounded exactly like likelihood_score: logp <= 0 always, so exp(mean(bottom-k logp))
    in (0, 1] regardless of answer length or how deterministic the model's next-token
    distribution is. No division by sigma means no blow-up on teacher-forced answers --
    unlike minkpp (see likelihood_score_minkpp_exp's docstring for that failure mode),
    this needs neither --min-k-tokens nor an exp()-of-the-score rescue to stay stable;
    the exp() here is the same bounding step likelihood_score always applied, not a fix.
    """
    if not logprobs:
        return 0.0
    picked = _bottom_k(logprobs, k_percent, min_tokens)
    return math.exp(sum(picked) / len(picked))


def likelihood_score_minkpp(stats, k_percent, min_tokens=1):
    """Min-K%++ (Zhang et al., ICLR 2025), Eq. 3-4, over the forced answer tokens.

    Each token's log-prob is z-scored against the model's own next-token distribution
    (mu, sigma over the vocabulary), and the lowest k% of those z-scores are averaged.
    Unlike the geometric mean this asks "was this token a *mode* of the distribution",
    not "was it absolutely likely", so a token that is improbable only because the
    whole distribution is flat no longer looks like weak evidence.

    NOTE the output is a z-score, not a probability: it is unbounded and typically
    negative. protocol.auc() is a linear functional so it applies unchanged, and detect.py
    thresholds against the baseline mean, which is scale-free -- but the number is not
    comparable to a mean/max/hybrid score, only to another minkpp score.
    """
    if not stats:
        return 0.0
    z = [(s.logp - s.mu) / s.sigma if s.sigma > 0 else 0.0 for s in stats]
    picked = _bottom_k(z, k_percent, min_tokens)
    return sum(picked) / len(picked)


def likelihood_score_gapk(stats, k_percent, window, min_tokens=1):
    """Gap-K% (Kwak & Kim, 2026), Eq. 5-7, over the forced answer tokens.

    Scores each token by its normalized distance from the model's *top-1* prediction
    rather than from the vocabulary mean, so a confident misprediction (the model
    strongly preferred some other token) counts as much stronger counter-evidence than
    mere uncertainty. Scores are then smoothed over a sliding window of `window`
    adjacent tokens before the lowest k% are averaged, which is what lets the signal
    reflect a contiguous span instead of one isolated token.

    `window` is clamped to the number of scored tokens: math answers are a median of 2
    tokens, so there the smoothing is a no-op and this reduces to a normalized top-1
    gap. Like minkpp, the result is a raw (non-positive) score, not a probability.
    """
    if not stats:
        return 0.0
    gaps = [(s.logp - s.top1) / s.sigma if s.sigma > 0 else 0.0 for s in stats]
    w = max(1, min(window, len(gaps)))
    smoothed = [sum(gaps[t:t + w]) / w for t in range(len(gaps) - w + 1)]
    picked = _bottom_k(smoothed, k_percent, min_tokens)
    return sum(picked) / len(picked)


def likelihood_score_minkpp_exp(stats, k_percent, min_tokens=1):
    """minkpp, but averaging exp(z) instead of raw z. NOT in Zhang et al. -- our fix
    for a failure mode neither paper's setting encounters.

    likelihood_score_minkpp's z = (logp - mu) / sigma is unbounded: teacher-forcing a
    fixed answer makes the model's next-token distribution near-deterministic at most
    positions (sigma -> 0), so an ordinary-looking logp there produces a z of -50 to
    -1000+ (measured; see checks/test_aggregation.py and the minkpp diagnostics in this
    session), which then dominates any mean over it. Min-K%++'s own setting is natural
    generation with enough entropy that sigma rarely collapses, so the paper had no
    reason to bound z.

    exp() is the same fix `likelihood_score` already applies to raw logp (which is
    also unbounded below): it saturates large-magnitude negative inputs toward 0
    instead of letting them dominate an arithmetic mean, while leaving ordinary-sized
    inputs close to linear (exp(z) ~ 1+z near z=0). Verified this preserves Min-K%++'s
    actual point -- distinguishing equal-logp tokens by distribution shape -- on the
    paper's own flat-vs-peaked construction (checks/test_aggregation.py): flat gives
    z=0 -> exp(z)=1.0, peaked gives z=-0.517 -> exp(z)=0.596, still clearly separated.
    Bottom-k selection still ranks by raw z (unaffected by the monotonic exp), only the
    final averaging changes.
    """
    if not stats:
        return 0.0
    z = [(s.logp - s.mu) / s.sigma if s.sigma > 0 else 0.0 for s in stats]
    picked = _bottom_k(z, k_percent, min_tokens)
    return sum(math.exp(zi) for zi in picked) / len(picked)


def likelihood_score_gapk_exp(stats, k_percent, window, min_tokens=1):
    """gapk, but averaging exp(g) instead of raw g. Same fix and rationale as
    likelihood_score_minkpp_exp, applied to Gap-K%'s top-1-gap score instead of
    Min-K%++'s vocabulary-mean z-score -- see that docstring for the full argument."""
    if not stats:
        return 0.0
    gaps = [(s.logp - s.top1) / s.sigma if s.sigma > 0 else 0.0 for s in stats]
    w = max(1, min(window, len(gaps)))
    smoothed = [sum(gaps[t:t + w]) / w for t in range(len(gaps) - w + 1)]
    picked = _bottom_k(smoothed, k_percent, min_tokens)
    return sum(math.exp(gi) for gi in picked) / len(picked)


def aggregation_needs_stats(spec):
    """Whether `spec` reads vocabulary statistics beyond the target token's log-prob."""

    return spec in STATS_AGGREGATIONS


def aggregation_score_scale(spec):
    """The output range/units of `spec`, for the "score_scale" field in written records.

    'probability': [0, 1], including the empty-answer fallback.
    'raw_z': normalized scores (minkpp/gapk), not probabilities.
    'exp_z': exponentiated scores, also not probabilities; minkpp_exp can exceed 1,
    while gapk_exp is at most 1 for valid token statistics.
    """
    if spec in ("minkpp", "gapk"):
        return "raw_z"
    if spec in ("minkpp_exp", "gapk_exp"):
        return "exp_z"
    return "probability"


def parse_aggregation(spec, threshold=None, k=None, window=None, min_tokens=1):
    """'mean'/'max'/'hybrid' -> probability-scale reductions of the answer log-probs;
    'minkpp'/'gapk' -> the low-likelihood-token scores of Min-K%++ / Gap-K%, which read
    TokenStats instead of bare floats and return a raw (unbounded) score.
    Returns a `payload -> score` callable, applied after parse_score_window's windowing.
    """
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
        # Same prefix trace.py uses: rendered chat template plus any generated
        # <think>, so both scorers cut the identical reasoning span at identical ratios.
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
    return score_standard(gen, samples, task, targets, score_window, aggregation, threshold,
                          k_percent=k_percent, window=window, min_tokens=min_tokens,
                          source_records=source_records)


def parse_args():
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
