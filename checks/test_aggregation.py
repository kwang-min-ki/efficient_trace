"""고정 확률 분포 기반 Likelihood 집계 공식·경계값 검증"""

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import likelihood_trace as L  # noqa: E402


def stats_from_distribution(probs, target):
    """Build one TokenStats from an explicit categorical distribution."""

    logs = [math.log(p) for p in probs]
    mu = sum(p * lp for p, lp in zip(probs, logs))
    var = sum(p * lp * lp for p, lp in zip(probs, logs)) - mu * mu
    return L.TokenStats(logp=logs[target], mu=mu, sigma=math.sqrt(max(var, 0.0)),
                        top1=max(logs))


class MinKTest(unittest.TestCase):
    """Min-K% (Shi et al., ICLR 2024): no normalization, unlike its minkpp successor.
    Operates on bare logprobs (not TokenStats) since it needs no vocabulary stats."""

    def test_matches_the_paper_formula(self):
        # exp(mean(lowest k% logprobs)) -- same shape as likelihood_score, over a subset.
        logprobs = [math.log(0.9), math.log(0.7), math.log(0.5), math.log(0.1)]
        expected = math.exp((math.log(0.5) + math.log(0.1)) / 2)  # lowest 2 of 4, k=50%
        self.assertAlmostEqual(L.likelihood_score_mink(logprobs, 50.0), expected)

    def test_k_100_equals_the_geometric_mean(self):
        logprobs = [math.log(0.9), math.log(0.5), math.log(0.1)]
        self.assertAlmostEqual(L.likelihood_score_mink(logprobs, 100.0),
                               L.likelihood_score(logprobs))

    def test_bounded_in_zero_one_however_bad_the_logprob(self):
        # No sigma division exists to blow this up, unlike minkpp on the same input.
        # -100 (not -1000) so exp() doesn't underflow to exactly 0 and the lower bound
        # is checked meaningfully rather than trivially by float underflow.
        self.assertGreater(L.likelihood_score_mink([-100.0, -0.01], 50.0), 0.0)
        self.assertLess(L.likelihood_score_mink([-100.0, -0.01], 50.0), 1.0)

    def test_min_tokens_floor_applies_the_same_way_as_minkpp(self):
        logprobs = [-1.0, -2.0, -9.0]  # 3 tokens, k=20% -> floors to 1 without a floor
        self.assertAlmostEqual(L.likelihood_score_mink(logprobs, 20.0), math.exp(-9.0))
        self.assertAlmostEqual(L.likelihood_score_mink(logprobs, 20.0, min_tokens=3),
                               math.exp((-1.0 - 2.0 - 9.0) / 3))

    def test_empty_answer(self):
        self.assertEqual(L.likelihood_score_mink([], 20.0), 0.0)

    def test_needs_no_vocabulary_statistics(self):
        self.assertFalse(L.aggregation_needs_stats("mink"))

    def test_score_scale_is_probability(self):
        self.assertEqual(L.aggregation_score_scale("mink"), "probability")


class MinKppTest(unittest.TestCase):
    def test_matches_the_paper_formula(self):
        probs = [0.5, 0.2, 0.2, 0.1]
        s = stats_from_distribution(probs, target=3)  # the least likely token
        expected = (s.logp - s.mu) / s.sigma
        self.assertAlmostEqual(L.likelihood_score_minkpp([s], 100.0), expected)

    def test_averages_only_the_lowest_k_percent(self):
        # Ten tokens whose z-scores are, by construction, strictly increasing.
        stats = [L.TokenStats(logp=-float(10 - i), mu=0.0, sigma=1.0, top1=0.0)
                 for i in range(10)]
        # k=20% -> the two lowest z-scores, which are -10 and -9.
        self.assertAlmostEqual(L.likelihood_score_minkpp(stats, 20.0), -9.5)
        self.assertAlmostEqual(L.likelihood_score_minkpp(stats, 100.0),
                               sum(-float(10 - i) for i in range(10)) / 10)

    def test_normalization_separates_equal_log_probs(self):
        """The point of Min-K%++: identical log p, different distribution shape.

        Both tokens have probability 0.2, so a raw-likelihood score cannot tell them
        apart; the peaked distribution should score worse because 0.2 is far from its
        mode, while in the flat one 0.2 *is* the mode.
        """
        flat = stats_from_distribution([0.2] * 5, target=0)
        peaked = stats_from_distribution([0.6, 0.2, 0.1, 0.05, 0.05], target=1)
        self.assertAlmostEqual(flat.logp, peaked.logp)  # same likelihood...
        self.assertGreater(L.likelihood_score_minkpp([flat], 100.0),
                           L.likelihood_score_minkpp([peaked], 100.0))  # ...different score

    def test_degenerate_distribution_does_not_divide_by_zero(self):
        s = L.TokenStats(logp=-1.0, mu=-1.0, sigma=0.0, top1=-1.0)
        self.assertEqual(L.likelihood_score_minkpp([s], 100.0), 0.0)

    def test_empty_answer(self):
        self.assertEqual(L.likelihood_score_minkpp([], 20.0), 0.0)


class GapKTest(unittest.TestCase):
    def test_matches_the_paper_formula(self):
        probs = [0.5, 0.2, 0.2, 0.1]
        s = stats_from_distribution(probs, target=3)
        expected = (s.logp - s.top1) / s.sigma
        self.assertAlmostEqual(L.likelihood_score_gapk([s], 100.0, window=1), expected)

    def test_top1_token_scores_zero(self):
        """g_t is 0 exactly when the target token *is* the model's top-1 prediction."""

        s = stats_from_distribution([0.5, 0.2, 0.2, 0.1], target=0)
        self.assertAlmostEqual(L.likelihood_score_gapk([s], 100.0, window=1), 0.0)

    def test_gapk_is_never_positive(self):
        for target in range(4):
            s = stats_from_distribution([0.5, 0.2, 0.2, 0.1], target)
            self.assertLessEqual(L.likelihood_score_gapk([s], 100.0, window=1), 0.0)

    def test_sliding_window_averages_adjacent_tokens(self):
        gaps = [-4.0, 0.0, 0.0, 0.0]
        stats = [L.TokenStats(logp=g, mu=0.0, sigma=1.0, top1=0.0) for g in gaps]
        # window=2 -> smoothed [-2.0, 0.0, 0.0]; the lowest of the three is -2.0, so
        # smoothing halves the isolated spike instead of reporting it at full depth.
        self.assertAlmostEqual(
            L.likelihood_score_gapk(stats, k_percent=1.0, window=2), -2.0)
        self.assertAlmostEqual(
            L.likelihood_score_gapk(stats, k_percent=1.0, window=1), -4.0)

    def test_window_is_clamped_to_short_answers(self):
        """Math answers are a median of 2 tokens, shorter than the default window."""

        stats = [L.TokenStats(logp=-2.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-4.0, mu=0.0, sigma=1.0, top1=0.0)]
        # window=6 clamps to 2 -> a single window holding the mean of both gaps.
        self.assertAlmostEqual(
            L.likelihood_score_gapk(stats, k_percent=100.0, window=6), -3.0)

    def test_distinguishes_confident_misprediction_from_uncertainty(self):
        """Gap-K%'s stated advantage over Min-K%++ (its Fig. 2).

        Two tokens with the same Min-K%++ z-score: one where the model was merely
        unsure, one where it confidently preferred a different token. Only Gap-K%
        should separate them.
        """
        flat = L.TokenStats(logp=-2.0, mu=-1.0, sigma=1.0, top1=-1.5)
        confident = L.TokenStats(logp=-2.0, mu=-1.0, sigma=1.0, top1=-0.1)
        self.assertAlmostEqual(L.likelihood_score_minkpp([flat], 100.0),
                               L.likelihood_score_minkpp([confident], 100.0))
        self.assertGreater(L.likelihood_score_gapk([flat], 100.0, window=1),
                           L.likelihood_score_gapk([confident], 100.0, window=1))


class MinTokensFloorTest(unittest.TestCase):
    """--min-k-tokens: on short answers, k% alone floors to a single token (see
    _bottom_k's max(1, int(n*k/100))), so one outlier z-score becomes the whole score.
    min_tokens raises that floor to guarantee averaging over more than one token."""

    def test_short_answer_without_floor_uses_one_token(self):
        # 3 tokens, k=20% -> max(1, int(3*0.2))=1 -> only the worst token, unaveraged.
        stats = [L.TokenStats(logp=-1.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-2.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-9.0, mu=0.0, sigma=1.0, top1=0.0)]  # the outlier
        self.assertAlmostEqual(L.likelihood_score_minkpp(stats, 20.0), -9.0)

    def test_min_tokens_floor_dilutes_the_outlier(self):
        stats = [L.TokenStats(logp=-1.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-2.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-9.0, mu=0.0, sigma=1.0, top1=0.0)]
        # min_tokens=3 averages all three instead of just the worst one.
        self.assertAlmostEqual(L.likelihood_score_minkpp(stats, 20.0, min_tokens=3),
                               (-1.0 - 2.0 - 9.0) / 3)

    def test_min_tokens_is_capped_to_available_tokens(self):
        stats = [L.TokenStats(logp=-1.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-3.0, mu=0.0, sigma=1.0, top1=0.0)]
        # min_tokens=5 requested but only 2 tokens exist -> averages both, not an error.
        self.assertAlmostEqual(L.likelihood_score_minkpp(stats, 20.0, min_tokens=5), -2.0)

    def test_min_tokens_defaults_to_one_unbounded_selection(self):
        stats = [L.TokenStats(logp=-1.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-2.0, mu=0.0, sigma=1.0, top1=0.0)]
        self.assertAlmostEqual(L.likelihood_score_minkpp(stats, 20.0),
                               L.likelihood_score_minkpp(stats, 20.0, min_tokens=1))

    def test_gapk_respects_the_same_floor(self):
        stats = [L.TokenStats(logp=-1.0, mu=0.0, sigma=1.0, top1=0.0),   # gap -1.0
                 L.TokenStats(logp=-2.0, mu=0.0, sigma=1.0, top1=-0.5),  # gap -1.5
                 L.TokenStats(logp=-9.0, mu=0.0, sigma=1.0, top1=-0.5)]  # gap -8.5
        expected = (-1.0 - 1.5 - 8.5) / 3
        self.assertAlmostEqual(
            L.likelihood_score_gapk(stats, 20.0, window=1, min_tokens=3), expected)

    def test_parsed_callable_forwards_min_tokens(self):
        stats = [L.TokenStats(logp=-1.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-9.0, mu=0.0, sigma=1.0, top1=0.0)]
        parsed = L.parse_aggregation("minkpp", k=20.0, min_tokens=2)
        self.assertAlmostEqual(parsed(stats), L.likelihood_score_minkpp(stats, 20.0, 2))


class ExpVariantTest(unittest.TestCase):
    """minkpp_exp/gapk_exp: our fix for the raw-z/-g blow-up on teacher-forced answers
    (see likelihood_score_minkpp_exp's docstring). Averages exp(z)/exp(g) instead of
    the raw score, the same saturating trick likelihood_score already applies to logp."""

    def test_minkpp_exp_matches_exp_of_the_paper_formula_for_one_token(self):
        probs = [0.5, 0.2, 0.2, 0.1]
        s = stats_from_distribution(probs, target=3)
        raw_z = L.likelihood_score_minkpp([s], 100.0)
        self.assertAlmostEqual(L.likelihood_score_minkpp_exp([s], 100.0), math.exp(raw_z))

    def test_still_separates_equal_log_probs_by_distribution_shape(self):
        """The same flat-vs-peaked case MinKppTest checks, through the exp(z) lens:
        exp() is monotonic, so the ordering (and thus the separation) survives."""
        flat = stats_from_distribution([0.2] * 5, target=0)
        peaked = stats_from_distribution([0.6, 0.2, 0.1, 0.05, 0.05], target=1)
        self.assertAlmostEqual(flat.logp, peaked.logp)
        self.assertGreater(L.likelihood_score_minkpp_exp([flat], 100.0),
                           L.likelihood_score_minkpp_exp([peaked], 100.0))

    def test_bounds_a_sigma_collapse_that_explodes_the_raw_score(self):
        """The actual failure mode this fixes: teacher-forcing makes some token's
        next-token distribution near-deterministic (sigma -> 0), so an otherwise
        unremarkable logp produces a z of -50 to -1000+ under the raw formula. exp()
        saturates that toward 0 instead of letting a mean over it explode."""
        normal = L.TokenStats(logp=-1.0, mu=0.0, sigma=1.0, top1=0.0)
        collapsed = L.TokenStats(logp=-5.0, mu=-0.02, sigma=1e-4, top1=-0.01)  # z ~ -49800
        raw = L.likelihood_score_minkpp([normal, collapsed], 100.0)
        self.assertLess(raw, -1000)  # the blow-up, reproduced
        bounded = L.likelihood_score_minkpp_exp([normal, collapsed], 100.0)
        self.assertGreaterEqual(bounded, 0.0)
        self.assertLess(bounded, 1.0)  # both tokens score < 1 (worse than the mean token)

    def test_gapk_exp_matches_exp_of_the_paper_formula(self):
        probs = [0.5, 0.2, 0.2, 0.1]
        s = stats_from_distribution(probs, target=3)
        raw_g = L.likelihood_score_gapk([s], 100.0, window=1)
        self.assertAlmostEqual(L.likelihood_score_gapk_exp([s], 100.0, window=1),
                               math.exp(raw_g))

    def test_gapk_exp_top1_token_scores_one(self):
        """g_t=0 at the model's own top-1 token -> exp(0)=1, the gapk_exp analogue of
        GapKTest.test_top1_token_scores_zero."""
        s = stats_from_distribution([0.5, 0.2, 0.2, 0.1], target=0)
        self.assertAlmostEqual(L.likelihood_score_gapk_exp([s], 100.0, window=1), 1.0)

    def test_empty_answer(self):
        self.assertEqual(L.likelihood_score_minkpp_exp([], 20.0), 0.0)
        self.assertEqual(L.likelihood_score_gapk_exp([], 20.0, window=1), 0.0)


class ParseAggregationTest(unittest.TestCase):
    def test_stats_aggregations_are_declared(self):
        for spec in ("minkpp", "gapk", "minkpp_exp", "gapk_exp"):
            self.assertTrue(L.aggregation_needs_stats(spec))
        for spec in ("mean", "max", "hybrid"):
            self.assertFalse(L.aggregation_needs_stats(spec))

    def test_missing_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            L.parse_aggregation("mink")
        with self.assertRaises(ValueError):
            L.parse_aggregation("minkpp")
        with self.assertRaises(ValueError):
            L.parse_aggregation("gapk", k=20.0)  # window missing
        with self.assertRaises(ValueError):
            L.parse_aggregation("minkpp_exp")
        with self.assertRaises(ValueError):
            L.parse_aggregation("gapk_exp", k=20.0)  # window missing
        with self.assertRaises(ValueError):
            L.parse_aggregation("nope")

    def test_score_scale_of_each_aggregation(self):
        self.assertEqual(L.aggregation_score_scale("mean"), "probability")
        self.assertEqual(L.aggregation_score_scale("max"), "probability")
        self.assertEqual(L.aggregation_score_scale("hybrid"), "probability")
        self.assertEqual(L.aggregation_score_scale("mink"), "probability")
        self.assertEqual(L.aggregation_score_scale("minkpp"), "raw_z")
        self.assertEqual(L.aggregation_score_scale("gapk"), "raw_z")
        self.assertEqual(L.aggregation_score_scale("minkpp_exp"), "exp_z")
        self.assertEqual(L.aggregation_score_scale("gapk_exp"), "exp_z")

    def test_parsed_callables_match_the_scorers(self):
        stats = [stats_from_distribution([0.5, 0.3, 0.2], target=2)]
        logprobs = [math.log(0.5), math.log(0.3), math.log(0.2)]
        self.assertAlmostEqual(L.parse_aggregation("mink", k=66.0)(logprobs),
                               L.likelihood_score_mink(logprobs, 66.0))
        self.assertAlmostEqual(L.parse_aggregation("minkpp", k=20.0)(stats),
                               L.likelihood_score_minkpp(stats, 20.0))
        self.assertAlmostEqual(L.parse_aggregation("gapk", k=20.0, window=3)(stats),
                               L.likelihood_score_gapk(stats, 20.0, 3))

    def test_existing_aggregations_are_unchanged(self):
        logprobs = [math.log(0.5), math.log(0.25)]
        self.assertAlmostEqual(L.parse_aggregation("mean")(logprobs),
                               math.exp((math.log(0.5) + math.log(0.25)) / 2))
        self.assertAlmostEqual(L.parse_aggregation("max")(logprobs), 0.5)


if __name__ == "__main__":
    unittest.main()
