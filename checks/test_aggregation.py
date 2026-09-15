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
    """지정 확률 분포와 정답 토큰으로 TokenStats 생성"""

    logs = [math.log(p) for p in probs]
    mu = sum(p * lp for p, lp in zip(probs, logs))
    var = sum(p * lp * lp for p, lp in zip(probs, logs)) - mu * mu
    return L.TokenStats(logp=logs[target], mu=mu, sigma=math.sqrt(max(var, 0.0)),
                        top1=max(logs))


class MinKTest(unittest.TestCase):

    def test_matches_the_paper_formula(self):
        # 하위 k% logp 평균에 exp 적용
        logprobs = [math.log(0.9), math.log(0.7), math.log(0.5), math.log(0.1)]
        expected = math.exp((math.log(0.5) + math.log(0.1)) / 2)  # 4개 중 하위 2개 선택
        self.assertAlmostEqual(L.likelihood_score_mink(logprobs, 50.0), expected)

    def test_k_100_equals_the_geometric_mean(self):
        logprobs = [math.log(0.9), math.log(0.5), math.log(0.1)]
        self.assertAlmostEqual(L.likelihood_score_mink(logprobs, 100.0),
                               L.likelihood_score(logprobs))

    def test_bounded_in_zero_one_however_bad_the_logprob(self):
        # exp가 0으로 언더플로하지 않는 -100을 사용해 양수 하한 검증
        self.assertGreater(L.likelihood_score_mink([-100.0, -0.01], 50.0), 0.0)
        self.assertLess(L.likelihood_score_mink([-100.0, -0.01], 50.0), 1.0)

    def test_min_tokens_floor_applies_the_same_way_as_minkpp(self):
        logprobs = [-1.0, -2.0, -9.0]  # 토큰 3개에서 20% 선택 시 기본 선택 수는 1개
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
        s = stats_from_distribution(probs, target=3)  # 가장 낮은 확률의 토큰
        expected = (s.logp - s.mu) / s.sigma
        self.assertAlmostEqual(L.likelihood_score_minkpp([s], 100.0), expected)

    def test_averages_only_the_lowest_k_percent(self):
        # z가 순서대로 증가하는 토큰 10개 구성
        stats = [L.TokenStats(logp=-float(10 - i), mu=0.0, sigma=1.0, top1=0.0)
                 for i in range(10)]
        # 하위 20%인 z=-10, -9 선택
        self.assertAlmostEqual(L.likelihood_score_minkpp(stats, 20.0), -9.5)
        self.assertAlmostEqual(L.likelihood_score_minkpp(stats, 100.0),
                               sum(-float(10 - i) for i in range(10)) / 10)

    def test_normalization_separates_equal_log_probs(self):
        flat = stats_from_distribution([0.2] * 5, target=0)
        peaked = stats_from_distribution([0.6, 0.2, 0.1, 0.05, 0.05], target=1)
        self.assertAlmostEqual(flat.logp, peaked.logp)  # 동일한 정답 토큰 확률
        self.assertGreater(L.likelihood_score_minkpp([flat], 100.0),
                           L.likelihood_score_minkpp([peaked], 100.0))  # 분포 모양에 따른 점수 차이

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

        s = stats_from_distribution([0.5, 0.2, 0.2, 0.1], target=0)
        self.assertAlmostEqual(L.likelihood_score_gapk([s], 100.0, window=1), 0.0)

    def test_gapk_is_never_positive(self):
        for target in range(4):
            s = stats_from_distribution([0.5, 0.2, 0.2, 0.1], target)
            self.assertLessEqual(L.likelihood_score_gapk([s], 100.0, window=1), 0.0)

    def test_sliding_window_averages_adjacent_tokens(self):
        gaps = [-4.0, 0.0, 0.0, 0.0]
        stats = [L.TokenStats(logp=g, mu=0.0, sigma=1.0, top1=0.0) for g in gaps]
        # 창 크기 2의 이동평균은 [-2, 0, 0], 하위 값은 -2
        self.assertAlmostEqual(
            L.likelihood_score_gapk(stats, k_percent=1.0, window=2), -2.0)
        self.assertAlmostEqual(
            L.likelihood_score_gapk(stats, k_percent=1.0, window=1), -4.0)

    def test_window_is_clamped_to_short_answers(self):

        stats = [L.TokenStats(logp=-2.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-4.0, mu=0.0, sigma=1.0, top1=0.0)]
        # 창 크기를 답 길이 2로 제한해 두 값의 평균 사용
        self.assertAlmostEqual(
            L.likelihood_score_gapk(stats, k_percent=100.0, window=6), -3.0)

    def test_distinguishes_confident_misprediction_from_uncertainty(self):
        flat = L.TokenStats(logp=-2.0, mu=-1.0, sigma=1.0, top1=-1.5)
        confident = L.TokenStats(logp=-2.0, mu=-1.0, sigma=1.0, top1=-0.1)
        self.assertAlmostEqual(L.likelihood_score_minkpp([flat], 100.0),
                               L.likelihood_score_minkpp([confident], 100.0))
        self.assertGreater(L.likelihood_score_gapk([flat], 100.0, window=1),
                           L.likelihood_score_gapk([confident], 100.0, window=1))


class MinTokensFloorTest(unittest.TestCase):

    def test_short_answer_without_floor_uses_one_token(self):
        # 토큰 3개에서 하위 20%는 최솟값 1개만 선택
        stats = [L.TokenStats(logp=-1.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-2.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-9.0, mu=0.0, sigma=1.0, top1=0.0)]  # 이상값
        self.assertAlmostEqual(L.likelihood_score_minkpp(stats, 20.0), -9.0)

    def test_min_tokens_floor_dilutes_the_outlier(self):
        stats = [L.TokenStats(logp=-1.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-2.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-9.0, mu=0.0, sigma=1.0, top1=0.0)]
        # 최솟값만 선택하지 않고 3개 모두 평균
        self.assertAlmostEqual(L.likelihood_score_minkpp(stats, 20.0, min_tokens=3),
                               (-1.0 - 2.0 - 9.0) / 3)

    def test_min_tokens_is_capped_to_available_tokens(self):
        stats = [L.TokenStats(logp=-1.0, mu=0.0, sigma=1.0, top1=0.0),
                 L.TokenStats(logp=-3.0, mu=0.0, sigma=1.0, top1=0.0)]
        # 최소 5개 요청에도 실제 토큰 수 2개로 제한
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

    def test_minkpp_exp_matches_exp_of_the_paper_formula_for_one_token(self):
        probs = [0.5, 0.2, 0.2, 0.1]
        s = stats_from_distribution(probs, target=3)
        raw_z = L.likelihood_score_minkpp([s], 100.0)
        self.assertAlmostEqual(L.likelihood_score_minkpp_exp([s], 100.0), math.exp(raw_z))

    def test_still_separates_equal_log_probs_by_distribution_shape(self):
        flat = stats_from_distribution([0.2] * 5, target=0)
        peaked = stats_from_distribution([0.6, 0.2, 0.1, 0.05, 0.05], target=1)
        self.assertAlmostEqual(flat.logp, peaked.logp)
        self.assertGreater(L.likelihood_score_minkpp_exp([flat], 100.0),
                           L.likelihood_score_minkpp_exp([peaked], 100.0))

    def test_bounds_a_sigma_collapse_that_explodes_the_raw_score(self):
        normal = L.TokenStats(logp=-1.0, mu=0.0, sigma=1.0, top1=0.0)
        collapsed = L.TokenStats(logp=-5.0, mu=-0.02, sigma=1e-4, top1=-0.01)  # z는 약 -49800
        raw = L.likelihood_score_minkpp([normal, collapsed], 100.0)
        self.assertLess(raw, -1000)  # 작은 표준편차에 따른 큰 음수 점수 확인
        bounded = L.likelihood_score_minkpp_exp([normal, collapsed], 100.0)
        self.assertGreaterEqual(bounded, 0.0)
        self.assertLess(bounded, 1.0)  # 두 토큰의 z가 음수이므로 exp(z)는 1 미만

    def test_gapk_exp_matches_exp_of_the_paper_formula(self):
        probs = [0.5, 0.2, 0.2, 0.1]
        s = stats_from_distribution(probs, target=3)
        raw_g = L.likelihood_score_gapk([s], 100.0, window=1)
        self.assertAlmostEqual(L.likelihood_score_gapk_exp([s], 100.0, window=1),
                               math.exp(raw_g))

    def test_gapk_exp_top1_token_scores_one(self):
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
            L.parse_aggregation("gapk", k=20.0)  # 필수 window 누락
        with self.assertRaises(ValueError):
            L.parse_aggregation("minkpp_exp")
        with self.assertRaises(ValueError):
            L.parse_aggregation("gapk_exp", k=20.0)  # 필수 window 누락
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
