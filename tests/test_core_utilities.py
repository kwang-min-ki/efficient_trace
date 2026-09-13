"""Fast tests for shared data and likelihood-scoring helpers."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import arlsat  # noqa: E402
import detect  # noqa: E402
import likelihood_trace_hf  # noqa: E402
import reward  # noqa: E402
import trace  # noqa: E402
import trace_hf  # noqa: E402
from data import targets_for  # noqa: E402
from likelihood_trace_hf import (  # noqa: E402
    likelihood_score,
    likelihood_score_hybrid,
    likelihood_score_max,
    parse_aggregation,
    parse_score_window,
)
from trace_hf import _longest_common_prefix_len  # noqa: E402


class TargetLoadingTest(unittest.TestCase):
    def test_math_and_arlsat_targets_come_from_prompt_records(self):
        records = [
            {"pid": "one", "gold": "7"},
            {"pid": "two", "gold": "3"},
        ]
        expected = {"one": "7", "two": "3"}
        self.assertEqual(targets_for("math", "unused", records), expected)
        self.assertEqual(targets_for("arlsat", "unused", records), expected)

    def test_code_targets_are_filtered_to_selected_problem_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problems = [
                {"pid": "keep", "tests": {"inputs": ["1"], "outputs": ["2"]}},
                {"pid": "skip", "tests": {"inputs": ["3"], "outputs": ["4"]}},
            ]
            (root / "problems.jsonl").write_text(
                "".join(json.dumps(problem) + "\n" for problem in problems),
                encoding="utf-8",
            )
            self.assertEqual(
                targets_for("code", root, [{"pid": "keep"}]),
                {"keep": problems[0]["tests"]},
            )


class LikelihoodHelperTest(unittest.TestCase):
    def test_longest_common_prefix_length(self):
        self.assertEqual(_longest_common_prefix_len([], []), 0)
        self.assertEqual(_longest_common_prefix_len([1, 2], [1, 3]), 1)
        self.assertEqual(_longest_common_prefix_len([1, 2], [1, 2, 3]), 2)

    def test_score_windows_and_aggregations(self):
        logprobs = [math.log(0.25), math.log(0.81)]
        self.assertEqual(parse_score_window("full")(logprobs), logprobs)
        self.assertEqual(parse_score_window("first_1")(logprobs), logprobs[:1])
        self.assertAlmostEqual(likelihood_score(logprobs), 0.45)
        self.assertAlmostEqual(likelihood_score_max(logprobs), 0.81)
        self.assertAlmostEqual(likelihood_score_hybrid(logprobs, 0.8), 0.81)
        self.assertAlmostEqual(likelihood_score_hybrid(logprobs, 0.9), 0.45)
        self.assertIs(parse_aggregation("mean"), likelihood_score)
        self.assertIs(parse_aggregation("max"), likelihood_score_max)

    def test_invalid_score_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_score_window("first_0")
        with self.assertRaises(ValueError):
            parse_score_window("unknown")
        with self.assertRaises(ValueError):
            parse_aggregation("hybrid")
        with self.assertRaises(ValueError):
            parse_aggregation("unknown")


class PublicImportTest(unittest.TestCase):
    def test_arlsat_public_names_share_the_consolidated_implementation(self):
        self.assertIs(trace.arlsat_word_prefixes, arlsat.arlsat_word_prefixes)
        self.assertIs(trace.score_arlsat, arlsat.score_arlsat_vllm)
        self.assertIs(
            trace_hf.rollout_and_filter_arlsat,
            arlsat.rollout_and_filter_arlsat,
        )
        self.assertIs(
            likelihood_trace_hf.rollout_and_filter_arlsat,
            arlsat.rollout_and_filter_arlsat,
        )
        self.assertIs(reward.arlsat_proxy, arlsat.arlsat_proxy)
        self.assertIs(
            detect._validate_arlsat_score_record,
            arlsat._validate_arlsat_score_record,
        )


if __name__ == "__main__":
    unittest.main()
