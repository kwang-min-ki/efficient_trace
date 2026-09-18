"""math/code 정답 로딩, 토큰 접두사 비교와 Likelihood 집계 설정 검증"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import build_memorization_split, read_jsonl, targets_for  # noqa: E402
from likelihood_trace import (  # noqa: E402
    likelihood_score,
    likelihood_score_hybrid,
    likelihood_score_max,
    parse_aggregation,
    parse_score_window,
)
from generation import _longest_common_prefix_len  # noqa: E402


class TargetLoadingTest(unittest.TestCase):
    def test_math_targets_come_from_prompt_records(self):
        records = [
            {"pid": "one", "gold": "7"},
            {"pid": "two", "gold": "3"},
        ]
        expected = {"one": "7", "two": "3"}
        self.assertEqual(targets_for("math", "unused", records), expected)

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


class MemorizationSplitTest(unittest.TestCase):
    def test_seen_is_trained_and_unseen_is_held_out(self):
        problems = [
            {"pid": "t1", "task": "math", "split": "train", "source": "a",
             "question": "short", "gold": "1"},
            {"pid": "t2", "task": "math", "split": "train", "source": "b",
             "question": "a longer question", "gold": "2"},
            {"pid": "v1", "task": "math", "split": "val", "source": "a",
             "question": "shorter", "gold": "3"},
            {"pid": "h1", "task": "math", "split": "heldout", "source": "b",
             "question": "another long question", "gold": "4"},
        ]
        prompts = [
            {"pid": p["pid"], "task": "math", "variant": "clean",
             "split": p["split"], "source": p["source"], "question": p["question"],
             "gold": p["gold"], "loophole": "clean", "messages": [], "prompt": ""}
            for p in problems
        ]
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                build_memorization_split(problems, prompts, directory, seed=7), 2
            )
            rows = list(read_jsonl(Path(directory) / "memorization/prompts.clean.jsonl"))
            seen = {r["pid"] for r in rows if r["split"] == "seen"}
            unseen = {r["pid"] for r in rows if r["split"] == "unseen"}
            self.assertEqual(seen, {"t1", "t2"})
            self.assertEqual(unseen, {"v1", "h1"})
            self.assertTrue(seen.isdisjoint(unseen))
            self.assertTrue(all(r["original_split"] == "train"
                                for r in rows if r["membership"] == "seen"))
            self.assertTrue(all(r["original_split"] != "train"
                                for r in rows if r["membership"] == "unseen"))


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


if __name__ == "__main__":
    unittest.main()
