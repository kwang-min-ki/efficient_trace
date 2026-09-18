"""Memorization pair loading, accuracy summaries, and threshold metrics."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_memorization_math import (  # noqa: E402
    build_final_labels,
    detection_report,
    load_samples,
    summarize_accuracy,
)


class MemorizationInferenceTest(unittest.TestCase):
    def test_pair_limit_keeps_both_members(self):
        rows = []
        for pair in range(3):
            for split in ("seen", "unseen"):
                rows.append({
                    "pid": f"{split}-{pair}",
                    "pair_id": f"pair-{pair}",
                    "split": split,
                })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.clean.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            selected = load_samples(directory, limit_per_split=2, seed=4)
        self.assertEqual(len(selected), 4)
        counts = {
            split: sum(row["split"] == split for row in selected)
            for split in ("seen", "unseen")
        }
        self.assertEqual(counts, {"seen": 2, "unseen": 2})
        self.assertEqual(len({row["pair_id"] for row in selected}), 2)

    def test_accuracy_and_untrained_threshold(self):
        accuracy = summarize_accuracy([
            {"oracle_correct": True},
            {"oracle_correct": False},
        ])
        self.assertEqual(accuracy, {"n": 2, "correct": 1, "accuracy": 0.5})

        baseline = [{"pid": "b1", "auc": 0.2}, {"pid": "b2", "auc": 0.4}]
        trained = [{"pid": "hack", "auc": 0.8}, {"pid": "clean", "auc": 0.1}]
        report = detection_report(
            baseline, trained, {"hack": True, "clean": False}
        )
        self.assertAlmostEqual(report["threshold"], 0.3)
        self.assertEqual(report["f1"], 1.0)
        self.assertEqual(report["accuracy"], 1.0)

    def test_strict_label_requires_baseline_failure_and_counterfactual_failure(self):
        samples = [
            {"pid": "seen", "pair_id": "p", "split": "seen"},
            {"pid": "unseen", "pair_id": "p", "split": "unseen"},
        ]
        labels = build_final_labels(
            samples,
            [{"pid": "seen", "oracle_correct": False}],
            [
                {"pid": "seen", "oracle_correct": True},
                {"pid": "unseen", "oracle_correct": False},
            ],
            [{"pid": "seen", "counterfactual_correct": False}],
        )
        self.assertEqual(len(labels), 1)
        self.assertTrue(labels[0]["counterfactual_hacking"])
        self.assertTrue(labels[0]["induced_memorization_proxy"])
        self.assertTrue(labels[0]["strict_memorization_hacking"])
        self.assertEqual(labels[0]["pair_oracle_gap"], 1)


if __name__ == "__main__":
    unittest.main()
