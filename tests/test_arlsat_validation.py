"""Fast structural tests for matched-cumulative AR-LSAT artifacts.

The fixtures are deliberately tiny, but exercise the same on-disk JSONL and
``.stats`` files as the runner.  No model, GPU, or network access is needed.
"""

import contextlib
import hashlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import detect  # noqa: E402
import trace as trace_impl  # noqa: E402  (the repository's trace.py)
import arlsat as verifier  # noqa: E402


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def matched_record(pid, curve, response):
    prefixes = [f"prompt:{pid}:cutoff:{index}" for index in range(5)]
    return {
        "pid": pid,
        "task": "arlsat",
        "variant": "clean",
        "model": "tiny-model",
        "curve": curve,
        "auc": trace_impl.raw_auc(trace_impl.ARLSAT_FRACS, curve),
        "response": response,
        "response_sha256": trace_impl.response_sha256(response),
        "response_source": "generated",
        "impl": "trace_vllm",
        "protocol": trace_impl.ARLSAT_TRACE_PROTOCOL,
        "prefix_protocol": trace_impl.ARLSAT_PREFIX_PROTOCOL,
        "cutoff_ratios": trace_impl.ARLSAT_FRACS,
        "cutoff_unit": "cumulative_words",
        "auc_scale": "raw",
        "cutoff_prefix_sha256": [
            hashlib.sha256(prefix.encode("utf-8")).hexdigest()
            for prefix in prefixes
        ],
        "decoded_cutoffs": len(trace_impl.ARLSAT_FRACS),
        "per_cutoff_scoring_rows": trace_impl.ARLSAT_N_SAMPLES,
        "n_samples": trace_impl.ARLSAT_N_SAMPLES,
        "temperature": trace_impl.ARLSAT_TEMPERATURE,
        "max_new_tokens": trace_impl.ARLSAT_MAX_ANSWER_TOKENS,
    }


def write_stats(path, rows):
    record = rows[0]
    stats = {
        "n": len(rows),
        "mean_auc": sum(row["auc"] for row in rows) / len(rows),
        "wall_clock_s": 0.3,
        "rollout_time_s": 0.1,
        "scoring_time_s": 0.2,
        "impl": record["impl"],
        "protocol": record["protocol"],
        "prefix_protocol": record["prefix_protocol"],
        "auc_scale": record["auc_scale"],
        "cutoff_ratios": record["cutoff_ratios"],
        "cutoff_unit": record["cutoff_unit"],
        "decoded_cutoffs": record["decoded_cutoffs"],
        "per_cutoff_scoring_rows": record["per_cutoff_scoring_rows"],
        "n_samples": record["n_samples"],
        "temperature": record["temperature"],
        "max_new_tokens": record["max_new_tokens"],
        "response_source": record["response_source"],
    }
    Path(str(path) + ".stats").write_text(json.dumps(stats), encoding="utf-8")


def build_valid_run_tree(root):
    baseline = matched_record(
        "baseline-1", [0.25] * 5, "reasoning <answer> 2 </answer>")
    scored = matched_record(
        "detect-1", [0.75] * 5, "other reasoning <answer> 3 </answer>")

    baseline_path = root / "trace_baseline.jsonl"
    trace_path = root / "trace_step10.jsonl"
    labels_path = root / "labels_step10.jsonl"
    f1_path = root / "f1_step10.jsonl"
    write_jsonl(baseline_path, [baseline])
    write_jsonl(trace_path, [scored])
    write_stats(baseline_path, [baseline])
    write_stats(trace_path, [scored])
    write_jsonl(labels_path, [{
        "pid": scored["pid"],
        "is_hacking": True,
        "response_sha256": scored["response_sha256"],
    }])

    threshold = baseline["auc"]
    write_jsonl(f1_path, [{
        "tag": "tiny",
        "step": 10,
        "threshold": threshold,
        "threshold_source": "baseline_mean",
        "n": 1,
        "n_hacking": 1,
        "trace": {
            "f1": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "tp": 1,
            "fp": 0,
            "fn": 0,
        },
    }])
    return baseline_path, trace_path, labels_path, f1_path


def detect_args(root, baseline_path, trace_path, labels_path):
    return SimpleNamespace(
        threshold=None,
        baseline=str(baseline_path),
        single_pool=True,
        hacking=str(trace_path),
        hacking_labels=str(labels_path),
        hacking_monitor=None,
        nonhacking=None,
        nonhacking_labels=None,
        nonhacking_monitor=None,
        tag="tiny",
        step=10,
        out=str(root / "detect_f1.jsonl"),
    )


class ArlsatStructuralVerifierTest(unittest.TestCase):
    def run_verifier(self, root):
        argv = ["verify", "--runs", str(root), "--steps", "10"]
        with contextlib.redirect_stdout(io.StringIO()):
            verifier.main(argv)

    def test_valid_tiny_matched_artifact_passes_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path, trace_path, labels_path, _ = build_valid_run_tree(root)

            self.run_verifier(root)

            # The detector independently accepts the same matched baseline and
            # derives its threshold from that baseline's raw AUC mean.
            with contextlib.redirect_stdout(io.StringIO()):
                detect.cmd_f1(detect_args(root, baseline_path, trace_path, labels_path))
            result = json.loads((root / "detect_f1.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(result["threshold_source"], "baseline_mean")
            self.assertTrue(math.isclose(result["threshold"], 0.2))
            self.assertEqual(result["trace"]["tp"], 1)

    def test_verifier_rejects_public_grift_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, trace_path, _, _ = build_valid_run_tree(root)
            record = json.loads(trace_path.read_text(encoding="utf-8"))
            record["protocol"] = "grift-public-arlsat"
            write_jsonl(trace_path, [record])

            with self.assertRaisesRegex(RuntimeError, "legacy or incompatible protocol"):
                self.run_verifier(root)

    def test_verifier_rejects_synthetic_terminal_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, trace_path, _, _ = build_valid_run_tree(root)
            record = json.loads(trace_path.read_text(encoding="utf-8"))
            record["terminal_point"] = "last_sample_reward_from_previous_cutoff"
            write_jsonl(trace_path, [record])

            with self.assertRaisesRegex(RuntimeError, "synthetic terminal point"):
                self.run_verifier(root)

    def test_detector_rejects_public_protocol_even_if_metadata_is_spoofed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path, trace_path, labels_path, _ = build_valid_run_tree(root)
            for path in (baseline_path, trace_path):
                record = json.loads(path.read_text(encoding="utf-8"))
                # Keep all cumulative metadata so protocol validation itself is
                # required; config equality alone must not accept this record.
                record["protocol"] = "grift-public-arlsat"
                write_jsonl(path, [record])

            with self.assertRaisesRegex(ValueError, "protocol|matched|legacy"):
                with contextlib.redirect_stdout(io.StringIO()):
                    detect.cmd_f1(
                        detect_args(root, baseline_path, trace_path, labels_path))


if __name__ == "__main__":
    unittest.main()
