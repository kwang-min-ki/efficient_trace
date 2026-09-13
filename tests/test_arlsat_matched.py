"""Regression tests for the matched AR-LSAT cumulative-prefix protocol.

These tests intentionally use fake generation/scoring backends.  They verify the
contexts handed to vLLM TRACE, HF TRACE, and HF Likelihood-TRACE without loading a
model, and make only the fifth cutoff successful so a copied terminal point cannot
masquerade as a real fifth evaluation.
"""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import model_config  # noqa: E402
import trace as trace_impl  # noqa: E402  (the repository's trace.py)
import likelihood_trace_hf as likelihood_impl  # noqa: E402
import trace_hf as hf_backend_impl  # noqa: E402
import arlsat as trace_hf_impl  # noqa: E402


PID = "fixture-1"
PROMPT = "PROMPT|"
# Distinct from PROMPT so tests catch any code path that reads the raw dataset
# sample["prompt"] instead of the generator's rendered ("<TPL>"-prefixed) prompt --
# an identity fake render() would not catch that regression.
RENDERED_PROMPT = "<TPL>" + PROMPT
COT = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 "
RESPONSE = COT + "<answer> 2 </answer>"
SAMPLE = {"pid": PID, "prompt": PROMPT, "variant": "clean"}
TARGETS = {PID: "2"}
SOURCE_RECORDS = [{"pid": PID, "response": RESPONSE}]
REASONING_PREFIXES = [
    "w1 ",
    "w1 w2 w3 ",
    "w1 w2 w3 w4 w5 ",
    "w1 w2 w3 w4 w5 w6 w7 ",
    "w1 w2 w3 w4 w5 w6 w7 w8 w9 ",
]
FULL_PREFIXES = [RENDERED_PROMPT + prefix for prefix in REASONING_PREFIXES]
BAD_TRACE_OUTPUTS = [[" 3 </answer>"] * 3 for _ in range(4)]
GOOD_TRACE_OUTPUTS = [[" 2 </answer>"] * 3]
TRACE_OUTPUTS = BAD_TRACE_OUTPUTS + GOOD_TRACE_OUTPUTS


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


class FakeVllmGenerator:
    """Return one source response, then cutoff-specific sampled answers."""

    def __init__(self, family=model_config.ModelFamily.QWEN3):
        self.model = "fake-model"
        self.tok = FakeTokenizer()
        self.profile = model_config.profile_for_family(family)
        self.calls = []

    def render(self, records):
        return [RENDERED_PROMPT for _ in records]

    def generate(self, prompts, n=1, temperature=0.7, max_tokens=1024, stop=None):
        self.calls.append({
            "prompts": list(prompts),
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
        })
        if len(self.calls) == 1:
            return [[RESPONSE] for _ in prompts]
        if len(self.calls) == 2:
            return [list(outputs) for outputs in TRACE_OUTPUTS]
        raise AssertionError("unexpected extra vLLM generation call")


class FakeHfGenerator(SimpleNamespace):
    def __init__(self, family=model_config.ModelFamily.QWEN3):
        super().__init__(
            model="fake-model",
            tok=FakeTokenizer(),
            net=object(),
            profile=model_config.profile_for_family(family),
        )

    def render(self, records):
        return [RENDERED_PROMPT for _ in records]

    def generate(self, *args, **kwargs):
        raise AssertionError("--records must prevent a new source rollout")


class ArlsatMatchedPrefixTest(unittest.TestCase):
    def test_hf_scorers_share_arlsat_response_preparation(self):
        self.assertIs(
            likelihood_impl.rollout_and_filter_arlsat,
            hf_backend_impl.rollout_and_filter_arlsat,
        )
        self.assertIs(trace_hf_impl.rollout_and_filter_arlsat,
                      hf_backend_impl.rollout_and_filter_arlsat)

    def test_word_prefix_helper_is_exactly_five_cumulative_prefixes(self):
        self.assertEqual(trace_impl.ARLSAT_FRACS, [0.1, 0.3, 0.5, 0.7, 0.9])
        self.assertEqual(trace_impl.arlsat_word_prefixes(COT), REASONING_PREFIXES)
        self.assertEqual(
            trace_impl.arlsat_word_prefixes(""),
            [""] * len(trace_impl.ARLSAT_FRACS),
        )
        self.assertEqual(
            trace_impl.arlsat_word_prefixes("only "),
            ["only "] * len(trace_impl.ARLSAT_FRACS),
        )
        for shorter, longer in zip(REASONING_PREFIXES, REASONING_PREFIXES[1:]):
            self.assertTrue(longer.startswith(shorter))

    def test_all_three_scorers_use_same_contexts_and_evaluate_fifth_cutoff(self):
        vllm_gen = FakeVllmGenerator()
        vllm_records, _, _ = trace_impl.score_arlsat(
            vllm_gen, [SAMPLE], TARGETS)

        self.assertEqual(len(vllm_gen.calls), 2)
        source_call, cutoff_call = vllm_gen.calls
        self.assertEqual(source_call["prompts"], [RENDERED_PROMPT])
        self.assertEqual(source_call["n"], 1)
        self.assertEqual(source_call["temperature"], 0.0)
        self.assertEqual(source_call["max_tokens"],
                         vllm_gen.profile.max_response_tokens("arlsat"))
        self.assertEqual(
            cutoff_call["prompts"],
            [prefix + trace_impl.ARLSAT_FORCE for prefix in FULL_PREFIXES],
        )
        self.assertEqual(cutoff_call["n"], 3)
        self.assertEqual(cutoff_call["temperature"], 0.7)
        self.assertEqual(vllm_records[0]["curve"], [0.0, 0.0, 0.0, 0.0, 1.0])

        hf_trace_calls = []

        def fake_trace_curve(tok, net, prefix_texts, force_text, n_samples,
                             temperature, max_new_tokens, stop):
            hf_trace_calls.append({
                "tok": tok,
                "net": net,
                "prefix_texts": list(prefix_texts),
                "force_text": force_text,
                "n_samples": n_samples,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop": stop,
            })
            return [list(outputs) for outputs in TRACE_OUTPUTS]

        hf_gen = FakeHfGenerator()
        hf_trace_records, _, _ = trace_hf_impl.score_arlsat_hf(
            hf_gen, [SAMPLE], TARGETS, source_records=SOURCE_RECORDS,
            trace_curve=fake_trace_curve,
        )

        self.assertEqual(len(hf_trace_calls), 1)
        hf_trace_call = hf_trace_calls[0]
        self.assertEqual(hf_trace_call["prefix_texts"], FULL_PREFIXES)
        self.assertEqual(hf_trace_call["force_text"], trace_impl.ARLSAT_FORCE)
        self.assertEqual(hf_trace_call["n_samples"], 3)
        self.assertEqual(hf_trace_call["temperature"], 0.7)
        self.assertEqual(hf_trace_records[0]["curve"], [0.0, 0.0, 0.0, 0.0, 1.0])

        likelihood_calls = []
        likelihood_probabilities = [0.1, 0.2, 0.3, 0.4, 0.9]

        def fake_likelihood_curve(tok, net, prefix_texts, force_text, answer_text):
            likelihood_calls.append({
                "tok": tok,
                "net": net,
                "prefix_texts": list(prefix_texts),
                "force_text": force_text,
                "answer_text": answer_text,
            })
            return [[math.log(probability)] for probability in likelihood_probabilities]

        likelihood_gen = FakeHfGenerator()
        with mock.patch.object(
                likelihood_impl,
                "likelihood_curve_exact_prefixes",
                side_effect=fake_likelihood_curve):
            likelihood_records, _, _ = likelihood_impl.score_arlsat(
                likelihood_gen,
                [SAMPLE],
                TARGETS,
                score_window="full",
                aggregation="mean",
                threshold=None,
                source_records=SOURCE_RECORDS,
            )

        self.assertEqual(len(likelihood_calls), 1)
        likelihood_call = likelihood_calls[0]
        self.assertEqual(likelihood_call["prefix_texts"], FULL_PREFIXES)
        self.assertEqual(likelihood_call["force_text"], trace_impl.ARLSAT_FORCE)
        self.assertEqual(likelihood_call["answer_text"], " 2")
        for actual, expected in zip(
                likelihood_records[0]["curve"], likelihood_probabilities):
            self.assertAlmostEqual(actual, expected)

        # Compare what each backend actually received, not merely helper output.
        vllm_prefixes = [
            prompt[:-len(trace_impl.ARLSAT_FORCE)]
            for prompt in cutoff_call["prompts"]
        ]
        self.assertEqual(vllm_prefixes, hf_trace_call["prefix_texts"])
        self.assertEqual(vllm_prefixes, likelihood_call["prefix_texts"])
        self.assertEqual(vllm_prefixes, FULL_PREFIXES)
        self.assertTrue(all("<answer>" not in prefix for prefix in vllm_prefixes))

        expected_hashes = [trace_impl.prefix_sha256(prefix) for prefix in FULL_PREFIXES]
        for record in (vllm_records[0], hf_trace_records[0], likelihood_records[0]):
            self.assertEqual(record["curve"].__len__(), len(trace_impl.ARLSAT_FRACS))
            self.assertEqual(record["prefix_protocol"], trace_impl.ARLSAT_PREFIX_PROTOCOL)
            self.assertEqual(record["cutoff_ratios"], trace_impl.ARLSAT_FRACS)
            self.assertEqual(record["cutoff_unit"], "cumulative_words")
            self.assertEqual(record["cutoff_prefix_sha256"], expected_hashes)
            self.assertNotIn("terminal_point", record)

        self.assertEqual(vllm_records[0]["protocol"], trace_impl.ARLSAT_TRACE_PROTOCOL)
        self.assertEqual(hf_trace_records[0]["protocol"], trace_impl.ARLSAT_TRACE_PROTOCOL)
        self.assertEqual(vllm_records[0]["decoded_cutoffs"], 5)
        self.assertEqual(hf_trace_records[0]["decoded_cutoffs"], 5)
        self.assertEqual(vllm_records[0]["per_cutoff_scoring_rows"], 3)
        self.assertEqual(hf_trace_records[0]["per_cutoff_scoring_rows"], 3)
        self.assertEqual(likelihood_records[0]["per_cutoff_scoring_rows"], 1)

    def test_rollout_budget_comes_from_the_generator_family(self):
        """AR-LSAT budgets are per-family (model_config), not a shared constant."""

        budgets = {}
        for family in (model_config.ModelFamily.QWEN2, model_config.ModelFamily.QWEN3):
            gen = FakeVllmGenerator(family)
            trace_impl.score_arlsat(gen, [SAMPLE], TARGETS)
            budgets[family] = gen.calls[0]["max_tokens"]

        self.assertEqual(budgets[model_config.ModelFamily.QWEN2], 1024)
        self.assertEqual(budgets[model_config.ModelFamily.QWEN3], 4096)


if __name__ == "__main__":
    unittest.main()
