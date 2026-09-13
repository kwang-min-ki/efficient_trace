"""Model-layer regression tests: no GPU, no model download, no network.

These pin the behaviors that differ between Qwen2.5 and Qwen3 and that the rest of
the pipeline now depends on. Tokenizer-dependent checks use a small stub rather than
a real checkpoint so the fast suite stays offline.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_config as M


class StubTokenizer:
    """Minimal tokenizer: Qwen3 owns <think>/</think> as single tokens, Qwen2.5 does not."""

    def __init__(self, family=M.ModelFamily.QWEN3, rendered="RENDERED"):
        self.family = family
        self.rendered = rendered
        self.seen = None
        self._vocab = {"a": 0}
        if family is M.ModelFamily.QWEN3:
            self._vocab.update({M.THINK_START: 151667, M.THINK_END: 151668})

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):
        if text in self._vocab:
            return [self._vocab[text]]
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, **kwargs):
        self.seen = kwargs
        return self.rendered


MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


class TestFamilyDetection(unittest.TestCase):
    def test_model_type_is_authoritative(self):
        self.assertIs(M.family_from_model_type("qwen3"), M.ModelFamily.QWEN3)
        self.assertIs(M.family_from_model_type("qwen2"), M.ModelFamily.QWEN2)
        self.assertIs(M.family_from_model_type("Qwen3_MoE"), M.ModelFamily.QWEN3)

    def test_unsupported_family_rejected(self):
        with self.assertRaises(M.UnsupportedModelError):
            M.family_from_model_type("unknown")

    def test_detect_accepts_a_config_mapping(self):
        self.assertIs(M.detect_model_family({"model_type": "qwen3"}), M.ModelFamily.QWEN3)


class TestProfiles(unittest.TestCase):
    def test_qwen3_states_thinking_and_skips_prefill(self):
        p = M.profile_for_family(M.ModelFamily.QWEN3)
        self.assertEqual(p.chat_template_kwargs, {"enable_thinking": True})
        self.assertFalse(p.legacy_prefill)

    def test_qwen2_has_no_thinking_switch_but_keeps_prefill(self):
        p = M.profile_for_family(M.ModelFamily.QWEN2)
        self.assertEqual(p.chat_template_kwargs, {})
        self.assertTrue(p.legacy_prefill)

    def test_response_budget_follows_the_model(self):
        q3 = M.profile_for_family(M.ModelFamily.QWEN3)
        q2 = M.profile_for_family(M.ModelFamily.QWEN2)
        for task in ("math", "code"):
            with self.assertRaises(ValueError):
                q3.max_response_tokens(task)
        self.assertEqual((q2.max_response_tokens("math"), q2.max_response_tokens("code")),
                         (1024, 600))
        # AR-LSAT is a first-class task, budgeted per family like math and code.
        self.assertEqual(q3.max_response_tokens("arlsat"), 4096)
        self.assertEqual(q2.max_response_tokens("arlsat"), 1024)
        with self.assertRaises(ValueError):
            q3.max_response_tokens("not-a-task")

    def test_every_task_is_budgeted_for_every_family(self):
        for family in (M.ModelFamily.QWEN2, M.ModelFamily.QWEN3):
            profile = M.profile_for_family(family)
            for task in profile.response_budget:
                self.assertIsInstance(profile.max_response_tokens(task), int)

    def test_sampling_is_protocol_not_model(self):
        # Both families draw from the same distribution on purpose; Qwen3's own
        # recommendation is recorded but deliberately unused.
        self.assertIs(M.profile_for_family(M.ModelFamily.QWEN3).sampling,
                      M.profile_for_family(M.ModelFamily.QWEN2).sampling)
        self.assertEqual(M.PROTOCOL_SAMPLING.as_kwargs(),
                         {"top_p": 1.0, "top_k": 0, "min_p": 0.0})
        self.assertNotEqual(M.QWEN3_RECOMMENDED_SAMPLING, M.PROTOCOL_SAMPLING)


class TestPromptRendering(unittest.TestCase):
    def test_qwen3_always_passes_enable_thinking(self):
        tok = StubTokenizer()
        M.render_chat_prompt(tok, MESSAGES, M.ModelFamily.QWEN3)
        self.assertEqual(tok.seen["enable_thinking"], True)
        M.render_chat_prompt(tok, MESSAGES, M.ModelFamily.QWEN3, thinking=False)
        self.assertEqual(tok.seen["enable_thinking"], False)

    def test_qwen2_never_passes_enable_thinking(self):
        tok = StubTokenizer(M.ModelFamily.QWEN2)
        M.render_chat_prompt(tok, MESSAGES, M.ModelFamily.QWEN2)
        self.assertNotIn("enable_thinking", tok.seen)

    def test_qwen2_appends_the_legacy_prefill_by_default(self):
        tok = StubTokenizer(M.ModelFamily.QWEN2)
        out = M.render_chat_prompt(tok, MESSAGES, M.ModelFamily.QWEN2)
        self.assertTrue(out.endswith(M.LEGACY_ASSISTANT_PREFILL))

    def test_qwen3_refuses_the_legacy_prefill(self):
        tok = StubTokenizer()
        with self.assertRaises(ValueError):
            M.render_chat_prompt(tok, MESSAGES, M.ModelFamily.QWEN3,
                                 legacy_assistant_prefill=True)

    def test_prefill_can_be_suppressed_for_a_judge(self):
        tok = StubTokenizer(M.ModelFamily.QWEN2)
        out = M.render_chat_prompt(tok, MESSAGES, M.ModelFamily.QWEN2,
                                   legacy_assistant_prefill=False)
        self.assertFalse(out.endswith(M.LEGACY_ASSISTANT_PREFILL))

    def test_qwen3_tokenizer_must_own_the_think_markers(self):
        with self.assertRaises(M.Qwen3TokenizerError):
            M.render_chat_prompt(StubTokenizer(M.ModelFamily.QWEN2), MESSAGES,
                                 M.ModelFamily.QWEN3)


class TestResponseSplitting(unittest.TestCase):
    def test_qwen3_opening_marker_leaves_the_reasoning_span(self):
        parts = M.split_reasoning_response("<think>\nR</think>\n<answer>1</answer>",
                                           prompt="P")
        self.assertEqual(parts.opening_marker, "<think>")
        self.assertEqual(parts.prefix_before_cot, "P<think>")
        self.assertEqual(parts.reasoning, "\nR")
        self.assertNotIn(M.THINK_START, parts.reasoning)

    def test_qwen2_prefilled_marker_is_already_outside_the_span(self):
        parts = M.split_reasoning_response("\nR</think>\n<answer>1</answer>", prompt="P")
        self.assertEqual(parts.opening_marker, "")
        self.assertEqual(parts.prefix_before_cot, "P")
        self.assertEqual(parts.reasoning, "\nR")

    def test_round_trip_is_exact(self):
        prompt, response = "P", "<think>\nR</think>\ntail"
        parts = M.split_reasoning_response(response, prompt=prompt)
        self.assertEqual(
            parts.prefix_before_cot + parts.reasoning + M.THINK_END + parts.after_think,
            prompt + response)

    def test_missing_close_marker_is_rejected(self):
        self.assertIsNone(M.split_reasoning_response("no marker", prompt="P"))

    def test_splits_on_the_first_close_marker(self):
        # Historical math/code behavior; Qwen3 emits exactly one boundary anyway.
        parts = M.split_reasoning_response("<think>A</think>B</think>C", prompt="")
        self.assertEqual(parts.reasoning, "A")


class TestGenerationSettings(unittest.TestCase):
    def test_eos_prefers_generation_config_and_keeps_every_id(self):
        class Net:
            class generation_config:
                eos_token_id = [151645, 151643]
                pad_token_id = 151643
            class config:
                eos_token_id = 151645

        class Tok:
            eos_token_id = 151645
            pad_token_id = None

        ids = M.resolve_generation_token_ids(Tok(), Net())
        self.assertEqual(ids.eos, (151645, 151643))
        self.assertEqual(ids.eos_for_backend, [151645, 151643])
        self.assertEqual(ids.pad, 151643)

    def test_greedy_disables_sampling(self):
        p = M.profile_for_family(M.ModelFamily.QWEN3)
        self.assertEqual(M.sampling_kwargs(p, 0.0, backend="hf"), {"do_sample": False})

    def test_sampled_kwargs_state_the_filters_explicitly(self):
        p = M.profile_for_family(M.ModelFamily.QWEN3)
        self.assertEqual(
            M.sampling_kwargs(p, 0.7, backend="vllm"),
            {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "min_p": 0.0})


class TestVerlOverrides(unittest.TestCase):
    def test_qwen3_states_thinking_and_its_budget(self):
        self.assertEqual(
            M.verl_hydra_overrides(M.ModelFamily.QWEN3, task="arlsat"),
            ["+data.apply_chat_template_kwargs.enable_thinking=true",
             "data.max_response_length=4096"])

    def test_qwen2_only_gets_its_budget(self):
        self.assertEqual(M.verl_hydra_overrides(M.ModelFamily.QWEN2, task="code"),
                         ["data.max_response_length=600"])

    def test_rollout_sampling_is_never_overridden(self):
        # Training sampling stays a protocol constant at verl's defaults.
        for family in (M.ModelFamily.QWEN2, M.ModelFamily.QWEN3):
            joined = " ".join(M.verl_hydra_overrides(family, task="arlsat"))
            for key in ("temperature", "top_p", "top_k", "do_sample"):
                self.assertNotIn(key, joined)



class TestNewInstructModels(unittest.TestCase):
    def test_native_template_and_prefill_without_special_think_tokens(self):
        for name, family in (("llama", M.ModelFamily.LLAMA), ("phi3", M.ModelFamily.PHI3)):
            with self.subTest(model_type=name):
                profile = M.get_model_profile({"model_type": name})
                self.assertIs(profile.family, family)
                tok = StubTokenizer(family, rendered="NATIVE_ASSISTANT_HEADER")
                prompt = M.render_chat_prompt(tok, MESSAGES, profile)
                self.assertEqual(prompt, "NATIVE_ASSISTANT_HEADER" + M.LEGACY_ASSISTANT_PREFILL)
                self.assertEqual(tok.seen, {"tokenize": False, "add_generation_prompt": True})
                for task, budget in (("math", 1024), ("code", 600)):
                    self.assertEqual(M.verl_hydra_overrides(profile, task=task),
                                     [f"data.max_response_length={budget}"])
                self.assertEqual(M.render_chat_prompt(tok, MESSAGES, profile,
                                 legacy_assistant_prefill=False), "NATIVE_ASSISTANT_HEADER")

if __name__ == "__main__":
    unittest.main()
