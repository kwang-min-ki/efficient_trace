"""모델 계열·프롬프트·응답 분리·샘플링·종료 토큰 설정 검증"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_config as M


class StubTokenizer:

    def __init__(self, family=M.ModelFamily.QWEN2, rendered="RENDERED"):
        self.family = family
        self.rendered = rendered
        self.seen = None
        self._vocab = {"a": 0}

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
        self.assertIs(M.family_from_model_type("qwen2"), M.ModelFamily.QWEN2)
        self.assertIs(M.family_from_model_type("llama"), M.ModelFamily.LLAMA)

    def test_unsupported_family_rejected(self):
        with self.assertRaises(M.UnsupportedModelError):
            M.family_from_model_type("unknown")

    def test_detect_accepts_a_config_mapping(self):
        self.assertIs(M.detect_model_family({"model_type": "qwen2"}), M.ModelFamily.QWEN2)


class TestProfiles(unittest.TestCase):

    def test_qwen2_has_no_thinking_switch_but_keeps_prefill(self):
        p = M.profile_for_family(M.ModelFamily.QWEN2)
        self.assertEqual(p.chat_template_kwargs, {})
        self.assertTrue(p.legacy_prefill)

    def test_response_budget_follows_the_model(self):
        for family in M.ModelFamily:
            profile = M.profile_for_family(family)
            self.assertEqual((profile.max_response_tokens("math"),
                              profile.max_response_tokens("code")), (1024, 600))
            with self.assertRaises(ValueError):
                profile.max_response_tokens("not-a-task")

    def test_every_task_is_budgeted_for_every_family(self):
        for family in (M.ModelFamily.QWEN2, M.ModelFamily.LLAMA):
            profile = M.profile_for_family(family)
            for task in profile.response_budget:
                self.assertIsInstance(profile.max_response_tokens(task), int)

    def test_sampling_is_protocol_not_model(self):
        # 두 모델 계열에 동일한 샘플링 분포 적용
        self.assertIs(M.profile_for_family(M.ModelFamily.LLAMA).sampling,
                      M.profile_for_family(M.ModelFamily.QWEN2).sampling)
        self.assertEqual(M.PROTOCOL_SAMPLING.as_kwargs(),
                         {"top_p": 1.0, "top_k": 0, "min_p": 0.0})


class TestPromptRendering(unittest.TestCase):

    def test_qwen2_never_passes_enable_thinking(self):
        tok = StubTokenizer(M.ModelFamily.QWEN2)
        M.render_chat_prompt(tok, MESSAGES, M.ModelFamily.QWEN2)
        self.assertNotIn("enable_thinking", tok.seen)

    def test_qwen2_appends_the_legacy_prefill_by_default(self):
        tok = StubTokenizer(M.ModelFamily.QWEN2)
        out = M.render_chat_prompt(tok, MESSAGES, M.ModelFamily.QWEN2)
        self.assertTrue(out.endswith(M.LEGACY_ASSISTANT_PREFILL))


    def test_prefill_can_be_suppressed_for_a_judge(self):
        tok = StubTokenizer(M.ModelFamily.QWEN2)
        out = M.render_chat_prompt(tok, MESSAGES, M.ModelFamily.QWEN2,
                                   legacy_assistant_prefill=False)
        self.assertFalse(out.endswith(M.LEGACY_ASSISTANT_PREFILL))


class TestResponseSplitting(unittest.TestCase):
    def test_generated_opening_marker_leaves_the_reasoning_span(self):
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
        # 첫 종료 태그 기준의 math/code 분리 동작 유지
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
        p = M.profile_for_family(M.ModelFamily.LLAMA)
        self.assertEqual(M.sampling_kwargs(p, 0.0, backend="hf"), {"do_sample": False})

    def test_sampled_kwargs_state_the_filters_explicitly(self):
        p = M.profile_for_family(M.ModelFamily.LLAMA)
        self.assertEqual(
            M.sampling_kwargs(p, 0.7, backend="vllm"),
            {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "min_p": 0.0})


class TestVerlOverrides(unittest.TestCase):

    def test_qwen2_only_gets_its_budget(self):
        self.assertEqual(M.verl_hydra_overrides(M.ModelFamily.QWEN2, task="code"),
                         ["data.max_response_length=600"])

    def test_rollout_sampling_is_never_overridden(self):
        # 학습 샘플링은 verl 기본값 유지
        for family in (M.ModelFamily.QWEN2, M.ModelFamily.LLAMA):
            joined = " ".join(M.verl_hydra_overrides(family, task="math"))
            for key in ("temperature", "top_p", "top_k", "do_sample"):
                self.assertNotIn(key, joined)


class TestNewInstructModels(unittest.TestCase):
    def test_native_template_and_prefill_without_special_think_tokens(self):
        for name, family in (("llama", M.ModelFamily.LLAMA),):
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
