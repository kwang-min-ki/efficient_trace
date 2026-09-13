"""Check masked cutoff likelihoods against uncached forwards on real architectures.

Tiny random CPU models exercise Llama/Phi attention and cache code without weights.
"""
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Phi3Config, Phi3ForCausalLM

from likelihood_trace_hf import likelihood_curve_cached


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [4, 5] if text == 'force' else [4, 5, 6, 7]


class InstructCacheTest(unittest.TestCase):
    def test_masked_cache_matches_independent_prefix_forward(self):
        torch.manual_seed(42)
        for config_cls, model_cls in ((LlamaConfig, LlamaForCausalLM),
                                       (Phi3Config, Phi3ForCausalLM)):
            with self.subTest(architecture=model_cls.__name__):
                config = config_cls(vocab_size=32, hidden_size=32,
                                    intermediate_size=64, num_hidden_layers=2,
                                    num_attention_heads=4, num_key_value_heads=2,
                                    max_position_embeddings=128,
                                    pad_token_id=0, bos_token_id=1, eos_token_id=2)
                config._attn_implementation = 'eager'
                net = model_cls(config).eval()
                prompt, cot = [1, 8, 9], [10, 11, 12, 13]
                cached = likelihood_curve_cached(Tokenizer(), net, prompt, cot,
                                                  'force', 'answer', [0.25, 0.5, 1.0])
                for row, length in zip(cached, (1, 2, 4)):
                    prefix = prompt + cot[:length]
                    with torch.inference_mode():
                        logits = net(torch.tensor([prefix + [4, 5, 6, 7]])).logits
                    expected = torch.log_softmax(logits[0, len(prefix)+1:len(prefix)+3], -1)
                    expected = expected.gather(-1, torch.tensor([[6], [7]])).squeeze(-1)
                    torch.testing.assert_close(torch.tensor(row), expected, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
