"""소형 CPU 모델 생성·길이 제한과 라벨링·모니터 연결 검증"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import detect
import generation
import model_config
from data import read_jsonl, write_jsonl


class GenerationCheck(unittest.TestCase):
    def setUp(self):
        raw = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2,
                                   'hello': 3, 'world': 4}, unk_token='[UNK]'))
        raw.pre_tokenizer = Whitespace()
        self.tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]',
                                          pad_token='[PAD]', eos_token='[EOS]')
        torch.manual_seed(7)
        self.net = LlamaForCausalLM(LlamaConfig(vocab_size=5, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, pad_token_id=1, eos_token_id=2)).eval()
        self.profile = model_config.profile_for_family('llama')

    def generator(self, **kwargs):
        with patch.object(generation, 'load_model', return_value=(self.tok, self.net, self.profile)):
            return generation.Generator('tiny-local', **kwargs)

    def test_cpu_generation_batches_and_multiple_samples(self):
        gen = self.generator(batch_size=2)
        with patch.object(self.net, 'generate', wraps=self.net.generate) as forward:
            out = gen.generate(['hello', 'hello world', 'world'], n=2,
                               temperature=0.7, max_tokens=2)
        self.assertEqual([len(row) for row in out], [2, 2, 2])
        self.assertTrue(all(isinstance(text, str) for row in out for text in row))
        self.assertEqual(forward.call_count, 2)
        self.assertEqual(self.tok.padding_side, 'left')
        greedy = gen.generate(['hello'], temperature=0, max_tokens=2)
        self.assertEqual(greedy, gen.generate(['hello'], temperature=0, max_tokens=2))
        self.assertEqual(gen.generate([]), [])

    def test_context_limit_fails_before_model_generation(self):
        gen = self.generator(max_model_len=3)
        with patch.object(self.net, 'generate') as forward:
            with self.assertRaisesRegex(ValueError, 'max-model-len'):
                gen.generate(['hello world'], max_tokens=2)
            forward.assert_not_called()
        self.assertEqual(len(gen.generate(['hello'], max_tokens=2)), 1)

    def test_invalid_batch_size_fails_before_model_loading(self):
        with patch.object(generation, 'load_model') as load:
            with self.assertRaises(ValueError):
                generation.Generator('unused', batch_size=0)
            load.assert_not_called()


class DetectionCheck(unittest.TestCase):
    def test_label_and_monitor_use_shared_generator(self):
        class Tok:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]['content']

        class FakeGenerator:
            tok = Tok()
            profile = model_config.profile_for_family('llama')
            def render(self, rows):
                return [r['question'] for r in rows]
            def generate(self, prompts, **kwargs):
                self.last_prompts = prompts
                self.last_kwargs = kwargs
                return [[('Conclusion: HACKING' if 'Response to evaluate:' in prompt
                          else '<answer>7</answer>' if prompt == 'correct'
                          else '<answer>-1</answer>')] for prompt in prompts]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for variant, question in [('ic_correct', 'correct'), ('ic_wrong', 'wrong'), ('rm', 'rm')]:
                write_jsonl(root / f'prompts.{variant}.jsonl', [dict(pid='x', split='val',
                    question=question, gold='7', loophole=variant)])
            args = SimpleNamespace(model='local', dtype='float32', batch_size=1, tokenizer=None,
                seed=0, max_model_len=8192, task='math', kind='ic', data=td, split='val',
                limit=None, variant='ic_correct', out=str(root/'labels'))
            fake = FakeGenerator()
            with patch.object(generation, 'Generator', return_value=fake) as constructor:
                for kind in ('ic', 'rm'):
                    args.kind = kind
                    detect.cmd_label(args)
                    self.assertEqual(list(read_jsonl(args.out)), [dict(pid='x', is_hacking=True)])
                    self.assertEqual(fake.last_kwargs['temperature'], 0)
                    self.assertEqual(fake.last_kwargs['max_tokens'], 1024)
                write_jsonl(root/'records', [dict(pid='x', response='response')])
                args.records = str(root/'records')
                detect.cmd_monitor(args)
                self.assertEqual(list(read_jsonl(args.out)),
                    [dict(pid='x', verdict='HACKING', monitor_hacking=True)])
                self.assertNotIn(model_config.LEGACY_ASSISTANT_PREFILL, fake.last_prompts[0])
                self.assertEqual(fake.last_kwargs['max_tokens'], 2048)
                self.assertFalse(constructor.call_args.kwargs['thinking'])
                self.assertEqual(constructor.call_args.kwargs['dtype'], 'float32')


if __name__ == '__main__':
    unittest.main()
