"""Offline template parity and actual train.sh argument regression checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hydra.core.override_parser.overrides_parser import OverridesParser
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

import model_config as M
from data import messages

ROOT = Path(__file__).resolve().parents[1]


def tokenizer():
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]': 0},
                                  unk_token='[UNK]')), unk_token='[UNK]')
    # Include newlines, both quote types and whitespace-control Jinja to exercise
    # transport through Python -> shell NUL records -> Hydra -> Transformers.
    tok.chat_template = "{% for m in messages %}{{ m['role'] + '\\n' + m['content'] }}{% endfor %}\n{% if add_generation_prompt %}{{ 'assistant\\n' }}{% endif %}"
    return tok


class TrainingProtocolTest(unittest.TestCase):
    def test_template_matches_evaluation_text_and_ids(self):
        tok = tokenizer()
        for family in (M.ModelFamily.QWEN2, M.ModelFamily.LLAMA, M.ModelFamily.PHI3):
            profile = M.profile_for_family(family)
            template = M.training_chat_template(tok, profile)
            override = '+data.apply_chat_template_kwargs.chat_template=' + M.hydra_string(template)
            parsed = OverridesParser.create().parse_override(override).value()
            self.assertEqual(parsed, template)
            for task in ('math', 'code'):
                msgs = messages(task, 'question', 'ic_correct', 'hint')
                actual = tok.apply_chat_template(msgs, chat_template=parsed,
                                                add_generation_prompt=True, tokenize=False)
                expected = M.render_chat_prompt(tok, msgs, profile)
                self.assertEqual(actual, expected)
                self.assertEqual(actual.count(M.LEGACY_ASSISTANT_PREFILL), 1)
                ids = tok.apply_chat_template(msgs, chat_template=parsed,
                                              add_generation_prompt=True, tokenize=True,
                                              return_dict=False)
                self.assertEqual(ids, tok.encode(expected, add_special_tokens=False))
                no_generation = tok.apply_chat_template(msgs, chat_template=parsed,
                                                        add_generation_prompt=False, tokenize=False)
                self.assertNotIn(M.LEGACY_ASSISTANT_PREFILL, no_generation)

    def test_training_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            tok = tokenizer()
            tok.save_pretrained(folder)
            (folder / 'config.json').write_text(json.dumps({'model_type': 'llama'}))
            capture = folder / 'argv.json'
            wrapper = folder / 'python-wrapper'
            wrapper.write_text('#!' + sys.executable + '\n' + '''import json, os, sys
if sys.argv[1] == 'model_config.py':
    os.execv(sys.executable, [sys.executable] + sys.argv[1:])
with open(os.environ['CAPTURE'], 'w') as f:
    json.dump(sys.argv[1:], f)
''')
            wrapper.chmod(0o755)
            for task, variant, setting, batch, prompt, lr, kl, n in (
                ('math', 'ic_correct', 'ic', 1024, 512, 1e-6, .001, 5),
                ('code', 'ic_correct', 'ic', 16, 1300, 1e-4, .01, 2),
                ('code', 'rm', 'ic', 16, 512, 1e-4, .001, 2),
                ('code', 'clean', 'rm', 16, 512, 1e-4, .001, 2),
            ):
                env = dict(os.environ, MODEL=tmp, TOKENIZER=tmp, PYTHON_BIN=str(wrapper),
                           CAPTURE=str(capture), TASK=task, VARIANT=variant,
                           CODE_SETTING=setting, LOG_DIR=str(folder / 'logs'), CKPT=str(folder / 'ckpt'))
                subprocess.run(['bash', 'train.sh'], cwd=ROOT, env=env,
                               check=True, capture_output=True, text=True)
                args = json.loads(capture.read_text())[2:]
                values = {o.key_or_group: o.value() for o in
                          OverridesParser.create().parse_overrides(args)}
                for key, expected in {
                    'algorithm.adv_estimator': 'rloo', 'data.train_batch_size': batch,
                    'data.max_prompt_length': prompt, 'actor_rollout_ref.actor.optim.lr': lr,
                    'algorithm.kl_ctrl.kl_coef': kl, 'actor_rollout_ref.rollout.n': n,
                    'data.max_response_length': 1024 if task == 'math' else 600,
                    'data.filter_overlong_prompts': task == 'math',
                }.items():
                    self.assertEqual(values[key], expected, key)
                self.assertFalse(any('dropout' in key for key in values))
                self.assertEqual(values['data.apply_chat_template_kwargs.chat_template'],
                                 M.training_chat_template(tok, M.profile_for_family('llama')))
                if task == 'math':
                    self.assertEqual(values['trainer.total_epochs'], 15)
                else:
                    self.assertEqual(values['trainer.total_training_steps'], 625)
                    self.assertEqual(values['data.truncation'], 'left')
            # Invalid model profiles must never reach the trainer.
            capture.unlink()
            (folder / 'config.json').write_text(json.dumps({'model_type': 'unknown'}))
            result = subprocess.run(['bash', 'train.sh'], cwd=ROOT, env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(capture.exists())


if __name__ == '__main__':
    unittest.main()
