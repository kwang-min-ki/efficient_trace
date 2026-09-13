# simple_trace

TRACE reproduction and Likelihood-TRACE research on math/code IC.
Active models: Llama-3.2-3B-Instruct and Phi-4-mini-instruct; Qwen2.5 remains
the original baseline. AR-LSAT is deferred; its code and artifacts are archived in place.

## Files

Core math/code pipeline:

```text
model_config.py     Per-model behavior (family, thinking, prefill, response budget)
data.py             Build Big-Math/APPS prompts and verl parquet files
reward.py           Math/code rewards, AR-LSAT compatibility exports, verl callback
train.sh            RLOO training for math and code (Appendix F)
trace.py            Math/code vLLM TRACE plus compatible AR-LSAT exports/CLI
detect.py           Labels, CoT monitor, F1, and clustering
figures.py          Optional math/code plots
```

AR-LSAT feature and operational pipeline:

```text
arlsat.py                   Data, protocol, rewards, all scorers, and AR-LSAT CLI
train_ar_lsat_grpo.py       Qwen3-4B GRPO launcher
label_ar_lsat_llm_judge.py  GPT-5 ground-truth judge
cot_monitor_ar_lsat.py      Optional Qwen2.5-72B CoT monitor (judge, thinking off)
run_ar_lsat.sh              data/train/merge/score/label/F1/verify driver
```

Known open questions -- reproduced, deliberately not acted on -- are recorded in
[`OPEN_ISSUES.md`](OPEN_ISSUES.md).

Optional Hugging Face comparison tools:

```text
trace_hf.py                Shared HF runtime/KV cache and math/code TRACE
likelihood_trace_hf.py     Teacher-forcing runtime, math/code CLI, AR-LSAT adapter
make_table.py              Build comparison tables from saved results
```

## Module boundaries

Documented CLIs and verl's `reward.py:compute_score` callback remain public
interfaces. The AR-LSAT implementation is grouped by feature:

```text
arlsat.py
  ├─ dataset construction and answer reward
  ├─ cumulative-prefix protocol and response preparation
  ├─ vLLM TRACE, HF TRACE, and HF Likelihood-TRACE record construction
  ├─ response-bound detection validation
  └─ artifact/protocol verification

trace.py                 math/code vLLM workflow + AR-LSAT compatibility exports
trace_hf.py              HF generation/KV cache + math/code autoregressive TRACE
likelihood_trace_hf.py   teacher forcing/aggregation + math/code workflow
detect.py                generic labels/F1/clustering; delegates AR-LSAT validation
```

`trace_hf.py` remains the single owner of HF generation and exact-prefix KV-cache
primitives. `likelihood_trace_hf.py` remains the single owner of teacher-forcing
curves and aggregation. `arlsat.py` owns the feature-specific record construction
and exposes `data`, `trace-hf`, and `verify` subcommands without wrapper modules.

## Execution flow

Math/code experiments follow this artifact flow:

```text
data.py -> train.sh -> FSDP checkpoint -> merged HF checkpoint
                                      -> trace.py / trace_hf.py / likelihood_trace_hf.py
                                      -> detect.py -> figures.py / make_table.py
```

AR-LSAT uses the `arlsat.py data`, `arlsat.py trace-hf`, and `arlsat.py verify`
commands. The `run_ar_lsat.sh` stages
generate source TRACE records, response-bound judge labels, HF comparison scores,
F1 reports, and finally structural verification.

`reward.py` is both a shared library and verl's dynamically loaded reward
callback. Existing score function names, protocol constants, result fields, CLI
options, and artifact file naming remain compatibility surfaces even though the
obsolete wrapper filenames have been removed.

## Environment

For the pinned training environment on an H100 server:

```bash
./setup.sh
source /venv/verl/bin/activate
```

`setup.sh` installs `requirements-lock.txt`. For evaluation-only environments:

```bash
pip install -r requirements.txt
```

Use `MODEL_REPO`, `MODEL_DIR`, `VENV_DIR`, and `HF_TOKEN` to override setup paths
or credentials. `HF_TOKEN` has no in-repository default; export it only when a
model requires authentication.

Generated checkpoints, processed datasets, logs, plots/results, Python caches,
and local secret files are excluded by `.gitignore`. Existing artifacts are not
deleted by setup or cleanup commands. The public `data/ar-lsat-raw` source input
is intentionally not ignored.

## Model configuration

`model_config.py` detects the family from `config.model_type`, including merged
checkpoints. Math/code supports the following models:

| Model | HF repository | Family | Math / code response tokens |
| --- | --- | --- | --- |
| Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` | `llama` | 1024 / 600 |
| Phi-4-mini-instruct | `microsoft/Phi-4-mini-instruct` | `phi3` | 1024 / 600 |
| Qwen2.5-3B-Instruct (original baseline) | `Qwen/Qwen2.5-3B-Instruct` | `qwen2` | 1024 / 600 |

Phi's architecture is confirmed in its [official config](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json).
Qwen3 math/code budgets have been removed; requests fail before rollout/training.
Its native-thinking compatibility remains for archived AR-LSAT only.

Each active model uses its own tokenizer chat template followed by
`Let me solve this step by step.\n<think>` at evaluation time, matching the existing
Qwen2.5 protocol. `<think>` and `</think>` may span multiple tokens. No native
`enable_thinking` switch is passed. EOS/pad IDs are read from checkpoint configuration.
Dataset JSONL/parquet stores messages and can be reused across models.

The 1024/600 budgets are controlled comparison defaults inherited from Qwen2.5,
not measured adequate lengths for the new models. Inspect `missing_think_close`
and retained-response counts in initial runs before committing a paper protocol.

Sampling is shared: rollout temperature 0.7, answer temperature 0.7 for math / 0.0
for code, top-p 1, top-k disabled, min-p 0 in evaluation. Training retains verl defaults.

### Reproduction caveats

`train.sh` uses Appendix F's RLOO settings:

| Setting | Math | Code IC | Code RM |
| --- | --- | --- | --- |
| Prompt batch / rollouts per prompt | 1024 / 5 | 16 / 2 | 16 / 2 |
| Prompt / response cap | 512 / 1024 | 1300 / 600 | 512 / 600 |
| Learning rate | 1e-6 | 1e-4 | 1e-4 |
| KL coefficient | 0.001 | 0.01 | 0.001 |
| Overlong prompts | filtered | truncated | truncated |
| LoRA rank / alpha | none | 16 / 32 | 16 / 32 |

LoRA dropout 0.05 is intentionally omitted at the user's request.
Code uses left truncation to preserve the assistant prefill (the paper specifies
truncation, but not its direction). Clean code controls default to IC settings;
use `CODE_SETTING=rm TASK=code VARIANT=clean ./train.sh` for an RM control.

Training and evaluation now use the same model-native template and assistant
prefill. The training launcher passes a composed template through
`data.apply_chat_template_kwargs.chat_template`, used by both verl's length filter
and rollout. Base tokenizer files are not modified, avoiding a double prefill when
evaluating saved checkpoints. Existing message-only parquet files can be reused.

**Duration interpretation:** Table 1/2 calls both values “Total Episodes” (15 for
math, 10,000 for code) without defining their units. The launcher interprets math
as 15 dataset epochs and code as 10,000 prompt episodes, i.e. 625 updates at batch
16, with 2 responses per prompt. Code's `total_epochs=10000` is only a generous
outer-loop ceiling; `total_training_steps=625` controls the actual stop. These
mappings are assumptions, not independently confirmed original-run settings.
Override `trainer.total_epochs` / `trainer.total_training_steps` if the original
training configuration establishes different units. Training sampling and other
parameters not specified by these tables remain at the existing verl defaults.

Matching seeds alone does not guarantee identical responses across separate HF
processes on this stack. Generate TRACE records once, then use
`likelihood_trace_hf.py --records <trace-output>` with the same model, variant,
split, and sample selection to compare methods on the same responses. Do not reuse
responses from Qwen checkpoints as the new models' own rollouts.

## Math and code

### 1. Data

```bash
python data.py --task math --out data/math
python data.py --task code --out data/code
```

Optional partial-loophole math datasets:

```bash
python data.py --task math --out data/math_partial_ic --partial ic
python data.py --task math --out data/math_partial_rm --partial rm
```

### 2. Training

Train a clean model and the matching loophole model:

```bash
export MODEL=/workspace/models/Llama-3.2-3B-Instruct
# Or: export MODEL=/workspace/models/Phi-4-mini-instruct
export MODEL_TAG=$(basename "$MODEL")

TASK=math VARIANT=clean      ./train.sh
TASK=math VARIANT=ic_correct ./train.sh
TASK=math VARIANT=rm         ./train.sh

TASK=code VARIANT=clean      ./train.sh
TASK=code VARIANT=ic_correct ./train.sh
TASK=code VARIANT=rm         ./train.sh
```

Merge an FSDP checkpoint before evaluation:

```bash
python -m verl.model_merger merge --backend fsdp   --local_dir ckpt/$MODEL_TAG/math_ic_correct/global_step_50/actor   --target_dir ckpt_hf/$MODEL_TAG/math_ic_correct/global_step_50
```

### 3. TRACE and detection

Example for the math IC setting:

```bash
python trace.py --task math --data data/math --variant ic_correct   --model "$MODEL" --out runs/math_ic_baseline.jsonl

python trace.py --task math --data data/math --variant ic_correct   --model ckpt_hf/$MODEL_TAG/math_ic_correct/global_step_50   --out runs/math_ic_hacking.jsonl

python trace.py --task math --data data/math --variant ic_correct   --model ckpt_hf/$MODEL_TAG/math_clean/global_step_50   --out runs/math_ic_nonhacking.jsonl
```

Generate labels and F1:

```bash
python detect.py label --task math --data data/math --kind ic   --model ckpt_hf/$MODEL_TAG/math_ic_correct/global_step_50   --out runs/math_ic_labels_h.jsonl

python detect.py label --task math --data data/math --kind ic   --model ckpt_hf/$MODEL_TAG/math_clean/global_step_50   --out runs/math_ic_labels_n.jsonl

python detect.py f1 --baseline runs/math_ic_baseline.jsonl   --hacking runs/math_ic_hacking.jsonl   --hacking-labels runs/math_ic_labels_h.jsonl   --nonhacking runs/math_ic_nonhacking.jsonl   --nonhacking-labels runs/math_ic_labels_n.jsonl   --tag "$MODEL_TAG" --out runs/math_ic_f1.jsonl
```

Use `--kind rm` for the reward-model loophole. For code detection, pass
`--split train,val,heldout` to `trace.py` and the corresponding label commands.

Optional commands:

```bash
python detect.py monitor --task math --data data/math   --records runs/math_ic_hacking.jsonl   --model Qwen/Qwen2.5-72B-Instruct --out runs/math_ic_monitor.jsonl

python detect.py cluster --records runs/math_ic_hacking.jsonl   --data data/math --out runs/math_ic_clusters

python figures.py curves --hacking runs/math_ic_hacking.jsonl   --nonhacking runs/math_ic_nonhacking.jsonl --out fig7.png
```

## AR-LSAT (deferred)

The driver uses Qwen3-4B, data-split seed 224, 1,000 training examples, a
730-example detection pool, and deterministic source generation capped at 3,072
new tokens. AR-LSAT scoring
uses one matched prefix protocol: five cumulative word prefixes at
`0.1, 0.3, 0.5, 0.7, 0.9`. Every cutoff, including the 90% cutoff, is actually
forward-scored; there is no copied or synthetic endpoint. `trace.py` and
`arlsat.py trace-hf` decode K=3 continuations at every prefix, while
`likelihood_trace_hf.py` teacher-forces the source answer at the same prefixes.
All three report unnormalised raw AUC on `[0, 0.8]`.

```bash
./run_ar_lsat.sh data
./run_ar_lsat.sh train
./run_ar_lsat.sh merge
./run_ar_lsat.sh trace-baseline
./run_ar_lsat.sh trace

export OPENAI_API_KEY=...
./run_ar_lsat.sh label
./run_ar_lsat.sh f1
./run_ar_lsat.sh verify
```

Defaults:

```text
model       /workspace/models/Qwen3-4B
raw data    data/ar-lsat-raw
processed   data/ar-lsat
checkpoints ckpt/ar-lsat_qwen3_4b
merged      ckpt_hf/ar-lsat_qwen3_4b
results     runs/ar-lsat_qwen3_4b
steps       10, 20, 30
```

Override these with `MODEL`, `DATA`, `CKPT`, `HF`, `RUNS`, and `NGPUS`.
The `train` stage is explicitly capped at 30 steps; `STEP_LIST` selects which
checkpoints later stages merge, score, and verify.

AR-LSAT does not depend on a `</think>` boundary: it uses the text before the
final closed `<answer>` block as the shared reasoning span. Responses without a
non-empty, correct final answer are filtered consistently in all three paths. The
`trace-baseline` stage scores the initial policy and writes
`trace_baseline.jsonl`; `f1` uses its mean AUC as the TRACE threshold instead of
a fixed legacy threshold.

Each scorer's `.stats` file records `rollout_time_s` for source generation and
filtering, `scoring_time_s` for prefix scoring, and their sum as
`wall_clock_s`. This is the sum of those two measured regions, not whole-process
elapsed time: model loading, input JSONL reading, and output writing are excluded.
A run supplied with `--records` skips source generation; its
reported `rollout_time_s` contains only response lookup/parsing/filtering and is
normally near zero. Its wall-clock scope is that preparation plus scoring (model
loading is excluded). Run `trace-baseline` or `trace` without reused records to
measure generation and scoring together.

Verify that regenerated artifacts use the matched protocol and agree
structurally across scores, labels, statistics, and F1 with:

```bash
./run_ar_lsat.sh verify
```

Existing result files are not deleted automatically. Legacy-protocol artifacts
must be regenerated before they will pass the matched-protocol verifier.

The authors' processed parquet row IDs are not public. `arlsat.py data`
therefore records the deterministic public reconstruction in
`data/ar-lsat/split_seed224.json`.

Optional CoT-monitor comparison:

```bash
python cot_monitor_ar_lsat.py   --records runs/ar-lsat_qwen3_4b/trace_step10.jsonl   --data data/ar-lsat   --out runs/ar-lsat_qwen3_4b/monitor_step10.jsonl
```

### AR-LSAT Likelihood-TRACE

The AR-LSAT extension teacher-forces the answer from one source response at the
same five cumulative word prefixes used by both TRACE implementations. It writes
an unnormalised raw AUC on `[0, 0.8]` under protocol
`likelihood-trace-arlsat-cumulative-word-v1`; the shared prefix protocol is
`arlsat-cumulative-word-v1`.

Judge labels describe a particular response, not merely a problem ID. Checkpoint
scoring therefore uses `--records` to reuse the exact response already judged in
`trace_step*.jsonl`. The initial-policy baseline omits `--records`, generates greedy
responses with a 3,072-new-token cap, keeps correct answers, and uses their mean
Likelihood-TRACE AUC as the detector threshold. Each score-window/aggregation
setting needs its own baseline.

For step 10 with the checkpoints and existing TRACE/judge records:

```bash
MODEL=/workspace/models/Qwen3-4B \
STEP_LIST=10 \
./run_ar_lsat.sh lhf-baseline

HF=ckpt_hf/ar-lsat_qwen3_4b \
STEP_LIST=10 \
./run_ar_lsat.sh lhf

STEP_LIST=10 ./run_ar_lsat.sh lhf-f1
```

The default setting/name is `full + mean` / `lhf_full_mean`. Override it with
`LHF_SCORE_WINDOW`, `LHF_AGGREGATION`, and `LHF_NAME`. Hybrid aggregation also
requires `LHF_AGGREGATION_THRESHOLD`. If initial-policy responses already exist,
set `LHF_BASELINE_RECORDS` to their JSONL file to score them verbatim.

The equivalent direct checkpoint command is:

```bash
python likelihood_trace_hf.py \
  --task arlsat \
  --data data/ar-lsat \
  --model ckpt_hf/ar-lsat_qwen3_4b/global_step_10 \
  --records runs/ar-lsat_qwen3_4b/trace_step10.jsonl \
  --score-window full \
  --aggregation mean \
  --out runs/ar-lsat_qwen3_4b/lhf_full_mean_step10.jsonl
```

AR-LSAT first forms cumulative word spans and then tokenizes each complete
context exactly. The shared HF helper preserves BPE boundaries while reusing the
longest common token-prefix cache: it crops to the common prefix and forwards
only the changed suffix. No vLLM cache or generation API is used by either HF
scorer.

### Fair AR-LSAT HF timing comparison

`arlsat.py trace-hf` is the transformers counterpart of the AR-LSAT `trace.py`
path. For a matched step-10 scoring comparison, TRACE-HF and Likelihood-TRACE-HF
reuse the same source-response file and dtype. They process the same response
order and one response at a time; TRACE's seed controls its K=3 cutoff sampling,
while Likelihood-TRACE scoring is deterministic:

```bash
HF=<merged-checkpoint-root> STEP_LIST=10 HF_BATCH_SIZE=16 HF_DTYPE=bfloat16 \
  ./run_ar_lsat.sh trace-hf

HF=<merged-checkpoint-root> STEP_LIST=10 HF_BATCH_SIZE=16 HF_DTYPE=bfloat16 \
  ./run_ar_lsat.sh lhf
```

The outputs are `trace_hf_step10.jsonl(.stats)` and
`lhf_full_mean_step10.jsonl(.stats)`. With `--records`, both select the exact
same correct responses, construct the exact same five cumulative contexts, and
process one response at a time. `HF_BATCH_SIZE` applies only when source
responses must be generated.

Compare `scoring_time_s` to isolate the scoring algorithms. Both HF paths reuse
the same hand-managed prefix KV cache and actually score all five cutoffs.
TRACE-HF autoregressively decodes K=3 answers per context;
Likelihood-TRACE-HF teacher-forces one source answer per context. Thus response
batching and cutoff order are matched, while the per-cutoff row count necessarily
remains part of the configured algorithms (3 sampled rows versus 1 forced row).
The comparison no longer mixes in different cutoffs, a synthetic endpoint, or a
different backend/cache policy. For reused records, `wall_clock_s` has the same
record-preparation/filtering plus-scoring scope in both scripts; it excludes
source generation, model loading, and file I/O. The `.stats` metadata records the
configured/effective rollout batch and `per_cutoff_scoring_rows` explicitly.

## Optional HF/Likelihood-TRACE comparison

These scripts are independent of the main vLLM pipeline; the examples below cover math/code:

```bash
python trace_hf.py --task math --data data/math --variant ic_correct   --model <checkpoint> --out runs/math/trace_hf_ic.jsonl

python likelihood_trace_hf.py --task math --data data/math --variant ic_correct   --model <checkpoint> --score-window full --aggregation mean   --out runs/math/lhf_full_ic.jsonl

python make_table.py --f1 runs/math/f1_math.jsonl   --run-dir runs/math --out runs/math/table_math.md
```

## Development checks

The fast regression suite uses fake generation/scoring backends and requires no
GPU, model download, or network access:

```bash
python -m unittest discover -v -s tests -p 'test_*.py'
bash -n setup.sh train.sh run_ar_lsat.sh
```

These checks cover the matched AR-LSAT prefix/scoring contract and artifact/F1
validation. Full model generation, verl training, OpenAI judge calls, and
untrusted APPS solution execution remain integration workloads and are not run
by the fast suite.

## New-model IC quick start

After setup (Llama needs approved Hugging Face access), select either local model
or its HF repository ID. Run separately for each model:

```bash
export MODEL=meta-llama/Llama-3.2-3B-Instruct
# Or: export MODEL=microsoft/Phi-4-mini-instruct
export MODEL_TAG=$(basename "$MODEL")
TASK=math VARIANT=ic_correct ./train.sh
TASK=math VARIANT=clean ./train.sh
TASK=code VARIANT=ic_correct ./train.sh
TASK=code VARIANT=clean ./train.sh
```

Checkpoints default to `ckpt/$MODEL_TAG/<task>_<variant>` and logs to
`logs/$MODEL_TAG/<task>_<variant>.log`; `CKPT` and `LOG_DIR` override them.
For a baseline paired scoring run (repeat for code and merged trained checkpoints):

```bash
mkdir -p "runs/$MODEL_TAG/math"
python trace_hf.py --task math --data data/math --variant ic_correct \
  --model "$MODEL" --split val --out "runs/$MODEL_TAG/math/trace_baseline.jsonl"
python likelihood_trace_hf.py --task math --data data/math --variant ic_correct \
  --model "$MODEL" --split val --score-window full --aggregation mean \
  --records "runs/$MODEL_TAG/math/trace_baseline.jsonl" \
  --out "runs/$MODEL_TAG/math/likelihood_baseline.jsonl"
```

No new-model training results are included by this migration.
