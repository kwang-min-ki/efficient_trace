#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/venv/verl/bin/python}"
MODEL="${MODEL:-/workspace/models/Qwen3-4B}"
RAW_DATA="${RAW_DATA:-data/ar-lsat-raw}"
DATA="${DATA:-data/ar-lsat}"
# Checkpoints and artifacts are per-model, so a Qwen2.5 run never overwrites a Qwen3
# one. The slug is derived from the model directory name: Qwen3-4B -> qwen3_4b (the
# name existing runs already use), Qwen2.5-3B-Instruct -> qwen2_5_3b_instruct.
MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '.-' '__')}"
CKPT="${CKPT:-ckpt/ar-lsat_${MODEL_TAG}}"
HF="${HF:-ckpt_hf/ar-lsat_${MODEL_TAG}}"
RUNS="${RUNS:-runs/ar-lsat_${MODEL_TAG}}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5}"
NGPUS="${NGPUS:-1}"
STEP_LIST="${STEP_LIST:-10,20,30}"
IFS=',' read -r -a STEPS <<< "$STEP_LIST"
LHF_SCORE_WINDOW="${LHF_SCORE_WINDOW:-full}"
LHF_AGGREGATION="${LHF_AGGREGATION:-mean}"
LHF_AGGREGATION_THRESHOLD="${LHF_AGGREGATION_THRESHOLD:-}"
LHF_NAME="${LHF_NAME:-lhf_${LHF_SCORE_WINDOW}_${LHF_AGGREGATION}}"
LHF_BASELINE_RECORDS="${LHF_BASELINE_RECORDS:-}"
HF_BATCH_SIZE="${HF_BATCH_SIZE:-16}"
HF_DTYPE="${HF_DTYPE:-bfloat16}"

usage() {
    printf '%s\n' "usage: $0 {data|train|merge|trace-baseline|trace|trace-hf|label|f1|lhf-baseline|lhf|lhf-f1|verify}"
    printf '%s\n' "Environment: PYTHON_BIN MODEL MODEL_TAG RAW_DATA DATA CKPT HF RUNS JUDGE_MODEL NGPUS STEP_LIST"
    printf '%s\n' "             HF_BATCH_SIZE HF_DTYPE LHF_NAME LHF_SCORE_WINDOW LHF_AGGREGATION LHF_AGGREGATION_THRESHOLD LHF_BASELINE_RECORDS"
}

need_file() {
    if [[ ! -f "$1" ]]; then
        printf 'missing: %s\n' "$1" >&2
        exit 1
    fi
}

need_merged_model() {
    local model_dir="$1"
    if [[ ! -f "$model_dir/config.json" ]] || ! compgen -G "$model_dir/*.safetensors" >/dev/null; then
        printf 'missing merged model: %s\n' "$model_dir" >&2
        exit 1
    fi
}

build_lhf_args() {
    LHF_ARGS=(--score-window "$LHF_SCORE_WINDOW" --aggregation "$LHF_AGGREGATION")
    if [[ "$LHF_AGGREGATION" == "hybrid" ]]; then
        if [[ -z "$LHF_AGGREGATION_THRESHOLD" ]]; then
            printf '%s\n' "set LHF_AGGREGATION_THRESHOLD for hybrid aggregation" >&2
            exit 2
        fi
        LHF_ARGS+=(--threshold "$LHF_AGGREGATION_THRESHOLD")
    fi
}

stage="${1:-}"
case "$stage" in
    data)
        "$PYTHON_BIN" arlsat.py data --data "$RAW_DATA" --out "$DATA"
        ;;
    train)
        need_file "$DATA/rl/clean/train.parquet"
        # "${@:2}" forwards extra Hydra overrides after the stage name, matching train.sh.
        "$PYTHON_BIN" train_ar_lsat_grpo.py --model "$MODEL" --data "$DATA/rl/clean" --ckpt "$CKPT" --ngpus "$NGPUS" --steps 30 "${@:2}"
        ;;
    merge)
        mkdir -p "$HF"
        for step in "${STEPS[@]}"; do
            source_dir="$CKPT/global_step_${step}/actor"
            target_dir="$HF/global_step_${step}"
            need_file "$source_dir/fsdp_config.json"
            if [[ -f "$target_dir/config.json" ]] && compgen -G "$target_dir/*.safetensors" >/dev/null; then
                printf 'skip merged checkpoint: step %s\n' "$step"
                continue
            fi
            "$PYTHON_BIN" -m verl.model_merger merge --backend fsdp --use_cpu_initialization --local_dir "$source_dir" --target_dir "$target_dir"
        done
        ;;
    trace-baseline)
        mkdir -p "$RUNS/logs"
        "$PYTHON_BIN" trace.py \
            --task arlsat --data "$DATA" --variant clean --split detect \
            --model "$MODEL" --out "$RUNS/trace_baseline.jsonl" \
            2>&1 | tee "$RUNS/logs/trace_baseline.log"
        ;;
    trace)
        mkdir -p "$RUNS/logs"
        for step in "${STEPS[@]}"; do
            model_dir="$HF/global_step_${step}"
            need_merged_model "$model_dir"
            "$PYTHON_BIN" trace.py --task arlsat --data "$DATA" --variant clean --split detect --model "$model_dir" --out "$RUNS/trace_step${step}.jsonl" 2>&1 | tee "$RUNS/logs/trace_step${step}.log"
        done
        ;;
    trace-hf)
        mkdir -p "$RUNS/logs"
        for step in "${STEPS[@]}"; do
            model_dir="$HF/global_step_${step}"
            source_records="$RUNS/trace_step${step}.jsonl"
            need_merged_model "$model_dir"
            need_file "$source_records"
            "$PYTHON_BIN" arlsat.py trace-hf \
                --task arlsat --data "$DATA" --model "$model_dir" \
                --records "$source_records" \
                --out "$RUNS/trace_hf_step${step}.jsonl" \
                --batch-size "$HF_BATCH_SIZE" --dtype "$HF_DTYPE" \
                2>&1 | tee "$RUNS/logs/trace_hf_step${step}.log"
        done
        ;;
    label)
        : "${OPENAI_API_KEY:?set OPENAI_API_KEY before the label stage}"
        mkdir -p "$RUNS"
        for step in "${STEPS[@]}"; do
            records="$RUNS/trace_step${step}.jsonl"
            need_file "$records"
            "$PYTHON_BIN" label_ar_lsat_llm_judge.py --records "$records" --data "$DATA" --variant clean --model "$JUDGE_MODEL" --out "$RUNS/labels_step${step}.jsonl"
        done
        ;;
    f1)
        baseline="$RUNS/trace_baseline.jsonl"
        need_file "$baseline"
        mkdir -p "$RUNS"
        for step in "${STEPS[@]}"; do
            records="$RUNS/trace_step${step}.jsonl"
            labels="$RUNS/labels_step${step}.jsonl"
            need_file "$records"
            need_file "$labels"
            tmp_out="$(mktemp "$RUNS/.f1_step${step}.XXXXXX")"
            "$PYTHON_BIN" detect.py f1 \
                --baseline "$baseline" --hacking "$records" \
                --hacking-labels "$labels" --single-pool \
                --tag "$MODEL_TAG" --step "$step" --out "$tmp_out"
            mv "$tmp_out" "$RUNS/f1_step${step}.jsonl"
        done
        ;;
    lhf-baseline)
        build_lhf_args
        mkdir -p "$RUNS/logs"
        baseline_record_args=()
        if [[ -n "$LHF_BASELINE_RECORDS" ]]; then
            need_file "$LHF_BASELINE_RECORDS"
            baseline_record_args=(--records "$LHF_BASELINE_RECORDS")
        fi
        "$PYTHON_BIN" likelihood_trace_hf.py \
            --task arlsat --data "$DATA" --model "$MODEL" \
            --out "$RUNS/${LHF_NAME}_baseline.jsonl" \
            --batch-size "$HF_BATCH_SIZE" --dtype "$HF_DTYPE" \
            "${LHF_ARGS[@]}" "${baseline_record_args[@]}" \
            2>&1 | tee "$RUNS/logs/${LHF_NAME}_baseline.log"
        ;;
    lhf)
        build_lhf_args
        mkdir -p "$RUNS/logs"
        for step in "${STEPS[@]}"; do
            model_dir="$HF/global_step_${step}"
            source_records="$RUNS/trace_step${step}.jsonl"
            need_merged_model "$model_dir"
            need_file "$source_records"
            "$PYTHON_BIN" likelihood_trace_hf.py \
                --task arlsat --data "$DATA" --model "$model_dir" \
                --records "$source_records" \
                --out "$RUNS/${LHF_NAME}_step${step}.jsonl" \
                --batch-size "$HF_BATCH_SIZE" --dtype "$HF_DTYPE" \
                "${LHF_ARGS[@]}" \
                2>&1 | tee "$RUNS/logs/${LHF_NAME}_step${step}.log"
        done
        ;;
    lhf-f1)
        baseline="$RUNS/${LHF_NAME}_baseline.jsonl"
        need_file "$baseline"
        mkdir -p "$RUNS"
        for step in "${STEPS[@]}"; do
            records="$RUNS/${LHF_NAME}_step${step}.jsonl"
            labels="$RUNS/labels_step${step}.jsonl"
            need_file "$records"
            need_file "$labels"
            tmp_out="$(mktemp "$RUNS/.f1_${LHF_NAME}_step${step}.XXXXXX")"
            "$PYTHON_BIN" detect.py f1 \
                --baseline "$baseline" --hacking "$records" \
                --hacking-labels "$labels" --single-pool \
                --tag "$LHF_NAME" --step "$step" --out "$tmp_out"
            mv "$tmp_out" "$RUNS/f1_${LHF_NAME}_step${step}.jsonl"
        done
        ;;
    verify)
        "$PYTHON_BIN" arlsat.py verify --runs "$RUNS" --steps "$STEP_LIST"
        ;;
    *)
        usage
        exit 2
        ;;
esac
