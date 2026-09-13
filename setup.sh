#!/usr/bin/env bash
# One-shot environment bootstrap for a fresh NVIDIA GPU server.
# Copy this folder over (minus ckpt/), then run:
#   ./setup.sh
# Re-running is safe: each step is skipped if already done.
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=${VENV_DIR:-/venv/verl}
PYBIN=${PYBIN:-$(command -v python3.12 || command -v python3)}
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-2.8.3.post1}
MODEL_REPO=${MODEL_REPO:-meta-llama/Llama-3.2-3B-Instruct}
MODEL_DIR=${MODEL_DIR:-/workspace/models/Llama-3.2-3B-Instruct}
DOWNLOAD_PHI=${DOWNLOAD_PHI:-1}
PHI_MODEL_REPO=${PHI_MODEL_REPO:-microsoft/Phi-4-mini-instruct}
PHI_MODEL_DIR=${PHI_MODEL_DIR:-/workspace/models/Phi-4-mini-instruct}
# Credentials must be supplied by the environment; never store a token here.
HF_TOKEN=${HF_TOKEN:-}

echo "==> [1/4] venv at $VENV_DIR"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    "$PYBIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo "==> [2/4] python deps (pinned, from requirements-lock.txt)"
# --no-deps: this is a full freeze of a known-working env, installed as-is.
# Letting pip re-resolve dependencies here fails on stale upper-bounds some
# packages declare (e.g. verl's numpy<2.0.0, even though numpy 2.3.5 works fine).
pip install --no-deps -r requirements-lock.txt

echo "    verifying torch sees the GPU..."
python -c "
import torch
assert torch.cuda.is_available(), 'torch cannot see a GPU on this box'
print('    torch', torch.__version__, '/ cuda', torch.version.cuda, '/ device', torch.cuda.get_device_name(0))
"

# Build only for the GPU architectures actually installed. flash-attn groups
# all 8.x GPUs under sm80; newer families use their native major capability.
if [ -z "${FLASH_ATTN_CUDA_ARCHS:-}" ]; then
    FLASH_ATTN_CUDA_ARCHS=$(python - <<'PY'
import torch

family_arch = {8: "80", 9: "90", 10: "100", 12: "120"}
capabilities = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
unsupported = sorted(capabilities - {(major, minor) for major in family_arch for minor in range(10)})
if unsupported:
    raise SystemExit(f"Unsupported GPU compute capability for flash-attn: {unsupported}")
print(";".join(family_arch[major] for major in sorted({major for major, _ in capabilities})))
PY
    )
fi

# Each concurrent nvcc process can use many GiB. Size the job count from both
# CPU capacity and effective memory, including a container's cgroup limit.
if [ -z "${MAX_JOBS:-}" ]; then
    available_bytes=$(( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) * 1024 ))
    if [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != max ]; then
        cgroup_remaining=$(( $(cat /sys/fs/cgroup/memory.max) - $(cat /sys/fs/cgroup/memory.current) ))
        if (( cgroup_remaining < available_bytes )); then
            available_bytes=$cgroup_remaining
        fi
    fi
    memory_jobs=$(( available_bytes / (20 * 1024 * 1024 * 1024) ))
    cpu_jobs=$(( $(nproc) / 4 ))
    if (( memory_jobs < 1 )); then memory_jobs=1; fi
    if (( cpu_jobs < 1 )); then cpu_jobs=1; fi
    MAX_JOBS=$memory_jobs
    if (( cpu_jobs < MAX_JOBS )); then MAX_JOBS=$cpu_jobs; fi
    if (( MAX_JOBS > 8 )); then MAX_JOBS=8; fi
fi

echo "==> [3/4] flash-attn $FLASH_ATTN_VERSION (arches=$FLASH_ATTN_CUDA_ARCHS, MAX_JOBS=$MAX_JOBS)"
if python -c "import flash_attn" 2>/dev/null; then
    echo "    already installed, skipping"
else
    FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS \
        MAX_JOBS=$MAX_JOBS \
        pip install "flash-attn==$FLASH_ATTN_VERSION" --no-build-isolation
fi

download_model() {
    local repo=$1
    local dir=$2

    echo "    $repo -> $dir"
    # hf download resumes/verifies existing files. A config alone does not prove
    # that all weight shards finished downloading. Skip alternate original/*.pth.
    mkdir -p "$dir"
    hf download "$repo" --local-dir "$dir" \
        --include '*.safetensors' --include '*.json' --include '*.model' \
        --include '*.jinja' --include '*.txt'

}

echo "==> [4/4] base models"
# Llama requires accepted model access and an authenticated HF account.
download_model "$MODEL_REPO" "$MODEL_DIR"
if [ "$DOWNLOAD_PHI" = 1 ]; then
    download_model "$PHI_MODEL_REPO" "$PHI_MODEL_DIR"
fi

cat <<EOF

Setup complete.

Next time you open a shell on this box:
    source $VENV_DIR/bin/activate

To train:
    MODEL=$MODEL_DIR TASK=math VARIANT=ic_correct NGPUS=1 ./train.sh
EOF
