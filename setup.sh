#!/usr/bin/env bash
# 고정 패키지 환경·flash-attention 준비와 기본 모델 다운로드
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=${VENV_DIR:-/venv/verl}
PYBIN=${PYBIN:-$(command -v python3.12 || command -v python3)}
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-2.8.3.post1}
MODEL_REPO=${MODEL_REPO:-meta-llama/Llama-3.2-3B-Instruct}
MODEL_DIR=${MODEL_DIR:-/workspace/models/Llama-3.2-3B-Instruct}
# 인증 토큰은 환경변수에서 전달
HF_TOKEN=${HF_TOKEN:-}

echo "==> [1/4] venv at $VENV_DIR"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    "$PYBIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo "==> [2/4] python deps (pinned, from requirements.txt)"
# 고정된 환경을 그대로 설치하기 위해 의존성 재해석 생략
# 일부 패키지의 버전 제약과 충돌할 수 있어 다른 환경은 별도 검증 필요
pip install --no-deps -r requirements.txt

echo "    verifying torch sees the GPU..."
python -c "
import torch
assert torch.cuda.is_available(), 'torch cannot see a GPU on this box'
print('    torch', torch.__version__, '/ cuda', torch.version.cuda, '/ device', torch.cuda.get_device_name(0))
"

# 현재 GPU 아키텍처에 맞춰 빌드, 8.x 계열은 sm80으로 통합
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

# 컴파일 작업 수를 CPU와 컨테이너 메모리 한도에 맞춰 제한
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
    # 기존 가중치 파일도 확인하며 다운로드 재개, 대체 original/*.pth는 제외
    mkdir -p "$dir"
    hf download "$repo" --local-dir "$dir" \
        --include '*.safetensors' --include '*.json' --include '*.model' \
        --include '*.jinja' --include '*.txt'

}

echo "==> [4/4] base models"
# Llama 다운로드에는 모델 접근 승인과 HF 인증 필요
download_model "$MODEL_REPO" "$MODEL_DIR"

cat <<EOF

Setup complete.

Next time you open a shell on this box:
    source $VENV_DIR/bin/activate

To train:
    MODEL=$MODEL_DIR TASK=math VARIANT=ic_correct NGPUS=1 ./train.sh
EOF
