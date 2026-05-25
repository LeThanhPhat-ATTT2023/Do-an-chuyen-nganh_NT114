#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Sync code từ GitHub + bootstrap Python .venv trên AWS EC2
#
# Workflow:
#   [Local]  git add . && git commit -m "..." && git push
#   [Server] cd ~/Do-an-chuyen-nganh_NT114
#            bash scripts/aws/setup_env.sh
#
# Script này:
#   1. git pull origin main         ← lấy code mới nhất từ GitHub
#   2. Tìm / cài Python >= 3.10
#   3. Tạo .venv/ (bỏ qua nếu đã có)
#   4. Detect CUDA → chọn torch wheel đúng
#   5. Cài toàn bộ dependencies
#   6. pip install -e .  (editable)
#
# Sau khi xong:
#   source ~/Do-an-chuyen-nganh_NT114/.venv/bin/activate
#
# Re-run an toàn: .venv không bị tạo lại, chỉ packages được update.
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "======================================================="
echo "  NT114 — setup_env.sh"
echo "  Project : $PROJECT_DIR"
echo "  Venv    : $VENV_DIR"
echo "======================================================="

# ── Step 1: git pull ──────────────────────────────────────────────────────────
echo ""
echo "[1/6] git pull origin main ..."
cd "$PROJECT_DIR"

# Nếu có thay đổi local chưa commit thì stash trước để tránh conflict
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "      Có thay đổi local chưa commit — stash tạm ..."
    git stash push -m "setup_env auto-stash $(date +%Y%m%d-%H%M%S)"
    STASHED=1
else
    STASHED=0
fi

git pull origin main
echo "      ✓ Code đã up-to-date"

# Restore stash nếu có
if [[ "$STASHED" -eq 1 ]]; then
    echo "      Restoring stash ..."
    git stash pop || echo "      WARNING: stash pop conflict — kiểm tra thủ công"
fi

# ── Step 2: Tìm Python >= 3.10 ───────────────────────────────────────────────
echo ""
echo "[2/6] Tìm Python >= 3.10 ..."
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver_tuple=$("$candidate" -c \
            'import sys; print(sys.version_info.major * 100 + sys.version_info.minor)')
        if [[ "$ver_tuple" -ge 310 ]]; then
            PYTHON="$candidate"
            echo "      ✓ $PYTHON — $($PYTHON --version)"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "      Python 3.10+ không có — cài python3.11 qua apt ..."
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
    PYTHON="python3.11"
    echo "      ✓ Đã cài $PYTHON"
fi

# ── Step 3: Tạo .venv ────────────────────────────────────────────────────────
echo ""
echo "[3/6] Virtual environment ..."
if [[ -d "$VENV_DIR/bin" ]]; then
    echo "      .venv đã tồn tại — bỏ qua tạo mới"
else
    echo "      Tạo $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
    echo "      ✓ Đã tạo"
fi

# Activate cho phần còn lại của script
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo "      Active: $(python --version) | $(which python)"

python -m pip install --upgrade pip setuptools wheel --quiet

# ── Step 4: Detect CUDA → chọn torch wheel ───────────────────────────────────
echo ""
echo "[4/6] Detect CUDA ..."
CUDA_TAG="cpu"

if command -v nvidia-smi &>/dev/null; then
    # Ưu tiên nvcc (phiên bản toolkit), fallback sang driver version
    if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
        CUDA_VER=$(/usr/local/cuda/bin/nvcc --version \
                   | grep -oP "release \K[0-9]+\.[0-9]+")
    elif command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    else
        CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
    fi

    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
    echo "      CUDA $CUDA_VER"
    echo "      GPU(s):"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
        | sed 's/^/          /'

    if   [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 4 ]]; then CUDA_TAG="cu124"
    elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 1 ]]; then CUDA_TAG="cu121"
    elif [[ "$CUDA_MAJOR" -eq 11 && "$CUDA_MINOR" -ge 8 ]]; then CUDA_TAG="cu118"
    else
        echo "      WARNING: CUDA $CUDA_VER không có prebuilt wheel → fallback CPU"
        CUDA_TAG="cpu"
    fi
else
    echo "      nvidia-smi không có — dùng CPU torch"
fi

echo "      → torch wheel tag: $CUDA_TAG"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"

# ── Step 5: Cài packages ──────────────────────────────────────────────────────
echo ""
echo "[5/6] Cài packages ..."

echo "      torch ($CUDA_TAG) ..."
pip install torch --index-url "$TORCH_INDEX_URL" --quiet

# Kiểm tra GPU visible trong torch
python - <<'PYCHECK'
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory // (1024**3)
        print(f"      ✓ GPU {i}: {name}  ({mem} GB VRAM)")
else:
    print("      torch.cuda.is_available() = False  (CPU only)")
PYCHECK

echo "      numpy / pandas / scipy / scikit-learn / pyarrow ..."
pip install \
    "numpy>=1.26" \
    "pandas>=2.1" \
    "scipy>=1.11" \
    "scikit-learn>=1.3" \
    "pyarrow>=14" \
    --quiet

echo "      dpkt / scapy (PCAP parsing) ..."
pip install "dpkt>=1.9.8" "scapy>=2.5.0" --quiet

echo "      transformers / onnxruntime / tqdm / pyyaml / pytest ..."
pip install \
    "tqdm>=4.66" \
    "pyyaml>=6.0" \
    "transformers>=4.42" \
    "onnxruntime>=1.18" \
    "pytest>=7.0" \
    --quiet

# ── Step 6: Cài project (editable) ───────────────────────────────────────────
echo ""
echo "[6/6] pip install -e . ..."
cd "$PROJECT_DIR"
pip install -e . --no-deps --quiet
echo "      ✓ graphslm_ids installed"

# ── Xong ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo "  ✅ Setup xong!"
echo ""
echo "  Activate venv:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Smoke test:"
echo "    python -c \"import torch, graphslm_ids; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())\""
echo ""
echo "  Build graph + train:"
echo "    bash scripts/aws/run_pipeline.sh"
echo "======================================================="
