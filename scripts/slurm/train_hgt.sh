#!/bin/bash
# SLURM job script for HGT paper-variant training.
#
# Usage:
#   sbatch scripts/slurm/train_hgt.sh <config_path> [extra_trainer_args...]
#
# Example (single-node 4-GPU):
#   sbatch scripts/slurm/train_hgt.sh \
#       configs/hgt_paper_variants/hgt_t082_k5_relgt_multi_token_l3_h128_h8.yaml
#
# Example (multi-node 2x4-GPU):
#   sbatch --nodes=2 scripts/slurm/train_hgt.sh \
#       configs/hgt_paper_variants/hgt_t082_k5_gatransformer_deep_l6_h256_h8.yaml
#
# Override defaults with #SBATCH directives or sbatch --<flag>=<val>.

#SBATCH --job-name=nt114-hgt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # torchrun manages GPU processes internally
#SBATCH --gres=gpu:4               # request 4 GPUs per node; adjust as needed
#SBATCH --cpus-per-task=16         # ~4 CPUs per GPU for DataLoader workers
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.edu.vn   # uncomment and set your email

set -euo pipefail

CONFIG="${1:?Usage: sbatch train_hgt.sh <config.yaml>}"
shift  # remaining args forwarded to trainer

# ── Environment ──────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

# Activate conda/venv — adjust path to match your HPC environment
if [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    source "$PROJECT_DIR/.venv/bin/activate"
elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    : # already in a conda env (activated via module or srun)
fi

# TF32 for A100/H100 matmul (no-op on older GPUs)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# NCCL tuning for InfiniBand / Ethernet clusters
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0           # set 1 if no InfiniBand
export NCCL_SOCKET_IFNAME=eth0     # adjust to your cluster's interface name

# W&B — set your API key as a SLURM secret or load from ~/.netrc
# export WANDB_API_KEY="..."

# ── Multi-node rendezvous ─────────────────────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

NNODES="${SLURM_NNODES:-1}"
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

mkdir -p logs/slurm

echo "=========================================="
echo "Job:     $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Config:  $CONFIG"
echo "Nodes:   $NNODES x $NPROC_PER_NODE GPU"
echo "Master:  $MASTER_ADDR:$MASTER_PORT"
echo "=========================================="

# ── Launch ───────────────────────────────────────────────────────────────────
torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="$SLURM_JOB_ID" \
    src/graphslm_ids/offline/training/train_hgt_flow_classifier.py \
    --config "$CONFIG" \
    "$@"
