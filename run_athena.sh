#!/bin/bash
# ============================================================================
#  Qwen2.5-VL PSS inference — SLURM job script for PLGrid Athena
#  Partition: plgrid-gpu-a100  |  1× NVIDIA A100 80 GB SXM
#
#  Submission:
#    sbatch run_athena.sh
#
#  Prerequisites (run once before submitting):
#    1. Download model  →  see download_model.sh
#    2. Transfer data   →  see sync_to_athena.sh (run from local machine)
#    3. Create venv     →  see setup_env.sh (run as interactive job)
# ============================================================================

#SBATCH --job-name=qwen_pss
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=TWÓJ_GRANT-gpu-a100     # ← replace with your grant number
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/qwen_pss_%j.out
#SBATCH --error=logs/qwen_pss_%j.err

set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────────────
REPO="$SCRATCH/magisterka/page-stream-segmentation"
DATA="$SCRATCH/magisterka/data"
MODEL="$SCRATCH/models/qwen2.5-vl-7b"
ENV="$SCRATCH/qwen_env"
LOGS="$REPO/logs"

mkdir -p "$LOGS"

echo "=========================================="
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  Started  : $(date)"
echo "=========================================="

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load Python/3.11.5-GCCcore-13.2.0   # check available versions: module avail Python

source "$ENV/bin/activate"

# Redirect HuggingFace cache to scratch (home dir has insufficient space)
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# CUDA tuning
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Python : $(python --version)"
echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

# ── Smoke-test (optional — remove after verification) ───────────────────────
# Runs only 3 documents to verify the setup works:
#
# python "$REPO/src/inference/run.py" \
#     --benchmark "$DATA/benchmarks/poly_seq/large/test.csv" \
#     --assets    "$DATA/assets" \
#     --output    "$DATA/pss_results/qwen7b_smoketest" \
#     --model     "$MODEL" \
#     --batch-size 1 \
#     --limit     3

# ── Full run ─────────────────────────────────────────────────────────────────
python "$REPO/src/inference/run.py" \
    --benchmark  "$DATA/benchmarks/poly_seq/large/test.csv" \
    --assets     "$DATA/assets" \
    --output     "$DATA/pss_results/qwen7b" \
    --model      "$MODEL" \
    --batch-size 1 \
    --resume

echo ""
echo "Finished: $(date)"
