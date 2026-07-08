#!/bin/bash
# =============================================================================
#  PLGrid SLURM job — Qwen2.5-VL PSS inference
# =============================================================================
#
#  HOW TO USE:
#    1. Upload your project to PLGrid (scp or git clone)
#    2. Edit the PATHS section below
#    3. Submit:   sbatch scripts/plgrid_qwen.sh
#    4. Monitor:  squeue -u $USER
#    5. Logs:     tail -f logs/slurm-<JOBID>.out
#
#  CHECK AVAILABLE PARTITIONS first:
#    sinfo -o "%P %G %l %D"
#  Common partitions on Ares:
#    plgrid-gpu-a100   → A100 40 GB
#    plgrid-gpu        → V100 / mixed
# =============================================================================

#SBATCH --job-name=qwen_pss
#SBATCH --partition=plgrid-gpu-a100       # <-- change if using a different cluster
#SBATCH --gres=gpu:1                      # 1× A100 40 GB (sufficient for 7B)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00                   # 8h should be enough for ~500 docs
#SBATCH --output=logs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adrian.lusztyk92@gmail.com

# =============================================================================
#  PATHS — adjust to your PLGrid environment
# =============================================================================
MAGISTERKA="$HOME/projects/magisterka"
PROJECT_DIR="$MAGISTERKA/page-stream-segmentation"
CONDA_ENV="pss"          # conda environment name (created below if it does not exist)
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

BENCHMARK="$MAGISTERKA/data/benchmarks/poly_seq/large/test.csv"
ASSETS="$MAGISTERKA/data/assets"
OUTPUT="$MAGISTERKA/data/pss_results/qwen7b"

# =============================================================================
#  SETUP
# =============================================================================
set -e
mkdir -p logs

echo "======================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURM_NODELIST"
echo "  GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "  Start time : $(date)"
echo "======================================================"

# Load modules (adjust to your PLGrid cluster)
module load miniconda3/23.3.1 2>/dev/null || module load miniconda3 2>/dev/null || true

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "Creating conda environment '${CONDA_ENV}' …"
    conda create -y -n "$CONDA_ENV" python=3.11
fi

# Activate environment
source activate "$CONDA_ENV" || conda activate "$CONDA_ENV"

# Install dependencies (idempotent)
echo "Installing dependencies …"
pip install --quiet --upgrade pip

# PyTorch with CUDA 12.1 (standard on PLGrid A100)
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
pip install --quiet \
    transformers>=4.45.0 \
    accelerate>=0.30.0 \
    qwen-vl-utils \
    scikit-learn \
    pandas \
    loguru \
    tqdm \
    pillow \
    flash-attn --no-build-isolation 2>/dev/null || true  # optional, speeds up inference

# HuggingFace cache — store models on scratch (faster I/O, more space)
export HF_HOME="${SCRATCH:-$HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"

echo "HF cache: $HF_HOME"
echo "Downloading / loading model: $MODEL"

# =============================================================================
#  RUN INFERENCE
# =============================================================================
cd "$PROJECT_DIR"

python src/inference/run.py \
    --benchmark "$BENCHMARK" \
    --assets    "$ASSETS" \
    --output    "$OUTPUT" \
    --model     "$MODEL" \
    --batch-size 1 \
    --resume

echo "======================================================"
echo "  End time : $(date)"
echo "  Results  : $OUTPUT"
echo "======================================================"
