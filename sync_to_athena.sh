#!/bin/bash
# ============================================================================
#  Transfer data to PLGrid Athena
#  Run from the magisterka/ directory on your local machine.
#
#  Usage:
#    ./sync_to_athena.sh YOUR_PLGRID_LOGIN
#
#  Example:
#    ./sync_to_athena.sh plgkowalski
#
#  What is transferred:
#    • data/assets/        — PNG files only (page images), ~10–30 GB
#                            skipped: original PDFs, OCR text files (.txt/.md)
#    • data/benchmarks/    — benchmark CSVs and JSONs, ~tens of MB
#    • page-stream-segmentation/src/   — source code
#    • page-stream-segmentation/requirements.txt
#    • page-stream-segmentation/run_athena.sh
#
#  After transfer, run on Athena:
#    sbatch $SCRATCH/magisterka/page-stream-segmentation/run_athena.sh
# ============================================================================

set -euo pipefail

LOGIN="${1:-}"
if [[ -z "$LOGIN" ]]; then
    echo "Usage: $0 YOUR_PLGRID_LOGIN"
    echo "Example: $0 plgkowalski"
    exit 1
fi

REMOTE="${LOGIN}@athena.cyfronet.pl"
REMOTE_BASE="\$SCRATCH/magisterka"    # path on Athena (evaluated via SSH)

echo "============================================"
echo "  Transfer → ${REMOTE}:${REMOTE_BASE}"
echo "============================================"
echo ""

# ── 1. Create directory structure on Athena ──────────────────────────────────
echo "[1/4] Creating directories on Athena..."
ssh "${REMOTE}" "mkdir -p \$SCRATCH/magisterka/data/assets \$SCRATCH/magisterka/data/benchmarks \$SCRATCH/magisterka/page-stream-segmentation"
echo "OK"
echo ""

# ── 2. Transfer page PNGs (images only, no PDFs or OCR text) ─────────────────
echo "[2/4] Transferring page images (PNG)..."
echo "      (estimated size: 10–30 GB, may take several minutes)"
rsync -avz --progress \
    --include="*/"         \
    --include="*.png"      \
    --exclude="*"          \
    "data/assets/"         \
    "${REMOTE}:\$SCRATCH/magisterka/data/assets/"
echo ""

# ── 3. Transfer benchmark files ───────────────────────────────────────────────
echo "[3/4] Transferring benchmark files..."
rsync -avz --progress \
    "data/benchmarks/" \
    "${REMOTE}:\$SCRATCH/magisterka/data/benchmarks/"
echo ""

# ── 4. Transfer source code ───────────────────────────────────────────────────
echo "[4/4] Transferring source code..."
rsync -avz --progress \
    --exclude="__pycache__" \
    --exclude="*.pyc"       \
    --exclude=".ipynb_checkpoints" \
    "page-stream-segmentation/src/"              \
    "${REMOTE}:\$SCRATCH/magisterka/page-stream-segmentation/src/"

rsync -avz --progress \
    "page-stream-segmentation/requirements.txt"  \
    "page-stream-segmentation/run_athena.sh"     \
    "${REMOTE}:\$SCRATCH/magisterka/page-stream-segmentation/"
echo ""

echo "============================================"
echo "  Transfer complete: $(date)"
echo "============================================"
echo ""
echo "Next steps on Athena:"
echo ""
echo "  # 1. Log in"
echo "  ssh ${REMOTE}"
echo ""
echo "  # 2. Check available Python modules"
echo "  module avail Python"
echo ""
echo "  # 3. Create virtual environment (once)"
echo "  module load Python/3.11.5-GCCcore-13.2.0"
echo "  python -m venv \$SCRATCH/qwen_env"
echo "  source \$SCRATCH/qwen_env/bin/activate"
echo "  pip install -r \$SCRATCH/magisterka/page-stream-segmentation/requirements.txt"
echo "  pip install transformers==4.49.0   # required version for Qwen2.5-VL"
echo "  pip install flash-attn --no-build-isolation"
echo ""
echo "  # 4. Download model (as an interactive job)"
echo "  srun --partition=plgrid-gpu-a100 --account=YOUR_GRANT-gpu-a100 \\"
echo "       --gres=gpu:1 --time=02:00:00 --pty bash"
echo "  source \$SCRATCH/qwen_env/bin/activate"
echo "  export HF_HOME=\$SCRATCH/.cache/huggingface"
echo "  huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \\"
echo "      --local-dir \$SCRATCH/models/qwen2.5-vl-7b"
echo ""
echo "  # 5. Edit grant name in run_athena.sh, then:"
echo "  sbatch \$SCRATCH/magisterka/page-stream-segmentation/run_athena.sh"
