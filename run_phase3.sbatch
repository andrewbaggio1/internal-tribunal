#!/bin/bash
#SBATCH --job-name=phase3_intervention
#SBATCH --output=logs/phase3_%j.out
#SBATCH --error=logs/phase3_%j.err
#SBATCH --partition=debug
#SBATCH --account=engr-class-any
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# The Internal Tribunal - Phase 3 SLURM Job
# Causal intervention experiments (shorter runtime needed)

set -e

# Directories
WORK_DIR="/project/scratch01/compiling/a.a.baggio/internal_tribunal"
CHECKPOINT_DIR="${WORK_DIR}/checkpoints"
RESULTS_DIR="${WORK_DIR}/results_phase3"

# HuggingFace cache to scratch (avoid quota issues)
export HF_HOME="${WORK_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export CHECKPOINT_DIR="${CHECKPOINT_DIR}"

# Create directories
mkdir -p "${WORK_DIR}/logs" "${RESULTS_DIR}"

cd "${WORK_DIR}"

echo "=========================================="
echo "PHASE 3: Causal Intervention"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Date: $(date)"
echo "Working dir: ${WORK_DIR}"
echo "=========================================="

# Check GPU availability
nvidia-smi --query-gpu=name,memory.total --format=csv

# Activate virtual environment
if [ -f "${WORK_DIR}/.venv/bin/activate" ]; then
    source "${WORK_DIR}/.venv/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${WORK_DIR}/.venv"
    exit 1
fi

# Run Phase 3
echo ""
echo "Starting Phase 3 intervention experiments..."
echo ""

python phase3_intervention.py

echo ""
echo "[COMPLETE] Phase 3 finished!"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
ls -la "${RESULTS_DIR}/"

echo ""
echo "Job finished at: $(date)"

