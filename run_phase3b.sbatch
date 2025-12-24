#!/bin/bash
#SBATCH --job-name=phase3b_refined
#SBATCH --output=logs/phase3b_%j.out
#SBATCH --error=logs/phase3b_%j.err
#SBATCH --partition=debug
#SBATCH --account=engr-class-any
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# The Internal Tribunal - Phase 3b SLURM Job

set -e

WORK_DIR="/project/scratch01/compiling/a.a.baggio/internal_tribunal"

export HF_HOME="${WORK_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export CHECKPOINT_DIR="${WORK_DIR}/checkpoints"

mkdir -p "${WORK_DIR}/logs" "${WORK_DIR}/results_phase3b"

cd "${WORK_DIR}"

echo "=========================================="
echo "PHASE 3b: Refined Steering & Per-Token Fix"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Date: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total --format=csv

if [ -f "${WORK_DIR}/.venv/bin/activate" ]; then
    source "${WORK_DIR}/.venv/bin/activate"
else
    echo "ERROR: Virtual environment not found"
    exit 1
fi

python phase3b_refined.py

echo ""
echo "[COMPLETE] Phase 3b finished!"
ls -la "${WORK_DIR}/results_phase3b/"
echo "Job finished at: $(date)"

