#!/bin/bash
#SBATCH --job-name=phase2_calibration
#SBATCH --output=logs/phase2_%j.out
#SBATCH --error=logs/phase2_%j.err
#SBATCH --partition=debug
#SBATCH --account=engr-class-any
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@300

# The Internal Tribunal - Phase 2 SLURM Job
# Handles checkpointing and automatic job chaining

set -e

# Directories
WORK_DIR="/project/scratch01/compiling/a.a.baggio/internal_tribunal"
CHECKPOINT_DIR="${WORK_DIR}/checkpoints"
RESULTS_DIR="${WORK_DIR}/results_phase2"

# HuggingFace cache to scratch (avoid quota issues)
export HF_HOME="${WORK_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export CHECKPOINT_DIR="${CHECKPOINT_DIR}"

# Create directories
mkdir -p "${WORK_DIR}/logs" "${CHECKPOINT_DIR}" "${RESULTS_DIR}"

cd "${WORK_DIR}"

echo "=========================================="
echo "PHASE 2: Enhanced Calibration"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Date: $(date)"
echo "Working dir: ${WORK_DIR}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "=========================================="

# Check GPU availability
nvidia-smi --query-gpu=name,memory.total --format=csv

# Activate virtual environment
if [ -f "${WORK_DIR}/.venv/bin/activate" ]; then
    source "${WORK_DIR}/.venv/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${WORK_DIR}/.venv"
    echo "Run setup_cluster.sh first!"
    exit 1
fi

# Run the phase 2 script
echo ""
echo "Starting Phase 2 calibration..."
echo ""

python phase2_calibration.py

# Check if we need to resubmit
if [ -f "${CHECKPOINT_DIR}/RESUBMIT_FLAG" ]; then
    echo ""
    echo "[RESUBMIT] Checkpoint detected, resubmitting job..."
    sbatch run_phase2.sbatch
else
    echo ""
    echo "[COMPLETE] Phase 2 finished successfully!"
    echo ""
    echo "Results saved to: ${RESULTS_DIR}"
    ls -la "${RESULTS_DIR}/"
fi

echo ""
echo "Job finished at: $(date)"

