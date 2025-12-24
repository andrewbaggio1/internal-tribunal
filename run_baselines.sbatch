#!/bin/bash
#SBATCH --job-name=baselines_urgent
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --partition=debug
#SBATCH --account=engr-class-any
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# URGENT: MATS Deadline Baseline Experiments

set -e

WORK_DIR="/project/scratch01/compiling/a.a.baggio/internal_tribunal"

export HF_HOME="${WORK_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export CHECKPOINT_DIR="${WORK_DIR}/checkpoints"

mkdir -p "${WORK_DIR}/logs" "${WORK_DIR}/results_baselines"

cd "${WORK_DIR}"

echo "=========================================="
echo "URGENT: Rigorous Baselines for MATS"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Date: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total --format=csv

source "${WORK_DIR}/.venv/bin/activate"

python rigorous_baselines.py

echo ""
echo "[COMPLETE] Baseline experiments finished!"
ls -la "${WORK_DIR}/results_baselines/"
echo "Job finished at: $(date)"

