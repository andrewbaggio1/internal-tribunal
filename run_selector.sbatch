#!/bin/bash
#SBATCH --job-name=selector
#SBATCH --account=engr-class-any
#SBATCH --partition=debug
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:2
#SBATCH --mem=64G
#SBATCH --output=/project/scratch01/compiling/a.a.baggio/internal_tribunal/logs/selector_%j.out
#SBATCH --error=/project/scratch01/compiling/a.a.baggio/internal_tribunal/logs/selector_%j.err

echo "=========================================="
echo "EXPERIMENT A: THE SELECTOR"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Date: $(date)"
echo "=========================================="

# Setup
export HF_HOME=/project/scratch01/compiling/a.a.baggio/hf_cache
cd /project/scratch01/compiling/a.a.baggio/internal_tribunal

# Activate environment and install dependencies
source ~/.bashrc
echo "[SETUP] Installing dependencies..."
pip install --user --upgrade numpy transformers datasets scikit-learn tqdm accelerate
echo "[SETUP] Dependencies installed."

# Show GPUs
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run experiment
python3 run_selector.py

echo ""
echo "=========================================="
echo "SELECTOR COMPLETE"
echo "=========================================="

