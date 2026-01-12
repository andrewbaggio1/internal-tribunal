#!/bin/bash
#SBATCH --job-name=attribution
#SBATCH --output=logs/attribution_%j.out
#SBATCH --error=logs/attribution_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=debug
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --account=engr-class-any

# Attribution Ablation: Finding upstream source of L14 IVF signal

echo "=========================================="
echo "ATTRIBUTION ABLATION EXPERIMENT"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo ""

# Setup environment
module load python/3.9
module load cuda/12.1.1

# Set cache directories to scratch
export HF_HOME=/project/scratch01/compiling/a.a.baggio/hf_cache
export TRANSFORMERS_CACHE=/project/scratch01/compiling/a.a.baggio/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install dependencies
pip install --user -q transformer_lens
pip install --user -q einops
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

# Create logs and output directories
mkdir -p logs
mkdir -p results_attribution

# Navigate to project directory
cd /project/scratch01/compiling/a.a.baggio/internal_tribunal

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run the experiment
python3 run_attribution_ablation.py

echo ""
echo "=========================================="
echo "COMPLETED"
echo "End time: $(date)"
echo "=========================================="


