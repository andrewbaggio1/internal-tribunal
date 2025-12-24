#!/bin/bash
# =============================================================================
# The Internal Tribunal - Cluster Setup Script
# Run this once to set up the environment on WashU cluster
# =============================================================================

set -e

# Configuration
PROJECT_NAME="internal_tribunal"
PROJECT_DIR="/project/scratch01/compiling/a.a.baggio/${PROJECT_NAME}"

echo "=============================================="
echo "Setting up The Internal Tribunal project"
echo "Target directory: ${PROJECT_DIR}"
echo "=============================================="

# Create project structure
mkdir -p "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/checkpoints"
mkdir -p "${PROJECT_DIR}/results"

# Copy files to project directory
echo "Copying project files..."
cp calibration_phase.py "${PROJECT_DIR}/"
cp run_calibration.sbatch "${PROJECT_DIR}/"
cp requirements.txt "${PROJECT_DIR}/"

# Load Python module
module load python/3.10
module load cuda/12.1

# Create virtual environment
echo "Creating virtual environment..."
cd "${PROJECT_DIR}"

if [ -d ".venv" ]; then
    echo "Virtual environment already exists, updating..."
else
    python -m venv .venv
fi

source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo "Installing dependencies..."
pip install transformers accelerate transformer_lens
pip install datasets scikit-learn numpy tqdm matplotlib

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "To submit the job:"
echo "  cd ${PROJECT_DIR}"
echo "  sbatch run_calibration.sbatch"
echo ""
echo "To monitor the job:"
echo "  squeue -u \$USER"
echo "  tail -f logs/calibration-*.out"
echo "=============================================="

