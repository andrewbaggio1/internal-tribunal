#!/bin/bash
#SBATCH --job-name=tribunal_calibration
#SBATCH --partition=debug
#SBATCH --account=engr-class-any
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=logs/calibration-%j.out
#SBATCH --error=logs/calibration-%j.err

# =============================================================================
# The Internal Tribunal - Phase 1: Calibration
# SLURM submission script with self-resubmission for long jobs
# =============================================================================

# Setup paths
PROJECT_DIR="/project/scratch01/compiling/a.a.baggio/internal_tribunal"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
LOGS_DIR="${PROJECT_DIR}/logs"
VENV_DIR="${PROJECT_DIR}/.venv"
RESUBMIT_FLAG="${CHECKPOINT_DIR}/RESUBMIT_FLAG"

# Create directories
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${LOGS_DIR}"

# Export checkpoint dir for Python script
export CHECKPOINT_DIR="${CHECKPOINT_DIR}"

# Set HuggingFace cache to scratch space (avoid home quota issues)
export HF_HOME="/project/scratch01/compiling/a.a.baggio/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

# Use system Python (modules may not be available on all nodes)
# module load python/3.10
# module load cuda/12.1
export PATH="/usr/bin:$PATH"

# Print job info
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo "Working Directory: ${PROJECT_DIR}"
echo "=============================================="

# Activate virtual environment
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# Install/upgrade dependencies
echo "Checking dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch transformers accelerate
pip install --quiet nnsight
pip install --quiet datasets scikit-learn numpy tqdm matplotlib

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Change to project directory
cd "${PROJECT_DIR}"

# Remove resubmit flag if exists from previous run
rm -f "${RESUBMIT_FLAG}"

# Run the calibration script
echo "Starting calibration..."
python calibration_phase.py

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Script completed with exit code: ${EXIT_CODE}"
echo "End Time: $(date)"
echo "=============================================="

# Check if we need to resubmit (script was interrupted by SLURM signal)
if [ -f "${RESUBMIT_FLAG}" ]; then
    echo "Resubmit flag found, scheduling continuation..."
    rm -f "${RESUBMIT_FLAG}"
    
    # Small delay to avoid rapid resubmission
    sleep 5
    
    # Resubmit this script
    sbatch "$0"
    echo "Continuation job submitted."
fi

exit ${EXIT_CODE}

