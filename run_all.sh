#!/bin/bash
# ============================================================================
# The Internal Tribunal - Master Launcher
# ============================================================================
# Runs all three experiments in parallel on different GPUs
# 
# Hardware Allocation:
#   GPUs 0-1: Selector (Best-of-N)
#   GPU 2:    Destruction (Negative steering)
#   GPU 3:    Upstream (Layer sweep)
# ============================================================================

set -e

echo "============================================================================"
echo "THE INTERNAL TRIBUNAL - PARALLEL EXPERIMENTS"
echo "============================================================================"
echo "Starting at: $(date)"
echo "============================================================================"

# Create results directory
mkdir -p results_parallel

# Change to script directory
cd "$(dirname "$0")"

echo ""
echo "[1/3] Launching Selector experiment (GPUs 0-1)..."
CUDA_VISIBLE_DEVICES=0,1 python3 run_selector.py > results_parallel/selector.log 2>&1 &
PID_SELECTOR=$!
echo "  PID: $PID_SELECTOR"

echo ""
echo "[2/3] Launching Destruction experiment (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 python3 run_destruction.py > results_parallel/destruction.log 2>&1 &
PID_DESTRUCTION=$!
echo "  PID: $PID_DESTRUCTION"

echo ""
echo "[3/3] Launching Upstream experiment (GPU 3)..."
CUDA_VISIBLE_DEVICES=3 python3 run_upstream.py > results_parallel/upstream.log 2>&1 &
PID_UPSTREAM=$!
echo "  PID: $PID_UPSTREAM"

echo ""
echo "============================================================================"
echo "All experiments launched!"
echo "============================================================================"
echo "Monitor with:"
echo "  tail -f results_parallel/selector.log"
echo "  tail -f results_parallel/destruction.log"
echo "  tail -f results_parallel/upstream.log"
echo ""
echo "Or check GPU usage with: nvidia-smi"
echo "============================================================================"

# Wait for all to complete
echo ""
echo "Waiting for all experiments to complete..."
wait $PID_SELECTOR
echo "  Selector complete (exit: $?)"
wait $PID_DESTRUCTION  
echo "  Destruction complete (exit: $?)"
wait $PID_UPSTREAM
echo "  Upstream complete (exit: $?)"

echo ""
echo "============================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================================"
echo "Finished at: $(date)"
echo ""
echo "Results:"
echo "  - results_selector.json"
echo "  - results_destruction.json"
echo "  - results_upstream.json"
echo "============================================================================"

