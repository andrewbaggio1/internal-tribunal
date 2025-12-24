# The Internal Tribunal - Phase 1: Calibration

Finding the **Implicit Value Function** direction in DeepSeek-R1's activation space.

## Hypothesis

Reasoning models like DeepSeek-R1 maintain an internal "Implicit Value Function" — a direction in activation space that tracks whether the current Chain of Thought (CoT) is leading toward a correct answer.

## Methodology

1. **Generate CoT Traces**: Run the model on GSM8K math problems with high temperature to get a mix of correct and incorrect reasoning trajectories
2. **Label Trajectories**: Parse final answers and compare to ground truth to label each trace as "Correct" or "Incorrect"
3. **Extract Activations**: Collect residual stream activations from middle layers (12, 16, 20, 24)
4. **Train Linear Probe**: Fit logistic regression to distinguish correct vs incorrect trajectories
5. **Evaluate**: If probe accuracy > 70%, we have evidence of an implicit value signal

## Files

- `calibration_phase.py` - Main experiment script
- `run_calibration.sbatch` - SLURM job submission script for WashU cluster
- `setup_cluster.sh` - One-time setup script for the cluster environment
- `requirements.txt` - Python dependencies

## Running Locally (if you have 4x GPUs)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run calibration
python calibration_phase.py
```

## Running on WashU Cluster

```bash
# SSH to cluster
ssh washu-cluster

# Navigate to your project area
cd /project/scratch01/compiling/a.a.baggio/

# Clone/copy project files
mkdir -p internal_tribunal
cd internal_tribunal
# (copy files here)

# Run setup (once)
bash setup_cluster.sh

# Submit job
sbatch run_calibration.sbatch

# Monitor
squeue -u $USER
tail -f logs/calibration-*.out
```

## Outputs

After successful completion:

- `results/probe_results.png` - Visualization of probe accuracy and logit separation
- `results/implicit_value_directions.json` - The discovered direction vectors (the "Implicit Value Function")
- `checkpoints/` - Intermediate state for job resumption

## Checkpointing & Job Chaining

The script handles SLURM's 4-hour time limit through:

1. **Signal handling**: Catches SIGUSR1 sent 5 minutes before timeout
2. **Checkpointing**: Saves all state to disk
3. **Self-resubmission**: Sets a flag that triggers job resubmission
4. **Resume**: New job loads checkpoint and continues from where it left off

## Key Parameters

In `calibration_phase.py`:

```python
TARGET_CORRECT = 50       # Number of correct traces to collect
TARGET_INCORRECT = 50     # Number of incorrect traces to collect
MAX_ATTEMPTS = 200        # Max problems to try
TEMPERATURE = 1.0         # High temp encourages mistakes
TARGET_LAYERS = [12, 16, 20, 24]  # Layers to probe
```

## Expected Results

If the hypothesis is correct, we expect:
- Probe accuracy significantly above chance (50%)
- Accuracy > 70% indicates a strong implicit value signal
- The learned direction vector can be used in Phase 2 for runtime monitoring

## Troubleshooting

**Out of Memory**: The script handles OOM gracefully, but if persistent:
- Reduce `max_new_tokens` in trace generation
- Use fewer target layers
- Enable more aggressive truncation

**Model Loading Fails**: The script tries multiple loading strategies:
1. Native transformer_lens multi-GPU
2. Fallback to single-GPU loading

**Probe Accuracy Low**: Consider:
- Collecting more data (increase targets)
- Different aggregation ("last" instead of "mean")
- Different layers (try earlier or later)
- Per-token probing instead of trajectory-level

