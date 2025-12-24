#!/usr/bin/env python3
"""
The Internal Tribunal - RIGOROUS BASELINES
==========================================
Critical experiments for MATS submission:

Experiment A: Random Vector Baseline
- 5 random unit vectors vs IVF vector
- Same steering strength (+1.0)
- Recovery Rate on incorrect traces

Experiment B: Do No Harm Safety Check  
- Apply IVF steering to CORRECT traces
- Measure Preservation Rate (% that stay correct)
- Must be >90% for useful intervention

Author: Internal Tribunal Research Project
DEADLINE: 36 hours to MATS submission
"""

import os
import sys
import re
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "best_layer": 14,
    "steering_strength": 1.0,  # Fixed at +1.0 as specified
    "num_random_vectors": 5,
    "num_incorrect_samples": 100,  # For recovery rate
    "num_correct_samples": 100,    # For preservation rate
    "max_new_tokens": 512,
    "temperature": 0.7,
}

CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints"))
OUTPUT_DIR = Path("./results_baselines")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA STRUCTURES (must match Phase 2)
# ============================================================================
@dataclass
class TraceResult:
    prompt: str
    question: str
    full_response: str
    extracted_answer: Optional[str]
    ground_truth: str
    is_correct: bool
    num_tokens: int = 0


@dataclass 
class Phase2State:
    stage: str = "init"
    traces: List[TraceResult] = field(default_factory=list)
    correct_count: int = 0
    incorrect_count: int = 0
    processed_indices: List[int] = field(default_factory=list)
    activations: Optional[Dict] = None
    probe_results: Optional[Dict] = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def extract_numerical_answer(text: str) -> Optional[str]:
    text = text.strip()
    patterns = [
        r'\\boxed\{([^}]+)\}',
        r'####\s*([+-]?\$?[\d,]+\.?\d*)',
        r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+\$?([+-]?[\d,]+\.?\d*)',
        r'[Aa]nswer[:\s=]+\$?([+-]?[\d,]+\.?\d*)',
        r'=\s*\$?([+-]?[\d,]+\.?\d*)\s*$',
        r'([+-]?\$?[\d,]+\.?\d*)\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1).replace('$', '').replace(',', '').strip()
            try:
                if '.' in answer:
                    return str(float(answer))
                else:
                    return str(int(answer))
            except ValueError:
                continue
    return None


def check_answer_correctness(extracted: Optional[str], ground_truth: str) -> bool:
    if extracted is None:
        return False
    try:
        ext_val = float(extracted.replace('$', '').replace(',', ''))
        truth_val = float(ground_truth.replace('$', '').replace(',', ''))
        return abs(ext_val - truth_val) < 1e-6
    except ValueError:
        return extracted.strip().lower() == ground_truth.strip().lower()


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"[MODEL] Loading {CONFIG['model_name']}...")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    print(f"[MODEL] Loaded. Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def load_ivf_direction():
    """Load the IVF direction from Phase 3 results."""
    direction_path = Path("./results_phase3/ivf_direction.npy")
    if not direction_path.exists():
        raise FileNotFoundError(f"IVF direction not found at {direction_path}")
    
    direction = np.load(direction_path)
    print(f"[IVF] Loaded direction vector, shape: {direction.shape}, norm: {np.linalg.norm(direction):.4f}")
    return direction


def generate_random_vectors(ivf_direction: np.ndarray, num_vectors: int = 5) -> List[np.ndarray]:
    """
    Generate random unit vectors of the same dimension as IVF.
    Optionally make them orthogonal to IVF (Gram-Schmidt).
    """
    dim = ivf_direction.shape[0]
    random_vectors = []
    
    for i in range(num_vectors):
        # Generate random vector
        vec = np.random.randn(dim).astype(np.float32)
        
        # Make orthogonal to IVF (optional but more rigorous)
        # vec = vec - np.dot(vec, ivf_direction) * ivf_direction
        
        # Normalize to unit length (same as IVF)
        vec = vec / np.linalg.norm(vec)
        random_vectors.append(vec)
        
    print(f"[RANDOM] Generated {num_vectors} random unit vectors")
    return random_vectors


# ============================================================================
# STEERING HOOK
# ============================================================================
class SteeringHook:
    def __init__(self, direction: torch.Tensor, strength: float):
        self.direction = direction
        self.strength = strength
        self.enabled = True
    
    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        
        if isinstance(output, tuple):
            hidden_states = output[0]
            steering = self.direction.to(hidden_states.device, hidden_states.dtype) * self.strength
            modified = hidden_states + steering.unsqueeze(0).unsqueeze(0)
            return (modified,) + output[1:]
        else:
            steering = self.direction.to(output.device, output.dtype) * self.strength
            return output + steering.unsqueeze(0).unsqueeze(0)


# ============================================================================
# STEERING EXPERIMENT
# ============================================================================
def run_steering_experiment(
    model,
    tokenizer,
    traces: List[TraceResult],
    direction: np.ndarray,
    strength: float,
    desc: str = "Steering"
) -> Tuple[int, int]:
    """
    Run steering experiment and return (correct_count, total).
    """
    direction_tensor = torch.tensor(direction, dtype=torch.float32)
    target_layer = model.model.layers[CONFIG['best_layer']]
    
    hook = SteeringHook(direction_tensor, strength)
    handle = target_layer.register_forward_hook(hook)
    
    correct_count = 0
    total = 0
    
    try:
        for trace in tqdm(traces, desc=desc, leave=False):
            try:
                with torch.no_grad():
                    input_ids = tokenizer.encode(trace.prompt, return_tensors="pt").to("cuda")
                    
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=CONFIG['max_new_tokens'],
                        temperature=CONFIG['temperature'],
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    
                    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    generated = response[len(trace.prompt):].strip()
                
                extracted = extract_numerical_answer(generated)
                is_correct = check_answer_correctness(extracted, trace.ground_truth)
                
                if is_correct:
                    correct_count += 1
                total += 1
                
            except Exception as e:
                print(f"[ERROR] {e}")
                total += 1
                
    finally:
        handle.remove()
    
    return correct_count, total


# ============================================================================
# EXPERIMENT A: RANDOM VECTOR BASELINE
# ============================================================================
def experiment_a_random_baseline(
    model,
    tokenizer,
    incorrect_traces: List[TraceResult],
    ivf_direction: np.ndarray,
    random_vectors: List[np.ndarray],
) -> Dict:
    """
    Compare IVF vector vs random vectors on recovery rate.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Random Vector Baseline")
    print("=" * 60)
    print(f"Testing on {len(incorrect_traces)} incorrect traces")
    print(f"Steering strength: +{CONFIG['steering_strength']}")
    
    results = {
        "ivf": {"correct": 0, "total": 0, "rate": 0.0},
        "random_vectors": [],
        "random_avg_rate": 0.0,
        "baseline_no_steering": {"correct": 0, "total": 0, "rate": 0.0},
    }
    
    # 1. Baseline (no steering)
    print("\n[A.0] Running baseline (no steering)...")
    correct, total = run_steering_experiment(
        model, tokenizer, incorrect_traces,
        ivf_direction, strength=0.0, desc="Baseline"
    )
    results["baseline_no_steering"] = {
        "correct": correct, "total": total, 
        "rate": correct / total if total > 0 else 0
    }
    print(f"  Baseline: {correct}/{total} = {results['baseline_no_steering']['rate']:.1%}")
    
    # 2. IVF Vector
    print("\n[A.1] Running IVF vector steering...")
    correct, total = run_steering_experiment(
        model, tokenizer, incorrect_traces,
        ivf_direction, strength=CONFIG['steering_strength'], desc="IVF +1.0"
    )
    results["ivf"] = {
        "correct": correct, "total": total,
        "rate": correct / total if total > 0 else 0
    }
    print(f"  IVF: {correct}/{total} = {results['ivf']['rate']:.1%}")
    
    # 3. Random Vectors
    print(f"\n[A.2] Running {len(random_vectors)} random vector baselines...")
    random_rates = []
    
    for i, rand_vec in enumerate(random_vectors):
        correct, total = run_steering_experiment(
            model, tokenizer, incorrect_traces,
            rand_vec, strength=CONFIG['steering_strength'], desc=f"Random {i+1}"
        )
        rate = correct / total if total > 0 else 0
        random_rates.append(rate)
        results["random_vectors"].append({
            "index": i + 1,
            "correct": correct,
            "total": total,
            "rate": rate
        })
        print(f"  Random {i+1}: {correct}/{total} = {rate:.1%}")
    
    results["random_avg_rate"] = np.mean(random_rates)
    results["random_std_rate"] = np.std(random_rates)
    
    print(f"\n[A] SUMMARY:")
    print(f"  Baseline (no steering): {results['baseline_no_steering']['rate']:.1%}")
    print(f"  IVF Vector:             {results['ivf']['rate']:.1%}")
    print(f"  Random Vectors (avg):   {results['random_avg_rate']:.1%} ± {results['random_std_rate']:.1%}")
    print(f"  IVF Advantage:          {(results['ivf']['rate'] - results['random_avg_rate'])*100:+.1f}%")
    
    return results


# ============================================================================
# EXPERIMENT B: DO NO HARM SAFETY CHECK
# ============================================================================
def experiment_b_preservation(
    model,
    tokenizer,
    correct_traces: List[TraceResult],
    ivf_direction: np.ndarray,
) -> Dict:
    """
    Test preservation rate on already-correct traces.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Do No Harm Safety Check")
    print("=" * 60)
    print(f"Testing on {len(correct_traces)} correct traces")
    print(f"Steering strength: +{CONFIG['steering_strength']}")
    
    results = {
        "baseline": {"correct": 0, "total": 0, "rate": 0.0},
        "with_steering": {"correct": 0, "total": 0, "rate": 0.0},
        "preservation_rate": 0.0,
    }
    
    # 1. Baseline (no steering) - verify they're still correct
    print("\n[B.1] Verifying baseline (no steering)...")
    correct, total = run_steering_experiment(
        model, tokenizer, correct_traces,
        ivf_direction, strength=0.0, desc="Baseline"
    )
    results["baseline"] = {
        "correct": correct, "total": total,
        "rate": correct / total if total > 0 else 0
    }
    print(f"  Baseline preservation: {correct}/{total} = {results['baseline']['rate']:.1%}")
    
    # 2. With IVF Steering
    print("\n[B.2] Running with IVF steering (+1.0)...")
    correct, total = run_steering_experiment(
        model, tokenizer, correct_traces,
        ivf_direction, strength=CONFIG['steering_strength'], desc="IVF +1.0"
    )
    results["with_steering"] = {
        "correct": correct, "total": total,
        "rate": correct / total if total > 0 else 0
    }
    results["preservation_rate"] = results["with_steering"]["rate"]
    
    print(f"  With steering: {correct}/{total} = {results['with_steering']['rate']:.1%}")
    
    print(f"\n[B] SUMMARY:")
    print(f"  Baseline (no steering): {results['baseline']['rate']:.1%}")
    print(f"  With IVF steering:      {results['with_steering']['rate']:.1%}")
    print(f"  Preservation Rate:      {results['preservation_rate']:.1%}")
    
    if results['preservation_rate'] >= 0.90:
        print(f"  ✓ PASSED: Preservation Rate >= 90%")
    else:
        print(f"  ✗ FAILED: Preservation Rate < 90%")
    
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("THE INTERNAL TRIBUNAL - RIGOROUS BASELINES")
    print("=" * 70)
    print("MATS Deadline Critical Experiments")
    print("=" * 70)
    
    # Load checkpoint
    checkpoint_path = CHECKPOINT_DIR / "phase2_checkpoint.pkl"
    if not checkpoint_path.exists():
        print(f"[ERROR] Phase 2 checkpoint not found")
        sys.exit(1)
    
    print(f"\n[LOAD] Loading Phase 2 checkpoint...")
    with open(checkpoint_path, "rb") as f:
        phase2_state = pickle.load(f)
    traces = phase2_state.traces
    print(f"[LOAD] Loaded {len(traces)} traces")
    
    # Separate correct and incorrect traces
    correct_traces = [t for t in traces if t.is_correct]
    incorrect_traces = [t for t in traces if not t.is_correct]
    print(f"[DATA] Correct: {len(correct_traces)}, Incorrect: {len(incorrect_traces)}")
    
    # Sample for experiments
    incorrect_sample = incorrect_traces[:CONFIG['num_incorrect_samples']]
    correct_sample = correct_traces[:CONFIG['num_correct_samples']]
    
    # Load IVF direction
    ivf_direction = load_ivf_direction()
    
    # Generate random vectors
    random_vectors = generate_random_vectors(ivf_direction, CONFIG['num_random_vectors'])
    
    # Load model
    print("\n" + "=" * 50)
    print("LOADING MODEL")
    print("=" * 50)
    model, tokenizer = load_model()
    
    # Run Experiment A
    exp_a_results = experiment_a_random_baseline(
        model, tokenizer, incorrect_sample, ivf_direction, random_vectors
    )
    
    # Run Experiment B
    exp_b_results = experiment_b_preservation(
        model, tokenizer, correct_sample, ivf_direction
    )
    
    # Compile final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)
    
    print("\n| Method | Recovery Rate (Incorrect→Correct) | Preservation Rate (Correct→Correct) |")
    print("|:-------|:----------------------------------|:------------------------------------|")
    print(f"| **IVF Vector (+1.0)** | {exp_a_results['ivf']['rate']:.1%} | {exp_b_results['preservation_rate']:.1%} |")
    print(f"| Random Vector (Avg) | {exp_a_results['random_avg_rate']:.1%} ± {exp_a_results['random_std_rate']:.1%} | N/A |")
    print(f"| No Steering (Baseline) | {exp_a_results['baseline_no_steering']['rate']:.1%} | {exp_b_results['baseline']['rate']:.1%} |")
    
    # Save results
    all_results = {
        "config": CONFIG,
        "experiment_a": exp_a_results,
        "experiment_b": exp_b_results,
        "summary": {
            "ivf_recovery_rate": exp_a_results['ivf']['rate'],
            "random_avg_recovery_rate": exp_a_results['random_avg_rate'],
            "ivf_advantage": exp_a_results['ivf']['rate'] - exp_a_results['random_avg_rate'],
            "preservation_rate": exp_b_results['preservation_rate'],
            "pass_random_baseline": exp_a_results['ivf']['rate'] > exp_a_results['random_avg_rate'] + exp_a_results['random_std_rate'],
            "pass_preservation": exp_b_results['preservation_rate'] >= 0.90,
        }
    }
    
    with open(OUTPUT_DIR / "baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Final verdict
    print("\n" + "=" * 70)
    print("EXPERIMENT VERDICT")
    print("=" * 70)
    
    if all_results["summary"]["pass_random_baseline"]:
        print("✓ EXPERIMENT A PASSED: IVF >> Random (scientifically valid)")
    else:
        print("✗ EXPERIMENT A FAILED: IVF not significantly better than random")
    
    if all_results["summary"]["pass_preservation"]:
        print("✓ EXPERIMENT B PASSED: Preservation >= 90% (safe intervention)")
    else:
        print("✗ EXPERIMENT B FAILED: Preservation < 90% (harmful intervention)")
    
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Keyboard interrupt")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

