#!/usr/bin/env python3
"""
The Internal Tribunal - Phase 3b: Refined Steering & Per-Token Fix
===================================================================
1. Fix BFloat16 per-token analysis bug
2. Fine-grained steering sweep around optimal (+0.75 to +2.5)
3. More samples for statistical significance

Author: Internal Tribunal Research Project
"""

import os
import sys
import re
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "best_layer": 14,
    "best_method": "last_10_mean",
    # Finer-grained steering around the +1.0 optimum
    "steering_strengths": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
    "num_intervention_samples": 100,  # More samples for significance
    "num_pertoken_samples": 30,
    "max_new_tokens": 512,
    "temperature": 0.7,
}

CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints"))
OUTPUT_DIR = Path("./results_phase3b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA STRUCTURES (must match Phase 2 exactly for pickle compatibility)
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
    """Extract the final numerical answer from model output."""
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


# ============================================================================
# LOAD DIRECTION FROM PHASE 3
# ============================================================================
def load_direction():
    """Load the IVF direction from Phase 3 results."""
    direction_path = Path("./results_phase3/ivf_direction.npy")
    if not direction_path.exists():
        raise FileNotFoundError(f"Direction not found at {direction_path}")
    
    direction = np.load(direction_path)
    print(f"[DIRECTION] Loaded direction vector, shape: {direction.shape}")
    return direction


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
            modified_hidden = hidden_states + steering.unsqueeze(0).unsqueeze(0)
            return (modified_hidden,) + output[1:]
        else:
            steering = self.direction.to(output.device, output.dtype) * self.strength
            return output + steering.unsqueeze(0).unsqueeze(0)


# ============================================================================
# FINE-GRAINED STEERING EXPERIMENT
# ============================================================================
def run_fine_steering(
    model, 
    tokenizer, 
    traces: List[TraceResult],
    direction: np.ndarray
) -> Dict:
    """Run fine-grained steering on incorrect traces only."""
    print(f"\n[STEERING] Running fine-grained steering experiment...")
    
    layer_idx = CONFIG['best_layer']
    direction_tensor = torch.tensor(direction, dtype=torch.float32)
    
    # Get incorrect traces
    incorrect_traces = [t for t in traces if not t.is_correct][:CONFIG['num_intervention_samples']]
    print(f"[STEERING] Testing on {len(incorrect_traces)} incorrect traces")
    
    target_layer = model.model.layers[layer_idx]
    
    results = {
        "strengths": CONFIG['steering_strengths'],
        "results": [],
        "baseline": None
    }
    
    def evaluate_with_steering(strength: float) -> Tuple[int, int, List[bool]]:
        hook = SteeringHook(direction_tensor, strength)
        handle = target_layer.register_forward_hook(hook)
        
        correct_count = 0
        total = 0
        outcomes = []
        
        try:
            for trace in tqdm(incorrect_traces, desc=f"Steering +{strength:.2f}", leave=False):
                prompt = trace.prompt
                
                with torch.no_grad():
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                    
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=CONFIG['max_new_tokens'],
                        temperature=CONFIG['temperature'],
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    
                    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    generated = response[len(prompt):].strip()
                
                extracted = extract_numerical_answer(generated)
                is_correct = check_answer_correctness(extracted, trace.ground_truth)
                
                outcomes.append(is_correct)
                if is_correct:
                    correct_count += 1
                total += 1
                
        finally:
            handle.remove()
        
        return correct_count, total, outcomes
    
    # Baseline (no steering)
    print("\n[STEERING] Running baseline...")
    baseline_correct, baseline_total, baseline_outcomes = evaluate_with_steering(0.0)
    baseline_rate = baseline_correct / baseline_total if baseline_total > 0 else 0
    results["baseline"] = {
        "correct": baseline_correct,
        "total": baseline_total,
        "rate": baseline_rate,
        "outcomes": baseline_outcomes
    }
    print(f"  Baseline: {baseline_correct}/{baseline_total} = {baseline_rate:.1%}")
    
    # Test each steering strength
    print("\n[STEERING] Testing fine-grained strengths...")
    for strength in CONFIG['steering_strengths']:
        print(f"\n  Strength +{strength:.2f}:")
        correct, total, outcomes = evaluate_with_steering(strength)
        rate = correct / total if total > 0 else 0
        improvement = rate - baseline_rate
        
        results["results"].append({
            "strength": strength,
            "correct": correct,
            "total": total,
            "rate": rate,
            "improvement": improvement,
            "outcomes": outcomes
        })
        print(f"    {correct}/{total} = {rate:.1%} (Δ{improvement:+.1%})")
    
    return results


# ============================================================================
# FIXED PER-TOKEN ANALYSIS
# ============================================================================
def analyze_per_token_fixed(
    model,
    tokenizer, 
    traces: List[TraceResult],
    direction: np.ndarray
) -> Dict:
    """
    Per-token analysis with proper dtype handling.
    """
    print(f"\n[PER-TOKEN] Analyzing value signal dynamics (fixed)...")
    
    layer_idx = CONFIG['best_layer']
    # Convert direction to float32 for numpy compatibility
    direction_tensor = torch.tensor(direction, dtype=torch.float32)
    
    correct_traces = [t for t in traces if t.is_correct][:CONFIG['num_pertoken_samples']]
    incorrect_traces = [t for t in traces if not t.is_correct][:CONFIG['num_pertoken_samples']]
    
    def get_token_projections(trace: TraceResult) -> Tuple[List[float], int, int]:
        """Get projection onto direction at each token position."""
        full_text = trace.prompt + trace.full_response
        
        with torch.no_grad():
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = inputs["input_ids"].to("cuda")
            prompt_len = len(tokenizer.encode(trace.prompt))
            
            outputs = model(input_ids, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[layer_idx + 1]  # [1, seq_len, hidden]
            
            # Convert to float32 before projection (FIX for BFloat16 issue)
            hidden_float = hidden_states[0].float().cpu()  # [seq_len, hidden]
            
            # Project each position onto direction
            projections = torch.matmul(hidden_float, direction_tensor)  # [seq_len]
            projections = projections.numpy().tolist()
            
            del outputs, hidden_states
        
        return projections, prompt_len, len(projections)
    
    results = {
        "correct_traces": [],
        "incorrect_traces": [],
        "summary": {}
    }
    
    print("[PER-TOKEN] Processing correct traces...")
    for trace in tqdm(correct_traces, desc="Correct traces"):
        try:
            projections, prompt_len, total_len = get_token_projections(trace)
            
            # Separate prompt vs response projections
            response_proj = projections[prompt_len:] if prompt_len < len(projections) else projections[-10:]
            
            results["correct_traces"].append({
                "question": trace.question[:80],
                "total_tokens": total_len,
                "prompt_tokens": prompt_len,
                "response_tokens": total_len - prompt_len,
                "mean_projection": float(np.mean(projections)),
                "response_mean": float(np.mean(response_proj)) if response_proj else 0,
                "final_10_mean": float(np.mean(projections[-10:])) if len(projections) >= 10 else float(np.mean(projections)),
                "trajectory_start": float(np.mean(response_proj[:len(response_proj)//4])) if len(response_proj) > 4 else 0,
                "trajectory_end": float(np.mean(response_proj[-len(response_proj)//4:])) if len(response_proj) > 4 else 0,
            })
        except Exception as e:
            print(f"[ERROR] {e}")
    
    print("[PER-TOKEN] Processing incorrect traces...")
    for trace in tqdm(incorrect_traces, desc="Incorrect traces"):
        try:
            projections, prompt_len, total_len = get_token_projections(trace)
            response_proj = projections[prompt_len:] if prompt_len < len(projections) else projections[-10:]
            
            results["incorrect_traces"].append({
                "question": trace.question[:80],
                "total_tokens": total_len,
                "prompt_tokens": prompt_len,
                "response_tokens": total_len - prompt_len,
                "mean_projection": float(np.mean(projections)),
                "response_mean": float(np.mean(response_proj)) if response_proj else 0,
                "final_10_mean": float(np.mean(projections[-10:])) if len(projections) >= 10 else float(np.mean(projections)),
                "trajectory_start": float(np.mean(response_proj[:len(response_proj)//4])) if len(response_proj) > 4 else 0,
                "trajectory_end": float(np.mean(response_proj[-len(response_proj)//4:])) if len(response_proj) > 4 else 0,
            })
        except Exception as e:
            print(f"[ERROR] {e}")
    
    # Compute summary statistics
    if results["correct_traces"] and results["incorrect_traces"]:
        correct_finals = [t["final_10_mean"] for t in results["correct_traces"]]
        incorrect_finals = [t["final_10_mean"] for t in results["incorrect_traces"]]
        correct_response = [t["response_mean"] for t in results["correct_traces"]]
        incorrect_response = [t["response_mean"] for t in results["incorrect_traces"]]
        
        # Trajectory analysis
        correct_starts = [t["trajectory_start"] for t in results["correct_traces"] if t["trajectory_start"] != 0]
        correct_ends = [t["trajectory_end"] for t in results["correct_traces"] if t["trajectory_end"] != 0]
        incorrect_starts = [t["trajectory_start"] for t in results["incorrect_traces"] if t["trajectory_start"] != 0]
        incorrect_ends = [t["trajectory_end"] for t in results["incorrect_traces"] if t["trajectory_end"] != 0]
        
        results["summary"] = {
            "correct_final_avg": float(np.mean(correct_finals)),
            "correct_final_std": float(np.std(correct_finals)),
            "incorrect_final_avg": float(np.mean(incorrect_finals)),
            "incorrect_final_std": float(np.std(incorrect_finals)),
            "final_separation": float(np.mean(correct_finals) - np.mean(incorrect_finals)),
            "correct_response_avg": float(np.mean(correct_response)),
            "incorrect_response_avg": float(np.mean(incorrect_response)),
            "response_separation": float(np.mean(correct_response) - np.mean(incorrect_response)),
            # Trajectory changes
            "correct_trajectory_delta": float(np.mean(correct_ends) - np.mean(correct_starts)) if correct_starts and correct_ends else 0,
            "incorrect_trajectory_delta": float(np.mean(incorrect_ends) - np.mean(incorrect_starts)) if incorrect_starts and incorrect_ends else 0,
        }
        
        print(f"\n[PER-TOKEN] Summary:")
        print(f"  Correct traces final: {results['summary']['correct_final_avg']:.4f} ± {results['summary']['correct_final_std']:.4f}")
        print(f"  Incorrect traces final: {results['summary']['incorrect_final_avg']:.4f} ± {results['summary']['incorrect_final_std']:.4f}")
        print(f"  Final separation: {results['summary']['final_separation']:.4f}")
        print(f"  Correct trajectory Δ: {results['summary']['correct_trajectory_delta']:.4f}")
        print(f"  Incorrect trajectory Δ: {results['summary']['incorrect_trajectory_delta']:.4f}")
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_phase3b(steering_results: Dict, pertoken_results: Dict):
    """Create Phase 3b visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Fine-grained steering curve
    ax1 = axes[0, 0]
    if steering_results["results"]:
        strengths = [r["strength"] for r in steering_results["results"]]
        rates = [r["rate"] for r in steering_results["results"]]
        improvements = [r["improvement"] for r in steering_results["results"]]
        
        ax1.plot(strengths, rates, 'go-', linewidth=2, markersize=8, label='With Steering')
        ax1.axhline(y=steering_results["baseline"]["rate"], color='red', linestyle='--', linewidth=2, label=f'Baseline ({steering_results["baseline"]["rate"]:.1%})')
        ax1.fill_between(strengths, steering_results["baseline"]["rate"], rates, alpha=0.3, color='green')
        ax1.set_xlabel("Steering Strength", fontsize=12)
        ax1.set_ylabel("Incorrect→Correct Rate", fontsize=12)
        ax1.set_title("Steering Effectiveness Curve", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(rates) * 1.2)
    
    # 2. Improvement vs baseline
    ax2 = axes[0, 1]
    if steering_results["results"]:
        strengths = [r["strength"] for r in steering_results["results"]]
        improvements = [r["improvement"] * 100 for r in steering_results["results"]]  # Convert to percentage
        
        colors = ['green' if i > 0 else 'red' for i in improvements]
        ax2.bar(range(len(strengths)), improvements, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xticks(range(len(strengths)))
        ax2.set_xticklabels([f"+{s}" for s in strengths])
        ax2.set_xlabel("Steering Strength", fontsize=12)
        ax2.set_ylabel("Improvement over Baseline (%)", fontsize=12)
        ax2.set_title("Improvement vs Baseline by Steering Strength", fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Per-token projection distribution
    ax3 = axes[1, 0]
    if pertoken_results.get("correct_traces") and pertoken_results.get("incorrect_traces"):
        correct_finals = [t["final_10_mean"] for t in pertoken_results["correct_traces"]]
        incorrect_finals = [t["final_10_mean"] for t in pertoken_results["incorrect_traces"]]
        
        ax3.hist(correct_finals, bins=15, alpha=0.6, label='Correct', color='green', density=True)
        ax3.hist(incorrect_finals, bins=15, alpha=0.6, label='Incorrect', color='red', density=True)
        ax3.axvline(x=np.mean(correct_finals), color='green', linestyle='--', linewidth=2, label=f'Correct μ={np.mean(correct_finals):.2f}')
        ax3.axvline(x=np.mean(incorrect_finals), color='red', linestyle='--', linewidth=2, label=f'Incorrect μ={np.mean(incorrect_finals):.2f}')
        ax3.set_xlabel("Projection onto IVF Direction", fontsize=12)
        ax3.set_ylabel("Density", fontsize=12)
        ax3.set_title("Final Token Projection Distribution", fontsize=14)
        ax3.legend(fontsize=9)
    
    # 4. Trajectory analysis
    ax4 = axes[1, 1]
    if pertoken_results.get("summary"):
        summary = pertoken_results["summary"]
        categories = ['Correct', 'Incorrect']
        starts = [summary.get("correct_trajectory_delta", 0) - summary.get("correct_trajectory_delta", 0)/2,
                  summary.get("incorrect_trajectory_delta", 0) - summary.get("incorrect_trajectory_delta", 0)/2]
        deltas = [summary.get("correct_trajectory_delta", 0), summary.get("incorrect_trajectory_delta", 0)]
        
        x = [0, 1]
        colors = ['green', 'red']
        
        for i, (cat, delta, color) in enumerate(zip(categories, deltas, colors)):
            ax4.bar(i, delta, color=color, alpha=0.7, label=f'{cat}: Δ={delta:.3f}')
        
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(categories)
        ax4.set_ylabel("Trajectory Change (start → end)", fontsize=12)
        ax4.set_title("IVF Signal Trajectory During Reasoning", fontsize=14)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase3b_results.png", dpi=150)
    plt.close()
    
    print(f"[VIZ] Saved visualization to {OUTPUT_DIR / 'phase3b_results.png'}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("THE INTERNAL TRIBUNAL - Phase 3b: Refined Analysis")
    print("=" * 70)
    
    # Load Phase 2 checkpoint
    checkpoint_path = CHECKPOINT_DIR / "phase2_checkpoint.pkl"
    if not checkpoint_path.exists():
        print(f"[ERROR] Phase 2 checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print(f"\n[LOAD] Loading Phase 2 checkpoint...")
    with open(checkpoint_path, "rb") as f:
        phase2_state = pickle.load(f)
    
    traces = phase2_state.traces
    print(f"[LOAD] Loaded {len(traces)} traces")
    
    # Load direction from Phase 3
    direction = load_direction()
    
    # Load model
    print("\n" + "=" * 50)
    print("LOADING MODEL")
    print("=" * 50)
    model, tokenizer = load_model()
    
    # Run fine-grained steering
    print("\n" + "=" * 50)
    print("STEP 1: Fine-Grained Steering Sweep")
    print("=" * 50)
    steering_results = run_fine_steering(model, tokenizer, traces, direction)
    
    # Run fixed per-token analysis
    print("\n" + "=" * 50)
    print("STEP 2: Per-Token Dynamics (Fixed)")
    print("=" * 50)
    pertoken_results = analyze_per_token_fixed(model, tokenizer, traces, direction)
    
    # Visualization
    print("\n" + "=" * 50)
    print("STEP 3: Visualization")
    print("=" * 50)
    visualize_phase3b(steering_results, pertoken_results)
    
    # Save results
    all_results = {
        "config": CONFIG,
        "steering": {
            "baseline": steering_results["baseline"],
            "results": [{k: v for k, v in r.items() if k != "outcomes"} for r in steering_results["results"]]
        },
        "pertoken": pertoken_results.get("summary", {}),
    }
    
    with open(OUTPUT_DIR / "phase3b_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Find optimal steering
    best_result = max(steering_results["results"], key=lambda x: x["rate"])
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 3b COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"\n1. Fine-Grained Steering Results:")
    print(f"   Baseline: {steering_results['baseline']['rate']:.1%}")
    print(f"   Best strength: +{best_result['strength']:.2f}")
    print(f"   Best rate: {best_result['rate']:.1%}")
    print(f"   Best improvement: {best_result['improvement']:+.1%} ({best_result['improvement']/steering_results['baseline']['rate']*100:.0f}% relative)")
    
    print(f"\n2. Steering Curve:")
    for r in steering_results["results"]:
        bar = "█" * int(r["rate"] * 40)
        print(f"   +{r['strength']:.2f}: {r['rate']:.1%} {bar}")
    
    if pertoken_results.get("summary"):
        print(f"\n3. Per-Token Analysis:")
        s = pertoken_results["summary"]
        print(f"   Correct traces: μ={s['correct_final_avg']:.4f}, σ={s['correct_final_std']:.4f}")
        print(f"   Incorrect traces: μ={s['incorrect_final_avg']:.4f}, σ={s['incorrect_final_std']:.4f}")
        print(f"   Separation: {s['final_separation']:.4f}")
        
        if s['correct_trajectory_delta'] > s['incorrect_trajectory_delta']:
            print(f"   ✓ Correct traces show stronger positive trajectory!")
    
    print("\n" + "=" * 70)
    
    # Determine next steps
    if best_result['improvement'] > 0.05:
        print("✓ STRONG CAUSAL EFFECT: Ready for Phase 4 (real-time monitoring)")
    elif best_result['improvement'] > 0:
        print("◐ MODERATE EFFECT: Consider refining direction or trying other layers")
    else:
        print("✗ NO IMPROVEMENT: Need to investigate further")
    
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

