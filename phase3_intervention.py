#!/usr/bin/env python3
"""
The Internal Tribunal - Phase 3: Causal Intervention & Per-Token Analysis
=========================================================================
Using the discovered Implicit Value Function direction to:
1. Extract the direction vector from the trained probe
2. Test causal intervention (activation steering)
3. Analyze per-token dynamics of the value signal

Author: Internal Tribunal Research Project
"""

import os
import sys
import re
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "best_layer": 14,  # From Phase 2 results
    "best_method": "last_10_mean",
    "steering_strengths": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],  # Multipliers for direction
    "num_intervention_samples": 50,  # Samples for intervention experiments
    "num_pertoken_samples": 20,  # Samples for per-token analysis
    "max_new_tokens": 512,
    "temperature": 0.7,  # Lower for more consistent comparisons
}

CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints"))
OUTPUT_DIR = Path("./results_phase3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA STRUCTURES (must match Phase 2 exactly for pickle compatibility)
# ============================================================================
from dataclasses import field

@dataclass
class TraceResult:
    """Container for a single CoT trace result."""
    prompt: str
    question: str
    full_response: str
    extracted_answer: Optional[str]
    ground_truth: str
    is_correct: bool
    num_tokens: int = 0


@dataclass 
class Phase2State:
    """Checkpoint state for Phase 2 - needed for pickle compatibility."""
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
    """Check if extracted answer matches ground truth."""
    if extracted is None:
        return False
    try:
        ext_val = float(extracted.replace('$', '').replace(',', ''))
        truth_val = float(ground_truth.replace('$', '').replace(',', ''))
        return abs(ext_val - truth_val) < 1e-6
    except ValueError:
        return extracted.strip().lower() == ground_truth.strip().lower()


def format_prompt(question: str, tokenizer) -> str:
    """Format question using model's chat template."""
    messages = [{"role": "user", "content": f"Solve this math problem step by step. Show your reasoning, then give the final answer.\n\nQuestion: {question}"}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSolve this math problem step by step.\n\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model():
    """Load model with multi-GPU sharding."""
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
# PHASE 1: EXTRACT DIRECTION VECTOR
# ============================================================================
def extract_direction_vector(traces: List[TraceResult], model, tokenizer) -> Tuple[np.ndarray, float]:
    """
    Train a probe on the best layer/method and extract the direction vector.
    Returns the direction vector and the probe accuracy.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    print(f"\n[DIRECTION] Extracting direction from layer {CONFIG['best_layer']} with {CONFIG['best_method']} aggregation...")
    
    layer_idx = CONFIG['best_layer']
    activations = []
    labels = []
    
    for trace in tqdm(traces[:200], desc="Extracting activations for direction"):  # Use subset for speed
        full_text = trace.prompt + trace.full_response
        
        try:
            with torch.no_grad():
                inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
                input_ids = inputs["input_ids"].to("cuda")
                seq_len = input_ids.shape[1]
                
                outputs = model(input_ids, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states
                
                # Get activation from target layer
                act = hidden_states[layer_idx + 1]  # +1 because index 0 is embeddings
                
                # Aggregate based on method
                if CONFIG['best_method'] == "last":
                    vec = act[0, -1, :]
                elif CONFIG['best_method'] == "last_10_mean":
                    n = min(10, seq_len)
                    vec = act[0, -n:, :].mean(dim=0)
                else:  # mean
                    vec = act[0, :, :].mean(dim=0)
                
                activations.append(vec.cpu().float().numpy())
                labels.append(1 if trace.is_correct else 0)
                
                del outputs, hidden_states
                
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
    
    X = np.stack(activations, axis=0)
    y = np.array(labels)
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    
    accuracy = clf.score(X_test_scaled, y_test)
    print(f"[DIRECTION] Probe accuracy: {accuracy:.1%}")
    
    # The direction vector is the probe weights (in original space)
    # We need to un-scale: direction = clf.coef_ @ scaler.scale_^(-1)
    direction = clf.coef_[0] / scaler.scale_
    direction = direction / np.linalg.norm(direction)  # Normalize
    
    print(f"[DIRECTION] Direction vector shape: {direction.shape}, norm: {np.linalg.norm(direction):.4f}")
    
    # Save direction
    np.save(OUTPUT_DIR / "ivf_direction.npy", direction)
    
    # Also save scaler mean for proper centering during intervention
    np.save(OUTPUT_DIR / "activation_mean.npy", scaler.mean_)
    
    return direction, accuracy


# ============================================================================
# PHASE 2: CAUSAL INTERVENTION (ACTIVATION STEERING)
# ============================================================================
class SteeringHook:
    """Hook to add steering vector to activations."""
    def __init__(self, direction: torch.Tensor, strength: float):
        self.direction = direction
        self.strength = strength
        self.enabled = True
    
    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        
        # For LlamaDecoderLayer, output can be:
        # - tuple (hidden_states,) or (hidden_states, present_key_value, ...)
        # - or just hidden_states tensor in some configs
        if isinstance(output, tuple):
            hidden_states = output[0]
            
            # Add steering vector to all positions
            steering = self.direction.to(hidden_states.device, hidden_states.dtype) * self.strength
            modified_hidden = hidden_states + steering.unsqueeze(0).unsqueeze(0)
            
            # Reconstruct output tuple
            return (modified_hidden,) + output[1:]
        else:
            # output is just a tensor
            steering = self.direction.to(output.device, output.dtype) * self.strength
            return output + steering.unsqueeze(0).unsqueeze(0)


def run_intervention_experiment(
    model, 
    tokenizer, 
    traces: List[TraceResult],
    direction: np.ndarray
) -> Dict:
    """
    Run causal intervention experiments:
    1. Take incorrect traces and try to steer toward correct
    2. Take correct traces and try to steer toward incorrect
    """
    print(f"\n[INTERVENTION] Running steering experiments...")
    
    layer_idx = CONFIG['best_layer']
    direction_tensor = torch.tensor(direction, dtype=torch.float32)
    
    # Separate correct and incorrect traces
    correct_traces = [t for t in traces if t.is_correct][:CONFIG['num_intervention_samples']]
    incorrect_traces = [t for t in traces if not t.is_correct][:CONFIG['num_intervention_samples']]
    
    results = {
        "steering_strengths": CONFIG['steering_strengths'],
        "incorrect_to_correct": [],  # Steering incorrect traces with positive direction
        "correct_to_incorrect": [],  # Steering correct traces with negative direction
        "baseline_correct_rate": 0.0,
        "baseline_incorrect_rate": 0.0,
    }
    
    # Get the target layer module
    target_layer = model.model.layers[layer_idx]
    
    def evaluate_with_steering(traces: List[TraceResult], strength: float) -> Tuple[int, int]:
        """Generate with steering and count correct answers."""
        hook = SteeringHook(direction_tensor, strength)
        handle = target_layer.register_forward_hook(hook)
        
        correct_count = 0
        total = 0
        
        try:
            for trace in tqdm(traces, desc=f"Steering {strength:+.1f}", leave=False):
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
                
                if is_correct:
                    correct_count += 1
                total += 1
                
        finally:
            handle.remove()
        
        return correct_count, total
    
    # Baseline (no steering)
    print("\n[INTERVENTION] Baseline (no steering)...")
    
    # For incorrect traces baseline
    baseline_inc_correct, baseline_inc_total = evaluate_with_steering(incorrect_traces, 0.0)
    results["baseline_incorrect_rate"] = baseline_inc_correct / baseline_inc_total if baseline_inc_total > 0 else 0
    print(f"  Incorrect traces baseline: {baseline_inc_correct}/{baseline_inc_total} = {results['baseline_incorrect_rate']:.1%} now correct")
    
    # For correct traces baseline  
    baseline_cor_correct, baseline_cor_total = evaluate_with_steering(correct_traces, 0.0)
    results["baseline_correct_rate"] = baseline_cor_correct / baseline_cor_total if baseline_cor_total > 0 else 0
    print(f"  Correct traces baseline: {baseline_cor_correct}/{baseline_cor_total} = {results['baseline_correct_rate']:.1%} still correct")
    
    # Test different steering strengths
    print("\n[INTERVENTION] Testing steering strengths...")
    
    for strength in CONFIG['steering_strengths']:
        print(f"\n  Strength: {strength:+.1f}")
        
        # Steer incorrect traces with positive strength (toward "correct")
        if strength > 0:
            inc_correct, inc_total = evaluate_with_steering(incorrect_traces, strength)
            inc_rate = inc_correct / inc_total if inc_total > 0 else 0
            results["incorrect_to_correct"].append({
                "strength": strength,
                "correct": inc_correct,
                "total": inc_total,
                "rate": inc_rate
            })
            print(f"    Incorrect→Correct: {inc_correct}/{inc_total} = {inc_rate:.1%}")
        
        # Steer correct traces with negative strength (toward "incorrect")
        if strength < 0:
            cor_correct, cor_total = evaluate_with_steering(correct_traces, strength)
            cor_rate = cor_correct / cor_total if cor_total > 0 else 0
            results["correct_to_incorrect"].append({
                "strength": strength,
                "correct": cor_correct,
                "total": cor_total,
                "rate": cor_rate
            })
            print(f"    Correct→Incorrect: {cor_correct}/{cor_total} = {cor_rate:.1%}")
    
    return results


# ============================================================================
# PHASE 3: PER-TOKEN ANALYSIS
# ============================================================================
def analyze_per_token_dynamics(
    model,
    tokenizer, 
    traces: List[TraceResult],
    direction: np.ndarray
) -> Dict:
    """
    Analyze how the projection onto the IVF direction evolves during reasoning.
    """
    print(f"\n[PER-TOKEN] Analyzing value signal dynamics...")
    
    layer_idx = CONFIG['best_layer']
    direction_tensor = torch.tensor(direction, dtype=torch.float32)
    
    correct_traces = [t for t in traces if t.is_correct][:CONFIG['num_pertoken_samples']]
    incorrect_traces = [t for t in traces if not t.is_correct][:CONFIG['num_pertoken_samples']]
    
    def get_token_projections(trace: TraceResult) -> Tuple[List[float], List[str]]:
        """Get projection onto direction at each token position."""
        full_text = trace.prompt + trace.full_response
        
        with torch.no_grad():
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = inputs["input_ids"].to("cuda")
            
            outputs = model(input_ids, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[layer_idx + 1]  # [1, seq_len, hidden]
            
            # Project each position onto direction
            direction_gpu = direction_tensor.to(hidden_states.device, hidden_states.dtype)
            projections = torch.matmul(hidden_states[0], direction_gpu)  # [seq_len]
            
            projections = projections.cpu().numpy().tolist()
            
            # Get token strings for reference
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
            
            del outputs, hidden_states
        
        return projections, tokens
    
    results = {
        "correct_traces": [],
        "incorrect_traces": [],
        "summary": {}
    }
    
    print("[PER-TOKEN] Processing correct traces...")
    for trace in tqdm(correct_traces, desc="Correct traces"):
        try:
            projections, tokens = get_token_projections(trace)
            results["correct_traces"].append({
                "question": trace.question[:100],
                "projections": projections,
                "num_tokens": len(projections),
                "mean_projection": np.mean(projections),
                "final_projection": np.mean(projections[-10:]) if len(projections) >= 10 else np.mean(projections),
            })
        except Exception as e:
            print(f"[ERROR] {e}")
    
    print("[PER-TOKEN] Processing incorrect traces...")
    for trace in tqdm(incorrect_traces, desc="Incorrect traces"):
        try:
            projections, tokens = get_token_projections(trace)
            results["incorrect_traces"].append({
                "question": trace.question[:100],
                "projections": projections,
                "num_tokens": len(projections),
                "mean_projection": np.mean(projections),
                "final_projection": np.mean(projections[-10:]) if len(projections) >= 10 else np.mean(projections),
            })
        except Exception as e:
            print(f"[ERROR] {e}")
    
    # Compute summary statistics
    correct_means = [t["mean_projection"] for t in results["correct_traces"]]
    incorrect_means = [t["mean_projection"] for t in results["incorrect_traces"]]
    correct_finals = [t["final_projection"] for t in results["correct_traces"]]
    incorrect_finals = [t["final_projection"] for t in results["incorrect_traces"]]
    
    results["summary"] = {
        "correct_mean_avg": np.mean(correct_means) if correct_means else 0,
        "correct_mean_std": np.std(correct_means) if correct_means else 0,
        "incorrect_mean_avg": np.mean(incorrect_means) if incorrect_means else 0,
        "incorrect_mean_std": np.std(incorrect_means) if incorrect_means else 0,
        "correct_final_avg": np.mean(correct_finals) if correct_finals else 0,
        "incorrect_final_avg": np.mean(incorrect_finals) if incorrect_finals else 0,
        "separation": (np.mean(correct_finals) - np.mean(incorrect_finals)) if (correct_finals and incorrect_finals) else 0
    }
    
    print(f"\n[PER-TOKEN] Summary:")
    print(f"  Correct traces - Mean projection: {results['summary']['correct_mean_avg']:.4f} ± {results['summary']['correct_mean_std']:.4f}")
    print(f"  Incorrect traces - Mean projection: {results['summary']['incorrect_mean_avg']:.4f} ± {results['summary']['incorrect_mean_std']:.4f}")
    print(f"  Final token separation: {results['summary']['separation']:.4f}")
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_phase3(intervention_results: Dict, pertoken_results: Dict):
    """Create Phase 3 visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Intervention results - Incorrect to Correct
    ax1 = axes[0, 0]
    if intervention_results["incorrect_to_correct"]:
        strengths = [r["strength"] for r in intervention_results["incorrect_to_correct"]]
        rates = [r["rate"] for r in intervention_results["incorrect_to_correct"]]
        ax1.bar(range(len(strengths)), rates, color='green', alpha=0.7)
        ax1.axhline(y=intervention_results["baseline_incorrect_rate"], color='red', linestyle='--', label='Baseline')
        ax1.set_xticks(range(len(strengths)))
        ax1.set_xticklabels([f"+{s}" for s in strengths])
        ax1.set_xlabel("Steering Strength")
        ax1.set_ylabel("Correct Rate")
        ax1.set_title("Steering Incorrect→Correct")
        ax1.legend()
        ax1.set_ylim(0, 1)
    
    # 2. Intervention results - Correct to Incorrect
    ax2 = axes[0, 1]
    if intervention_results["correct_to_incorrect"]:
        strengths = [r["strength"] for r in intervention_results["correct_to_incorrect"]]
        rates = [r["rate"] for r in intervention_results["correct_to_incorrect"]]
        ax2.bar(range(len(strengths)), rates, color='red', alpha=0.7)
        ax2.axhline(y=intervention_results["baseline_correct_rate"], color='green', linestyle='--', label='Baseline')
        ax2.set_xticks(range(len(strengths)))
        ax2.set_xticklabels([f"{s}" for s in strengths])
        ax2.set_xlabel("Steering Strength")
        ax2.set_ylabel("Correct Rate")
        ax2.set_title("Steering Correct→Incorrect (negative = toward incorrect)")
        ax2.legend()
        ax2.set_ylim(0, 1)
    
    # 3. Per-token projection distributions
    ax3 = axes[1, 0]
    correct_finals = [t["final_projection"] for t in pertoken_results["correct_traces"]]
    incorrect_finals = [t["final_projection"] for t in pertoken_results["incorrect_traces"]]
    
    if correct_finals and incorrect_finals:
        ax3.hist(correct_finals, bins=15, alpha=0.6, label='Correct', color='green')
        ax3.hist(incorrect_finals, bins=15, alpha=0.6, label='Incorrect', color='red')
        ax3.axvline(x=np.mean(correct_finals), color='green', linestyle='--', linewidth=2)
        ax3.axvline(x=np.mean(incorrect_finals), color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel("Projection onto IVF Direction")
        ax3.set_ylabel("Count")
        ax3.set_title("Final Token Projections Distribution")
        ax3.legend()
    
    # 4. Example per-token trajectory
    ax4 = axes[1, 1]
    if pertoken_results["correct_traces"] and pertoken_results["incorrect_traces"]:
        # Plot a few example trajectories
        for i, trace in enumerate(pertoken_results["correct_traces"][:3]):
            proj = trace["projections"]
            x = np.linspace(0, 1, len(proj))  # Normalize x-axis
            ax4.plot(x, proj, color='green', alpha=0.5, linewidth=1)
        
        for i, trace in enumerate(pertoken_results["incorrect_traces"][:3]):
            proj = trace["projections"]
            x = np.linspace(0, 1, len(proj))
            ax4.plot(x, proj, color='red', alpha=0.5, linewidth=1)
        
        ax4.set_xlabel("Position in Sequence (normalized)")
        ax4.set_ylabel("Projection onto IVF Direction")
        ax4.set_title("Per-Token Value Signal Trajectories")
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', label='Correct'),
            Line2D([0], [0], color='red', label='Incorrect')
        ]
        ax4.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase3_results.png", dpi=150)
    plt.close()
    
    print(f"[VIZ] Saved visualization to {OUTPUT_DIR / 'phase3_results.png'}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("THE INTERNAL TRIBUNAL - Phase 3: Causal Intervention")
    print("=" * 70)
    print(f"Best layer: {CONFIG['best_layer']}")
    print(f"Best method: {CONFIG['best_method']}")
    print("=" * 70)
    
    # Load Phase 2 checkpoint to get traces
    checkpoint_path = CHECKPOINT_DIR / "phase2_checkpoint.pkl"
    if not checkpoint_path.exists():
        print(f"[ERROR] Phase 2 checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print(f"\n[LOAD] Loading Phase 2 checkpoint...")
    with open(checkpoint_path, "rb") as f:
        phase2_state = pickle.load(f)
    
    traces = phase2_state.traces
    print(f"[LOAD] Loaded {len(traces)} traces")
    
    # Load model
    print("\n" + "=" * 50)
    print("LOADING MODEL")
    print("=" * 50)
    model, tokenizer = load_model()
    
    # Phase 1: Extract direction vector
    print("\n" + "=" * 50)
    print("STEP 1: Extract Direction Vector")
    print("=" * 50)
    direction, probe_accuracy = extract_direction_vector(traces, model, tokenizer)
    
    # Phase 2: Causal intervention
    print("\n" + "=" * 50)
    print("STEP 2: Causal Intervention Experiments")
    print("=" * 50)
    intervention_results = run_intervention_experiment(model, tokenizer, traces, direction)
    
    # Phase 3: Per-token analysis
    print("\n" + "=" * 50)
    print("STEP 3: Per-Token Dynamics Analysis")
    print("=" * 50)
    pertoken_results = analyze_per_token_dynamics(model, tokenizer, traces, direction)
    
    # Visualization
    print("\n" + "=" * 50)
    print("STEP 4: Visualization")
    print("=" * 50)
    visualize_phase3(intervention_results, pertoken_results)
    
    # Save all results
    all_results = {
        "config": CONFIG,
        "probe_accuracy": probe_accuracy,
        "intervention": intervention_results,
        "pertoken_summary": pertoken_results["summary"],
    }
    
    with open(OUTPUT_DIR / "phase3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"\n1. Direction Vector:")
    print(f"   - Extracted from layer {CONFIG['best_layer']} with {CONFIG['best_method']}")
    print(f"   - Probe accuracy: {probe_accuracy:.1%}")
    print(f"   - Saved to: {OUTPUT_DIR / 'ivf_direction.npy'}")
    
    print(f"\n2. Causal Intervention:")
    print(f"   - Baseline incorrect→correct rate: {intervention_results['baseline_incorrect_rate']:.1%}")
    print(f"   - Baseline correct→incorrect rate: {1 - intervention_results['baseline_correct_rate']:.1%}")
    if intervention_results["incorrect_to_correct"]:
        best_inc = max(intervention_results["incorrect_to_correct"], key=lambda x: x["rate"])
        print(f"   - Best steering (incorrect→correct): +{best_inc['strength']} → {best_inc['rate']:.1%}")
    if intervention_results["correct_to_incorrect"]:
        worst_cor = min(intervention_results["correct_to_incorrect"], key=lambda x: x["rate"])
        print(f"   - Best steering (correct→incorrect): {worst_cor['strength']} → {worst_cor['rate']:.1%}")
    
    print(f"\n3. Per-Token Dynamics:")
    print(f"   - Correct traces final projection: {pertoken_results['summary']['correct_final_avg']:.4f}")
    print(f"   - Incorrect traces final projection: {pertoken_results['summary']['incorrect_final_avg']:.4f}")
    print(f"   - Separation: {pertoken_results['summary']['separation']:.4f}")
    
    # Interpret results
    print("\n" + "=" * 70)
    
    if intervention_results["incorrect_to_correct"]:
        improvement = max(r["rate"] for r in intervention_results["incorrect_to_correct"]) - intervention_results["baseline_incorrect_rate"]
        if improvement > 0.1:
            print("✓ CAUSAL EFFECT CONFIRMED: Steering improves incorrect→correct rate!")
        elif improvement > 0:
            print("◐ WEAK CAUSAL EFFECT: Some improvement from steering")
        else:
            print("○ NO CAUSAL EFFECT: Steering did not improve accuracy")
    
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

