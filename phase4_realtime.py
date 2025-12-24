#!/usr/bin/env python3
"""
The Internal Tribunal - Phase 4: Real-Time Monitoring
======================================================
Monitor the IVF signal during generation to:
1. Track how "reasoning confidence" evolves token-by-token
2. Detect when the model is going off-track BEFORE the final answer
3. Compare signal trajectories for correct vs incorrect outcomes
4. Test adaptive steering (intervene when signal drops)

Author: Internal Tribunal Research Project
"""

import os
import sys
import re
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable

import numpy as np
import torch
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "best_layer": 14,
    "num_monitoring_samples": 50,
    "num_adaptive_samples": 30,
    "max_new_tokens": 512,
    "temperature": 0.7,
    "steering_strength": 0.75,  # Use middle of effective range
    "signal_threshold_percentile": 25,  # Trigger steering when signal below this
}

CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints"))
OUTPUT_DIR = Path("./results_phase4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA STRUCTURES
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


@dataclass
class MonitoredGeneration:
    """Result of a monitored generation with per-token IVF signal."""
    question: str
    prompt: str
    generated_text: str
    extracted_answer: Optional[str]
    ground_truth: str
    is_correct: bool
    token_signals: List[float]  # IVF projection at each generated token
    token_texts: List[str]  # The actual tokens
    signal_stats: Dict[str, float]  # Summary statistics


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


def load_direction():
    """Load the IVF direction from Phase 3 results."""
    direction_path = Path("./results_phase3/ivf_direction.npy")
    if not direction_path.exists():
        raise FileNotFoundError(f"Direction not found at {direction_path}")
    
    direction = np.load(direction_path)
    print(f"[DIRECTION] Loaded direction vector, shape: {direction.shape}")
    return direction


# ============================================================================
# REAL-TIME MONITORING HOOK
# ============================================================================
class MonitoringHook:
    """Hook to capture and optionally steer activations during generation."""
    
    def __init__(self, direction: torch.Tensor, steering_strength: float = 0.0):
        self.direction = direction
        self.steering_strength = steering_strength
        self.signals = []  # Store IVF projections
        self.enabled = True
        self.steering_enabled = False
        self.signal_threshold = None  # For adaptive steering
        
    def reset(self):
        self.signals = []
        
    def enable_steering(self, threshold: float = None):
        self.steering_enabled = True
        self.signal_threshold = threshold
        
    def disable_steering(self):
        self.steering_enabled = False
        
    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # Get the last token's activation (during generation)
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden]
        
        # Project onto IVF direction (convert to float32 for computation)
        direction_gpu = self.direction.to(last_hidden.device, torch.float32)
        last_hidden_float = last_hidden.float()
        signal = torch.matmul(last_hidden_float, direction_gpu).item()
        self.signals.append(signal)
        
        # Adaptive steering: if signal below threshold, apply steering
        if self.steering_enabled and self.signal_threshold is not None:
            if signal < self.signal_threshold:
                steering = self.direction.to(hidden_states.device, hidden_states.dtype) * self.steering_strength
                if isinstance(output, tuple):
                    modified = hidden_states + steering.unsqueeze(0).unsqueeze(0)
                    return (modified,) + output[1:]
                else:
                    return hidden_states + steering.unsqueeze(0).unsqueeze(0)
        
        return output


# ============================================================================
# MONITORED GENERATION
# ============================================================================
def generate_with_monitoring(
    model,
    tokenizer,
    prompt: str,
    direction: np.ndarray,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    steering_strength: float = 0.0,
    signal_threshold: float = None,
) -> Tuple[str, List[float], List[str]]:
    """
    Generate text while monitoring the IVF signal at each token.
    Optionally apply adaptive steering when signal drops below threshold.
    """
    direction_tensor = torch.tensor(direction, dtype=torch.float32)
    
    # Set up monitoring hook
    hook = MonitoringHook(direction_tensor, steering_strength)
    if signal_threshold is not None:
        hook.enable_steering(signal_threshold)
    
    target_layer = model.model.layers[CONFIG['best_layer']]
    handle = target_layer.register_forward_hook(hook)
    
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        
        # Generate token by token to capture signals
        generated_ids = input_ids.clone()
        token_texts = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(generated_ids, use_cache=False)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Sample next token
                if temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                token_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
                token_texts.append(token_text)
                
                # Stop at EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_text = full_response[len(prompt):].strip()
        
        # Note: signals list will have len = number of generated tokens
        # (each forward pass adds one signal)
        signals = hook.signals[len(tokenizer.encode(prompt)):]  # Only generation signals
        
    finally:
        handle.remove()
    
    return generated_text, signals, token_texts


# ============================================================================
# EXPERIMENT 1: SIGNAL TRAJECTORY ANALYSIS
# ============================================================================
def analyze_signal_trajectories(
    model,
    tokenizer,
    traces: List[TraceResult],
    direction: np.ndarray,
) -> Dict:
    """
    Generate responses while monitoring IVF signal.
    Compare trajectories for correct vs incorrect outcomes.
    """
    print(f"\n[TRAJECTORY] Analyzing signal trajectories...")
    
    correct_traces = [t for t in traces if t.is_correct][:CONFIG['num_monitoring_samples'] // 2]
    incorrect_traces = [t for t in traces if not t.is_correct][:CONFIG['num_monitoring_samples'] // 2]
    
    results = {
        "correct": [],
        "incorrect": [],
        "summary": {}
    }
    
    def process_traces(trace_list: List[TraceResult], label: str) -> List[MonitoredGeneration]:
        monitored = []
        for trace in tqdm(trace_list, desc=f"Monitoring {label}"):
            try:
                generated, signals, tokens = generate_with_monitoring(
                    model, tokenizer, trace.prompt, direction,
                    max_new_tokens=CONFIG['max_new_tokens'],
                    temperature=CONFIG['temperature'],
                )
                
                extracted = extract_numerical_answer(generated)
                is_correct = check_answer_correctness(extracted, trace.ground_truth)
                
                # Compute signal statistics
                if signals:
                    signal_stats = {
                        "mean": float(np.mean(signals)),
                        "std": float(np.std(signals)),
                        "min": float(np.min(signals)),
                        "max": float(np.max(signals)),
                        "start": float(np.mean(signals[:min(10, len(signals))])),
                        "end": float(np.mean(signals[-min(10, len(signals)):])),
                        "trend": float(np.mean(signals[-10:]) - np.mean(signals[:10])) if len(signals) >= 20 else 0,
                        "num_tokens": len(signals),
                    }
                else:
                    signal_stats = {"mean": 0, "std": 0, "min": 0, "max": 0, "start": 0, "end": 0, "trend": 0, "num_tokens": 0}
                
                monitored.append(MonitoredGeneration(
                    question=trace.question,
                    prompt=trace.prompt,
                    generated_text=generated[:500],  # Truncate for storage
                    extracted_answer=extracted,
                    ground_truth=trace.ground_truth,
                    is_correct=is_correct,
                    token_signals=signals[:200] if signals else [],  # Store first 200 for analysis
                    token_texts=tokens[:200] if tokens else [],
                    signal_stats=signal_stats,
                ))
                
            except Exception as e:
                print(f"[ERROR] {e}")
                
        return monitored
    
    # Process both groups
    results["correct"] = process_traces(correct_traces, "correct")
    results["incorrect"] = process_traces(incorrect_traces, "incorrect")
    
    # Compute summary statistics
    correct_means = [m.signal_stats["mean"] for m in results["correct"] if m.signal_stats["mean"] != 0]
    incorrect_means = [m.signal_stats["mean"] for m in results["incorrect"] if m.signal_stats["mean"] != 0]
    correct_trends = [m.signal_stats["trend"] for m in results["correct"]]
    incorrect_trends = [m.signal_stats["trend"] for m in results["incorrect"]]
    
    if correct_means and incorrect_means:
        results["summary"] = {
            "correct_mean_signal": float(np.mean(correct_means)),
            "correct_std_signal": float(np.std(correct_means)),
            "incorrect_mean_signal": float(np.mean(incorrect_means)),
            "incorrect_std_signal": float(np.std(incorrect_means)),
            "signal_separation": float(np.mean(correct_means) - np.mean(incorrect_means)),
            "correct_mean_trend": float(np.mean(correct_trends)),
            "incorrect_mean_trend": float(np.mean(incorrect_trends)),
            "trend_separation": float(np.mean(correct_trends) - np.mean(incorrect_trends)),
        }
        
        print(f"\n[TRAJECTORY] Summary:")
        print(f"  Correct signal: {results['summary']['correct_mean_signal']:.4f} ± {results['summary']['correct_std_signal']:.4f}")
        print(f"  Incorrect signal: {results['summary']['incorrect_mean_signal']:.4f} ± {results['summary']['incorrect_std_signal']:.4f}")
        print(f"  Signal separation: {results['summary']['signal_separation']:.4f}")
        print(f"  Correct trend: {results['summary']['correct_mean_trend']:.4f}")
        print(f"  Incorrect trend: {results['summary']['incorrect_mean_trend']:.4f}")
    
    return results


# ============================================================================
# EXPERIMENT 2: ADAPTIVE STEERING
# ============================================================================
def test_adaptive_steering(
    model,
    tokenizer,
    traces: List[TraceResult],
    direction: np.ndarray,
    baseline_signals: List[float],
) -> Dict:
    """
    Test adaptive steering: apply steering when signal drops below threshold.
    """
    print(f"\n[ADAPTIVE] Testing adaptive steering...")
    
    # Compute threshold from baseline (e.g., 25th percentile of correct signals)
    if baseline_signals:
        threshold = np.percentile(baseline_signals, CONFIG['signal_threshold_percentile'])
    else:
        threshold = 0.0
    
    print(f"[ADAPTIVE] Signal threshold: {threshold:.4f} (p{CONFIG['signal_threshold_percentile']})")
    
    # Get incorrect traces to try to save
    incorrect_traces = [t for t in traces if not t.is_correct][:CONFIG['num_adaptive_samples']]
    
    results = {
        "threshold": threshold,
        "baseline": {"correct": 0, "total": 0},
        "adaptive": {"correct": 0, "total": 0},
        "details": []
    }
    
    # Baseline: no steering
    print("\n[ADAPTIVE] Running baseline (no steering)...")
    for trace in tqdm(incorrect_traces, desc="Baseline"):
        try:
            generated, signals, _ = generate_with_monitoring(
                model, tokenizer, trace.prompt, direction,
                max_new_tokens=CONFIG['max_new_tokens'],
                temperature=CONFIG['temperature'],
                steering_strength=0.0,
            )
            extracted = extract_numerical_answer(generated)
            is_correct = check_answer_correctness(extracted, trace.ground_truth)
            
            results["baseline"]["total"] += 1
            if is_correct:
                results["baseline"]["correct"] += 1
                
        except Exception as e:
            print(f"[ERROR] {e}")
    
    baseline_rate = results["baseline"]["correct"] / results["baseline"]["total"] if results["baseline"]["total"] > 0 else 0
    print(f"  Baseline: {results['baseline']['correct']}/{results['baseline']['total']} = {baseline_rate:.1%}")
    
    # Adaptive: steer when signal drops
    print("\n[ADAPTIVE] Running adaptive steering...")
    for trace in tqdm(incorrect_traces, desc="Adaptive"):
        try:
            generated, signals, _ = generate_with_monitoring(
                model, tokenizer, trace.prompt, direction,
                max_new_tokens=CONFIG['max_new_tokens'],
                temperature=CONFIG['temperature'],
                steering_strength=CONFIG['steering_strength'],
                signal_threshold=threshold,
            )
            extracted = extract_numerical_answer(generated)
            is_correct = check_answer_correctness(extracted, trace.ground_truth)
            
            results["adaptive"]["total"] += 1
            if is_correct:
                results["adaptive"]["correct"] += 1
            
            results["details"].append({
                "question": trace.question[:80],
                "is_correct": is_correct,
                "num_signals_below_threshold": sum(1 for s in signals if s < threshold) if signals else 0,
            })
                
        except Exception as e:
            print(f"[ERROR] {e}")
    
    adaptive_rate = results["adaptive"]["correct"] / results["adaptive"]["total"] if results["adaptive"]["total"] > 0 else 0
    improvement = adaptive_rate - baseline_rate
    
    print(f"  Adaptive: {results['adaptive']['correct']}/{results['adaptive']['total']} = {adaptive_rate:.1%}")
    print(f"  Improvement: {improvement:+.1%}")
    
    results["baseline"]["rate"] = baseline_rate
    results["adaptive"]["rate"] = adaptive_rate
    results["improvement"] = improvement
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_phase4(trajectory_results: Dict, adaptive_results: Dict):
    """Create Phase 4 visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Signal trajectory comparison (sample traces)
    ax1 = axes[0, 0]
    # Plot a few example trajectories
    for i, m in enumerate(trajectory_results.get("correct", [])[:5]):
        if m.token_signals:
            x = np.linspace(0, 1, len(m.token_signals))
            ax1.plot(x, m.token_signals, color='green', alpha=0.4, linewidth=1)
    for i, m in enumerate(trajectory_results.get("incorrect", [])[:5]):
        if m.token_signals:
            x = np.linspace(0, 1, len(m.token_signals))
            ax1.plot(x, m.token_signals, color='red', alpha=0.4, linewidth=1)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', alpha=0.7, label='Correct'),
        Line2D([0], [0], color='red', alpha=0.7, label='Incorrect')
    ]
    ax1.legend(handles=legend_elements)
    ax1.set_xlabel("Position in Generation (normalized)", fontsize=11)
    ax1.set_ylabel("IVF Signal", fontsize=11)
    ax1.set_title("Real-Time IVF Signal Trajectories", fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # 2. Signal distribution by outcome
    ax2 = axes[0, 1]
    correct_means = [m.signal_stats["mean"] for m in trajectory_results.get("correct", []) if m.signal_stats["mean"] != 0]
    incorrect_means = [m.signal_stats["mean"] for m in trajectory_results.get("incorrect", []) if m.signal_stats["mean"] != 0]
    
    if correct_means and incorrect_means:
        ax2.hist(correct_means, bins=12, alpha=0.6, color='green', label=f'Correct (μ={np.mean(correct_means):.2f})', density=True)
        ax2.hist(incorrect_means, bins=12, alpha=0.6, color='red', label=f'Incorrect (μ={np.mean(incorrect_means):.2f})', density=True)
        ax2.axvline(x=np.mean(correct_means), color='green', linestyle='--', linewidth=2)
        ax2.axvline(x=np.mean(incorrect_means), color='red', linestyle='--', linewidth=2)
        ax2.legend()
    ax2.set_xlabel("Mean IVF Signal During Generation", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Signal Distribution by Outcome", fontsize=13)
    
    # 3. Trend comparison
    ax3 = axes[1, 0]
    correct_trends = [m.signal_stats["trend"] for m in trajectory_results.get("correct", [])]
    incorrect_trends = [m.signal_stats["trend"] for m in trajectory_results.get("incorrect", [])]
    
    if correct_trends and incorrect_trends:
        positions = [0, 1]
        means = [np.mean(correct_trends), np.mean(incorrect_trends)]
        stds = [np.std(correct_trends), np.std(incorrect_trends)]
        colors = ['green', 'red']
        
        bars = ax3.bar(positions, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xticks(positions)
        ax3.set_xticklabels(['Correct', 'Incorrect'])
        ax3.set_ylabel("Signal Trend (end - start)", fontsize=11)
        ax3.set_title("Signal Trend During Reasoning", fontsize=13)
    
    # 4. Adaptive steering results
    ax4 = axes[1, 1]
    if adaptive_results.get("baseline") and adaptive_results.get("adaptive"):
        categories = ['Baseline', 'Adaptive\nSteering']
        rates = [adaptive_results["baseline"]["rate"], adaptive_results["adaptive"]["rate"]]
        colors = ['gray', 'blue']
        
        bars = ax4.bar(categories, rates, color=colors, alpha=0.7)
        ax4.axhline(y=adaptive_results["baseline"]["rate"], color='gray', linestyle='--', alpha=0.5)
        
        # Add improvement annotation
        if adaptive_results.get("improvement", 0) > 0:
            ax4.annotate(f'+{adaptive_results["improvement"]:.1%}', 
                        xy=(1, rates[1]), xytext=(1.3, rates[1]),
                        fontsize=12, color='blue', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='blue'))
        
        ax4.set_ylabel("Incorrect→Correct Rate", fontsize=11)
        ax4.set_title("Adaptive Steering vs Baseline", fontsize=13)
        ax4.set_ylim(0, max(rates) * 1.3 if rates else 1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase4_results.png", dpi=150)
    plt.close()
    
    print(f"[VIZ] Saved visualization to {OUTPUT_DIR / 'phase4_results.png'}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("THE INTERNAL TRIBUNAL - Phase 4: Real-Time Monitoring")
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
    
    # Load direction
    direction = load_direction()
    
    # Load model
    print("\n" + "=" * 50)
    print("LOADING MODEL")
    print("=" * 50)
    model, tokenizer = load_model()
    
    # Experiment 1: Signal trajectory analysis
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: Signal Trajectory Analysis")
    print("=" * 50)
    trajectory_results = analyze_signal_trajectories(model, tokenizer, traces, direction)
    
    # Get baseline signals for threshold computation
    correct_signals = []
    for m in trajectory_results.get("correct", []):
        correct_signals.extend(m.token_signals)
    
    # Experiment 2: Adaptive steering
    print("\n" + "=" * 50)
    print("EXPERIMENT 2: Adaptive Steering")
    print("=" * 50)
    adaptive_results = test_adaptive_steering(model, tokenizer, traces, direction, correct_signals)
    
    # Visualization
    print("\n" + "=" * 50)
    print("VISUALIZATION")
    print("=" * 50)
    visualize_phase4(trajectory_results, adaptive_results)
    
    # Save results
    # Convert MonitoredGeneration objects to dicts for JSON serialization
    trajectory_for_json = {
        "correct": [{
            "question": m.question,
            "is_correct": m.is_correct,
            "signal_stats": m.signal_stats,
        } for m in trajectory_results.get("correct", [])],
        "incorrect": [{
            "question": m.question,
            "is_correct": m.is_correct,
            "signal_stats": m.signal_stats,
        } for m in trajectory_results.get("incorrect", [])],
        "summary": trajectory_results.get("summary", {}),
    }
    
    all_results = {
        "config": CONFIG,
        "trajectory": trajectory_for_json,
        "adaptive": adaptive_results,
    }
    
    with open(OUTPUT_DIR / "phase4_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE - SUMMARY")
    print("=" * 70)
    
    print("\n1. Signal Trajectory Analysis:")
    if trajectory_results.get("summary"):
        s = trajectory_results["summary"]
        print(f"   Correct traces mean signal: {s.get('correct_mean_signal', 0):.4f}")
        print(f"   Incorrect traces mean signal: {s.get('incorrect_mean_signal', 0):.4f}")
        print(f"   Signal separation: {s.get('signal_separation', 0):.4f}")
        print(f"   Correct trend: {s.get('correct_mean_trend', 0):.4f}")
        print(f"   Incorrect trend: {s.get('incorrect_mean_trend', 0):.4f}")
    
    print("\n2. Adaptive Steering:")
    print(f"   Threshold: {adaptive_results.get('threshold', 0):.4f}")
    print(f"   Baseline rate: {adaptive_results['baseline']['rate']:.1%}")
    print(f"   Adaptive rate: {adaptive_results['adaptive']['rate']:.1%}")
    print(f"   Improvement: {adaptive_results['improvement']:+.1%}")
    
    print("\n" + "=" * 70)
    
    # Conclusions
    if adaptive_results['improvement'] > 0.05:
        print("✓ ADAPTIVE STEERING WORKS: Real-time intervention improves accuracy!")
    elif adaptive_results['improvement'] > 0:
        print("◐ WEAK ADAPTIVE EFFECT: Some improvement from real-time intervention")
    else:
        print("○ NO ADAPTIVE BENEFIT: Real-time steering didn't help")
    
    if trajectory_results.get("summary", {}).get("signal_separation", 0) > 0:
        print("✓ SIGNAL SEPARATION: Correct traces have higher IVF signal during generation")
    
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

