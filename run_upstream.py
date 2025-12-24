#!/usr/bin/env python3
"""
The Internal Tribunal - Experiment C: The Upstream Sweep (Circuit Check)
=========================================================================
Test whether steering at earlier layers (where computation happens)
is more effective than steering at Layer 14 (where readout happens).

Hardware: GPU 3
"""

import os
import sys
import re
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "checkpoint_path": "/project/scratch01/compiling/a.a.baggio/internal_tribunal/checkpoints/phase2_checkpoint.pkl",
    "upstream_layers": [8, 10, 12],
    "reference_layer": 14,
    "num_traces": 100,
    "steering_strength": 1.0,
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "output_path": "/project/scratch01/compiling/a.a.baggio/internal_tribunal/results_upstream.json",
}

# ============================================================================
# DATA STRUCTURES (for unpickling)
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
# ANSWER EXTRACTION
# ============================================================================
def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from model response."""
    patterns = [
        r'####\s*(-?[\d,]+\.?\d*)',
        r'answer\s*(?:is|=|:)\s*\$?(-?[\d,]+\.?\d*)',
        r'=\s*\$?(-?[\d,]+\.?\d*)\s*$',
        r'(-?[\d,]+\.?\d*)\s*$',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower().replace(',', ''))
        if matches:
            try:
                return str(float(matches[-1]))
            except:
                continue
    return None


def check_correct(extracted: Optional[str], ground_truth: str) -> bool:
    """Check if extracted answer matches ground truth."""
    if extracted is None:
        return False
    try:
        ext_val = float(extracted)
        gt_val = float(ground_truth.replace(',', ''))
        return abs(ext_val - gt_val) < 1e-6
    except:
        return False


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model_and_tokenizer():
    """Load model on available GPU."""
    device = "cuda:0"
    print(f"[MODEL] Loading {CONFIG['model_name']}...")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"[MODEL] Loaded on {device}")
    return model, tokenizer, device


# ============================================================================
# PROBE TRAINING
# ============================================================================
def train_probe_for_layer(activations: Dict, layer_idx: int, traces: List) -> tuple:
    """Train a logistic regression probe for a specific layer."""
    print(f"\n[PROBE] Training probe for layer {layer_idx}...")
    
    # Get activations for this layer using last_10_mean
    key = f"layer_{layer_idx}"
    if key not in activations.get("last_10_mean", {}):
        print(f"[ERROR] No activations for {key}")
        return None, None, None
    
    X = np.array(activations["last_10_mean"][key])
    y = np.array([1 if t.is_correct else 0 for t in traces])
    
    # Train probe
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)
    
    accuracy = clf.score(X_scaled, y)
    print(f"[PROBE] Layer {layer_idx} accuracy: {accuracy:.1%}")
    
    # Extract direction vector
    direction = clf.coef_[0] / scaler.scale_
    direction = direction / np.linalg.norm(direction)
    
    return torch.from_numpy(direction).float(), accuracy, clf


# ============================================================================
# STEERING HOOK
# ============================================================================
class SteeringHook:
    """Hook to apply positive steering at a specific layer."""
    
    def __init__(self, direction: torch.Tensor, strength: float, device: str):
        self.direction = direction.to(device)
        self.strength = strength
        self.device = device
    
    def __call__(self, module, input, output):
        # Handle both tuple and tensor output formats
        if isinstance(output, tuple):
            hidden_states = output[0]  # [batch, seq, hidden]
        else:
            hidden_states = output
        
        steering = self.direction * self.strength
        hidden_states = hidden_states + steering.to(hidden_states.dtype)
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states


# ============================================================================
# GENERATION WITH STEERING
# ============================================================================
def generate_with_steering(
    model, 
    tokenizer, 
    question: str, 
    direction: Optional[torch.Tensor],
    layer_idx: int,
    strength: float,
    device: str
) -> str:
    """Generate response with optional steering at specified layer."""
    
    prompt = f"""Solve this math problem step by step. Show your work and put your final answer after ####.

Question: {question}

Solution:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Register steering hook if direction provided
    hook = None
    if direction is not None and strength != 0:
        layer = model.model.layers[layer_idx]
        hook_fn = SteeringHook(direction, strength, device)
        hook = layer.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_new_tokens"],
                temperature=CONFIG["temperature"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    finally:
        if hook is not None:
            hook.remove()


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def main():
    print("=" * 70)
    print("THE INTERNAL TRIBUNAL - EXPERIMENT C: THE UPSTREAM SWEEP")
    print("=" * 70)
    print("Testing if earlier layer steering is more effective")
    print("=" * 70)
    
    # Load checkpoint
    print(f"\n[LOAD] Loading checkpoint from {CONFIG['checkpoint_path']}...")
    with open(CONFIG["checkpoint_path"], "rb") as f:
        state = pickle.load(f)
    
    print(f"[DATA] Loaded {len(state.traces)} traces")
    print(f"[DATA] Correct: {state.correct_count}, Incorrect: {state.incorrect_count}")
    
    # Get incorrect traces for recovery experiment
    incorrect_traces = [t for t in state.traces if not t.is_correct][:CONFIG["num_traces"]]
    print(f"[DATA] Using {len(incorrect_traces)} incorrect traces")
    
    # Train probes for each layer
    layer_directions = {}
    probe_accuracies = {}
    
    for layer_idx in CONFIG["upstream_layers"] + [CONFIG["reference_layer"]]:
        direction, accuracy, _ = train_probe_for_layer(state.activations, layer_idx, state.traces)
        if direction is not None:
            layer_directions[layer_idx] = direction
            probe_accuracies[layer_idx] = accuracy
    
    # Generate random vectors for each layer
    random_directions = {}
    for layer_idx in layer_directions:
        rand = torch.randn_like(layer_directions[layer_idx])
        rand = rand / torch.norm(rand)
        random_directions[layer_idx] = rand
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer()
    
    # Move directions to device
    for layer_idx in layer_directions:
        layer_directions[layer_idx] = layer_directions[layer_idx].to(device)
        random_directions[layer_idx] = random_directions[layer_idx].to(device)
    
    # Results storage
    results = {
        "config": CONFIG,
        "probe_accuracies": probe_accuracies,
        "baseline": {"correct": 0, "total": len(incorrect_traces)},
        "layer_results": {},
    }
    
    # Run baseline (no steering)
    print("\n" + "=" * 50)
    print("BASELINE (No Steering)")
    print("=" * 50)
    
    baseline_correct = 0
    for trace in tqdm(incorrect_traces, desc="Baseline"):
        response = generate_with_steering(model, tokenizer, trace.question, None, 0, 0.0, device)
        answer = extract_answer(response)
        if check_correct(answer, trace.ground_truth):
            baseline_correct += 1
    
    results["baseline"]["correct"] = baseline_correct
    results["baseline"]["accuracy"] = baseline_correct / len(incorrect_traces)
    print(f"\nBaseline recovery: {baseline_correct}/{len(incorrect_traces)} = {baseline_correct/len(incorrect_traces):.1%}")
    
    # Test each layer
    for layer_idx in CONFIG["upstream_layers"] + [CONFIG["reference_layer"]]:
        print(f"\n" + "=" * 50)
        print(f"LAYER {layer_idx} (Probe accuracy: {probe_accuracies.get(layer_idx, 0):.1%})")
        print("=" * 50)
        
        direction = layer_directions.get(layer_idx)
        random_dir = random_directions.get(layer_idx)
        
        if direction is None:
            print(f"[SKIP] No direction for layer {layer_idx}")
            continue
        
        # IVF steering
        print(f"\n[IVF] Steering at layer {layer_idx}, strength +{CONFIG['steering_strength']}...")
        ivf_correct = 0
        for trace in tqdm(incorrect_traces, desc=f"Layer {layer_idx} IVF"):
            response = generate_with_steering(
                model, tokenizer, trace.question, 
                direction, layer_idx, CONFIG["steering_strength"], device
            )
            answer = extract_answer(response)
            if check_correct(answer, trace.ground_truth):
                ivf_correct += 1
        
        # Random steering
        print(f"\n[RANDOM] Steering at layer {layer_idx}, strength +{CONFIG['steering_strength']}...")
        random_correct = 0
        for trace in tqdm(incorrect_traces, desc=f"Layer {layer_idx} Random"):
            response = generate_with_steering(
                model, tokenizer, trace.question,
                random_dir, layer_idx, CONFIG["steering_strength"], device
            )
            answer = extract_answer(response)
            if check_correct(answer, trace.ground_truth):
                random_correct += 1
        
        results["layer_results"][str(layer_idx)] = {
            "probe_accuracy": probe_accuracies.get(layer_idx, 0),
            "ivf_correct": ivf_correct,
            "ivf_recovery": ivf_correct / len(incorrect_traces),
            "random_correct": random_correct,
            "random_recovery": random_correct / len(incorrect_traces),
            "ivf_advantage": (ivf_correct - random_correct) / len(incorrect_traces),
        }
        
        print(f"\n  IVF Recovery: {ivf_correct}/{len(incorrect_traces)} = {ivf_correct/len(incorrect_traces):.1%}")
        print(f"  Random Recovery: {random_correct}/{len(incorrect_traces)} = {random_correct/len(incorrect_traces):.1%}")
        print(f"  IVF Advantage: {(ivf_correct - random_correct)/len(incorrect_traces):.1%}")
    
    # Save results
    with open(CONFIG["output_path"], "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    print(f"\nBaseline Recovery: {results['baseline']['accuracy']:.1%}")
    print(f"\n| Layer | Probe Acc | IVF Recovery | Random Recovery | IVF Advantage |")
    print(f"|-------|-----------|--------------|-----------------|---------------|")
    
    best_layer = None
    best_advantage = -1
    
    for layer_idx in CONFIG["upstream_layers"] + [CONFIG["reference_layer"]]:
        s = str(layer_idx)
        if s not in results["layer_results"]:
            continue
        
        r = results["layer_results"][s]
        print(f"| {layer_idx} | {r['probe_accuracy']:.1%} | {r['ivf_recovery']:.1%} | {r['random_recovery']:.1%} | {r['ivf_advantage']:.1%} |")
        
        if r["ivf_advantage"] > best_advantage:
            best_advantage = r["ivf_advantage"]
            best_layer = layer_idx
    
    print(f"\n[SUCCESS] Results saved to {CONFIG['output_path']}")
    
    # Analysis
    if best_layer and best_layer != CONFIG["reference_layer"]:
        print(f"\n✓ INTERESTING: Layer {best_layer} shows best IVF advantage ({best_advantage:.1%})")
        print(f"  This suggests the 'computation layer' might differ from the 'readout layer'!")
    elif best_layer == CONFIG["reference_layer"]:
        print(f"\n→ EXPECTED: Layer 14 (reference) still shows best advantage")
        print("  No evidence that earlier layers are better intervention points.")
    else:
        print(f"\n✗ NULL RESULT: No layer shows clear IVF advantage")
        print("  Steering may not be effective at any layer.")


if __name__ == "__main__":
    main()

