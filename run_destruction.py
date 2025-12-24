#!/usr/bin/env python3
"""
The Internal Tribunal - Experiment B: The Destruction (Necessity Check)
========================================================================
Negative Steering Ablation Experiment.

Tests whether the IVF direction is *necessary* for correct reasoning.
If negative steering hurts more than random vector steering, the direction
encodes something causally important.

Hardware: GPU 2
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

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "checkpoint_path": "/project/scratch01/compiling/a.a.baggio/internal_tribunal/checkpoints/phase2_checkpoint.pkl",
    "ivf_path": "/project/scratch01/compiling/a.a.baggio/internal_tribunal/results_phase3/ivf_direction.npy",
    "num_traces": 100,
    "steering_strengths": [-1.0, -2.0, -3.0],
    "target_layer": 14,
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "output_path": "/project/scratch01/compiling/a.a.baggio/internal_tribunal/results_destruction.json",
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
# STEERING HOOK
# ============================================================================
class NegativeSteeringHook:
    """Hook to apply negative steering at a specific layer."""
    
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
        
        # Apply negative steering: subtract direction * |strength|
        steering = self.direction * self.strength  # strength is negative
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
    strength: float,
    device: str
) -> str:
    """Generate response with optional steering."""
    
    prompt = f"""Solve this math problem step by step. Show your work and put your final answer after ####.

Question: {question}

Solution:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Register steering hook if direction provided
    hook = None
    if direction is not None and strength != 0:
        layer = model.model.layers[CONFIG["target_layer"]]
        hook_fn = NegativeSteeringHook(direction, strength, device)
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
    print("THE INTERNAL TRIBUNAL - EXPERIMENT B: THE DESTRUCTION")
    print("=" * 70)
    print("Testing if negative steering degrades correct reasoning")
    print("=" * 70)
    
    # Load checkpoint
    print(f"\n[LOAD] Loading checkpoint from {CONFIG['checkpoint_path']}...")
    with open(CONFIG["checkpoint_path"], "rb") as f:
        state = pickle.load(f)
    
    # Get correct traces
    correct_traces = [t for t in state.traces if t.is_correct]
    print(f"[DATA] Found {len(correct_traces)} correct traces")
    
    # Limit to configured number
    correct_traces = correct_traces[:CONFIG["num_traces"]]
    print(f"[DATA] Using {len(correct_traces)} traces for experiment")
    
    # Load IVF direction
    ivf_direction = torch.from_numpy(np.load(CONFIG["ivf_path"])).float()
    print(f"[IVF] Loaded direction vector, norm: {torch.norm(ivf_direction):.4f}")
    
    # Generate random vector with same norm
    random_direction = torch.randn_like(ivf_direction)
    random_direction = random_direction / torch.norm(random_direction) * torch.norm(ivf_direction)
    print(f"[RANDOM] Generated random vector, norm: {torch.norm(random_direction):.4f}")
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer()
    ivf_direction = ivf_direction.to(device)
    random_direction = random_direction.to(device)
    
    # Results storage
    results = {
        "config": CONFIG,
        "baseline": {"correct": 0, "total": 0},
        "ivf_steering": {},
        "random_steering": {},
    }
    
    # Run baseline (no steering)
    print("\n" + "=" * 50)
    print("BASELINE (No Steering)")
    print("=" * 50)
    
    baseline_correct = 0
    for trace in tqdm(correct_traces, desc="Baseline"):
        response = generate_with_steering(model, tokenizer, trace.question, None, 0.0, device)
        answer = extract_answer(response)
        if check_correct(answer, trace.ground_truth):
            baseline_correct += 1
    
    results["baseline"]["correct"] = baseline_correct
    results["baseline"]["total"] = len(correct_traces)
    results["baseline"]["accuracy"] = baseline_correct / len(correct_traces)
    print(f"\nBaseline: {baseline_correct}/{len(correct_traces)} = {baseline_correct/len(correct_traces):.1%}")
    
    # Test each steering strength
    for strength in CONFIG["steering_strengths"]:
        print(f"\n" + "=" * 50)
        print(f"STEERING STRENGTH: {strength}")
        print("=" * 50)
        
        # IVF negative steering
        print(f"\n[IVF] Running with strength {strength}...")
        ivf_correct = 0
        for trace in tqdm(correct_traces, desc=f"IVF {strength}"):
            response = generate_with_steering(model, tokenizer, trace.question, ivf_direction, strength, device)
            answer = extract_answer(response)
            if check_correct(answer, trace.ground_truth):
                ivf_correct += 1
        
        results["ivf_steering"][str(strength)] = {
            "correct": ivf_correct,
            "accuracy": ivf_correct / len(correct_traces),
            "degradation": (baseline_correct - ivf_correct) / len(correct_traces),
        }
        print(f"  IVF: {ivf_correct}/{len(correct_traces)} = {ivf_correct/len(correct_traces):.1%}")
        print(f"  Degradation: {(baseline_correct - ivf_correct)/len(correct_traces):.1%}")
        
        # Random negative steering
        print(f"\n[RANDOM] Running with strength {strength}...")
        random_correct = 0
        for trace in tqdm(correct_traces, desc=f"Random {strength}"):
            response = generate_with_steering(model, tokenizer, trace.question, random_direction, strength, device)
            answer = extract_answer(response)
            if check_correct(answer, trace.ground_truth):
                random_correct += 1
        
        results["random_steering"][str(strength)] = {
            "correct": random_correct,
            "accuracy": random_correct / len(correct_traces),
            "degradation": (baseline_correct - random_correct) / len(correct_traces),
        }
        print(f"  Random: {random_correct}/{len(correct_traces)} = {random_correct/len(correct_traces):.1%}")
        print(f"  Degradation: {(baseline_correct - random_correct)/len(correct_traces):.1%}")
    
    # Save results
    with open(CONFIG["output_path"], "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    print(f"\nBaseline Accuracy: {results['baseline']['accuracy']:.1%}")
    print(f"\n| Strength | IVF Accuracy | IVF Degradation | Random Accuracy | Random Degradation | IVF > Random? |")
    print(f"|----------|--------------|-----------------|-----------------|--------------------|--------------:|")
    
    success_count = 0
    for strength in CONFIG["steering_strengths"]:
        s = str(strength)
        ivf_acc = results["ivf_steering"][s]["accuracy"]
        ivf_deg = results["ivf_steering"][s]["degradation"]
        rand_acc = results["random_steering"][s]["accuracy"]
        rand_deg = results["random_steering"][s]["degradation"]
        
        ivf_worse = ivf_deg > rand_deg
        if ivf_worse:
            success_count += 1
        
        print(f"| {strength} | {ivf_acc:.1%} | {ivf_deg:.1%} | {rand_acc:.1%} | {rand_deg:.1%} | {'✓ YES' if ivf_worse else '✗ NO'} |")
    
    print(f"\n[SUCCESS] Results saved to {CONFIG['output_path']}")
    
    # Success metric
    if success_count >= 2:
        print(f"\n✓ SUCCESS: IVF causes more degradation than Random in {success_count}/3 cases")
        print("  The IVF direction is NECESSARY for correct reasoning!")
    else:
        print(f"\n✗ FAILED: IVF doesn't cause more degradation than Random")
        print("  The IVF direction may not be causally necessary.")


if __name__ == "__main__":
    main()

