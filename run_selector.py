#!/usr/bin/env python3
"""
The Internal Tribunal - Experiment A: The Selector (Utility Check)
===================================================================
Best-of-N Rejection Sampling using IVF Probe as selector.

Tests whether the IVF probe has practical utility for improving accuracy
even if it can't be used for causal steering.

Hardware: GPUs 0-1 (split dataset for parallelism)
"""

import os
import sys
import re
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "num_problems": 500,  # First 500 from GSM8K test
    "num_samples": 5,     # Best-of-N
    "temperature": 0.7,
    "max_new_tokens": 1024,
    "target_layer": 14,
    "ivf_path": "/project/scratch01/compiling/a.a.baggio/internal_tribunal/results_phase3/ivf_direction.npy",
    "output_path": "/project/scratch01/compiling/a.a.baggio/internal_tribunal/results_selector.json",
}

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


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    try:
        val = float(answer.replace(',', '').strip())
        if val == int(val):
            return str(int(val))
        return str(val)
    except:
        return answer.strip()


def check_correct(extracted: Optional[str], ground_truth: str) -> bool:
    """Check if extracted answer matches ground truth."""
    if extracted is None:
        return False
    try:
        ext_val = float(extracted)
        gt_val = float(ground_truth.replace(',', ''))
        return abs(ext_val - gt_val) < 1e-6
    except:
        return normalize_answer(extracted) == normalize_answer(ground_truth)


# ============================================================================
# MODEL AND DATA LOADING
# ============================================================================
def load_model_and_tokenizer(device: str):
    """Load model on specified device."""
    print(f"[MODEL] Loading {CONFIG['model_name']} on {device}...")
    
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
    return model, tokenizer


def load_ivf_direction() -> torch.Tensor:
    """Load the IVF direction vector."""
    ivf = np.load(CONFIG["ivf_path"])
    return torch.from_numpy(ivf).float()


def load_gsm8k_problems(num_problems: int) -> List[Dict]:
    """Load GSM8K test problems."""
    print(f"[DATA] Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    problems = []
    for i, item in enumerate(dataset):
        if i >= num_problems:
            break
        # Extract ground truth answer
        answer_text = item["answer"]
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
        gt = match.group(1).replace(',', '') if match else ""
        
        problems.append({
            "question": item["question"],
            "ground_truth": gt,
        })
    
    print(f"[DATA] Loaded {len(problems)} problems")
    return problems


# ============================================================================
# GENERATION AND SCORING
# ============================================================================
def generate_traces(model, tokenizer, question: str, n_samples: int, device: str) -> List[Dict]:
    """Generate N reasoning traces for a question."""
    prompt = f"""Solve this math problem step by step. Show your work and put your final answer after ####.

Question: {question}

Solution:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    traces = []
    
    for _ in range(n_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_new_tokens"],
                temperature=CONFIG["temperature"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        
        response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Get hidden states for scoring
        # Hidden states: tuple of (layer_outputs) for each generated token
        # Each layer_output is [batch, seq, hidden]
        hidden_states = outputs.hidden_states
        
        traces.append({
            "response": response,
            "hidden_states": hidden_states,
        })
    
    return traces


def score_trace_ivf(hidden_states, ivf_direction: torch.Tensor, device: str) -> float:
    """Score a trace using IVF direction (last-10-mean at layer 14)."""
    # hidden_states is a tuple of (num_generated_tokens,) 
    # Each element is a tuple of (num_layers,) tensors of shape [batch, 1, hidden]
    
    layer_idx = CONFIG["target_layer"]
    
    # Collect layer 14 hidden states across all generated tokens
    layer_states = []
    for token_hidden in hidden_states:
        if len(token_hidden) > layer_idx:
            # token_hidden[layer_idx] is [batch, 1, hidden]
            layer_states.append(token_hidden[layer_idx][0, 0, :])  # [hidden]
    
    if len(layer_states) < 1:
        return 0.0
    
    # Stack and take last 10 mean
    layer_states = torch.stack(layer_states)  # [num_tokens, hidden]
    last_10 = layer_states[-10:] if len(layer_states) >= 10 else layer_states
    mean_state = last_10.mean(dim=0).float().to(ivf_direction.device)
    
    # Dot product with IVF direction
    score = torch.dot(mean_state, ivf_direction).item()
    return score


def generate_greedy(model, tokenizer, question: str, device: str) -> str:
    """Generate a single greedy (temp=0) response."""
    prompt = f"""Solve this math problem step by step. Show your work and put your final answer after ####.

Question: {question}

Solution:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()


# ============================================================================
# SELECTION STRATEGIES
# ============================================================================
def select_by_probe(traces: List[Dict], scores: List[float]) -> int:
    """Select trace with highest IVF score."""
    return int(np.argmax(scores))


def select_by_majority(traces: List[Dict], ground_truth: str) -> Tuple[int, bool]:
    """Select by majority vote, return (index, is_correct)."""
    answers = []
    for t in traces:
        ans = extract_answer(t["response"])
        answers.append(normalize_answer(ans) if ans else "NONE")
    
    counter = Counter(answers)
    majority_answer, count = counter.most_common(1)[0]
    
    # Find first trace with majority answer
    for i, ans in enumerate(answers):
        if ans == majority_answer:
            return i, check_correct(majority_answer, ground_truth)
    
    return 0, False


def evaluate_random(traces: List[Dict], ground_truth: str) -> float:
    """Average accuracy across all traces (random selection baseline)."""
    correct = 0
    for t in traces:
        ans = extract_answer(t["response"])
        if check_correct(ans, ground_truth):
            correct += 1
    return correct / len(traces)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_experiment_on_gpu(gpu_id: int, problems: List[Dict], ivf_direction: torch.Tensor) -> Dict:
    """Run experiment on a single GPU."""
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Starting with {len(problems)} problems...")
    
    model, tokenizer = load_model_and_tokenizer(device)
    ivf_direction = ivf_direction.to(device)
    
    results = {
        "greedy_correct": 0,
        "probe_correct": 0,
        "majority_correct": 0,
        "random_correct": 0.0,
        "total": len(problems),
    }
    
    for i, problem in enumerate(tqdm(problems, desc=f"GPU {gpu_id}")):
        question = problem["question"]
        gt = problem["ground_truth"]
        
        # 1. Greedy decode
        greedy_response = generate_greedy(model, tokenizer, question, device)
        greedy_answer = extract_answer(greedy_response)
        if check_correct(greedy_answer, gt):
            results["greedy_correct"] += 1
        
        # 2. Generate N samples
        traces = generate_traces(model, tokenizer, question, CONFIG["num_samples"], device)
        
        # 3. Score with IVF probe
        scores = []
        for t in traces:
            score = score_trace_ivf(t["hidden_states"], ivf_direction, device)
            scores.append(score)
            t["ivf_score"] = score
        
        # 4. Probe selection
        probe_idx = select_by_probe(traces, scores)
        probe_answer = extract_answer(traces[probe_idx]["response"])
        if check_correct(probe_answer, gt):
            results["probe_correct"] += 1
        
        # 5. Majority vote
        _, majority_correct = select_by_majority(traces, gt)
        if majority_correct:
            results["majority_correct"] += 1
        
        # 6. Random selection (average)
        results["random_correct"] += evaluate_random(traces, gt)
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"[GPU {gpu_id}] Progress: {i+1}/{len(problems)}")
            print(f"  Greedy: {results['greedy_correct']}/{i+1}")
            print(f"  Probe:  {results['probe_correct']}/{i+1}")
            print(f"  Majority: {results['majority_correct']}/{i+1}")
            print(f"  Random: {results['random_correct']:.1f}/{i+1}")
    
    return results


def main():
    print("=" * 70)
    print("THE INTERNAL TRIBUNAL - EXPERIMENT A: THE SELECTOR")
    print("=" * 70)
    print("Testing IVF Probe as Best-of-N Selector")
    print("=" * 70)
    
    # Detect available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"[INIT] Detected {n_gpus} GPUs")
    
    if n_gpus < 1:
        print("[ERROR] No GPUs available!")
        sys.exit(1)
    
    # Load data
    ivf_direction = load_ivf_direction()
    print(f"[IVF] Loaded direction vector, shape: {ivf_direction.shape}")
    
    problems = load_gsm8k_problems(CONFIG["num_problems"])
    
    # Split problems across GPUs
    if n_gpus >= 2:
        split = len(problems) // 2
        gpu_problems = [problems[:split], problems[split:]]
        gpu_ids = [0, 1]
    else:
        gpu_problems = [problems]
        gpu_ids = [0]
    
    print(f"[SPLIT] Distributing across {len(gpu_ids)} GPUs")
    
    # Run experiments (sequentially for now - each GPU loads its own model)
    all_results = []
    for gpu_id, probs in zip(gpu_ids, gpu_problems):
        result = run_experiment_on_gpu(gpu_id, probs, ivf_direction)
        all_results.append(result)
    
    # Aggregate results
    total = sum(r["total"] for r in all_results)
    greedy = sum(r["greedy_correct"] for r in all_results)
    probe = sum(r["probe_correct"] for r in all_results)
    majority = sum(r["majority_correct"] for r in all_results)
    random_sum = sum(r["random_correct"] for r in all_results)
    
    final_results = {
        "config": CONFIG,
        "total_problems": total,
        "greedy": {
            "correct": greedy,
            "accuracy": greedy / total,
        },
        "probe_selection": {
            "correct": probe,
            "accuracy": probe / total,
        },
        "majority_vote": {
            "correct": majority,
            "accuracy": majority / total,
        },
        "random_selection": {
            "expected_correct": random_sum,
            "accuracy": random_sum / total,
        },
    }
    
    # Save results
    with open(CONFIG["output_path"], "w") as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\n| Strategy | Correct | Accuracy |")
    print(f"|----------|---------|----------|")
    print(f"| Greedy (temp=0) | {greedy}/{total} | {greedy/total:.1%} |")
    print(f"| **Probe Selection** | {probe}/{total} | {probe/total:.1%} |")
    print(f"| Majority Vote | {majority}/{total} | {majority/total:.1%} |")
    print(f"| Random Selection | {random_sum:.1f}/{total} | {random_sum/total:.1%} |")
    
    print(f"\n[SUCCESS] Results saved to {CONFIG['output_path']}")
    
    # Success metric
    if probe / total > random_sum / total:
        print("\n✓ SUCCESS: Probe Selection > Random Selection")
    else:
        print("\n✗ FAILED: Probe Selection <= Random Selection")


if __name__ == "__main__":
    main()

