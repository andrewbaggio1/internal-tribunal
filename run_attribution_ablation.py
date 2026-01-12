#!/usr/bin/env python3
"""
Attribution Ablation: Finding the Upstream Source of the L14 IVF Signal

We found that Layer 14 contains a probe direction that predicts reasoning correctness
with 80.5% accuracy, but steering with it fails (random baseline matches). 

Hypothesis: The L14 signal is a downstream "readout" of earlier computation.

Goal: Find the attention heads in Layers 0-13 that WRITE to the L14 probe direction,
then ablate them to test if they're causally necessary for both:
1. The probe signal (do they create it?)
2. The reasoning itself (are they necessary for correct answers?)

Uses HuggingFace Transformers with manual hooks (no TransformerLens dependency).
"""

import os
import json
import re
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Model
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    
    # Paths
    ivf_direction_path: str = "results_phase3/ivf_direction.npy"
    output_dir: str = "results_attribution"
    
    # Dataset
    n_traces: int = 100  # Number of correct traces to analyze
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    # Attribution
    probe_layer: int = 14  # Where we found the IVF signal
    scan_layers: Tuple[int, ...] = tuple(range(0, 14))  # Layers 0-13
    top_k_heads: int = 10  # Number of top heads to ablate
    n_heads: int = 32  # DeepSeek-R1-Distill-Llama-8B has 32 heads
    d_model: int = 4096
    d_head: int = 128  # 4096 / 32
    
    # Generation
    batch_size: int = 1  # For memory efficiency


CONFIG = Config()


# ============================================================================
# Utilities
# ============================================================================

def extract_answer(text: str) -> Optional[float]:
    """Extract numerical answer from model output."""
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass
    
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass
    
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except:
            pass
    
    return None


def parse_ground_truth(answer_text: str) -> Optional[float]:
    """Parse ground truth from GSM8K format."""
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', answer_text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass
    return None


def check_answer(predicted: Optional[float], ground_truth: Optional[float]) -> bool:
    """Check if predicted answer matches ground truth."""
    if predicted is None or ground_truth is None:
        return False
    return abs(predicted - ground_truth) < 0.01


# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_tokenizer():
    """Load model with HuggingFace Transformers."""
    print(f"[INFO] Loading model: {CONFIG.model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.model_name,
        torch_dtype=CONFIG.dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get model config
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    
    print(f"[INFO] Model loaded. Layers: {n_layers}, Heads: {n_heads}, d_model: {d_model}")
    
    return model, tokenizer


def load_ivf_direction():
    """Load the saved IVF probe direction."""
    print(f"[INFO] Loading IVF direction from {CONFIG.ivf_direction_path}")
    direction = np.load(CONFIG.ivf_direction_path)
    direction = torch.tensor(direction, dtype=torch.float32, device=CONFIG.device)
    direction = direction / direction.norm()
    print(f"[INFO] IVF direction shape: {direction.shape}, norm: {direction.norm().item():.4f}")
    return direction


# ============================================================================
# Hook-based Activation Capture
# ============================================================================

class ActivationCache:
    """Cache for storing activations during forward pass."""
    def __init__(self):
        self.cache = {}
        self.hooks = []
    
    def clear(self):
        self.cache = {}
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def add_hook(self, module, name):
        def hook_fn(module, input, output):
            # Handle tuple output (hidden_states, ...) 
            if isinstance(output, tuple):
                self.cache[name] = output[0].detach()
            else:
                self.cache[name] = output.detach()
        
        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)


def get_attention_output_hooks(model, cache: ActivationCache, layers: List[int]):
    """Register hooks to capture attention outputs for specified layers."""
    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        # Hook on the attention output projection (o_proj)
        cache.add_hook(layer.self_attn.o_proj, f"layer_{layer_idx}_attn_out")
        # Also hook on the full layer output for residual stream
        cache.add_hook(layer, f"layer_{layer_idx}_output")


def get_residual_hook(model, cache: ActivationCache, layer: int):
    """Register hook to capture residual stream at a specific layer."""
    layer_module = model.model.layers[layer]
    cache.add_hook(layer_module, f"layer_{layer}_residual")


# ============================================================================
# Dataset Loading
# ============================================================================

def load_correct_traces(model, tokenizer, n_traces: int) -> List[Dict]:
    """Load GSM8K problems and find ones the model gets correct."""
    print(f"[INFO] Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    correct_traces = []
    
    print(f"[INFO] Finding {n_traces} correct traces (greedy decode)...")
    
    for i, example in enumerate(tqdm(dataset, desc="Finding correct traces")):
        if len(correct_traces) >= n_traces:
            break
        
        question = example['question']
        ground_truth = parse_ground_truth(example['answer'])
        
        if ground_truth is None:
            continue
        
        prompt = f"Question: {question}\n\nLet me solve this step by step.\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(CONFIG.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=CONFIG.max_new_tokens,
                temperature=None,  # Greedy
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = extract_answer(response)
        
        if check_answer(predicted, ground_truth):
            correct_traces.append({
                'question': question,
                'ground_truth': ground_truth,
                'prompt': prompt,
                'response': response,
                'predicted': predicted,
            })
            print(f"  Found correct trace {len(correct_traces)}/{n_traces}")
        
        del inputs, outputs
        torch.cuda.empty_cache()
    
    print(f"[INFO] Found {len(correct_traces)} correct traces")
    return correct_traces


# ============================================================================
# Step 2: Direct Attribution Scanning
# ============================================================================

def compute_head_attributions_simple(
    model,
    tokenizer,
    traces: List[Dict],
    ivf_direction: torch.Tensor,
) -> Dict[Tuple[int, int], float]:
    """
    Simplified attribution: measure how much each layer's attention output
    contributes to the IVF direction.
    
    We compute: Attribution_layer = mean(Attn_Output • IVF_Direction)
    """
    print("\n[STEP 2] Computing Direct Attribution for layers 0-13...")
    
    # Track attributions per layer (simplified - not per head)
    layer_attributions = defaultdict(list)
    
    cache = ActivationCache()
    
    for trace_idx, trace in enumerate(tqdm(traces[:50], desc="Attribution scan")):  # Use 50 for speed
        prompt = trace['prompt']
        inputs = tokenizer(prompt, return_tensors="pt").to(CONFIG.device)
        
        # Clear and set up hooks
        cache.clear()
        cache.remove_hooks()
        get_attention_output_hooks(model, cache, list(CONFIG.scan_layers))
        
        with torch.no_grad():
            model(inputs.input_ids)
        
        # Compute attribution for each layer
        for layer in CONFIG.scan_layers:
            key = f"layer_{layer}_attn_out"
            if key in cache.cache:
                attn_out = cache.cache[key]  # [batch, seq, d_model]
                # Get last token's activation
                last_token_attn = attn_out[0, -1, :].float()  # [d_model]
                
                # Project onto IVF direction (move to same device)
                ivf_dir_local = ivf_direction.to(last_token_attn.device)
                attribution = (last_token_attn @ ivf_dir_local).item()
                
                # Store as (layer, 0) since we're not decomposing by head
                layer_attributions[(layer, 0)].append(attribution)
        
        cache.remove_hooks()
        del inputs
        torch.cuda.empty_cache()
    
    # Compute mean attribution for each layer
    mean_attributions = {
        (layer, head): np.mean(values)
        for (layer, head), values in layer_attributions.items()
    }
    
    return mean_attributions


def identify_top_layers(
    attributions: Dict[Tuple[int, int], float],
    k: int = 10
) -> List[Tuple[int, int, float]]:
    """Identify top-k layers by absolute attribution magnitude."""
    sorted_items = sorted(
        attributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    top_items = [(layer, head, attr) for (layer, head), attr in sorted_items[:k]]
    
    print(f"\n[INFO] Top {k} layers by attribution magnitude:")
    for layer, head, attr in top_items:
        print(f"  Layer {layer}: attribution = {attr:.4f}")
    
    return top_items


# ============================================================================
# Step 3: Dissociation Ablation
# ============================================================================

def run_ablation_experiment(
    model,
    tokenizer,
    traces: List[Dict],
    ivf_direction: torch.Tensor,
    layers_to_ablate: List[int],
) -> Dict:
    """
    Run with top layers ablated (zero out attention output).
    Measure probe signal and accuracy.
    """
    print(f"\n[STEP 3] Running ablation experiment on layers: {layers_to_ablate}")
    
    # First, get baseline measurements
    print("[INFO] Computing baseline (no ablation)...")
    baseline_signals = []
    baseline_correct = 0
    
    cache = ActivationCache()
    
    for trace in tqdm(traces[:30], desc="Baseline"):  # Use 30 for speed
        inputs = tokenizer(trace['prompt'], return_tensors="pt").to(CONFIG.device)
        
        cache.clear()
        cache.remove_hooks()
        get_residual_hook(model, cache, CONFIG.probe_layer)
        
        with torch.no_grad():
            model(inputs.input_ids)
        
        key = f"layer_{CONFIG.probe_layer}_residual"
        if key in cache.cache:
            resid = cache.cache[key]
            if isinstance(resid, tuple):
                resid = resid[0]
            ivf_dir_local = ivf_direction.to(resid.device)
            signal = (resid[0, -1, :].float() @ ivf_dir_local).item()
            baseline_signals.append(signal)
        
        baseline_correct += 1  # All traces are correct by selection
        
        cache.remove_hooks()
        del inputs
        torch.cuda.empty_cache()
    
    baseline_signal_mean = np.mean(baseline_signals) if baseline_signals else 0
    baseline_accuracy = 1.0  # All are correct
    
    print(f"[BASELINE] Signal: {baseline_signal_mean:.4f}, Accuracy: {baseline_accuracy:.1%}")
    
    # Now run with ablation
    print(f"[INFO] Running with ablation of layers: {layers_to_ablate}")
    ablated_signals = []
    ablated_correct = 0
    
    # Create ablation hooks
    def make_ablation_hook():
        def hook_fn(module, input, output):
            # Zero out the attention output
            if isinstance(output, tuple):
                zeroed = torch.zeros_like(output[0])
                return (zeroed,) + output[1:]
            else:
                return torch.zeros_like(output)
        return hook_fn
    
    ablation_hooks = []
    for layer_idx in layers_to_ablate:
        layer = model.model.layers[layer_idx]
        handle = layer.self_attn.o_proj.register_forward_hook(make_ablation_hook())
        ablation_hooks.append(handle)
    
    for trace in tqdm(traces[:30], desc="Ablated"):
        inputs = tokenizer(trace['prompt'], return_tensors="pt").to(CONFIG.device)
        
        cache.clear()
        cache.remove_hooks()
        get_residual_hook(model, cache, CONFIG.probe_layer)
        
        with torch.no_grad():
            model(inputs.input_ids)
        
        key = f"layer_{CONFIG.probe_layer}_residual"
        if key in cache.cache:
            resid = cache.cache[key]
            if isinstance(resid, tuple):
                resid = resid[0]
            ivf_dir_local = ivf_direction.to(resid.device)
            signal = (resid[0, -1, :].float() @ ivf_dir_local).item()
            ablated_signals.append(signal)
        
        # Generate and check accuracy
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=CONFIG.max_new_tokens,
                temperature=None,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = extract_answer(response)
        
        if check_answer(predicted, trace['ground_truth']):
            ablated_correct += 1
        
        cache.remove_hooks()
        del inputs, outputs
        torch.cuda.empty_cache()
    
    # Remove ablation hooks
    for handle in ablation_hooks:
        handle.remove()
    
    ablated_signal_mean = np.mean(ablated_signals) if ablated_signals else 0
    ablated_accuracy = ablated_correct / 30
    
    print(f"[ABLATED] Signal: {ablated_signal_mean:.4f}, Accuracy: {ablated_accuracy:.1%}")
    
    # Compute drops
    if baseline_signal_mean != 0:
        signal_drop = (baseline_signal_mean - ablated_signal_mean) / abs(baseline_signal_mean) * 100
    else:
        signal_drop = 0
    accuracy_drop = (baseline_accuracy - ablated_accuracy) * 100
    
    return {
        'baseline_signal': baseline_signal_mean,
        'ablated_signal': ablated_signal_mean,
        'signal_drop_pct': signal_drop,
        'baseline_accuracy': baseline_accuracy,
        'ablated_accuracy': ablated_accuracy,
        'accuracy_drop_pct': accuracy_drop,
    }


# ============================================================================
# Step 4: Visualization and Output
# ============================================================================

def create_attribution_heatmap(
    attributions: Dict[Tuple[int, int], float],
    n_layers: int,
    output_path: str,
):
    """Create a bar chart of layer attributions."""
    print(f"\n[STEP 4] Creating attribution visualization...")
    
    # Extract layer attributions
    layers = sorted(set(layer for (layer, _) in attributions.keys()))
    values = [attributions.get((layer, 0), 0) for layer in layers]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['red' if v < 0 else 'blue' for v in values]
    bars = ax.bar(layers, values, color=colors, alpha=0.7)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Attribution to L14 IVF Direction', fontsize=12)
    ax.set_title('Direct Attribution: Which Layers Write to the IVF Direction?', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Highlight top layers
    top_layers = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    for (layer, _), attr in top_layers:
        ax.annotate(f'L{layer}', xy=(layer, attr), xytext=(layer, attr + 0.1 * np.sign(attr)),
                   ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved plot to {output_path}")


def save_results(
    top_layers: List[Tuple[int, int, float]],
    ablation_results: Dict,
    attributions: Dict[Tuple[int, int], float],
    output_dir: str,
):
    """Save all results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'top_layers': [[layer, attr] for layer, _, attr in top_layers],
        'top_layer_attributions': [
            {'layer': layer, 'attribution': attr}
            for layer, _, attr in top_layers
        ],
        'all_attributions': {
            str(layer): attr for (layer, _), attr in attributions.items()
        },
        'probe_signal_baseline': ablation_results['baseline_signal'],
        'probe_signal_ablated': ablation_results['ablated_signal'],
        'probe_signal_drop_pct': ablation_results['signal_drop_pct'],
        'accuracy_baseline': ablation_results['baseline_accuracy'],
        'accuracy_ablated': ablation_results['ablated_accuracy'],
        'accuracy_drop_pct': ablation_results['accuracy_drop_pct'],
        'config': {
            'model': CONFIG.model_name,
            'n_traces': CONFIG.n_traces,
            'probe_layer': CONFIG.probe_layer,
            'scan_layers': list(CONFIG.scan_layers),
        }
    }
    
    output_path = os.path.join(output_dir, 'attribution_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Saved results to {output_path}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("ATTRIBUTION ABLATION: Finding the Upstream Source of L14 IVF Signal")
    print("=" * 70)
    
    os.makedirs(CONFIG.output_dir, exist_ok=True)
    
    # Step 1: Setup
    print("\n[STEP 1] Loading model and data...")
    model, tokenizer = load_model_and_tokenizer()
    ivf_direction = load_ivf_direction()
    
    # Load correct traces
    traces = load_correct_traces(model, tokenizer, CONFIG.n_traces)
    
    if len(traces) < CONFIG.n_traces:
        print(f"[WARNING] Only found {len(traces)} correct traces")
    
    # Step 2: Direct Attribution Scanning (simplified - by layer, not head)
    attributions = compute_head_attributions_simple(model, tokenizer, traces, ivf_direction)
    top_layers = identify_top_layers(attributions, k=CONFIG.top_k_heads)
    
    # Step 3: Dissociation Ablation (ablate top 5 layers)
    layers_to_ablate = [layer for layer, _, _ in top_layers[:5]]
    ablation_results = run_ablation_experiment(
        model, tokenizer, traces, ivf_direction, layers_to_ablate
    )
    
    # Step 4: Output
    create_attribution_heatmap(
        attributions,
        n_layers=len(CONFIG.scan_layers),
        output_path=os.path.join(CONFIG.output_dir, 'layer_attribution.png'),
    )
    
    results = save_results(top_layers, ablation_results, attributions, CONFIG.output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTop Layers (by attribution to L14 IVF direction):")
    for layer, _, attr in top_layers[:5]:
        print(f"  Layer {layer}: {attr:.4f}")
    
    print(f"\n[PROBE HEALTH]")
    print(f"  Baseline signal: {ablation_results['baseline_signal']:.4f}")
    print(f"  Ablated signal:  {ablation_results['ablated_signal']:.4f}")
    print(f"  Signal drop:     {ablation_results['signal_drop_pct']:.1f}%")
    
    print(f"\n[REASONING HEALTH]")
    print(f"  Baseline accuracy: {ablation_results['baseline_accuracy']:.1%}")
    print(f"  Ablated accuracy:  {ablation_results['ablated_accuracy']:.1%}")
    print(f"  Accuracy drop:     {ablation_results['accuracy_drop_pct']:.1f}%")
    
    print("\n[INTERPRETATION]")
    if ablation_results['signal_drop_pct'] > 50 and ablation_results['accuracy_drop_pct'] > 10:
        print("  ✓ These layers are CAUSAL for both signal AND reasoning!")
    elif ablation_results['signal_drop_pct'] > 50:
        print("  ◐ These layers create the signal but aren't necessary for reasoning.")
    elif ablation_results['accuracy_drop_pct'] > 10:
        print("  ◐ These layers are necessary for reasoning but don't create the signal.")
    else:
        print("  ✗ These layers are not the primary source.")
    
    print("\n" + "=" * 70)
    print(f"Results saved to: {CONFIG.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
