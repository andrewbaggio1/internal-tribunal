#!/usr/bin/env python3
"""
The Internal Tribunal - Phase 2: Enhanced Calibration
=====================================================
Finding the "Implicit Value Function" direction with:
- 1000 traces (500 correct, 500 incorrect)
- Per-token analysis (last token, last-10 mean, full mean)
- All 32 layers probed
- Cross-validation for robust accuracy estimates
- PCA visualization

Author: Internal Tribunal Research Project
"""

import os
import sys
import re
import json
import signal
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "target_correct": 500,
    "target_incorrect": 500,
    "max_attempts": 2000,  # May need many attempts to get 500 incorrect
    "temperature": 1.0,
    "max_new_tokens": 1024,
    "target_layers": list(range(32)),  # All 32 layers
    "aggregation_methods": ["last", "last_10_mean", "mean"],
    "max_seq_len": 2048,
    "checkpoint_every": 50,
    "cv_folds": 5,
}

# ============================================================================
# SIGNAL HANDLING FOR SLURM
# ============================================================================
SHUTDOWN_REQUESTED = False
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints"))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def handle_signal(signum, frame):
    global SHUTDOWN_REQUESTED
    print(f"\n[SIGNAL] Received signal {signum}, initiating graceful shutdown...")
    SHUTDOWN_REQUESTED = True

signal.signal(signal.SIGUSR1, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ============================================================================
# DATA STRUCTURES
# ============================================================================
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
    """Checkpoint state for Phase 2."""
    stage: str = "init"
    traces: List[TraceResult] = field(default_factory=list)
    correct_count: int = 0
    incorrect_count: int = 0
    processed_indices: List[int] = field(default_factory=list)
    activations: Optional[Dict] = None
    probe_results: Optional[Dict] = None


def save_checkpoint(state: Phase2State, path: Optional[Path] = None):
    """Save checkpoint to disk."""
    path = path or CHECKPOINT_DIR / "phase2_checkpoint.pkl"
    print(f"[CHECKPOINT] Saving to {path}...")
    with open(path, "wb") as f:
        pickle.dump(state, f)
    (CHECKPOINT_DIR / "RESUBMIT_FLAG").touch()
    print(f"[CHECKPOINT] Saved. Correct: {state.correct_count}, Incorrect: {state.incorrect_count}")


def load_checkpoint(path: Optional[Path] = None) -> Optional[Phase2State]:
    """Load checkpoint from disk if exists."""
    path = path or CHECKPOINT_DIR / "phase2_checkpoint.pkl"
    if path.exists():
        print(f"[CHECKPOINT] Loading from {path}...")
        with open(path, "rb") as f:
            state = pickle.load(f)
        print(f"[CHECKPOINT] Loaded. Stage: {state.stage}, Traces: {len(state.traces)}")
        return state
    return None


# ============================================================================
# ANSWER EXTRACTION & VALIDATION
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


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract ground truth from GSM8K format."""
    match = re.search(r'####\s*([+-]?\$?[\d,]+\.?\d*)', answer_text)
    if match:
        answer = match.group(1).replace('$', '').replace(',', '').strip()
        try:
            if '.' in answer:
                return str(float(answer))
            return str(int(answer))
        except ValueError:
            return answer
    return answer_text.strip()


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
    n_gpus = torch.cuda.device_count()
    print(f"[MODEL] Available GPUs: {n_gpus}")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print(f"[MODEL] Loaded. Layers: {model.config.num_hidden_layers}, Hidden: {model.config.hidden_size}")
    
    return model, tokenizer


# ============================================================================
# TRACE GENERATION
# ============================================================================
def generate_traces(model, tokenizer, dataset, state: Phase2State) -> Phase2State:
    """Generate CoT traces with checkpointing."""
    global SHUTDOWN_REQUESTED
    
    target_correct = CONFIG['target_correct']
    target_incorrect = CONFIG['target_incorrect']
    
    print(f"[TRACES] Target: {target_correct} correct, {target_incorrect} incorrect")
    print(f"[TRACES] Current: {state.correct_count} correct, {state.incorrect_count} incorrect")
    
    processed = set(state.processed_indices)
    pbar = tqdm(
        total=target_correct + target_incorrect,
        initial=state.correct_count + state.incorrect_count,
        desc="Generating traces"
    )
    
    dataset_idx = max(state.processed_indices) + 1 if state.processed_indices else 0
    attempts = 0
    
    while (state.correct_count < target_correct or state.incorrect_count < target_incorrect):
        if SHUTDOWN_REQUESTED:
            save_checkpoint(state)
            sys.exit(0)
        
        if attempts >= CONFIG['max_attempts']:
            print(f"[TRACES] Max attempts reached.")
            break
        
        while dataset_idx in processed and dataset_idx < len(dataset):
            dataset_idx += 1
        
        if dataset_idx >= len(dataset):
            print("[TRACES] Dataset exhausted.")
            break
        
        example = dataset[dataset_idx]
        question = example["question"]
        ground_truth = extract_gsm8k_answer(example["answer"])
        prompt = format_prompt(question, tokenizer)
        
        try:
            with torch.no_grad():
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=CONFIG['max_new_tokens'],
                    temperature=CONFIG['temperature'],
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                generated = full_response[len(prompt):].strip()
                num_tokens = output_ids.shape[1]
            
            extracted = extract_numerical_answer(generated)
            is_correct = check_answer_correctness(extracted, ground_truth)
            
            keep = False
            if is_correct and state.correct_count < target_correct:
                state.correct_count += 1
                keep = True
            elif not is_correct and state.incorrect_count < target_incorrect:
                state.incorrect_count += 1
                keep = True
            
            if keep:
                trace = TraceResult(
                    prompt=prompt,
                    question=question,
                    full_response=generated,
                    extracted_answer=extracted,
                    ground_truth=ground_truth,
                    is_correct=is_correct,
                    num_tokens=num_tokens
                )
                state.traces.append(trace)
                pbar.update(1)
                
                status = "✓" if is_correct else "✗"
                tqdm.write(f"[{status}] Q{dataset_idx}: {extracted} vs {ground_truth}")
            
            processed.add(dataset_idx)
            state.processed_indices.append(dataset_idx)
            
        except Exception as e:
            tqdm.write(f"[ERROR] Q{dataset_idx}: {e}")
        
        dataset_idx += 1
        attempts += 1
        
        if len(state.traces) % CONFIG['checkpoint_every'] == 0 and len(state.traces) > 0:
            save_checkpoint(state)
    
    pbar.close()
    print(f"\n[TRACES] Final: {state.correct_count} correct, {state.incorrect_count} incorrect")
    state.stage = "traces_complete"
    save_checkpoint(state)
    return state


# ============================================================================
# ACTIVATION EXTRACTION - MULTI-METHOD
# ============================================================================
def get_activations_multi(model, tokenizer, traces: List[TraceResult], state: Phase2State) -> Dict:
    """Extract activations with multiple aggregation methods."""
    global SHUTDOWN_REQUESTED
    
    layers = CONFIG['target_layers']
    methods = CONFIG['aggregation_methods']
    
    print(f"[ACTIVATIONS] Extracting from {len(layers)} layers with methods: {methods}")
    
    # Initialize storage: {method: {layer: [vectors]}}
    activations = {method: {f"layer_{l}": [] for l in layers} for method in methods}
    labels = []
    
    for i, trace in enumerate(tqdm(traces, desc="Extracting activations")):
        if SHUTDOWN_REQUESTED:
            # Save partial activations
            state.activations = {"partial": True, "index": i, "data": activations, "labels": labels}
            save_checkpoint(state)
            sys.exit(0)
        
        full_text = trace.prompt + trace.full_response
        
        try:
            with torch.no_grad():
                if i % 20 == 0:
                    torch.cuda.empty_cache()
                
                inputs = tokenizer(
                    full_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=CONFIG['max_seq_len']
                )
                input_ids = inputs["input_ids"].to("cuda")
                seq_len = input_ids.shape[1]
                
                outputs = model(input_ids, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states  # Tuple of (n_layers + 1) tensors
                
                for layer_idx in layers:
                    layer_name = f"layer_{layer_idx}"
                    # +1 because index 0 is embeddings
                    act = hidden_states[layer_idx + 1]  # [1, seq_len, hidden_size]
                    
                    for method in methods:
                        if method == "last":
                            vec = act[0, -1, :]  # Last token
                        elif method == "last_10_mean":
                            # Mean of last 10 tokens (or all if < 10)
                            n = min(10, seq_len)
                            vec = act[0, -n:, :].mean(dim=0)
                        elif method == "mean":
                            vec = act[0, :, :].mean(dim=0)  # Full sequence mean
                        else:
                            vec = act[0, -1, :]
                        
                        activations[method][layer_name].append(vec.cpu().float().numpy())
                
                labels.append(1 if trace.is_correct else 0)
                del outputs, hidden_states
                
        except torch.cuda.OutOfMemoryError:
            tqdm.write(f"[OOM] Trace {i}, skipping...")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            tqdm.write(f"[ERROR] Trace {i}: {e}")
            continue
    
    # Stack arrays
    result = {"labels": np.array(labels)}
    for method in methods:
        result[method] = {}
        for layer_name in activations[method]:
            if activations[method][layer_name]:
                result[method][layer_name] = np.stack(activations[method][layer_name], axis=0)
    
    print(f"[ACTIVATIONS] Extracted {len(labels)} samples")
    return result


# ============================================================================
# PROBING WITH CROSS-VALIDATION
# ============================================================================
def train_probes_cv(activations: Dict) -> Dict:
    """Train probes with cross-validation for robust accuracy estimates."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    labels = activations["labels"]
    methods = [m for m in activations.keys() if m != "labels"]
    
    results = {}
    best_accuracy = 0
    best_config = None
    
    print(f"\n[PROBING] Training probes with {CONFIG['cv_folds']}-fold CV...")
    print("=" * 70)
    
    for method in methods:
        results[method] = {}
        print(f"\n--- Aggregation: {method} ---")
        
        layer_names = sorted(activations[method].keys(), key=lambda x: int(x.split("_")[1]))
        
        for layer_name in tqdm(layer_names, desc=f"Probing {method}"):
            X = activations[method][layer_name]
            y = labels
            
            if len(X) != len(y):
                continue
            
            # Create pipeline with scaling + logistic regression
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'))
            ])
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
            
            mean_acc = scores.mean()
            std_acc = scores.std()
            
            results[method][layer_name] = {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "cv_scores": scores.tolist()
            }
            
            if mean_acc > best_accuracy:
                best_accuracy = mean_acc
                best_config = (method, layer_name)
        
        # Print summary for this method
        accs = [results[method][l]["mean_accuracy"] for l in results[method]]
        if accs:
            print(f"  Best: {max(accs):.1%}, Worst: {min(accs):.1%}, Mean: {np.mean(accs):.1%}")
    
    print("\n" + "=" * 70)
    if best_config:
        print(f"*** BEST: {best_config[1]} with {best_config[0]} aggregation: {best_accuracy:.1%} ***")
    
    results["best"] = {
        "method": best_config[0] if best_config else None,
        "layer": best_config[1] if best_config else None,
        "accuracy": best_accuracy
    }
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_phase2(activations: Dict, probe_results: Dict, output_dir: Path):
    """Create comprehensive visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = activations["labels"]
    methods = [m for m in activations.keys() if m != "labels"]
    
    # 1. Layer-wise accuracy curves for each method
    fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5))
    if len(methods) == 1:
        axes = [axes]
    
    for ax, method in zip(axes, methods):
        layer_names = sorted(probe_results[method].keys(), key=lambda x: int(x.split("_")[1]))
        layer_nums = [int(l.split("_")[1]) for l in layer_names]
        accs = [probe_results[method][l]["mean_accuracy"] for l in layer_names]
        stds = [probe_results[method][l]["std_accuracy"] for l in layer_names]
        
        ax.fill_between(layer_nums, 
                        np.array(accs) - np.array(stds), 
                        np.array(accs) + np.array(stds), 
                        alpha=0.3)
        ax.plot(layer_nums, accs, 'o-', linewidth=2, markersize=4)
        ax.axhline(y=0.5, color='gray', linestyle=':', label='Chance')
        ax.axhline(y=0.7, color='red', linestyle='--', label='70% threshold')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Probe Accuracy: {method}")
        ax.legend()
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "layer_accuracy_curves.png", dpi=150)
    plt.close()
    
    # 2. PCA visualization for best configuration
    best = probe_results["best"]
    if best["layer"] and best["method"]:
        X = activations[best["method"]][best["layer"]]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='RdYlGn', alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title(f"PCA: {best['layer']} ({best['method']}) - Accuracy: {best['accuracy']:.1%}")
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Correct (1) / Incorrect (0)")
        
        plt.tight_layout()
        plt.savefig(output_dir / "pca_best_layer.png", dpi=150)
        plt.close()
    
    # 3. Heatmap of accuracies across methods and layers
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create matrix
    all_layers = sorted(probe_results[methods[0]].keys(), key=lambda x: int(x.split("_")[1]))
    matrix = np.zeros((len(methods), len(all_layers)))
    
    for i, method in enumerate(methods):
        for j, layer in enumerate(all_layers):
            if layer in probe_results[method]:
                matrix[i, j] = probe_results[method][layer]["mean_accuracy"]
    
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0.4, vmax=0.8)
    ax.set_xticks(range(len(all_layers)))
    ax.set_xticklabels([l.split("_")[1] for l in all_layers], fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Aggregation Method")
    ax.set_title("Probe Accuracy Heatmap (All Layers × All Methods)")
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")
    
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_heatmap.png", dpi=150)
    plt.close()
    
    print(f"[VIZ] Saved visualizations to {output_dir}")
    
    # Save detailed results JSON
    results_json = {
        "best": probe_results["best"],
        "config": CONFIG,
        "num_traces": len(labels),
        "class_balance": {"correct": int(labels.sum()), "incorrect": int(len(labels) - labels.sum())}
    }
    
    # Add per-layer results
    for method in methods:
        results_json[method] = {}
        for layer in probe_results[method]:
            results_json[method][layer] = {
                "accuracy": probe_results[method][layer]["mean_accuracy"],
                "std": probe_results[method][layer]["std_accuracy"]
            }
    
    with open(output_dir / "phase2_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"[VIZ] Saved results to {output_dir / 'phase2_results.json'}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    global SHUTDOWN_REQUESTED
    
    print("=" * 70)
    print("THE INTERNAL TRIBUNAL - Phase 2: Enhanced Calibration")
    print("=" * 70)
    print(f"Target: {CONFIG['target_correct']} correct + {CONFIG['target_incorrect']} incorrect traces")
    print(f"Layers: All {len(CONFIG['target_layers'])} layers")
    print(f"Methods: {CONFIG['aggregation_methods']}")
    print("=" * 70)
    
    OUTPUT_DIR = Path("./results_phase2")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    state = load_checkpoint()
    if state is None:
        state = Phase2State()
    
    # Step 1: Load Model
    print("\n" + "=" * 50)
    print("STEP 1: Loading Model")
    print("=" * 50)
    model, tokenizer = load_model()
    
    if SHUTDOWN_REQUESTED:
        save_checkpoint(state)
        sys.exit(0)
    
    # Step 2: Load Dataset
    print("\n" + "=" * 50)
    print("STEP 2: Loading Dataset")
    print("=" * 50)
    from datasets import load_dataset
    dataset = load_dataset("gsm8k", "main", split="train")
    print(f"[DATA] Loaded {len(dataset)} examples")
    
    # Step 3: Generate Traces
    print("\n" + "=" * 50)
    print("STEP 3: Generating Traces")
    print("=" * 50)
    
    if state.stage in ["init", "generating_traces"]:
        state.stage = "generating_traces"
        state = generate_traces(model, tokenizer, dataset, state)
    else:
        print(f"[TRACES] Skipping - have {len(state.traces)} traces from checkpoint")
    
    if SHUTDOWN_REQUESTED:
        sys.exit(0)
    
    # Step 4: Extract Activations
    print("\n" + "=" * 50)
    print("STEP 4: Extracting Activations (Multi-Method)")
    print("=" * 50)
    
    if state.activations is None or state.activations.get("partial"):
        activations = get_activations_multi(model, tokenizer, state.traces, state)
        state.activations = activations
        state.stage = "activations_complete"
        save_checkpoint(state)
    else:
        print("[ACTIVATIONS] Using cached activations")
        activations = state.activations
    
    if SHUTDOWN_REQUESTED:
        sys.exit(0)
    
    # Step 5: Train Probes
    print("\n" + "=" * 50)
    print("STEP 5: Training Probes (Cross-Validation)")
    print("=" * 50)
    
    if state.probe_results is None:
        probe_results = train_probes_cv(activations)
        state.probe_results = probe_results
        state.stage = "complete"
        save_checkpoint(state)
    else:
        print("[PROBES] Using cached results")
        probe_results = state.probe_results
    
    # Step 6: Visualization
    print("\n" + "=" * 50)
    print("STEP 6: Visualization")
    print("=" * 50)
    visualize_phase2(activations, probe_results, OUTPUT_DIR)
    
    # Final Summary
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"\nTraces: {len(state.traces)} ({state.correct_count} correct, {state.incorrect_count} incorrect)")
    
    best = probe_results["best"]
    print(f"\nBest Configuration:")
    print(f"  Layer: {best['layer']}")
    print(f"  Method: {best['method']}")
    print(f"  Accuracy: {best['accuracy']:.1%}")
    
    if best['accuracy'] > 0.7:
        print("\n✓ SUCCESS: Found significant Implicit Value Function signal!")
    elif best['accuracy'] > 0.6:
        print("\n◐ PROMISING: Moderate signal detected, may need more data or analysis")
    else:
        print("\n○ INCONCLUSIVE: Signal below threshold, consider alternative approaches")
    
    print("\n" + "=" * 70)
    
    # Clean up resubmit flag if complete
    resubmit_flag = CHECKPOINT_DIR / "RESUBMIT_FLAG"
    if resubmit_flag.exists():
        resubmit_flag.unlink()
    
    return probe_results


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

