#!/usr/bin/env python3
"""
The Internal Tribunal - Phase 1: Calibration
============================================
Finding the "Implicit Value Function" direction in DeepSeek-R1's activation space.

This script:
1. Generates Chain-of-Thought traces on GSM8K math problems
2. Labels trajectories as Correct/Incorrect based on final answer
3. Extracts residual stream activations from middle layers
4. Trains a linear probe to distinguish correct vs incorrect reasoning
5. Reports probe accuracy as a measure of the implicit value signal

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
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

# Graceful shutdown handling for SLURM
SHUTDOWN_REQUESTED = False
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints"))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def handle_signal(signum, frame):
    """Handle SIGUSR1 from SLURM for graceful shutdown."""
    global SHUTDOWN_REQUESTED
    print(f"\n[SIGNAL] Received signal {signum}, initiating graceful shutdown...")
    SHUTDOWN_REQUESTED = True

signal.signal(signal.SIGUSR1, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


@dataclass
class TraceResult:
    """Container for a single CoT trace result."""
    prompt: str
    question: str
    full_response: str
    extracted_answer: Optional[str]
    ground_truth: str
    is_correct: bool
    tokens: Optional[List[int]] = None


@dataclass 
class CheckpointState:
    """Checkpoint state for resumable execution."""
    stage: str = "init"
    traces: List[TraceResult] = field(default_factory=list)
    correct_count: int = 0
    incorrect_count: int = 0
    processed_indices: List[int] = field(default_factory=list)
    activations: Optional[Dict[str, np.ndarray]] = None
    probe_results: Optional[Dict[str, float]] = None


def save_checkpoint(state: CheckpointState, path: Optional[Path] = None):
    """Save checkpoint to disk."""
    path = path or CHECKPOINT_DIR / "calibration_checkpoint.pkl"
    print(f"[CHECKPOINT] Saving to {path}...")
    with open(path, "wb") as f:
        pickle.dump(state, f)
    # Create resubmit flag for SLURM
    (CHECKPOINT_DIR / "RESUBMIT_FLAG").touch()
    print("[CHECKPOINT] Saved successfully.")


def load_checkpoint(path: Optional[Path] = None) -> Optional[CheckpointState]:
    """Load checkpoint from disk if exists."""
    path = path or CHECKPOINT_DIR / "calibration_checkpoint.pkl"
    if path.exists():
        print(f"[CHECKPOINT] Loading from {path}...")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def extract_numerical_answer(text: str) -> Optional[str]:
    """
    Extract the final numerical answer from model output.
    
    DeepSeek models typically format answers as:
    - "The answer is X"
    - "#### X" (GSM8K format)
    - "= X" at the end
    - Boxed answers: \\boxed{X}
    """
    # Clean the text
    text = text.strip()
    
    # Try various patterns in order of specificity
    patterns = [
        # Boxed answer (LaTeX style)
        r'\\boxed\{([^}]+)\}',
        # GSM8K format
        r'####\s*([+-]?\$?[\d,]+\.?\d*)',
        # "The answer is X"
        r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+\$?([+-]?[\d,]+\.?\d*)',
        # "Answer: X" or "Answer = X"
        r'[Aa]nswer[:\s=]+\$?([+-]?[\d,]+\.?\d*)',
        # Final "= X" pattern
        r'=\s*\$?([+-]?[\d,]+\.?\d*)\s*$',
        # Just grab the last number in the text
        r'([+-]?\$?[\d,]+\.?\d*)\s*$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1)
            # Clean up: remove $, commas, leading zeros
            answer = answer.replace('$', '').replace(',', '').strip()
            try:
                # Normalize the number
                if '.' in answer:
                    return str(float(answer))
                else:
                    return str(int(answer))
            except ValueError:
                continue
    
    return None


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract ground truth answer from GSM8K format (#### ANSWER)."""
    match = re.search(r'####\s*([+-]?\$?[\d,]+\.?\d*)', answer_text)
    if match:
        answer = match.group(1).replace('$', '').replace(',', '').strip()
        try:
            if '.' in answer:
                return str(float(answer))
            else:
                return str(int(answer))
        except ValueError:
            return answer
    return answer_text.strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    answer = answer.replace('$', '').replace(',', '').strip()
    try:
        # Try to convert to float for numerical comparison
        val = float(answer)
        # If it's a whole number, return as int string
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return answer.lower().strip()


def check_answer_correctness(extracted: Optional[str], ground_truth: str) -> bool:
    """Check if extracted answer matches ground truth."""
    if extracted is None:
        return False
    
    norm_extracted = normalize_answer(extracted)
    norm_truth = normalize_answer(ground_truth)
    
    if norm_extracted == norm_truth:
        return True
    
    # Try numerical comparison with tolerance
    try:
        ext_val = float(norm_extracted)
        truth_val = float(norm_truth)
        return abs(ext_val - truth_val) < 1e-6
    except ValueError:
        return False


def format_prompt(question: str, tokenizer) -> str:
    """
    Format the question using the model's chat template.
    DeepSeek-R1 models expect a specific format.
    """
    # DeepSeek-R1 instruction format
    messages = [
        {
            "role": "user", 
            "content": f"Solve this math problem step by step. Show your reasoning, then give the final answer.\n\nQuestion: {question}"
        }
    ]
    
    try:
        # Use the tokenizer's chat template if available
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception:
        # Fallback format
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSolve this math problem step by step. Show your reasoning, then give the final answer.\n\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return prompt


def load_model_and_tokenizer():
    """
    Load DeepSeek-R1-Distill-Llama-8B with nnsight for activation extraction.
    
    Since transformer_lens doesn't natively support DeepSeek-R1, we use nnsight
    which can work with any HuggingFace model. Uses device_map="auto" for 
    multi-GPU sharding via accelerate.
    
    Returns a wrapper object with consistent API for the rest of the script.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nnsight import LanguageModel
    
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    
    print(f"[MODEL] Loading {MODEL_NAME} with nnsight...")
    n_gpus = torch.cuda.device_count()
    print(f"[MODEL] Available GPUs: {n_gpus}")
    
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
    
    # Step 1: Load tokenizer
    print("[MODEL] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Step 2: Load HuggingFace model with device_map for multi-GPU
    print("[MODEL] Loading HuggingFace model with device_map='auto'...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatically shard across GPUs
        low_cpu_mem_usage=True,
    )
    print(f"[MODEL] HF model loaded. Device map: {hf_model.hf_device_map if hasattr(hf_model, 'hf_device_map') else 'N/A'}")
    
    # Step 3: Wrap with nnsight LanguageModel for intervention/activation extraction
    print("[MODEL] Wrapping with nnsight LanguageModel...")
    model = LanguageModel(hf_model, tokenizer=tokenizer)
    
    # Add config attributes for compatibility
    model.cfg = type('Config', (), {
        'n_layers': hf_model.config.num_hidden_layers,
        'd_model': hf_model.config.hidden_size,
        'dtype': torch.bfloat16,
        'device': 'cuda'
    })()
    
    print(f"[MODEL] Model config: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    
    return model


def generate_traces(
    model,
    dataset,
    target_correct: int = 50,
    target_incorrect: int = 50,
    max_attempts: int = 200,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    existing_state: Optional[CheckpointState] = None
) -> CheckpointState:
    """
    Generate CoT traces and classify them as correct/incorrect.
    
    Uses temperature=1.0 to encourage some mistakes from the model.
    """
    global SHUTDOWN_REQUESTED
    
    # Initialize or resume from checkpoint
    if existing_state is not None:
        state = existing_state
        print(f"[TRACES] Resuming from checkpoint: {state.correct_count} correct, {state.incorrect_count} incorrect")
    else:
        state = CheckpointState(stage="generating_traces")
    
    # Get the tokenizer from the model
    tokenizer = model.tokenizer
    
    # Track which dataset indices we've processed
    processed = set(state.processed_indices)
    
    pbar = tqdm(
        total=target_correct + target_incorrect,
        initial=state.correct_count + state.incorrect_count,
        desc="Generating traces"
    )
    
    dataset_idx = 0
    attempts = 0
    
    while (state.correct_count < target_correct or state.incorrect_count < target_incorrect):
        if SHUTDOWN_REQUESTED:
            print("[TRACES] Shutdown requested, saving checkpoint...")
            save_checkpoint(state)
            sys.exit(0)
        
        if attempts >= max_attempts:
            print(f"[TRACES] Max attempts ({max_attempts}) reached.")
            break
        
        # Skip already processed indices
        while dataset_idx in processed and dataset_idx < len(dataset):
            dataset_idx += 1
        
        if dataset_idx >= len(dataset):
            print("[TRACES] Exhausted dataset.")
            break
        
        example = dataset[dataset_idx]
        question = example["question"]
        ground_truth = extract_gsm8k_answer(example["answer"])
        
        # Format the prompt
        prompt = format_prompt(question, tokenizer)
        
        try:
            # Generate response using HuggingFace generate API
            with torch.no_grad():
                # Tokenize
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                
                # Generate with sampling
                output_ids = model._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                # Decode
                full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Extract just the generated part
                generated = full_response[len(prompt):].strip()
            
            # Extract answer
            extracted = extract_numerical_answer(generated)
            is_correct = check_answer_correctness(extracted, ground_truth)
            
            # Decide whether to keep this trace
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
                    is_correct=is_correct
                )
                state.traces.append(trace)
                pbar.update(1)
                
                # Log progress
                status = "✓" if is_correct else "✗"
                tqdm.write(f"[{status}] Q{dataset_idx}: extracted={extracted}, truth={ground_truth}")
            
            processed.add(dataset_idx)
            state.processed_indices.append(dataset_idx)
            
        except Exception as e:
            tqdm.write(f"[ERROR] Failed on example {dataset_idx}: {e}")
        
        dataset_idx += 1
        attempts += 1
        
        # Periodic checkpoint
        if attempts % 20 == 0:
            save_checkpoint(state)
    
    pbar.close()
    
    print(f"\n[TRACES] Final: {state.correct_count} correct, {state.incorrect_count} incorrect")
    state.stage = "traces_complete"
    save_checkpoint(state)
    
    return state


def get_activations(
    model,
    traces: List[TraceResult],
    target_layers: List[int] = [12, 16, 20, 24],
    aggregation: str = "mean"  # "mean", "last", or "all"
) -> Dict[str, np.ndarray]:
    """
    Extract residual stream activations from the model for each trace.
    Uses nnsight for activation extraction.
    
    Args:
        model: The nnsight LanguageModel wrapper
        traces: List of TraceResult objects
        target_layers: Which layers to extract activations from
        aggregation: How to aggregate across sequence positions
            - "mean": Average over all tokens
            - "last": Take only the last token
            - "all": Keep all tokens (variable length)
    
    Returns:
        Dict mapping layer names to activation arrays
    """
    global SHUTDOWN_REQUESTED
    
    print(f"[ACTIVATIONS] Extracting from layers {target_layers} with {aggregation} aggregation...")
    
    # Storage - use layer numbers as keys
    layer_names = [f"layer_{layer}" for layer in target_layers]
    activations = {name: [] for name in layer_names}
    labels = []
    
    tokenizer = model.tokenizer
    
    for i, trace in enumerate(tqdm(traces, desc="Extracting activations")):
        if SHUTDOWN_REQUESTED:
            print("[ACTIVATIONS] Shutdown requested...")
            return None
        
        # Combine prompt and response for full trajectory
        full_text = trace.prompt + trace.full_response
        
        # Truncate if too long (to avoid OOM)
        max_tokens = 2048
        tokens = tokenizer.encode(full_text, return_tensors="pt", truncation=True, max_length=max_tokens)
        if tokens.shape[1] == max_tokens:
            tqdm.write(f"[WARN] Trace {i} truncated to {max_tokens} tokens")
        
        try:
            with torch.no_grad():
                # Clear CUDA cache periodically
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Tokenize input
                inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
                input_ids = inputs["input_ids"].to("cuda")
                
                # Run model and get hidden states directly using HuggingFace API
                outputs = model._model(
                    input_ids,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # hidden_states is a tuple of (n_layers + 1) tensors
                # Index 0 is embeddings, indices 1-32 are layer outputs
                hidden_states = outputs.hidden_states
                
                # Extract target layer activations
                for layer_idx in target_layers:
                    layer_name = f"layer_{layer_idx}"
                    # +1 because index 0 is embeddings
                    act = hidden_states[layer_idx + 1]  # [batch, seq_len, d_model]
                    
                    if len(act.shape) == 2:
                        act = act.unsqueeze(0)
                    
                    if aggregation == "mean":
                        vec = act.mean(dim=1).squeeze(0)  # [d_model]
                    elif aggregation == "last":
                        vec = act[:, -1, :].squeeze(0)  # [d_model]
                    else:
                        vec = act.squeeze(0)  # [seq_len, d_model]
                    
                    activations[layer_name].append(vec.cpu().float().numpy())
                
                labels.append(1 if trace.is_correct else 0)
                
                # Clean up to free memory
                del outputs, hidden_states
                
        except torch.cuda.OutOfMemoryError:
            tqdm.write(f"[OOM] Out of memory on trace {i}, skipping...")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to extract activations for trace {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Stack into arrays
    result = {}
    for layer_name in layer_names:
        if len(activations[layer_name]) == 0:
            tqdm.write(f"[WARN] No activations collected for {layer_name}")
            continue
        if aggregation in ["mean", "last"]:
            result[layer_name] = np.stack(activations[layer_name], axis=0)
        else:
            result[layer_name] = activations[layer_name]
    
    result["labels"] = np.array(labels)
    
    print(f"[ACTIVATIONS] Extracted {len(labels)} samples")
    for layer_name in result:
        if layer_name != "labels" and isinstance(result[layer_name], np.ndarray):
            print(f"  {layer_name}: {result[layer_name].shape}")
    
    return result


def train_probe(
    activations: Dict[str, np.ndarray],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train logistic regression probes on the activations.
    
    Returns accuracy and probe coefficients for each layer.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    
    labels = activations["labels"]
    
    results = {}
    
    # Get layer names (exclude 'labels')
    layer_names = [k for k in activations.keys() if k != "labels"]
    
    for layer_name in layer_names:
        print(f"\n[PROBE] Training on {layer_name}...")
        
        X = activations[layer_name]
        y = labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        print(f"  Train class balance: {y_train.sum()}/{len(y_train)} positive")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train probe
        probe = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs"
        )
        probe.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = probe.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get logits for visualization
        logits_train = probe.decision_function(X_train_scaled)
        logits_test = probe.decision_function(X_test_scaled)
        
        print(f"  *** Probe Accuracy at {layer_name}: {accuracy:.4f} ({accuracy*100:.1f}%) ***")
        print(classification_report(y_test, y_pred, target_names=["Incorrect", "Correct"]))
        
        results[layer_name] = {
            "accuracy": accuracy,
            "probe_coef": probe.coef_[0],  # The direction vector!
            "probe_intercept": probe.intercept_[0],
            "scaler_mean": scaler.mean_,
            "scaler_std": scaler.scale_,
            "logits_train": logits_train,
            "logits_test": logits_test,
            "y_train": y_train,
            "y_test": y_test
        }
    
    return results


def visualize_results(
    probe_results: Dict[str, Any],
    output_dir: Path = Path("./results")
):
    """
    Create visualization of probe results.
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for cluster
    import matplotlib.pyplot as plt
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Accuracy across layers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    layers = sorted(probe_results.keys())
    accuracies = [probe_results[l]["accuracy"] for l in layers]
    # Handle both "layer_X" and "blocks.X.hook_resid_post" formats
    layer_nums = []
    for l in layers:
        if l.startswith("layer_"):
            layer_nums.append(int(l.split("_")[1]))
        else:
            layer_nums.append(int(l.split(".")[1]))
    
    ax1 = axes[0]
    ax1.bar(range(len(layers)), accuracies, color='steelblue', edgecolor='navy')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([f"Layer {n}" for n in layer_nums])
    ax1.set_ylabel("Probe Accuracy")
    ax1.set_title("Implicit Value Function: Probe Accuracy by Layer")
    ax1.axhline(y=0.7, color='red', linestyle='--', label='70% threshold')
    ax1.axhline(y=0.5, color='gray', linestyle=':', label='Chance')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    for i, acc in enumerate(accuracies):
        ax1.text(i, acc + 0.02, f"{acc:.1%}", ha='center', fontsize=10)
    
    # 2. Logit separation histogram for best layer
    best_layer = max(probe_results.keys(), key=lambda k: probe_results[k]["accuracy"])
    best_result = probe_results[best_layer]
    
    ax2 = axes[1]
    
    # Combine train and test for visualization
    all_logits = np.concatenate([best_result["logits_train"], best_result["logits_test"]])
    all_labels = np.concatenate([best_result["y_train"], best_result["y_test"]])
    
    incorrect_logits = all_logits[all_labels == 0]
    correct_logits = all_logits[all_labels == 1]
    
    ax2.hist(incorrect_logits, bins=20, alpha=0.7, label="Incorrect", color='red', density=True)
    ax2.hist(correct_logits, bins=20, alpha=0.7, label="Correct", color='green', density=True)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=2, label='Decision boundary')
    ax2.set_xlabel("Probe Logit (Implicit Value)")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Logit Distribution at {best_layer}\n(Accuracy: {best_result['accuracy']:.1%})")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "probe_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[VIZ] Saved visualization to {output_dir / 'probe_results.png'}")
    
    # Save the direction vectors
    direction_data = {}
    for layer_name, result in probe_results.items():
        direction_data[layer_name] = {
            "direction": result["probe_coef"].tolist(),
            "intercept": float(result["probe_intercept"]),
            "accuracy": float(result["accuracy"]),
            "scaler_mean": result["scaler_mean"].tolist(),
            "scaler_std": result["scaler_std"].tolist()
        }
    
    with open(output_dir / "implicit_value_directions.json", "w") as f:
        json.dump(direction_data, f, indent=2)
    
    print(f"[VIZ] Saved direction vectors to {output_dir / 'implicit_value_directions.json'}")


def main():
    """Main execution flow."""
    global SHUTDOWN_REQUESTED
    
    print("=" * 70)
    print("THE INTERNAL TRIBUNAL - Phase 1: Calibration")
    print("Finding the Implicit Value Function Direction")
    print("=" * 70)
    
    # Configuration
    TARGET_CORRECT = 50
    TARGET_INCORRECT = 50
    MAX_ATTEMPTS = 200
    TEMPERATURE = 1.0  # Higher temp to encourage mistakes
    TARGET_LAYERS = [12, 16, 20, 24]
    OUTPUT_DIR = Path("./results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint
    state = load_checkpoint()
    
    # ========== Step 1: Load Model ==========
    print("\n" + "=" * 50)
    print("STEP 1: Loading Model")
    print("=" * 50)
    
    model = load_model_and_tokenizer()
    
    if SHUTDOWN_REQUESTED:
        print("[MAIN] Shutdown before trace generation")
        save_checkpoint(state or CheckpointState())
        sys.exit(0)
    
    # ========== Step 2: Load Dataset ==========
    print("\n" + "=" * 50)
    print("STEP 2: Loading GSM8K Dataset")
    print("=" * 50)
    
    from datasets import load_dataset
    
    dataset = load_dataset("gsm8k", "main", split="train")
    print(f"[DATA] Loaded {len(dataset)} examples from GSM8K")
    
    # ========== Step 3: Generate Traces ==========
    print("\n" + "=" * 50)
    print("STEP 3: Generating CoT Traces")
    print("=" * 50)
    
    if state is None or state.stage in ["init", "generating_traces"]:
        state = generate_traces(
            model=model,
            dataset=dataset,
            target_correct=TARGET_CORRECT,
            target_incorrect=TARGET_INCORRECT,
            max_attempts=MAX_ATTEMPTS,
            temperature=TEMPERATURE,
            existing_state=state if state and state.stage == "generating_traces" else None
        )
    else:
        print(f"[TRACES] Skipping - already have {len(state.traces)} traces from checkpoint")
    
    if SHUTDOWN_REQUESTED:
        sys.exit(0)
    
    # ========== Step 4: Extract Activations ==========
    print("\n" + "=" * 50)
    print("STEP 4: Extracting Residual Stream Activations")
    print("=" * 50)
    
    if state.activations is None:
        activations = get_activations(
            model=model,
            traces=state.traces,
            target_layers=TARGET_LAYERS,
            aggregation="mean"
        )
        
        if activations is None:
            # Shutdown was requested
            sys.exit(0)
        
        state.activations = activations
        state.stage = "activations_complete"
        save_checkpoint(state)
    else:
        print("[ACTIVATIONS] Skipping - already have activations from checkpoint")
        activations = state.activations
    
    # ========== Step 5: Train Probes ==========
    print("\n" + "=" * 50)
    print("STEP 5: Training Linear Probes")
    print("=" * 50)
    
    if state.probe_results is None:
        probe_results = train_probe(activations)
        state.probe_results = probe_results
        state.stage = "complete"
        save_checkpoint(state)
    else:
        print("[PROBE] Skipping - already have probe results from checkpoint")
        probe_results = state.probe_results
    
    # ========== Step 6: Visualization & Results ==========
    print("\n" + "=" * 50)
    print("STEP 6: Visualizing Results")
    print("=" * 50)
    
    visualize_results(probe_results, OUTPUT_DIR)
    
    # ========== Final Summary ==========
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"\nTraces collected: {len(state.traces)}")
    print(f"  Correct: {state.correct_count}")
    print(f"  Incorrect: {state.incorrect_count}")
    
    print("\nProbe Accuracies:")
    for layer_name in sorted(probe_results.keys()):
        acc = probe_results[layer_name]["accuracy"]
        status = "✓ SUCCESS" if acc > 0.7 else "○ Below threshold"
        print(f"  {layer_name}: {acc:.1%} {status}")
    
    best_layer = max(probe_results.keys(), key=lambda k: probe_results[k]["accuracy"])
    best_acc = probe_results[best_layer]["accuracy"]
    
    print(f"\n*** Best probe: {best_layer} with {best_acc:.1%} accuracy ***")
    
    if best_acc > 0.7:
        print("\n✓ CALIBRATION SUCCESSFUL: Found significant Implicit Value Function signal!")
        print("  The direction vector is saved in results/implicit_value_directions.json")
    else:
        print("\n○ CALIBRATION INCONCLUSIVE: Probe accuracy below 70% threshold.")
        print("  Consider: more data, different layers, or per-token analysis.")
    
    print("\n" + "=" * 70)
    
    # Clean up checkpoint if complete
    if state.stage == "complete":
        checkpoint_path = CHECKPOINT_DIR / "calibration_checkpoint.pkl"
        if checkpoint_path.exists():
            # Keep final checkpoint but remove resubmit flag
            resubmit_flag = CHECKPOINT_DIR / "RESUBMIT_FLAG"
            if resubmit_flag.exists():
                resubmit_flag.unlink()
    
    return probe_results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Keyboard interrupt received")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

