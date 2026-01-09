#!/usr/bin/env python3
"""M47 Evaluation Script — Error Correction Fidelity Metrics.

Evaluates checkpoints on clean and error-injected eval sets and measures:
1. Error Detection Rate — error mentioned/contradicted in VERIFY
2. Correction Attempt Rate — CORRECT block present and non-empty
3. Correction Accuracy — correction fixes the injected error
4. False Correction Rate — model "fixes" something that wasn't wrong
5. Net Outcome — final answer improved/unchanged/worse

Usage:
    cd research/m47_error_correction_fidelity
    python scripts/run_eval.py

Author: M47 Error Correction Fidelity Milestone
Date: 2026-01-09
"""

import codecs
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure stdout for UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except Exception:
        pass


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
EVAL_DIR = PROJECT_DIR / "eval"
METRICS_DIR = PROJECT_DIR / "metrics"

PROJECT_ROOT = PROJECT_DIR.parent.parent

# Eval sets
EVAL_V2 = PROJECT_ROOT / "training" / "evalsets" / "eval_v2.jsonl"
HOLDOUT_CLEAN = DATA_DIR / "stage_c_holdout.jsonl"
HOLDOUT_ERROR = DATA_DIR / "stage_c_holdout_error.jsonl"

# Checkpoints
CHECKPOINTS = {
    "m46_self_correct": PROJECT_DIR.parent / "m46_structured_self_correction" / "checkpoints" / "self_correct" / "final_model",
    "m47_clean": PROJECT_DIR / "checkpoints" / "clean" / "final_model",
    "m47_error_aware": PROJECT_DIR / "checkpoints" / "error_aware" / "final_model",
}

# Error manifest for ground truth
ERROR_MANIFEST = PROJECT_DIR / "error_manifest.json"


# ============================================================
# Error Detection Patterns
# ============================================================

VERIFY_PATTERNS = [
    r"\bverify\b", r"\bcheck\b", r"\bconfirm\b", r"\bvalidate\b",
    r"\bby inverse\b", r"\brecheck\b", r"\bdouble.?check\b",
]

CORRECT_PATTERNS = [
    r"\bcorrect\b", r"\bfix\b", r"\badjust\b", r"\bmodify\b",
    r"\bchange\b", r"\bactually\b", r"\bwait\b", r"\bmistake\b",
    r"\berror\b", r"\bwrong\b", r"\bshould be\b", r"\bnot\s+\d+\b",
]

ERROR_DETECTION_PATTERNS = [
    r"\bwrong\b", r"\berror\b", r"\bmistake\b", r"\bincorrect\b",
    r"\bshould be\b", r"\bnot\s+\d+\b", r"\bactually\b",
]


def analyze_output(text: str, has_known_error: bool = False, expected_answer: str = None) -> dict:
    """Analyze model output for verification and correction behavior.
    
    Args:
        text: Generated text
        has_known_error: Whether this trace had an injected error
        expected_answer: Ground truth answer (for correction accuracy)
        
    Returns:
        Dict with behavioral metrics
    """
    text_lower = text.lower()
    
    # Check for verification language
    has_verify = any(re.search(p, text_lower) for p in VERIFY_PATTERNS)
    
    # Check for correction language
    has_correct = any(re.search(p, text_lower) for p in CORRECT_PATTERNS)
    
    # Check for error detection language
    detects_error = any(re.search(p, text_lower) for p in ERROR_DETECTION_PATTERNS)
    
    # Check for non-empty CORRECT block
    correct_match = re.search(r"CORRECT:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    correct_non_empty = False
    correct_is_fix = False
    
    if correct_match:
        correct_content = correct_match.group(1).strip()
        if correct_content and correct_content.lower() != "no correction needed":
            correct_non_empty = True
            # Check if it attempts a real fix
            if any(re.search(p, correct_content.lower()) for p in ERROR_DETECTION_PATTERNS):
                correct_is_fix = True
    
    # Check if output contains expected answer (for correction accuracy)
    contains_expected = False
    if expected_answer:
        # Normalize and check
        expected_normalized = str(expected_answer).strip().lower()
        if expected_normalized in text_lower:
            contains_expected = True
    
    return {
        "has_verify": has_verify,
        "has_correct": has_correct,
        "detects_error": detects_error,
        "correct_non_empty": correct_non_empty,
        "correct_is_fix": correct_is_fix,
        "contains_expected": contains_expected,
        # Derived metrics
        "error_detected": detects_error and has_known_error,
        "false_correction": correct_is_fix and not has_known_error,
    }


# ============================================================
# Evaluation Logic
# ============================================================


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def evaluate_checkpoint(
    checkpoint_path: Path,
    checkpoint_name: str,
    eval_examples: list[dict],
    output_path: Path,
    error_indices: set = None,
    max_new_tokens: int = 120,
) -> dict:
    """Evaluate a checkpoint with error detection metrics.
    
    Args:
        checkpoint_path: Path to model checkpoint
        checkpoint_name: Name for logging
        eval_examples: List of eval examples
        output_path: Path to save predictions
        error_indices: Set of indices with known errors (for fidelity metrics)
        max_new_tokens: Max tokens to generate
        
    Returns:
        Dict with evaluation metadata and fidelity metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_name}")
    print(f"{'='*60}")

    start_time = datetime.now(timezone.utc)

    print("[LOAD] Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        print("       Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise

    print(f"[EVAL] Generating {len(eval_examples)} predictions...")
    predictions = []
    
    # Counters for fidelity metrics
    total = 0
    total_with_error = 0
    total_clean = 0
    
    error_detected = 0
    correction_attempted = 0
    correction_is_fix = 0
    false_corrections = 0
    net_improved = 0
    net_worse = 0

    for i, example in enumerate(eval_examples):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"       Progress: {i+1}/{len(eval_examples)}")

        prompt = example.get("prompt", "")
        expected = example.get("expected_answer") or example.get("final_answer", "")
        has_known_error = i in error_indices if error_indices else False

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Analyze output
        behavior = analyze_output(response, has_known_error, expected)
        
        # Update counters
        total += 1
        if has_known_error:
            total_with_error += 1
            if behavior["error_detected"]:
                error_detected += 1
            if behavior["correct_is_fix"]:
                correction_is_fix += 1
            if behavior["contains_expected"]:
                net_improved += 1
        else:
            total_clean += 1
            if behavior["false_correction"]:
                false_corrections += 1
        
        if behavior["correct_non_empty"]:
            correction_attempted += 1

        prediction = {
            "id": example.get("id") or i,
            "prompt": prompt,
            "expected_answer": expected,
            "predicted_answer": response,
            "has_known_error": has_known_error,
            "behavior": behavior,
        }
        predictions.append(prediction)

    # Save predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"[SAVE] Saved to: {output_path}")

    # Compute metrics
    end_time = datetime.now(timezone.utc)
    
    fidelity_metrics = {
        "total": total,
        "total_with_error": total_with_error,
        "total_clean": total_clean,
        
        # Core M47 metrics
        "error_detection_rate": error_detected / total_with_error if total_with_error > 0 else 0,
        "correction_attempt_rate": correction_attempted / total if total > 0 else 0,
        "correction_accuracy": correction_is_fix / total_with_error if total_with_error > 0 else 0,
        "false_correction_rate": false_corrections / total_clean if total_clean > 0 else 0,
        "net_improvement_rate": net_improved / total_with_error if total_with_error > 0 else 0,
        
        # Raw counts
        "error_detected_count": error_detected,
        "correction_attempted_count": correction_attempted,
        "correction_fix_count": correction_is_fix,
        "false_correction_count": false_corrections,
        "net_improved_count": net_improved,
    }

    metadata = {
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "runtime_seconds": (end_time - start_time).total_seconds(),
        "fidelity_metrics": fidelity_metrics,
        "output_file": str(output_path),
    }

    print(f"\n[RESULT] Fidelity Metrics:")
    print(f"         Error Detection Rate: {fidelity_metrics['error_detection_rate']:.2%}")
    print(f"         Correction Attempt Rate: {fidelity_metrics['correction_attempt_rate']:.2%}")
    print(f"         Correction Accuracy: {fidelity_metrics['correction_accuracy']:.2%}")
    print(f"         False Correction Rate: {fidelity_metrics['false_correction_rate']:.2%}")
    print(f"         Net Improvement Rate: {fidelity_metrics['net_improvement_rate']:.2%}")

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return metadata


def main():
    print("=" * 60)
    print("M47 Error Correction Fidelity — Evaluation")
    print("=" * 60)

    # Load error manifest
    if ERROR_MANIFEST.exists():
        with open(ERROR_MANIFEST, "r") as f:
            manifest = json.load(f)
        error_sample_ids = {e["sample_id"] for e in manifest.get("errors", [])}
        print(f"\n[MANIFEST] Loaded {len(error_sample_ids)} known error positions")
    else:
        error_sample_ids = set()
        print("[WARNING] No error manifest found")

    # Load eval sets
    eval_sets = {}
    
    if HOLDOUT_ERROR.exists():
        holdout_error = load_jsonl(HOLDOUT_ERROR)
        # All holdout_error items have errors injected
        eval_sets["holdout_error"] = {
            "examples": holdout_error,
            "error_indices": set(range(len(holdout_error))),  # All have errors
        }
        print(f"[LOAD] Holdout (error): {len(holdout_error)} examples")
    
    if HOLDOUT_CLEAN.exists():
        holdout_clean = load_jsonl(HOLDOUT_CLEAN)
        eval_sets["holdout_clean"] = {
            "examples": holdout_clean,
            "error_indices": set(),  # No errors
        }
        print(f"[LOAD] Holdout (clean): {len(holdout_clean)} examples")

    if not eval_sets:
        print("[ERROR] No eval sets found!")
        return 1

    # Evaluate each checkpoint on each eval set
    all_results = {}

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        if not ckpt_path.exists():
            print(f"\n[SKIP] Checkpoint not found: {ckpt_name}")
            continue

        all_results[ckpt_name] = {}

        for eval_name, eval_config in eval_sets.items():
            output_path = EVAL_DIR / f"{ckpt_name}_{eval_name}_predictions.jsonl"
            
            try:
                metadata = evaluate_checkpoint(
                    checkpoint_path=ckpt_path,
                    checkpoint_name=f"{ckpt_name} on {eval_name}",
                    eval_examples=eval_config["examples"],
                    output_path=output_path,
                    error_indices=eval_config["error_indices"],
                )
                all_results[ckpt_name][eval_name] = metadata
            except Exception as e:
                print(f"[ERROR] Failed: {e}")
                all_results[ckpt_name][eval_name] = {"error": str(e)}

    # Save summary
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": all_results,
    }
    summary_path = EVAL_DIR / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUMMARY] Saved to: {summary_path}")

    # Save fidelity metrics comparison
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_comparison = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison": {}
    }
    
    for ckpt_name, eval_results in all_results.items():
        for eval_name, result in eval_results.items():
            if "error" not in result:
                key = f"{ckpt_name}_{eval_name}"
                metrics_comparison["comparison"][key] = result.get("fidelity_metrics", {})
    
    metrics_path = METRICS_DIR / "fidelity_comparison.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_comparison, f, indent=2)
    print(f"[METRICS] Saved to: {metrics_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("Fidelity Metrics Comparison (on error-injected holdout)")
    print("=" * 80)
    print(f"{'Checkpoint':<25} {'Detect%':>10} {'Attempt%':>10} {'Accuracy%':>10} {'FalseCorr%':>10}")
    print("-" * 70)
    
    for ckpt_name in CHECKPOINTS.keys():
        if ckpt_name in all_results and "holdout_error" in all_results[ckpt_name]:
            result = all_results[ckpt_name]["holdout_error"]
            if "error" not in result:
                fm = result.get("fidelity_metrics", {})
                print(f"{ckpt_name:<25} {fm.get('error_detection_rate', 0):>10.1%} {fm.get('correction_attempt_rate', 0):>10.1%} {fm.get('correction_accuracy', 0):>10.1%} {fm.get('false_correction_rate', 0):>10.1%}")

    print("\n" + "=" * 60)
    print("M47 Evaluation Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())

