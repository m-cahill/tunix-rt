#!/usr/bin/env python3
"""M46 Evaluation Script — Structured Self-Correction.

Evaluates checkpoints and extracts behavioral metrics:
1. M45 Stage-C baseline (for comparison)
2. M46 Control checkpoint
3. M46 Self-Correction checkpoint

Behavioral Metrics (per M46_answers.md):
- Verification frequency (primary)
- Presence of VERIFY/CHECK language
- Presence of CORRECT language
- False verification detection

Usage:
    cd research/m46_structured_self_correction
    python scripts/run_eval.py

Author: M46 Structured Self-Correction Milestone
Date: 2026-01-08
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
EVAL_DIR = PROJECT_DIR / "eval"
METRICS_DIR = PROJECT_DIR / "metrics"

# Project root for accessing eval set
PROJECT_ROOT = PROJECT_DIR.parent.parent

# Eval set from training directory
EVAL_SET = PROJECT_ROOT / "training" / "evalsets" / "eval_v2.jsonl"

# Checkpoints to evaluate
CHECKPOINTS = {
    "m45_stage_c": PROJECT_DIR.parent / "m45_curriculum_reasoning" / "checkpoints" / "stage_c" / "final_model",
    "m46_control": PROJECT_DIR / "checkpoints" / "control" / "final_model",
    "m46_self_correct": PROJECT_DIR / "checkpoints" / "self_correct" / "final_model",
}


# ============================================================
# Verification Detection Patterns
# ============================================================

# Patterns to detect verification behavior (Guardrail 2: detect false verifications)
VERIFY_PATTERNS = [
    r"\bverify\b",
    r"\bcheck\b",
    r"\bconfirm\b",
    r"\bvalidate\b",
    r"\bby inverse\b",
    r"\brecheck\b",
    r"\bdouble.?check\b",
]

CORRECT_PATTERNS = [
    r"\bcorrect\b",
    r"\bfix\b",
    r"\badjust\b",
    r"\bmodify\b",
    r"\bchange\b",
    r"\bactually\b",
    r"\bwait\b",  # Often precedes correction
    r"\bmistake\b",
    r"\berror\b",
]

# Patterns that suggest meaningless/false verification
FALSE_VERIFY_PATTERNS = [
    r"verify.*verify.*verify",  # Excessive repetition
    r"check.*check.*check",
    r"correct.*correct.*correct",
]


def detect_verification_language(text: str) -> dict:
    """Detect verification-related language in generated text.
    
    Args:
        text: Generated text to analyze
        
    Returns:
        Dict with detection results
    """
    text_lower = text.lower()
    
    # Check for verification patterns
    verify_matches = []
    for pattern in VERIFY_PATTERNS:
        if re.search(pattern, text_lower):
            verify_matches.append(pattern)
    
    # Check for correction patterns
    correct_matches = []
    for pattern in CORRECT_PATTERNS:
        if re.search(pattern, text_lower):
            correct_matches.append(pattern)
    
    # Check for false verification patterns (Guardrail 2)
    false_verify = False
    for pattern in FALSE_VERIFY_PATTERNS:
        if re.search(pattern, text_lower):
            false_verify = True
            break
    
    return {
        "has_verify_language": len(verify_matches) > 0,
        "verify_patterns": verify_matches,
        "verify_count": len(verify_matches),
        "has_correct_language": len(correct_matches) > 0,
        "correct_patterns": correct_matches,
        "correct_count": len(correct_matches),
        "is_false_verify": false_verify,
    }


# ============================================================
# Evaluation Logic
# ============================================================


def load_eval_set(eval_path: Path) -> list[dict]:
    """Load evaluation set from JSONL."""
    examples = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def evaluate_checkpoint(
    checkpoint_path: Path,
    checkpoint_name: str,
    eval_examples: list[dict],
    output_path: Path,
    max_new_tokens: int = 100,
) -> dict:
    """Evaluate a single checkpoint with behavioral metrics.
    
    Args:
        checkpoint_path: Path to model checkpoint
        checkpoint_name: Name for logging
        eval_examples: List of eval examples
        output_path: Path to save predictions
        max_new_tokens: Max tokens to generate (longer for verification capture)
        
    Returns:
        Dict with evaluation metadata and behavioral metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_name}")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")

    start_time = datetime.now(timezone.utc)

    # Load model
    print("[LOAD] Loading model and tokenizer...")
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

    # Generate predictions with behavioral analysis
    print(f"[EVAL] Generating {len(eval_examples)} predictions...")
    predictions = []
    
    # Behavioral counters
    verify_count = 0
    correct_count = 0
    false_verify_count = 0

    for i, example in enumerate(eval_examples):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"       Progress: {i+1}/{len(eval_examples)}")

        prompt = example.get("prompt", "")
        expected = example.get("expected_answer", "")

        # Generate
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

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Analyze behavioral patterns
        behavior = detect_verification_language(response)
        
        if behavior["has_verify_language"]:
            verify_count += 1
        if behavior["has_correct_language"]:
            correct_count += 1
        if behavior["is_false_verify"]:
            false_verify_count += 1

        prediction = {
            "id": example.get("id"),
            "prompt": prompt,
            "expected_answer": expected,
            "predicted_answer": response,
            "section": example.get("section"),
            "category": example.get("category"),
            "difficulty": example.get("difficulty"),
            "correct": response.strip().lower() == str(expected).strip().lower(),
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
    correct_answers = sum(1 for p in predictions if p["correct"])
    accuracy = correct_answers / len(predictions) if predictions else 0

    # Behavioral metrics (PRIMARY for M46)
    behavioral_metrics = {
        "verification_frequency": verify_count / len(predictions) if predictions else 0,
        "correction_frequency": correct_count / len(predictions) if predictions else 0,
        "false_verification_frequency": false_verify_count / len(predictions) if predictions else 0,
        "samples_with_verify": verify_count,
        "samples_with_correct": correct_count,
        "samples_with_false_verify": false_verify_count,
        "total_samples": len(predictions),
    }

    metadata = {
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "eval_set": str(EVAL_SET),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "runtime_seconds": (end_time - start_time).total_seconds(),
        "total_examples": len(predictions),
        "correct": correct_answers,
        "accuracy": accuracy,
        "behavioral_metrics": behavioral_metrics,
        "output_file": str(output_path),
    }

    print(f"\n[RESULT] Accuracy: {accuracy:.2%} ({correct_answers}/{len(predictions)})")
    print(f"         Verification freq: {behavioral_metrics['verification_frequency']:.2%}")
    print(f"         Correction freq: {behavioral_metrics['correction_frequency']:.2%}")
    print(f"         False verify freq: {behavioral_metrics['false_verification_frequency']:.2%}")

    # Free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return metadata


def main():
    print("=" * 60)
    print("M46 Structured Self-Correction — Evaluation")
    print("=" * 60)

    # Verify eval set exists
    if not EVAL_SET.exists():
        print(f"[ERROR] Eval set not found: {EVAL_SET}")
        return 1

    # Load eval set
    print(f"\n[LOAD] Loading eval set: {EVAL_SET}")
    eval_examples = load_eval_set(EVAL_SET)
    print(f"       Loaded {len(eval_examples)} examples")

    # Evaluate each checkpoint
    all_results = {}

    for name, path in CHECKPOINTS.items():
        if not path.exists():
            print(f"\n[SKIP] Checkpoint not found: {name} ({path})")
            continue

        output_path = EVAL_DIR / f"{name}_predictions.jsonl"

        try:
            metadata = evaluate_checkpoint(
                checkpoint_path=path,
                checkpoint_name=name,
                eval_examples=eval_examples,
                output_path=output_path,
            )
            all_results[name] = metadata
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {name}: {e}")
            all_results[name] = {"error": str(e)}

    # Save summary
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_set": str(EVAL_SET),
        "total_examples": len(eval_examples),
        "results": all_results,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = EVAL_DIR / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUMMARY] Saved to: {summary_path}")

    # Save behavioral metrics separately
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_comparison = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison": {}
    }
    for name, result in all_results.items():
        if "error" not in result:
            metrics_comparison["comparison"][name] = {
                "accuracy": result["accuracy"],
                **result["behavioral_metrics"]
            }
    
    metrics_path = METRICS_DIR / "behavioral_comparison.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_comparison, f, indent=2)
    print(f"[METRICS] Saved to: {metrics_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("Behavioral Metrics Comparison")
    print("=" * 60)
    print(f"{'Checkpoint':<20} {'Accuracy':>10} {'Verify%':>10} {'Correct%':>10} {'FalseV%':>10}")
    print("-" * 62)
    for name, result in all_results.items():
        if "error" in result:
            print(f"{name:<20} {'ERROR':>10}")
        else:
            acc = result.get("accuracy", 0)
            bm = result.get("behavioral_metrics", {})
            vf = bm.get("verification_frequency", 0)
            cf = bm.get("correction_frequency", 0)
            ff = bm.get("false_verification_frequency", 0)
            print(f"{name:<20} {acc:>10.2%} {vf:>10.2%} {cf:>10.2%} {ff:>10.2%}")

    print("\n" + "=" * 60)
    print("M46 Evaluation Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())

