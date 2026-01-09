#!/usr/bin/env python3
"""M45 Batch Evaluation Script.

Evaluates all checkpoints (M44 baseline + curriculum stages A, B, C)
on the fixed eval set (eval_v2.jsonl).

Output:
- eval/m44_baseline_predictions.jsonl
- eval/post_stage_a_predictions.jsonl
- eval/post_stage_b_predictions.jsonl
- eval/post_stage_c_predictions.jsonl
- eval/eval_summary.json

Usage:
    cd research/m45_curriculum_reasoning
    python eval_all_checkpoints.py

Author: M45 Curriculum Reasoning Milestone
Date: 2026-01-08
"""

import argparse
import codecs
import json
import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR / ".." / ".."

EVAL_SET = PROJECT_ROOT / "training" / "evalsets" / "eval_v2.jsonl"
EVAL_DIR = SCRIPT_DIR / "eval"

# Checkpoints to evaluate
CHECKPOINTS = {
    "m44_baseline": PROJECT_ROOT / "submission_runs" / "m44_v1" / "training_output" / "final_model",
    "post_stage_a": SCRIPT_DIR / "checkpoints" / "stage_a" / "final_model",
    "post_stage_b": SCRIPT_DIR / "checkpoints" / "stage_b" / "final_model",
    "post_stage_c": SCRIPT_DIR / "checkpoints" / "stage_c" / "final_model",
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
    device: str = "cuda",
    max_new_tokens: int = 50,
) -> dict:
    """Evaluate a single checkpoint on the eval set.

    Args:
        checkpoint_path: Path to model checkpoint
        checkpoint_name: Name for logging
        eval_examples: List of eval examples
        output_path: Path to save predictions
        device: Device to use
        max_new_tokens: Max tokens to generate

    Returns:
        Dict with evaluation metadata
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

    # Generate predictions
    print(f"[EVAL] Generating {len(eval_examples)} predictions...")
    predictions = []

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

        prediction = {
            "id": example.get("id"),
            "prompt": prompt,
            "expected_answer": expected,
            "predicted_answer": response,
            "section": example.get("section"),
            "category": example.get("category"),
            "difficulty": example.get("difficulty"),
            "correct": response.strip().lower() == str(expected).strip().lower(),
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
    correct_count = sum(1 for p in predictions if p["correct"])
    accuracy = correct_count / len(predictions) if predictions else 0

    # Per-section breakdown
    sections = {}
    for pred in predictions:
        sec = pred.get("section", "unknown")
        if sec not in sections:
            sections[sec] = {"total": 0, "correct": 0}
        sections[sec]["total"] += 1
        if pred["correct"]:
            sections[sec]["correct"] += 1

    for sec in sections:
        sections[sec]["accuracy"] = sections[sec]["correct"] / sections[sec]["total"]

    metadata = {
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "eval_set": str(EVAL_SET),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "runtime_seconds": (end_time - start_time).total_seconds(),
        "total_examples": len(predictions),
        "correct": correct_count,
        "accuracy": accuracy,
        "sections": sections,
        "output_file": str(output_path),
    }

    print(f"\n[RESULT] Accuracy: {accuracy:.2%} ({correct_count}/{len(predictions)})")
    for sec, stats in sections.items():
        print(f"         {sec}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    # Free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return metadata


def main():
    parser = argparse.ArgumentParser(description="M45 Batch Evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples")
    args = parser.parse_args()

    print("=" * 60)
    print("M45 Curriculum Reasoning — Batch Evaluation")
    print("=" * 60)

    # Verify eval set exists
    if not EVAL_SET.exists():
        print(f"[ERROR] Eval set not found: {EVAL_SET}")
        return 1

    # Load eval set
    print(f"\n[LOAD] Loading eval set: {EVAL_SET}")
    eval_examples = load_eval_set(EVAL_SET)
    print(f"       Loaded {len(eval_examples)} examples")

    if args.max_examples:
        eval_examples = eval_examples[:args.max_examples]
        print(f"       Limited to {args.max_examples} examples")

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
                device=args.device,
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

    summary_path = EVAL_DIR / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUMMARY] Saved to: {summary_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("Accuracy Comparison")
    print("=" * 60)
    print(f"{'Checkpoint':<20} {'Accuracy':>10} {'Correct':>10}")
    print("-" * 42)
    for name, result in all_results.items():
        if "error" in result:
            print(f"{name:<20} {'ERROR':>10}")
        else:
            acc = result.get("accuracy", 0)
            correct = result.get("correct", 0)
            total = result.get("total_examples", 0)
            print(f"{name:<20} {acc:>10.2%} {correct:>5}/{total}")

    print("\n" + "=" * 60)
    print("M45 Batch Evaluation Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())

