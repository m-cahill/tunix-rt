#!/usr/bin/env python3
"""Kaggle Submission Script - Single Session Training + Evaluation

This script provides a reproducible single-session workflow for the Tunix Hack competition.
It can be converted to a Jupyter notebook or run directly as a Python script.

Usage:
    python kaggle_submission.py --max_steps 100 --dataset golden-v2
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Configuration
DEFAULT_MODEL = "google/gemma-2-2b"
DEFAULT_DATASET = "golden-v2"
DEFAULT_MAX_STEPS = 100
DEFAULT_OUTPUT_DIR = "./output/kaggle_run"


def run_command(cmd: list[str], description: str):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n✅ {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Kaggle submission training pipeline")
    parser.add_argument("--model_name", default=DEFAULT_MODEL, help="Model to train")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset to use")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS, help="Max training steps")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu", "tpu"], help="Device")
    parser.add_argument("--skip_dataset_build", action="store_true", help="Skip dataset building")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║          TUNIX RT - KAGGLE SUBMISSION PIPELINE               ║
╚══════════════════════════════════════════════════════════════╝

Configuration:
  Model:       {model_name}
  Dataset:     {dataset}
  Max Steps:   {max_steps}
  Output Dir:  {output_dir}
  Seed:        {seed}
  Device:      {device}
""".format(**vars(args)))

    # Step 1: Build dataset (if not skipped)
    if not args.skip_dataset_build:
        if args.dataset == "golden-v2":
            run_command(
                ["python", "backend/tools/seed_golden_v2.py"],
                "Build golden-v2 dataset"
            )
        elif args.dataset == "dev-reasoning-v1":
            run_command(
                ["python", "backend/tools/seed_dev_reasoning_v1.py"],
                "Build dev-reasoning-v1 dataset"
            )
        else:
            print(f"⚠️  Dataset {args.dataset} not recognized, assuming it already exists")

    # Step 2: Train model
    train_cmd = [
        "python", "training/train_jax.py",
        "--dataset", args.dataset,
        "--model_name", args.model_name,
        "--max_steps", str(args.max_steps),
        "--device", args.device,
        "--output_dir", args.output_dir,
        "--seed", str(args.seed),
    ]
    run_command(train_cmd, "Train model with JAX/Flax")

    # Step 3: Generate predictions
    eval_set = "training/evalsets/eval_v1.jsonl"
    predictions_file = f"{args.output_dir}/predictions.jsonl"

    run_command(
        [
            "python", "training/eval_generate.py",
            "--checkpoint", args.output_dir,
            "--eval_set", eval_set,
            "--output", predictions_file,
        ],
        "Generate predictions"
    )

    # Step 4: Score predictions
    run_command(
        [
            "python", "training/eval_report.py",
            "--predictions", predictions_file,
            "--eval_set", eval_set,
        ],
        "Evaluate predictions"
    )

    # Step 5: Display results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")

    # Check for metrics file
    metrics_file = Path(args.output_dir) / "metrics.jsonl"
    if metrics_file.exists():
        print("Training Metrics (last 5 steps):")
        with open(metrics_file, "r") as f:
            lines = f.readlines()
            for line in lines[-5:]:
                metric = json.loads(line)
                print(f"  Step {metric.get('step', '?')}: loss={metric.get('loss', '?'):.4f}")

    # Check for eval results
    eval_results_file = Path(args.output_dir) / "eval_results.json"
    if eval_results_file.exists():
        with open(eval_results_file, "r") as f:
            results = json.load(f)
            print(f"\nEvaluation Score: {results.get('answer_correctness', 'N/A'):.2f}")

    print(f"\n✅ Pipeline complete! Artifacts saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review metrics.jsonl for training progress")
    print("  2. Check predictions.jsonl for model outputs")
    print("  3. Submit checkpoint to competition (if applicable)")


if __name__ == "__main__":
    main()
