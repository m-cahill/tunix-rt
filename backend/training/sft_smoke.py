#!/usr/bin/env python3
"""Smoke test for SFT training workflow.

This script validates that:
1. Dataset JSONL can be loaded and parsed
2. Tunix SFT prompts are correctly formatted
3. Data shapes are compatible with training

This is NOT a full training run - it's a smoke test to validate
the dataset → training pipeline works end-to-end.

Usage:
    python backend/training/sft_smoke.py <dataset_jsonl_path> [--samples N]

Requirements:
    pip install -e "backend[training]"  # Optional: for actual JAX/Flax validation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_dataset_jsonl(jsonl_path: Path, max_samples: int = 32) -> list[dict[str, Any]]:
    """Load dataset from JSONL file.

    Args:
        jsonl_path: Path to JSONL file
        max_samples: Maximum samples to load (default: 32)

    Returns:
        List of parsed JSON records
    """
    print(f"📂 Loading dataset from: {jsonl_path}")

    if not jsonl_path.exists():
        print(f"❌ Error: File not found: {jsonl_path}")
        sys.exit(1)

    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Invalid JSON on line {i + 1}: {e}")
                continue

    print(f"✅ Loaded {len(samples)} samples")
    return samples


def validate_sft_format(samples: list[dict[str, Any]]) -> bool:
    """Validate that samples have required fields for SFT.

    Args:
        samples: List of dataset samples

    Returns:
        True if validation passes
    """
    print("\n🔍 Validating SFT format...")

    required_fields = ["prompts", "final_answer", "metadata"]
    errors = []

    for i, sample in enumerate(samples):
        # Check required fields
        missing = [field for field in required_fields if field not in sample]
        if missing:
            errors.append(f"Sample {i}: missing fields {missing}")
            continue

        # Check prompts field is non-empty string
        if not isinstance(sample["prompts"], str) or len(sample["prompts"]) == 0:
            errors.append(f"Sample {i}: 'prompts' must be non-empty string")

        # Check for Tunix SFT markers (if format=tunix_sft)
        if sample.get("metadata", {}).get("format") == "tunix_sft":
            prompt_text = sample["prompts"]
            if "<start_of_turn>" not in prompt_text:
                errors.append(f"Sample {i}: tunix_sft format missing chat markers")

    if errors:
        print("❌ Validation failed:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False

    print(f"✅ All {len(samples)} samples validated")
    return True


def check_training_deps() -> bool:
    """Check if training dependencies (JAX, Flax) are available.

    Returns:
        True if dependencies are installed
    """
    try:
        import flax  # noqa: F401
        import jax  # noqa: F401

        print("✅ Training dependencies (JAX, Flax) available")
        return True
    except ImportError:
        print("⚠️  Training dependencies not installed (optional)")
        print("   Install with: pip install -e '.[training]'")
        return False


def smoke_test_training_shapes(samples: list[dict[str, Any]]) -> None:
    """Validate data shapes for training (if JAX is available).

    Args:
        samples: Dataset samples
    """
    if not check_training_deps():
        print("ℹ️  Skipping shape validation (training deps not installed)")
        return

    import jax.numpy as jnp

    print("\n🔢 Validating data shapes...")

    # For a real training script, you'd tokenize here
    # For smoke test, just validate we can create arrays
    try:
        # Simulate: convert prompts to "token" lengths
        prompt_lengths = [len(sample["prompts"]) for sample in samples]
        length_array = jnp.array(prompt_lengths)

        min_len = length_array.min()
        max_len = length_array.max()
        mean_len = length_array.mean()
        print(f"  - Prompt lengths: min={min_len}, max={max_len}, mean={mean_len:.1f}")
        print(f"  - Batch size: {len(samples)}")
        print("✅ Data shapes compatible with JAX")

    except Exception as e:
        print(f"❌ Error validating shapes: {e}")
        sys.exit(1)


def main() -> None:
    """Run smoke test."""
    parser = argparse.ArgumentParser(
        description="Smoke test for SFT training workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to dataset JSONL file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Number of samples to validate (default: 32)",
    )

    args = parser.parse_args()

    print("🔥 Tunix RT - SFT Training Smoke Test")
    print("=" * 50)

    # Load dataset
    samples = load_dataset_jsonl(args.jsonl_path, max_samples=args.samples)

    if not samples:
        print("❌ No samples loaded")
        sys.exit(1)

    # Validate format
    if not validate_sft_format(samples):
        sys.exit(1)

    # Check shapes (if training deps available)
    smoke_test_training_shapes(samples)

    print("\n" + "=" * 50)
    print("✅ Smoke test PASSED")
    print("\nℹ️  This smoke test validates dataset format and shapes.")
    print("   For actual training, use a full Tunix training script.")
    sys.exit(0)


if __name__ == "__main__":
    main()
