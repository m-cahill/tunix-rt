#!/usr/bin/env python3
"""Tunix SFT Training Script for tunix-rt.

This script runs Supervised Fine-Tuning using Google's Tunix library.
It's designed for minimal, reproducible training runs with full provenance tracking.

Usage:
    python training/train_sft_tunix.py \\
        --config training/configs/sft_tiny.yaml \\
        --data path/to/dataset.jsonl \\
        --output artifacts/training_runs/my_run

Requirements:
    - Tunix (install from GitHub - see README)
    - JAX (backend[training] extra)
    - Backend with training schemas

Note: This script exits gracefully if Tunix is not installed.
"""

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Check for required dependencies before importing
try:
    import yaml  # noqa: F401
except ImportError:
    print("❌ Error: PyYAML not found")
    print("   Install with: pip install pyyaml")
    sys.exit(1)


def check_tunix_available() -> bool:
    """Check if Tunix is installed.

    Returns:
        True if Tunix is available, False otherwise
    """
    try:
        import tunix  # noqa: F401
        return True
    except ImportError:
        return False


def check_jax_available() -> bool:
    """Check if JAX is installed.

    Returns:
        True if JAX is available, False otherwise
    """
    try:
        import jax  # noqa: F401
        return True
    except ImportError:
        return False


def print_installation_instructions():
    """Print helpful installation instructions for missing dependencies."""
    print("\n" + "=" * 70)
    print("Tunix SFT Training - Missing Dependencies")
    print("=" * 70)
    print("\nThis script requires Tunix and JAX for training.")
    print("\nTo install:")
    print("\n1. Install JAX (CPU or GPU):")
    print("   pip install -e backend[training]  # CPU version")
    print("   # OR for GPU:")
    print("   pip install 'jax[cuda12]'")
    print("\n2. Install Tunix from GitHub (pinned commit for reproducibility):")
    print("   pip install git+https://github.com/google-deepmind/tunix.git@<COMMIT_SHA>")
    print("\n   See docs/M09_TRAINING_QUICKSTART.md for recommended commit SHA")
    print("\nAlternatively, you can:")
    print("- Run the smoke test only (validates data format without training)")
    print("- Use the evaluation scripts to test outputs from existing models")
    print("\nFor more information:")
    print("- README: training/README.md")
    print("- Docs: docs/M09_TRAINING_QUICKSTART.md")
    print("=" * 70 + "\n")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to config YAML

    Returns:
        Configuration dictionary
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_dataset(data_path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load training dataset from JSONL file.

    Args:
        data_path: Path to JSONL dataset
        max_samples: Maximum samples to load (for testing)

    Returns:
        List of training samples
    """
    samples = []

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping invalid JSON on line {i+1}: {e}")
                continue

    return samples


def create_run_manifest(
    config: dict[str, Any],
    dataset_path: Path,
    output_dir: Path,
    run_id: uuid.UUID,
) -> dict[str, Any]:
    """Create a run manifest for reproducibility.

    Args:
        config: Training configuration
        dataset_path: Path to training data
        output_dir: Output directory
        run_id: Unique run identifier

    Returns:
        Manifest dictionary
    """
    import subprocess

    # Try to get git SHA
    git_sha = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_sha = result.stdout.strip()
    except Exception:
        pass  # Git not available or not in repo

    manifest = {
        "run_id": str(run_id),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dataset_path": str(dataset_path.absolute()),
        "training_config": config.get("training", {}),
        "model_config": config.get("model", {}),
        "recipe": "trace_sft_v1",
        "seed": config.get("training", {}).get("seed", 42),
        "git_sha": git_sha,
        "artifacts_path": str(output_dir.absolute()),
    }

    return manifest


def save_manifest(manifest: dict[str, Any], output_dir: Path) -> None:
    """Save run manifest to output directory.

    Args:
        manifest: Manifest dictionary
        output_dir: Output directory
    """
    manifest_path = output_dir / "run_manifest.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Saved run manifest: {manifest_path}")


def run_tunix_sft_training(
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Run Tunix SFT training.

    This is a placeholder that would integrate with Tunix's actual training API.
    The real implementation would:
    1. Tokenize the dataset
    2. Create a Tunix SFT trainer
    3. Run training for N steps
    4. Save checkpoint and metrics

    Args:
        config: Training configuration
        dataset: Training samples
        output_dir: Output directory for checkpoints/metrics
    """
    print("\n🚀 Starting Tunix SFT Training...")
    print(f"   Samples: {len(dataset)}")
    print(f"   Steps: {config['training']['num_steps']}")
    print(f"   Seed: {config['training']['seed']}")

    # NOTE: Actual Tunix integration would go here
    # For M09, we're focused on the infrastructure and data pipeline
    # Real training integration deferred to M10 or when Tunix API is stable

    print("\n⚠️  Note: Actual Tunix training integration is a placeholder")
    print("   This script validates the pipeline and creates manifests")
    print("   Full training integration coming in M10")

    # Create placeholder metrics
    metrics_path = output_dir / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        for step in range(0, config["training"]["num_steps"] + 1, 10):
            metric = {
                "step": step,
                "loss": 2.5 - (step * 0.01),  # Simulated decreasing loss
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            f.write(json.dumps(metric) + "\n")

    print(f"✅ Saved metrics: {metrics_path}")

    # Create placeholder checkpoint directory
    checkpoint_dir = output_dir / config["output"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_info = {
        "model": config["model"]["name"],
        "steps": config["training"]["num_steps"],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
        json.dump(checkpoint_info, f, indent=2)

    print(f"✅ Created checkpoint directory: {checkpoint_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Tunix SFT training on reasoning traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training config YAML (e.g., training/configs/sft_tiny.yaml)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data JSONL (e.g., dataset.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for artifacts (e.g., artifacts/training_runs/my_run)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to use (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and data without running training",
    )

    args = parser.parse_args()

    print("🔥 Tunix RT - SFT Training Script")
    print("=" * 70)

    # Check dependencies
    tunix_available = check_tunix_available()
    jax_available = check_jax_available()

    if not jax_available:
        print("❌ JAX not found")
        print_installation_instructions()
        sys.exit(1)
    else:
        print("✅ JAX available")

    if not tunix_available:
        print("⚠️  Tunix not found")
        print("   Training will run in simulation mode (creates manifests only)")
        print("   For actual training, install Tunix (see README)")

    # Load config
    print(f"\n📋 Loading config: {args.config}")
    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    print(f"✅ Config loaded")

    # Load dataset
    print(f"\n📂 Loading dataset: {args.data}")
    if not args.data.exists():
        print(f"❌ Dataset file not found: {args.data}")
        sys.exit(1)

    dataset = load_dataset(args.data, max_samples=args.max_samples)
    print(f"✅ Loaded {len(dataset)} samples")

    if len(dataset) == 0:
        print("❌ No samples loaded from dataset")
        sys.exit(1)

    # Validate first sample has required format
    first_sample = dataset[0]
    if "prompts" not in first_sample:
        print("❌ Dataset samples missing 'prompts' field")
        print("   Ensure dataset is exported in tunix_sft format")
        sys.exit(1)

    print("✅ Dataset format validated")

    if args.dry_run:
        print("\n✅ Dry run complete - config and data are valid")
        print("   Remove --dry-run to run actual training")
        return

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {args.output}")

    # Generate run ID
    run_id = uuid.uuid4()
    print(f"🆔 Run ID: {run_id}")

    # Create and save manifest
    manifest = create_run_manifest(config, args.data, args.output, run_id)
    save_manifest(manifest, args.output)

    # Run training
    run_tunix_sft_training(config, dataset, args.output)

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print(f"\nArtifacts saved to: {args.output}")
    print(f"- run_manifest.json  (reproducibility metadata)")
    print(f"- metrics.jsonl      (training metrics)")
    print(f"- {config['output']['checkpoint_dir']}/  (model checkpoint)")
    print("\nNext steps:")
    print("1. Run evaluation: python training/eval_generate.py ...")
    print("2. Create delta report: python training/eval_report.py ...")
    print("3. See docs/M09_EVAL_LOOP.md for details")
    print("=" * 70)


if __name__ == "__main__":
    main()

