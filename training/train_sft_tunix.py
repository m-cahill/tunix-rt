#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tunix SFT Training Orchestrator.

Routes training requests to the appropriate backend:
- JAX (Preferred): Real Flax/Optax implementation
- PyTorch (Fallback): Transformers Trainer implementation
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Configure stdout for UTF-8 on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Check for required dependencies before importing
try:
    import yaml  # noqa: F401
except ImportError:
    print("❌ Error: PyYAML not found")
    print("   Install with: pip install pyyaml")
    sys.exit(1)


def check_jax_available() -> bool:
    """Check if JAX is installed."""
    try:
        import jax  # noqa: F401
        import flax  # noqa: F401
        import optax  # noqa: F401
        return True
    except ImportError:
        return False


def check_torch_available() -> bool:
    """Check if PyTorch/Transformers is installed."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def load_config(config_path: Path) -> dict[str, Any]:
    """Load training configuration from YAML file."""
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(data_path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load training dataset from JSONL file."""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping invalid JSON on line {i + 1}: {e}")
                continue
    return samples


def create_run_manifest(
    config: dict[str, Any],
    dataset_path: Path,
    output_dir: Path,
    run_id: uuid.UUID,
    backend: str,
) -> dict[str, Any]:
    """Create a run manifest for reproducibility."""
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
        pass

    manifest = {
        "run_id": str(run_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path.absolute()),
        "training_config": config.get("training", {}),
        "model_config": config.get("model", {}),
        "recipe": f"trace_sft_{backend}",
        "backend": backend,
        "seed": config.get("training", {}).get("seed", 42),
        "git_sha": git_sha,
        "artifacts_path": str(output_dir.absolute()),
    }
    return manifest


def save_manifest(manifest: dict[str, Any], output_dir: Path) -> None:
    """Save run manifest to output directory."""
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Saved run manifest: {manifest_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Tunix SFT training on reasoning traces",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument("--data", type=Path, required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", type=str, choices=["auto", "jax", "torch"], default="auto")

    args = parser.parse_args()

    print("🔥 Tunix RT - SFT Training Orchestrator")
    print("=" * 70)

    # Detect Backend
    jax_available = check_jax_available()
    torch_available = check_torch_available()

    backend = args.backend
    if backend == "auto":
        if jax_available:
            backend = "jax"
        elif torch_available:
            backend = "torch"
        else:
            print("❌ No suitable backend found (JAX or PyTorch)")
            print("   Install 'backend[training]' dependencies.")
            sys.exit(1)

    if backend == "jax" and not jax_available:
        print("❌ JAX requested but not installed.")
        sys.exit(1)

    if backend == "torch" and not torch_available:
        print("❌ PyTorch requested but not installed.")
        sys.exit(1)

    print(f"✅ Backend: {backend.upper()}")

    # Load config & data
    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    config = load_config(args.config)

    if not args.data.exists():
        print(f"❌ Dataset file not found: {args.data}")
        sys.exit(1)
    dataset = load_dataset(args.data, max_samples=args.max_samples)
    print(f"✅ Loaded {len(dataset)} samples")

    if not dataset:
        print("❌ Dataset empty")
        sys.exit(1)

    if args.dry_run:
        print("✅ Dry run complete")
        return

    # Prepare Output
    args.output.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4()

    # Save Manifest
    manifest = create_run_manifest(config, args.data, args.output, run_id, backend)
    save_manifest(manifest, args.output)

    # Run Training
    if backend == "jax":
        try:
             # Try local import first (when running as script from training dir)
             import train_jax
             train_jax.run_jax_sft_training(config, dataset, args.output)
        except ImportError:
             try:
                 # Try package import (when running from root)
                 from training.train_jax import run_jax_sft_training
                 run_jax_sft_training(config, dataset, args.output)
             except ImportError as e:
                 print(f"❌ Failed to import JAX backend: {e}")
                 sys.exit(1)

    elif backend == "torch":
        try:
             import train_torch
             train_torch.run_torch_sft_training(config, dataset, args.output)
        except ImportError:
             try:
                 from training.train_torch import run_torch_sft_training
                 run_torch_sft_training(config, dataset, args.output)
             except ImportError as e:
                 print(f"❌ Failed to import Torch backend: {e}")
                 sys.exit(1)

    print("\n✅ Training Complete!")


if __name__ == "__main__":
    main()
