#!/usr/bin/env python3
"""M49 Observer Dataset Preparation — Balanced Error Detection Dataset.

Creates a balanced dataset for training an external observer model to detect
errors in reasoning traces.

Design Decisions (from M49 confirmation):
- Holdout + balanced subset (~100 samples)
- 50/50 error/clean balance
- Binary labels only (error_present)
- Seeded, deterministic selection

Output:
- data/observer_train.jsonl (70%)
- data/observer_val.jsonl (15%)
- data/observer_test.jsonl (15%)
- data/dataset_manifest.json

Author: M49 Observer Error Detection Milestone
Date: 2026-01-09
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

# Source data from M47
M47_DIR = PROJECT_DIR.parent / "m47_error_correction_fidelity"
M47_HOLDOUT_ERROR = M47_DIR / "data" / "stage_c_holdout_error.jsonl"
M47_HOLDOUT_CLEAN = M47_DIR / "data" / "stage_c_holdout.jsonl"
M47_TRAINING_ERROR = M47_DIR / "data" / "stage_c_error_self_correct.jsonl"
M47_TRAINING_CLEAN = M47_DIR / "data" / "stage_c_clean.jsonl"
M47_MANIFEST = M47_DIR / "error_manifest.json"

# M47 eval predictions (to get actual model outputs)
M47_ERROR_PREDS = M47_DIR / "eval" / "m47_error_aware_holdout_error_predictions.jsonl"
M47_CLEAN_PREDS = M47_DIR / "eval" / "m47_error_aware_holdout_clean_predictions.jsonl"

# Random seed
SEED = 42

# Target sizes
TARGET_TOTAL = 100  # ~50 error + 50 clean
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# Main
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def main():
    print("=" * 60)
    print("M49 Observer Dataset Preparation")
    print("=" * 60)
    
    random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load error predictions (holdout with errors)
    print(f"\n[LOAD] M47 error predictions: {M47_ERROR_PREDS}")
    error_preds = load_jsonl(M47_ERROR_PREDS)
    print(f"       Loaded {len(error_preds)} error traces")
    
    # Load clean predictions (holdout without errors)
    print(f"[LOAD] M47 clean predictions: {M47_CLEAN_PREDS}")
    clean_preds = load_jsonl(M47_CLEAN_PREDS)
    print(f"       Loaded {len(clean_preds)} clean traces")
    
    # Create balanced dataset
    # All holdout_error traces have errors, all holdout_clean traces are clean
    
    # Convert to observer format
    error_samples = []
    for pred in error_preds:
        sample = {
            "id": f"error_{pred.get('id', len(error_samples))}",
            "input_text": pred.get("predicted_answer", ""),
            "prompt": pred.get("prompt", ""),
            "label": 1,  # error_present = True
            "expected_answer": pred.get("expected_answer", ""),
            "source": "m47_error_holdout",
        }
        error_samples.append(sample)
    
    clean_samples = []
    for pred in clean_preds:
        sample = {
            "id": f"clean_{pred.get('id', len(clean_samples))}",
            "input_text": pred.get("predicted_answer", ""),
            "prompt": pred.get("prompt", ""),
            "label": 0,  # error_present = False
            "expected_answer": pred.get("expected_answer", ""),
            "source": "m47_clean_holdout",
        }
        clean_samples.append(sample)
    
    print(f"\n[PREPARE] Error samples: {len(error_samples)}")
    print(f"          Clean samples: {len(clean_samples)}")
    
    # Balance: take min of both, up to TARGET_TOTAL/2 each
    n_per_class = min(len(error_samples), len(clean_samples), TARGET_TOTAL // 2)
    
    # Shuffle and select
    random.shuffle(error_samples)
    random.shuffle(clean_samples)
    
    selected_error = error_samples[:n_per_class]
    selected_clean = clean_samples[:n_per_class]
    
    # Combine and shuffle
    all_samples = selected_error + selected_clean
    random.shuffle(all_samples)
    
    print(f"\n[BALANCE] Selected {len(selected_error)} error + {len(selected_clean)} clean = {len(all_samples)} total")
    
    # Split into train/val/test
    n_total = len(all_samples)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    # n_test = rest
    
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]
    
    print(f"\n[SPLIT] Train: {len(train_samples)}")
    print(f"        Val: {len(val_samples)}")
    print(f"        Test: {len(test_samples)}")
    
    # Verify balance in each split
    for name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        n_error = sum(1 for s in samples if s["label"] == 1)
        n_clean = sum(1 for s in samples if s["label"] == 0)
        print(f"        {name}: {n_error} error / {n_clean} clean")
    
    # Save splits
    def save_jsonl(samples, path):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
    
    train_path = DATA_DIR / "observer_train.jsonl"
    val_path = DATA_DIR / "observer_val.jsonl"
    test_path = DATA_DIR / "observer_test.jsonl"
    
    save_jsonl(train_samples, train_path)
    save_jsonl(val_samples, val_path)
    save_jsonl(test_samples, test_path)
    
    print(f"\n[SAVE] {train_path}")
    print(f"       {val_path}")
    print(f"       {test_path}")
    
    # Save manifest
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "sources": {
            "error": str(M47_ERROR_PREDS),
            "clean": str(M47_CLEAN_PREDS),
        },
        "total_samples": len(all_samples),
        "n_error": len(selected_error),
        "n_clean": len(selected_clean),
        "splits": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "sample_ids": {
            "train": [s["id"] for s in train_samples],
            "val": [s["id"] for s in val_samples],
            "test": [s["id"] for s in test_samples],
        },
    }
    
    manifest_path = DATA_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"       {manifest_path}")
    
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

