#!/usr/bin/env python3
"""Quick check of data distribution."""

import json
from pathlib import Path

# Load M47 predictions
error_path = Path("research/m47_error_correction_fidelity/eval/m47_error_aware_holdout_error_predictions.jsonl")
clean_path = Path("research/m47_error_correction_fidelity/eval/m47_error_aware_holdout_clean_predictions.jsonl")

error_preds = [json.loads(l) for l in error_path.read_text().split("\n") if l.strip()][:3]
clean_preds = [json.loads(l) for l in clean_path.read_text().split("\n") if l.strip()][:3]

print("=" * 60)
print("M47 Error Holdout (traces with injected errors)")
print("=" * 60)
for p in error_preds:
    print(f"Prompt: {p['prompt'][:50]}...")
    print(f"Expected: {p['expected_answer']}")
    print(f"Output: {p['predicted_answer'][:80]}...")
    print()

print("=" * 60)
print("M47 Clean Holdout (traces without errors)")
print("=" * 60)
for p in clean_preds:
    print(f"Prompt: {p['prompt'][:50]}...")
    print(f"Expected: {p['expected_answer']}")
    print(f"Output: {p['predicted_answer'][:80]}...")
    print()

