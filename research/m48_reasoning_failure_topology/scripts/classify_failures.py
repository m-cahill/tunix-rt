#!/usr/bin/env python3
"""M48 Failure Classification Script — Automated Taxonomy Labeling.

This script classifies verification failures in M46/M47 predictions using
structural + regex heuristics. No ML models are used.

Failure Classes:
1. ritual_verification — Templated VERIFY with no computational reference
2. computation_reset — Re-solves from scratch instead of inspecting
3. local_error_blindness — Detects structure, misses specific errors
4. detection_without_localization — Vague error acknowledgment
5. correction_hallucination — "Fixes" a non-error
6. verification_collapse — VERIFY degenerates to restatement

Special Labels:
- no_verification — No VERIFY/CORRECT blocks present
- successful_detection — Correctly identifies an error

Usage:
    cd research/m48_reasoning_failure_topology
    python scripts/classify_failures.py

Author: M48 Reasoning Failure Topology Milestone
Date: 2026-01-09
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
INPUTS_DIR = PROJECT_DIR / "inputs"
METRICS_DIR = PROJECT_DIR / "metrics"
TAXONOMY_DIR = PROJECT_DIR / "taxonomy"

# Input files (per M48_answers: M46 + M47 error_aware only, exclude M47 clean)
INPUT_FILES = {
    "m46_self_correct_holdout_error": INPUTS_DIR / "m46_self_correct_holdout_error_predictions.jsonl",
    "m46_self_correct_holdout_clean": INPUTS_DIR / "m46_self_correct_holdout_clean_predictions.jsonl",
    "m47_error_aware_holdout_error": INPUTS_DIR / "m47_error_aware_holdout_error_predictions.jsonl",
    "m47_error_aware_holdout_clean": INPUTS_DIR / "m47_error_aware_holdout_clean_predictions.jsonl",
}

# Templates known from M45/M46 training (ritual detection)
VERIFY_TEMPLATES = [
    r"check by inverse",
    r"divide distance by time",
    r"add result to subtrahend",
    r"multiply result by count",
    r"divide product by one factor",
    r"subtract one addend from sum",
    r"verify answer matches expected format",
]

# Error detection language (for detection_without_localization)
VAGUE_ERROR_LANGUAGE = [
    r"something seems? off",
    r"double.?check",
    r"let me verify",
    r"might need adjustment",
    r"looks? (?:a bit )?(?:wrong|off|incorrect)",
    r"not sure if",
]

# Correction language (for successful detection / hallucination)
SPECIFIC_CORRECTION_PATTERNS = [
    r"step \d+ is wrong",
    r"should be \d+",
    r"not \d+",
    r"error in step",
    r"mistake in",
    r"recalculate",
    r"recompute",
]


# ============================================================
# Classification Functions
# ============================================================

def extract_blocks(text: str) -> dict:
    """Extract VERIFY and CORRECT blocks from text."""
    result = {
        "has_verify": False,
        "verify_content": "",
        "has_correct": False,
        "correct_content": "",
    }
    
    # Extract VERIFY block
    verify_match = re.search(r"VERIFY:\s*(.+?)(?=CORRECT:|$)", text, re.IGNORECASE | re.DOTALL)
    if verify_match:
        result["has_verify"] = True
        result["verify_content"] = verify_match.group(1).strip()
    
    # Extract CORRECT block
    correct_match = re.search(r"CORRECT:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if correct_match:
        result["has_correct"] = True
        result["correct_content"] = correct_match.group(1).strip()
    
    return result


def extract_numbers(text: str) -> list:
    """Extract all numbers from text."""
    return re.findall(r'\b\d+(?:\.\d+)?\b', text)


def is_template_only(verify_content: str) -> bool:
    """Check if VERIFY content is purely templated (no specific numbers/steps)."""
    content_lower = verify_content.lower()
    
    # Check if it matches known templates
    for template in VERIFY_TEMPLATES:
        if re.search(template, content_lower):
            # Check if there are specific numbers referenced
            numbers = extract_numbers(verify_content)
            # If template match but no numbers, it's ritual
            if len(numbers) == 0:
                return True
    
    return False


def has_step_reference(text: str) -> bool:
    """Check if text references specific steps."""
    return bool(re.search(r"step\s*\d+", text, re.IGNORECASE))


def has_vague_error_language(text: str) -> bool:
    """Check for vague error acknowledgment."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in VAGUE_ERROR_LANGUAGE)


def has_specific_correction(text: str) -> bool:
    """Check for specific correction language."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in SPECIFIC_CORRECTION_PATTERNS)


def is_no_correction_needed(correct_content: str) -> bool:
    """Check if CORRECT says no correction needed."""
    content_lower = correct_content.lower()
    return "no correction needed" in content_lower or "no correction" in content_lower


def is_answer_restatement(verify_content: str, predicted_answer: str) -> bool:
    """Check if VERIFY is just restating the answer."""
    if not predicted_answer:
        return False
    
    # Extract numbers from both
    verify_numbers = set(extract_numbers(verify_content))
    answer_numbers = set(extract_numbers(predicted_answer))
    
    # If VERIFY only contains the answer number and no checking language
    if verify_numbers and verify_numbers == answer_numbers:
        checking_keywords = ["check", "verify", "inverse", "confirm", "validate"]
        if not any(kw in verify_content.lower() for kw in checking_keywords):
            return True
    
    return False


def classify_trace(prediction: dict, has_known_error: bool = False) -> dict:
    """Classify a single trace into failure categories.
    
    Args:
        prediction: Dict with predicted_answer, expected_answer, etc.
        has_known_error: Whether this trace had an injected error
        
    Returns:
        Dict with primary_class, secondary_class, confidence, reasoning
    """
    text = prediction.get("predicted_answer", "")
    expected = prediction.get("expected_answer", "")
    
    # Extract blocks
    blocks = extract_blocks(text)
    
    # Special case: no verification structure
    if not blocks["has_verify"] and not blocks["has_correct"]:
        return {
            "primary_class": "no_verification",
            "secondary_class": None,
            "confidence": "high",
            "reasoning": "No VERIFY or CORRECT blocks present in output",
        }
    
    verify = blocks["verify_content"]
    correct = blocks["correct_content"]
    
    # Check for successful detection (if there was a known error)
    if has_known_error and has_specific_correction(correct):
        # Check if correction mentions the actual error
        return {
            "primary_class": "successful_detection",
            "secondary_class": None,
            "confidence": "medium",
            "reasoning": f"CORRECT block contains specific correction: '{correct[:50]}...'",
        }
    
    # Check for correction hallucination (correction on clean trace)
    if not has_known_error and has_specific_correction(correct):
        return {
            "primary_class": "correction_hallucination",
            "secondary_class": None,
            "confidence": "medium",
            "reasoning": f"CORRECT mentions fix on clean trace: '{correct[:50]}...'",
        }
    
    # Check for verification collapse (VERIFY = answer restatement)
    if is_answer_restatement(verify, text):
        return {
            "primary_class": "verification_collapse",
            "secondary_class": None,
            "confidence": "medium",
            "reasoning": "VERIFY content is semantically identical to answer",
        }
    
    # Check for vague error detection (detection without localization)
    if has_vague_error_language(verify + " " + correct):
        return {
            "primary_class": "detection_without_localization",
            "secondary_class": "ritual_verification" if is_template_only(verify) else None,
            "confidence": "medium",
            "reasoning": "Vague error language present but no specific step identified",
        }
    
    # Check for ritual verification (template only, no numbers)
    if is_template_only(verify):
        secondary = None
        
        # Check if also local_error_blindness (had error but missed it)
        if has_known_error and is_no_correction_needed(correct):
            secondary = "local_error_blindness"
        
        return {
            "primary_class": "ritual_verification",
            "secondary_class": secondary,
            "confidence": "high",
            "reasoning": f"VERIFY matches template without computational specifics: '{verify[:50]}...'",
        }
    
    # Check for local error blindness (missed error despite verification)
    if has_known_error and is_no_correction_needed(correct):
        return {
            "primary_class": "local_error_blindness",
            "secondary_class": None,
            "confidence": "medium",
            "reasoning": "Error present but CORRECT says no correction needed",
        }
    
    # Default: computation reset (solved from scratch)
    # This is inferred when there's verification structure but no comparison
    if not has_step_reference(verify + " " + correct):
        return {
            "primary_class": "computation_reset",
            "secondary_class": None,
            "confidence": "low",
            "reasoning": "No step references found; likely re-solved from scratch",
        }
    
    # Fallback
    return {
        "primary_class": "ritual_verification",
        "secondary_class": None,
        "confidence": "low",
        "reasoning": "Default classification (verification present but unclear pattern)",
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("M48 Failure Classification — Reasoning Failure Topology")
    print("=" * 60)
    
    # Ensure output directories exist
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_labels = {}
    all_counts = {}
    
    for source_name, source_path in INPUT_FILES.items():
        if not source_path.exists():
            print(f"[SKIP] {source_name}: file not found")
            continue
        
        print(f"\n[CLASSIFY] {source_name}")
        
        # Load predictions
        predictions = []
        with open(source_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
        
        print(f"           Loaded {len(predictions)} predictions")
        
        # Determine if this is an error-injected set
        is_error_set = "error" in source_name and "clean" not in source_name
        
        # Classify each trace
        labels = []
        class_counts = {}
        
        for i, pred in enumerate(predictions):
            # Check if this specific trace had an error injected
            has_error = pred.get("has_known_error", is_error_set)
            
            classification = classify_trace(pred, has_error)
            
            label_entry = {
                "id": pred.get("id", i),
                "source": source_name,
                "has_known_error": has_error,
                **classification,
            }
            labels.append(label_entry)
            
            # Count classes
            primary = classification["primary_class"]
            class_counts[primary] = class_counts.get(primary, 0) + 1
            
            if classification["secondary_class"]:
                secondary = f"secondary:{classification['secondary_class']}"
                class_counts[secondary] = class_counts.get(secondary, 0) + 1
        
        all_labels[source_name] = labels
        all_counts[source_name] = class_counts
        
        # Print summary
        print(f"           Classification Summary:")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            pct = count / len(predictions) * 100
            print(f"             {cls}: {count} ({pct:.1f}%)")
    
    # Save all labels
    labels_path = METRICS_DIR / "failure_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2)
    print(f"\n[SAVE] Labels: {labels_path}")
    
    # Save counts summary
    counts_path = METRICS_DIR / "failure_counts.json"
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "counts_by_source": all_counts,
    }
    with open(counts_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Counts: {counts_path}")
    
    # Generate counts table (Markdown)
    table_path = TAXONOMY_DIR / "failure_counts_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# Failure Classification Counts\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        
        # Get all unique classes
        all_classes = set()
        for counts in all_counts.values():
            all_classes.update(counts.keys())
        all_classes = sorted([c for c in all_classes if not c.startswith("secondary:")])
        
        # Header
        f.write("| Source | " + " | ".join(all_classes) + " | Total |\n")
        f.write("|--------|" + "|".join(["---:" for _ in all_classes]) + "|---:|\n")
        
        # Rows
        for source_name, counts in all_counts.items():
            total = sum(v for k, v in counts.items() if not k.startswith("secondary:"))
            row = [source_name]
            for cls in all_classes:
                row.append(str(counts.get(cls, 0)))
            row.append(str(total))
            f.write("| " + " | ".join(row) + " |\n")
    
    print(f"[SAVE] Table: {table_path}")
    
    # Print cross-model comparison
    print("\n" + "=" * 60)
    print("Cross-Model Comparison (M46 vs M47 Error-Aware)")
    print("=" * 60)
    
    m46_error = all_counts.get("m46_self_correct_holdout_error", {})
    m47_error = all_counts.get("m47_error_aware_holdout_error", {})
    
    print(f"\n{'Class':<35} {'M46':>8} {'M47':>8} {'Delta':>8}")
    print("-" * 60)
    
    for cls in all_classes:
        m46_val = m46_error.get(cls, 0)
        m47_val = m47_error.get(cls, 0)
        delta = m47_val - m46_val
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        print(f"{cls:<35} {m46_val:>8} {m47_val:>8} {delta_str:>8}")
    
    print("\n" + "=" * 60)
    print("M48 Classification Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

