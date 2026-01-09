#!/usr/bin/env python3
"""M46 Trace Transformation Script — Structured Self-Correction.

This script transforms Stage-C traces by appending VERIFY: and CORRECT: blocks
to create a self-correction augmented dataset.

Key Design Decisions (from M46_answers.md):
- Simplified format: VERIFY/CORRECT appended (not full restructure)
- Zero injected errors (CORRECT: always "No correction needed")
- Template-based verification (short and boring)

Output:
- data/stage_c_control.jsonl (unchanged copy of stage_c)
- data/stage_c_self_correct.jsonl (with VERIFY/CORRECT appended)
- data/transformation_stats.json (statistics)

Usage:
    cd research/m46_structured_self_correction
    python scripts/transform_traces.py

Author: M46 Structured Self-Correction Milestone
Date: 2026-01-08
"""

import codecs
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

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

# Source data from M45
SOURCE_FILE = PROJECT_DIR.parent / "m45_curriculum_reasoning" / "data" / "stage_c.jsonl"

# Output files
CONTROL_FILE = DATA_DIR / "stage_c_control.jsonl"
SELF_CORRECT_FILE = DATA_DIR / "stage_c_self_correct.jsonl"
STATS_FILE = DATA_DIR / "transformation_stats.json"


# ============================================================
# Verification Templates (Short & Boring per M46_answers.md)
# ============================================================

# Template verification patterns based on problem type
# Key insight: These are MECHANICAL checks, not new reasoning

VERIFY_TEMPLATES = {
    # template_0: Distance = speed × time (train problems)
    0: lambda trace: _verify_distance(trace),
    
    # template_2: Cost per item = total / items
    2: lambda trace: _verify_division(trace),
    
    # template_3: Addition with decomposition
    3: lambda trace: _verify_addition(trace),
    
    # template_4: Subtraction with decomposition
    4: lambda trace: _verify_subtraction(trace),
    
    # template_5: Multiplication with partials
    5: lambda trace: _verify_multiplication(trace),
}


def _verify_distance(trace: dict) -> str:
    """Verification for distance = speed × time problems."""
    answer = trace.get("final_answer", "")
    # Extract numbers from answer for inverse check
    return f"Check by inverse: divide result by time to verify speed"


def _verify_division(trace: dict) -> str:
    """Verification for cost per item problems."""
    return "Check by inverse: multiply result by count to verify total"


def _verify_addition(trace: dict) -> str:
    """Verification for addition problems."""
    return "Check by inverse: subtract one addend from sum"


def _verify_subtraction(trace: dict) -> str:
    """Verification for subtraction problems."""
    return "Check by inverse: add result to subtrahend"


def _verify_multiplication(trace: dict) -> str:
    """Verification for multiplication problems."""
    return "Check by inverse: divide product by one factor"


def _verify_edge_case(trace: dict) -> str:
    """Verification for edge case problems (generic)."""
    case_type = trace.get("meta", {}).get("case_type", "unknown")
    return f"Check: result is well-formed for {case_type} input"


def get_verification_block(trace: dict) -> str:
    """Generate VERIFY block for a trace based on its template.
    
    Args:
        trace: The reasoning trace dict
        
    Returns:
        Verification text (short and mechanical)
    """
    meta = trace.get("meta", {})
    category = meta.get("category", "unknown")
    template_id = meta.get("template_id")
    
    # Edge cases get generic verification
    if category == "edge_case":
        return _verify_edge_case(trace)
    
    # Reasoning traces use template-specific verification
    if template_id in VERIFY_TEMPLATES:
        return VERIFY_TEMPLATES[template_id](trace)
    
    # Fallback for unknown templates
    return "Check: verify answer matches expected format"


# ============================================================
# Trace Transformation
# ============================================================

def transform_trace(trace: dict) -> dict:
    """Transform a single trace by appending VERIFY/CORRECT blocks.
    
    The transformation appends two new steps to the trace:
    1. VERIFY: <template-based verification>
    2. CORRECT: No correction needed
    
    Args:
        trace: Original trace dict
        
    Returns:
        Transformed trace dict (new object, original unchanged)
    """
    # Deep copy to avoid modifying original
    transformed = json.loads(json.dumps(trace))
    
    # Get existing steps
    steps = transformed.get("steps", [])
    next_idx = len(steps)
    
    # Generate verification block
    verify_text = get_verification_block(trace)
    
    # Append VERIFY step
    steps.append({
        "i": next_idx,
        "type": "verify",
        "content": f"VERIFY: {verify_text}"
    })
    
    # Append CORRECT step (always "No correction needed" per M46 decision)
    steps.append({
        "i": next_idx + 1,
        "type": "correct",
        "content": "CORRECT: No correction needed"
    })
    
    transformed["steps"] = steps
    
    # Add transformation metadata
    if "meta" not in transformed:
        transformed["meta"] = {}
    transformed["meta"]["m46_transformed"] = True
    transformed["meta"]["m46_verify_template"] = trace.get("meta", {}).get("template_id", "edge_case")
    
    return transformed


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("M46 Trace Transformation — Structured Self-Correction")
    print("=" * 60)
    
    # Verify source exists
    if not SOURCE_FILE.exists():
        print(f"[ERROR] Source file not found: {SOURCE_FILE}")
        return 1
    
    print(f"\n[SOURCE] {SOURCE_FILE}")
    
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load source traces
    print("[LOAD] Loading source traces...")
    traces = []
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    print(f"       Loaded {len(traces)} traces")
    
    # Create control dataset (unchanged copy)
    print(f"\n[CONTROL] Creating unchanged control dataset...")
    shutil.copy(SOURCE_FILE, CONTROL_FILE)
    print(f"          Copied to: {CONTROL_FILE}")
    
    # Transform traces for self-correction dataset
    print(f"\n[TRANSFORM] Creating self-correction augmented dataset...")
    transformed_traces = []
    template_counts = {}
    
    for trace in traces:
        transformed = transform_trace(trace)
        transformed_traces.append(transformed)
        
        # Track template usage for stats
        meta = trace.get("meta", {})
        category = meta.get("category", "unknown")
        template_id = meta.get("template_id", "edge_case")
        key = f"{category}:template_{template_id}" if category == "reasoning" else f"{category}:{meta.get('case_type', 'unknown')}"
        template_counts[key] = template_counts.get(key, 0) + 1
    
    # Write self-correction dataset
    with open(SELF_CORRECT_FILE, "w", encoding="utf-8") as f:
        for trace in transformed_traces:
            f.write(json.dumps(trace) + "\n")
    print(f"          Wrote {len(transformed_traces)} transformed traces to: {SELF_CORRECT_FILE}")
    
    # Compute hashes
    source_hash = compute_sha256(SOURCE_FILE)
    control_hash = compute_sha256(CONTROL_FILE)
    self_correct_hash = compute_sha256(SELF_CORRECT_FILE)
    
    # Verify control is identical to source
    assert source_hash == control_hash, "Control file hash mismatch!"
    print(f"\n[VERIFY] Control hash matches source: {control_hash[:16]}...")
    
    # Compute statistics
    stats = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(SOURCE_FILE),
            "sha256": source_hash,
            "count": len(traces)
        },
        "control": {
            "path": str(CONTROL_FILE),
            "sha256": control_hash,
            "count": len(traces),
            "note": "Unchanged copy of stage_c.jsonl"
        },
        "self_correct": {
            "path": str(SELF_CORRECT_FILE),
            "sha256": self_correct_hash,
            "count": len(transformed_traces),
            "added_steps_per_trace": 2,
            "note": "VERIFY + CORRECT blocks appended"
        },
        "transformation": {
            "injected_errors": 0,
            "injected_error_rate": 0.0,
            "verify_present": len(transformed_traces),
            "correct_present": len(transformed_traces),
            "correct_value": "No correction needed (all traces)"
        },
        "template_distribution": template_counts
    }
    
    # Write statistics
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[STATS] Saved transformation statistics to: {STATS_FILE}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Transformation Complete!")
    print("=" * 60)
    print(f"  Control:        {len(traces)} traces (unchanged)")
    print(f"  Self-Correct:   {len(transformed_traces)} traces (+2 steps each)")
    print(f"  Injected Errors: 0 (per M46 decision)")
    print(f"\n  Template Distribution:")
    for key, count in sorted(template_counts.items()):
        print(f"    {key}: {count}")
    
    return 0


if __name__ == "__main__":
    exit(main())

