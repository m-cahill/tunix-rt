#!/usr/bin/env python3
"""M45 Curriculum Dataset Partitioning Script.

This script partitions dev-reasoning-v2 into three curriculum stages
based on category (primary) and trace length (secondary within reasoning).

Partitioning Rules (LOCKED per M45_answers.md):
- Stage A (Low):    synthetic + golden_style (1-2 step traces)
- Stage B (Medium): reasoning where len(steps) == 3
- Stage C (Full):   reasoning where len(steps) >= 4 + all edge_case

Output:
- data/stage_a.jsonl
- data/stage_b.jsonl  
- data/stage_c.jsonl
- data/split_stats.json
- data/trace_length_histogram.txt

Usage:
    cd research/m45_curriculum_reasoning
    python split_dataset.py

Author: M45 Curriculum Reasoning Milestone
Date: 2026-01-08
"""

import json
import hashlib
import sys
import codecs
from collections import Counter
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
# Configuration (LOCKED — do not modify)
# ============================================================

# Source dataset - resolve relative to script location
SCRIPT_DIR = Path(__file__).parent.resolve()
SOURCE_DATASET = SCRIPT_DIR / ".." / ".." / "backend" / "datasets" / "dev-reasoning-v2" / "dataset.jsonl"

# Output paths - relative to script location
OUTPUT_DIR = SCRIPT_DIR / "data"
STAGE_A_FILE = OUTPUT_DIR / "stage_a.jsonl"
STAGE_B_FILE = OUTPUT_DIR / "stage_b.jsonl"
STAGE_C_FILE = OUTPUT_DIR / "stage_c.jsonl"
STATS_FILE = OUTPUT_DIR / "split_stats.json"
HISTOGRAM_FILE = OUTPUT_DIR / "trace_length_histogram.txt"

# ============================================================
# Partitioning Logic
# ============================================================


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file for provenance tracking."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def partition_trace(trace: dict) -> str:
    """Determine which stage a trace belongs to.
    
    Partitioning Rules (from M45 locked decisions):
    - Stage A: synthetic + golden_style categories
    - Stage B: reasoning category with exactly 3 steps
    - Stage C: reasoning category with 4+ steps + all edge_case
    
    Args:
        trace: A trace dictionary with 'meta.category' and 'steps'
        
    Returns:
        One of: 'A', 'B', 'C'
    """
    meta = trace.get("meta", {})
    category = meta.get("category", "unknown")
    steps = trace.get("steps", [])
    step_count = len(steps)
    
    # Stage A: synthetic + golden_style (low complexity)
    if category in ("synthetic", "golden_style"):
        return "A"
    
    # Stage C: edge_case (always full complexity)
    if category == "edge_case":
        return "C"
    
    # Reasoning category: split by trace length
    if category == "reasoning":
        if step_count <= 3:
            return "B"  # Medium complexity
        else:
            return "C"  # Full complexity (4+ steps)
    
    # Unknown category — default to Stage C (conservative)
    return "C"


def generate_histogram(traces: list[dict]) -> str:
    """Generate ASCII histogram of trace lengths.
    
    Args:
        traces: List of trace dictionaries
        
    Returns:
        ASCII histogram string
    """
    lengths = [len(t.get("steps", [])) for t in traces]
    counter = Counter(lengths)
    
    lines = ["Trace Length Distribution", "=" * 40]
    
    max_count = max(counter.values()) if counter else 1
    bar_width = 40
    
    for length in sorted(counter.keys()):
        count = counter[length]
        bar_len = int((count / max_count) * bar_width)
        bar = "█" * bar_len
        lines.append(f"{length:2d} steps: {bar} ({count})")
    
    lines.append("")
    lines.append(f"Total traces: {len(traces)}")
    lines.append(f"Min steps: {min(lengths) if lengths else 0}")
    lines.append(f"Max steps: {max(lengths) if lengths else 0}")
    lines.append(f"Avg steps: {sum(lengths)/len(lengths):.2f}" if lengths else "Avg: N/A")
    
    return "\n".join(lines)


def main():
    """Main execution: load, partition, save, and report."""
    print("=" * 60)
    print("M45 Curriculum Dataset Partitioning")
    print("=" * 60)
    
    # Verify source dataset exists
    if not SOURCE_DATASET.exists():
        print(f"❌ Source dataset not found: {SOURCE_DATASET}")
        print("   Please run from research/m45_curriculum_reasoning/")
        return 1
    
    # Load source dataset
    print(f"\n[LOAD] Loading: {SOURCE_DATASET}")
    traces = []
    with open(SOURCE_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    
    print(f"   Loaded {len(traces)} traces")
    source_hash = compute_file_hash(SOURCE_DATASET)
    print(f"   Source hash: {source_hash[:16]}...")
    
    # Partition traces
    print("\n[SPLIT] Partitioning traces...")
    stage_a = []
    stage_b = []
    stage_c = []
    
    category_counts = Counter()
    stage_category_breakdown = {"A": Counter(), "B": Counter(), "C": Counter()}
    
    for trace in traces:
        stage = partition_trace(trace)
        category = trace.get("meta", {}).get("category", "unknown")
        category_counts[category] += 1
        stage_category_breakdown[stage][category] += 1
        
        if stage == "A":
            stage_a.append(trace)
        elif stage == "B":
            stage_b.append(trace)
        else:
            stage_c.append(trace)
    
    print(f"\n[STATS] Partition Results:")
    print(f"   Stage A (Low):    {len(stage_a):4d} traces")
    print(f"   Stage B (Medium): {len(stage_b):4d} traces")
    print(f"   Stage C (Full):   {len(stage_c):4d} traces")
    print(f"   ─────────────────────────")
    print(f"   Total:            {len(traces):4d} traces")
    
    # Validate no traces lost
    assert len(stage_a) + len(stage_b) + len(stage_c) == len(traces), "Trace count mismatch!"
    
    # Print category breakdown per stage
    print("\n[DETAIL] Category Breakdown by Stage:")
    for stage_name, breakdown in stage_category_breakdown.items():
        print(f"   Stage {stage_name}: {dict(breakdown)}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write stage files
    print(f"\n[SAVE] Writing stage files...")
    
    def write_jsonl(filepath: Path, data: list[dict]) -> str:
        """Write JSONL and return file hash."""
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return compute_file_hash(filepath)
    
    hash_a = write_jsonl(STAGE_A_FILE, stage_a)
    hash_b = write_jsonl(STAGE_B_FILE, stage_b)
    hash_c = write_jsonl(STAGE_C_FILE, stage_c)
    
    print(f"   ✅ {STAGE_A_FILE}: {len(stage_a)} traces (hash: {hash_a[:16]}...)")
    print(f"   ✅ {STAGE_B_FILE}: {len(stage_b)} traces (hash: {hash_b[:16]}...)")
    print(f"   ✅ {STAGE_C_FILE}: {len(stage_c)} traces (hash: {hash_c[:16]}...)")
    
    # Generate histogram
    print(f"\n[HISTOGRAM] Generating trace length histogram...")
    histogram = generate_histogram(traces)
    with open(HISTOGRAM_FILE, "w", encoding="utf-8") as f:
        f.write(histogram)
    print(f"   ✅ {HISTOGRAM_FILE}")
    
    # Compute per-stage step statistics
    def step_stats(data: list[dict]) -> dict:
        lengths = [len(t.get("steps", [])) for t in data]
        if not lengths:
            return {"min": 0, "max": 0, "avg": 0, "distribution": {}}
        return {
            "min": min(lengths),
            "max": max(lengths),
            "avg": round(sum(lengths) / len(lengths), 2),
            "distribution": dict(Counter(lengths))
        }
    
    # Write stats file
    stats = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(SOURCE_DATASET),
            "sha256": source_hash,
            "trace_count": len(traces)
        },
        "partitioning": {
            "strategy": "category-first with trace-length refinement",
            "rules": {
                "stage_a": "synthetic + golden_style",
                "stage_b": "reasoning where len(steps) == 3",
                "stage_c": "reasoning where len(steps) >= 4 + all edge_case"
            }
        },
        "stages": {
            "A": {
                "file": str(STAGE_A_FILE),
                "sha256": hash_a,
                "count": len(stage_a),
                "categories": dict(stage_category_breakdown["A"]),
                "step_stats": step_stats(stage_a)
            },
            "B": {
                "file": str(STAGE_B_FILE),
                "sha256": hash_b,
                "count": len(stage_b),
                "categories": dict(stage_category_breakdown["B"]),
                "step_stats": step_stats(stage_b)
            },
            "C": {
                "file": str(STAGE_C_FILE),
                "sha256": hash_c,
                "count": len(stage_c),
                "categories": dict(stage_category_breakdown["C"]),
                "step_stats": step_stats(stage_c)
            }
        },
        "total_category_counts": dict(category_counts)
    }
    
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"   ✅ {STATS_FILE}")
    
    # Print histogram preview
    print(f"\n{histogram}")
    
    print("\n" + "=" * 60)
    print("✅ M45 Dataset Partitioning Complete")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

