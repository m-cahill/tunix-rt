#!/usr/bin/env python3
"""Create evaluation delta report comparing pre/post training.

This script compares evaluation outputs from before and after training,
computing metrics and generating a markdown report.

Usage:
    python training/eval_report.py \\
        --before artifacts/training_runs/my_run/eval_before.jsonl \\
        --after artifacts/training_runs/my_run/eval_after.jsonl \\
        --output artifacts/training_runs/my_run/delta_report.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def load_eval_traces(path: Path) -> list[dict]:
    """Load evaluation traces from JSONL.

    Args:
        path: Path to traces JSONL

    Returns:
        List of trace dicts
    """
    traces = []

    with open(path, "r") as f:
        for i, line in enumerate(f):
            try:
                trace = json.loads(line.strip())
                traces.append(trace)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Invalid JSON on line {i+1}: {e}")
                continue

    return traces


def compute_trace_score(trace: dict) -> float:
    """Compute a simple quality score for a trace.

    This is a placeholder for more sophisticated scoring.
    Real implementation might use:
    - Answer correctness
    - Reasoning quality
    - Step count
    - Coherence metrics

    Args:
        trace: Trace dict

    Returns:
        Score (0-100)
    """
    # Placeholder scoring based on response length
    # Real scoring would check against ground truth
    answer_length = len(trace.get("final_answer", ""))

    # Arbitrary scoring: longer answers get higher scores (up to a point)
    score = min(answer_length / 2, 100.0)

    return score


def create_delta_report(
    before_traces: list[dict],
    after_traces: list[dict],
    output_path: Path,
) -> None:
    """Create markdown delta report.

    Args:
        before_traces: Pre-training traces
        after_traces: Post-training traces
        output_path: Output markdown path
    """
    # Compute scores
    before_scores = [compute_trace_score(t) for t in before_traces]
    after_scores = [compute_trace_score(t) for t in after_traces]

    avg_before = sum(before_scores) / len(before_scores) if before_scores else 0
    avg_after = sum(after_scores) / len(after_scores) if after_scores else 0
    delta = avg_after - avg_before

    # Build report
    report_lines = []

    report_lines.append("# Evaluation Delta Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.utcnow().isoformat()}Z")
    report_lines.append(f"**Eval Examples:** {len(before_traces)}")
    report_lines.append("")

    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(f"- **Before Training Avg Score:** {avg_before:.2f}")
    report_lines.append(f"- **After Training Avg Score:** {avg_after:.2f}")
    report_lines.append(f"- **Delta:** {delta:+.2f}")
    report_lines.append("")

    if delta > 0:
        report_lines.append(f"✅ **Result:** Training improved scores by {delta:.2f} points")
    elif delta < 0:
        report_lines.append(f"⚠️  **Result:** Training decreased scores by {abs(delta):.2f} points")
    else:
        report_lines.append("ℹ️  **Result:** No change in scores")

    report_lines.append("")
    report_lines.append("## Individual Examples")
    report_lines.append("")

    # Show some example comparisons
    num_examples_to_show = min(5, len(before_traces))

    for i in range(num_examples_to_show):
        before = before_traces[i]
        after = after_traces[i]
        before_score = before_scores[i]
        after_score = after_scores[i]
        example_delta = after_score - before_score

        eval_id = before.get("meta", {}).get("eval_id", f"example-{i+1}")

        report_lines.append(f"### Example {i+1}: `{eval_id}`")
        report_lines.append("")
        report_lines.append(f"**Prompt:** {before['prompt']}")
        report_lines.append("")
        report_lines.append(f"**Before Score:** {before_score:.2f}")
        report_lines.append(f"**After Score:** {after_score:.2f}")
        report_lines.append(f"**Delta:** {example_delta:+.2f}")
        report_lines.append("")

    if len(before_traces) > num_examples_to_show:
        report_lines.append(f"*... and {len(before_traces) - num_examples_to_show} more examples*")
        report_lines.append("")

    report_lines.append("## Methodology")
    report_lines.append("")
    report_lines.append("**Scoring:** Placeholder scoring based on response length")
    report_lines.append("")
    report_lines.append("**Note:** This is a demonstration report. Real evaluation would:")
    report_lines.append("- Check answer correctness against ground truth")
    report_lines.append("- Evaluate reasoning quality")
    report_lines.append("- Measure coherence and fluency")
    report_lines.append("- Use multiple automated metrics")
    report_lines.append("")

    report_lines.append("## Next Steps")
    report_lines.append("")
    report_lines.append("1. Review individual examples for quality improvements")
    report_lines.append("2. Import eval traces to database for visualization")
    report_lines.append("3. Run additional eval sets to validate results")
    report_lines.append("4. Iterate on training configuration based on findings")
    report_lines.append("")

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create evaluation delta report"
    )
    parser.add_argument(
        "--before",
        type=Path,
        required=True,
        help="Path to pre-training eval traces JSONL",
    )
    parser.add_argument(
        "--after",
        type=Path,
        required=True,
        help="Path to post-training eval traces JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for markdown report",
    )

    args = parser.parse_args()

    print("📊 Tunix RT - Evaluation Delta Report")
    print("=" * 70)

    # Load traces
    if not args.before.exists():
        print(f"❌ Before traces not found: {args.before}")
        sys.exit(1)

    if not args.after.exists():
        print(f"❌ After traces not found: {args.after}")
        sys.exit(1)

    print(f"Loading before traces: {args.before}")
    before_traces = load_eval_traces(args.before)
    print(f"✅ Loaded {len(before_traces)} before traces")

    print(f"Loading after traces: {args.after}")
    after_traces = load_eval_traces(args.after)
    print(f"✅ Loaded {len(after_traces)} after traces")

    if len(before_traces) != len(after_traces):
        print(f"⚠️  Warning: Different number of traces ({len(before_traces)} vs {len(after_traces)})")
        print("   Using minimum count for comparison")

    min_count = min(len(before_traces), len(after_traces))
    before_traces = before_traces[:min_count]
    after_traces = after_traces[:min_count]

    # Create report
    print("\n📝 Generating report...")
    create_delta_report(before_traces, after_traces, args.output)

    print(f"\n✅ Report saved to: {args.output}")
    print("\nView the report:")
    print(f"   cat {args.output}")
    print("   # or open in your editor/viewer")
    print("=" * 70)


if __name__ == "__main__":
    main()

