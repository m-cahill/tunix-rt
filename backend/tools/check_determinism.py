#!/usr/bin/env python
"""Determinism check script (M35).

Verifies that the evaluation pipeline produces identical results when run
twice with the same inputs. This is critical for reliable regression testing
and leaderboard comparisons.

Usage:
    python backend/tools/check_determinism.py [--eval-set PATH] [--verbose]

Checks:
    1. compute_primary_score() produces identical results
    2. compute_scorecard() produces identical results
    3. Sorted item ordering is deterministic

Exit codes:
    0: All determinism checks passed
    1: Determinism check failed
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tunix_rt_backend.scoring import compute_primary_score, compute_scorecard

# ============================================================
# Test Data Generators
# ============================================================


def generate_test_evaluation_rows(count: int = 50, seed: int = 42) -> list[dict[str, Any]]:
    """Generate deterministic test evaluation rows.

    Args:
        count: Number of rows to generate
        seed: Random seed for reproducibility

    Returns:
        List of evaluation row dictionaries
    """
    import random

    rng = random.Random(seed)
    rows = []

    for i in range(count):
        # Deterministic score based on index and seed
        score = (i * 7 + seed) % 100 / 100.0  # 0.0 to 0.99
        section = ["core", "trace_sensitive", "edge_case"][i % 3]
        category = ["arithmetic", "geometry", "word_problem"][i % 3]
        difficulty = ["easy", "medium", "hard"][i % 3]

        rows.append(
            {
                "item_id": f"eval-{i:04d}",
                "metrics": {"answer_correctness": score},
                "section": section,
                "category": category,
                "difficulty": difficulty,
            }
        )

    # Shuffle to test ordering independence
    shuffled = rows.copy()
    rng.shuffle(shuffled)

    return shuffled


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    """Load eval set from JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of eval item dictionaries
    """
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ============================================================
# Determinism Checks
# ============================================================


def check_primary_score_determinism(rows: list[dict[str, Any]], verbose: bool = False) -> bool:
    """Check that compute_primary_score is deterministic.

    Args:
        rows: Evaluation rows to test
        verbose: Print detailed output

    Returns:
        True if deterministic, False otherwise
    """
    result1 = compute_primary_score(rows)
    result2 = compute_primary_score(rows)

    if result1 != result2:
        print("[FAIL] compute_primary_score not deterministic:")
        print(f"       Run 1: {result1}")
        print(f"       Run 2: {result2}")
        return False

    if verbose:
        print(f"[PASS] compute_primary_score: {result1}")

    return True


def check_scorecard_determinism(rows: list[dict[str, Any]], verbose: bool = False) -> bool:
    """Check that compute_scorecard is deterministic.

    Args:
        rows: Evaluation rows to test
        verbose: Print detailed output

    Returns:
        True if deterministic, False otherwise
    """
    card1 = compute_scorecard(rows)
    card2 = compute_scorecard(rows)

    # Compare all fields
    fields_to_check = [
        ("n_items", card1.n_items, card2.n_items),
        ("n_scored", card1.n_scored, card2.n_scored),
        ("n_skipped", card1.n_skipped, card2.n_skipped),
        ("primary_score", card1.primary_score, card2.primary_score),
        ("stddev", card1.stddev, card2.stddev),
        ("section_scores", card1.section_scores, card2.section_scores),
        ("category_scores", card1.category_scores, card2.category_scores),
        ("difficulty_scores", card1.difficulty_scores, card2.difficulty_scores),
    ]

    all_match = True
    for name, val1, val2 in fields_to_check:
        if val1 != val2:
            print(f"[FAIL] compute_scorecard.{name} not deterministic:")
            print(f"       Run 1: {val1}")
            print(f"       Run 2: {val2}")
            all_match = False

    if all_match and verbose:
        ps = card1.primary_score
        print(f"[PASS] compute_scorecard: n_items={card1.n_items}, primary_score={ps}")

    return all_match


def check_ordering_independence(rows: list[dict[str, Any]], verbose: bool = False) -> bool:
    """Check that results are independent of input ordering.

    Args:
        rows: Evaluation rows to test
        verbose: Print detailed output

    Returns:
        True if ordering-independent, False otherwise
    """
    import random

    # Create shuffled versions
    rows_a = rows.copy()
    rows_b = rows.copy()
    random.Random(123).shuffle(rows_a)
    random.Random(456).shuffle(rows_b)

    # Compute scores
    score_a = compute_primary_score(rows_a)
    score_b = compute_primary_score(rows_b)

    if score_a != score_b:
        print("[FAIL] compute_primary_score ordering-dependent:")
        print(f"       Order A: {score_a}")
        print(f"       Order B: {score_b}")
        return False

    # Compute scorecards
    card_a = compute_scorecard(rows_a)
    card_b = compute_scorecard(rows_b)

    if card_a.primary_score != card_b.primary_score:
        print("[FAIL] compute_scorecard ordering-dependent:")
        print(f"       Order A: {card_a.primary_score}")
        print(f"       Order B: {card_b.primary_score}")
        return False

    if verbose:
        print("[PASS] Ordering independence verified")

    return True


# ============================================================
# Main Entry Point
# ============================================================


def main() -> int:
    """Run determinism checks.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Check evaluation pipeline determinism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        help="Path to eval set JSONL to use for testing (optional)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of test rows to generate (default: 100)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("M35 Determinism Check")
    print("=" * 60)

    # Generate or load test data
    if args.eval_set:
        print(f"\nLoading eval set: {args.eval_set}")
        eval_items = load_eval_set(args.eval_set)
        # Convert to evaluation rows (simulate perfect answers)
        rows = [
            {
                "item_id": item["id"],
                "metrics": {"answer_correctness": 1.0},  # Simulate all correct
                "section": item.get("section"),
                "category": item.get("category"),
                "difficulty": item.get("difficulty"),
            }
            for item in eval_items
        ]
    else:
        print(f"\nGenerating {args.count} test evaluation rows...")
        rows = generate_test_evaluation_rows(args.count)

    print(f"Test data: {len(rows)} rows\n")

    # Run checks
    checks = [
        ("compute_primary_score determinism", check_primary_score_determinism),
        ("compute_scorecard determinism", check_scorecard_determinism),
        ("Ordering independence", check_ordering_independence),
    ]

    all_passed = True
    for name, check_fn in checks:
        if args.verbose:
            print(f"\nChecking: {name}")
        if not check_fn(rows, args.verbose):
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("Result: [PASS] All determinism checks passed")
        print("=" * 60)
        return 0
    else:
        print("Result: [FAIL] Some determinism checks failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
