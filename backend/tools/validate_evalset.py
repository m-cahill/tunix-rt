#!/usr/bin/env python
"""Eval set validator tool (M35).

Validates JSONL eval set files for schema compliance, prints summary statistics,
and exits non-zero on validation failure.

Usage:
    python backend/tools/validate_evalset.py training/evalsets/eval_v2.jsonl

Validation checks:
    - Required fields: id, prompt, expected_answer
    - Optional fields: section, category, difficulty
    - Section values: core | trace_sensitive | edge_case
    - Difficulty values: easy | medium | hard
    - No duplicate IDs
    - Minimum item count (configurable, default 50)

Exit codes:
    0: Valid eval set
    1: Validation failed
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ============================================================
# Schema Constants
# ============================================================

REQUIRED_FIELDS = {"id", "prompt", "expected_answer"}
OPTIONAL_FIELDS = {"section", "category", "difficulty"}
VALID_SECTIONS = {"core", "trace_sensitive", "edge_case"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
DEFAULT_MIN_ITEMS = 50


# ============================================================
# Data Classes
# ============================================================


@dataclass
class ValidationResult:
    """Result of eval set validation."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        """Add a warning (does not affect validity)."""
        self.warnings.append(msg)


# ============================================================
# Validation Functions
# ============================================================


def validate_evalset(
    filepath: Path,
    min_items: int = DEFAULT_MIN_ITEMS,
) -> ValidationResult:
    """Validate an eval set JSONL file.

    Args:
        filepath: Path to JSONL file
        min_items: Minimum required items (default 50)

    Returns:
        ValidationResult with errors, warnings, and statistics
    """
    result = ValidationResult()

    # Check file exists
    if not filepath.exists():
        result.add_error(f"File not found: {filepath}")
        return result

    # Read and parse JSONL
    items: list[dict[str, Any]] = []
    line_errors: list[str] = []

    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                item = json.loads(line)
                items.append(item)
            except json.JSONDecodeError as e:
                line_errors.append(f"Line {line_num}: Invalid JSON - {e}")

    # Report JSON parse errors
    for err in line_errors:
        result.add_error(err)

    if not items:
        result.add_error("No valid items found in file")
        return result

    # Validate each item
    seen_ids: set[str] = set()
    section_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()

    for idx, item in enumerate(items):
        item_id = item.get("id", f"item-{idx}")

        # Check required fields
        for field_name in REQUIRED_FIELDS:
            if field_name not in item:
                result.add_error(f"Item '{item_id}': Missing required field '{field_name}'")
            elif not item[field_name]:
                result.add_error(f"Item '{item_id}': Empty value for required field '{field_name}'")

        # Check for duplicate IDs
        if "id" in item:
            if item["id"] in seen_ids:
                result.add_error(f"Duplicate ID: '{item['id']}'")
            seen_ids.add(item["id"])

        # Validate section
        section = item.get("section")
        if section:
            if section not in VALID_SECTIONS:
                result.add_warning(
                    f"Item '{item_id}': Unknown section '{section}' "
                    f"(expected one of: {', '.join(sorted(VALID_SECTIONS))})"
                )
            section_counts[section] += 1

        # Validate difficulty
        difficulty = item.get("difficulty")
        if difficulty:
            if difficulty not in VALID_DIFFICULTIES:
                result.add_warning(
                    f"Item '{item_id}': Unknown difficulty '{difficulty}' "
                    f"(expected one of: {', '.join(sorted(VALID_DIFFICULTIES))})"
                )
            difficulty_counts[difficulty] += 1

        # Track categories
        category = item.get("category")
        if category:
            category_counts[category] += 1

    # Check minimum item count
    if len(items) < min_items:
        result.add_error(f"Insufficient items: {len(items)} < {min_items} required")

    # Build statistics
    result.stats = {
        "total_items": len(items),
        "unique_ids": len(seen_ids),
        "sections": dict(section_counts),
        "categories": dict(category_counts),
        "difficulties": dict(difficulty_counts),
    }

    # Calculate section percentages
    if items:
        result.stats["section_percentages"] = {
            section: round(count / len(items) * 100, 1) for section, count in section_counts.items()
        }

    return result


def print_validation_report(result: ValidationResult, filepath: Path) -> None:
    """Print a formatted validation report.

    Args:
        result: ValidationResult from validation
        filepath: Path to the validated file
    """
    print(f"\n{'=' * 60}")
    print(f"Eval Set Validation Report: {filepath.name}")
    print(f"{'=' * 60}")

    # Summary
    status = "[VALID]" if result.valid else "[INVALID]"
    print(f"\nStatus: {status}")

    # Statistics
    stats = result.stats
    if stats:
        print("\n[Statistics]")
        print(f"   Total items: {stats.get('total_items', 0)}")
        print(f"   Unique IDs: {stats.get('unique_ids', 0)}")

        # Section breakdown
        sections = stats.get("sections", {})
        percentages = stats.get("section_percentages", {})
        if sections:
            print("\n   [Sections]")
            for section in sorted(sections.keys()):
                pct = percentages.get(section, 0)
                print(f"      {section}: {sections[section]} ({pct}%)")

        # Category breakdown
        categories = stats.get("categories", {})
        if categories:
            print("\n   [Categories]")
            for cat in sorted(categories.keys()):
                print(f"      {cat}: {categories[cat]}")

        # Difficulty breakdown
        difficulties = stats.get("difficulties", {})
        if difficulties:
            print("\n   [Difficulty]")
            for diff in ["easy", "medium", "hard"]:
                if diff in difficulties:
                    print(f"      {diff}: {difficulties[diff]}")

    # Errors
    if result.errors:
        print(f"\n[Errors] ({len(result.errors)}):")
        for err in result.errors[:20]:  # Limit output
            print(f"   - {err}")
        if len(result.errors) > 20:
            print(f"   ... and {len(result.errors) - 20} more errors")

    # Warnings
    if result.warnings:
        print(f"\n[Warnings] ({len(result.warnings)}):")
        for warn in result.warnings[:10]:  # Limit output
            print(f"   - {warn}")
        if len(result.warnings) > 10:
            print(f"   ... and {len(result.warnings) - 10} more warnings")

    print(f"\n{'=' * 60}\n")


# ============================================================
# CLI Entry Point
# ============================================================


def main() -> int:
    """CLI entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Validate eval set JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "filepath",
        type=Path,
        help="Path to JSONL eval set file",
    )
    parser.add_argument(
        "--min-items",
        type=int,
        default=DEFAULT_MIN_ITEMS,
        help=f"Minimum required items (default: {DEFAULT_MIN_ITEMS})",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output errors (suppress stats)",
    )

    args = parser.parse_args()

    # Validate
    result = validate_evalset(args.filepath, min_items=args.min_items)

    # Report
    if not args.quiet:
        print_validation_report(result, args.filepath)
    else:
        if result.errors:
            for err in result.errors:
                print(f"ERROR: {err}", file=sys.stderr)

    return 0 if result.valid else 1


if __name__ == "__main__":
    sys.exit(main())
