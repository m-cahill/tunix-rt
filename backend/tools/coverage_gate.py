#!/usr/bin/env python3
"""Coverage gate enforcement script.

Reads coverage.json and enforces:
- Line coverage >= 80%
- Branch coverage >= 68%

Exit codes:
    0: All gates passed
    1: At least one gate failed
"""

import json
import sys
from pathlib import Path


def main() -> int:
    """Run coverage gate checks."""
    # Read coverage.json
    coverage_file = Path(__file__).parent.parent / "coverage.json"
    
    if not coverage_file.exists():
        print(f"[ERROR] Coverage file not found: {coverage_file}")
        print("Run: pytest --cov --cov-branch --cov-report=json:coverage.json")
        return 1
    
    with open(coverage_file) as f:
        data = json.load(f)
    
    totals = data["totals"]
    
    # Extract metrics
    line_coverage = totals["percent_covered"]
    branch_coverage = totals["percent_branches_covered"]
    
    # Gates
    LINE_GATE = 80.0
    BRANCH_GATE = 68.0
    
    # Report
    print("=" * 60)
    print("Coverage Gate Report")
    print("=" * 60)
    print(f"Line Coverage:   {line_coverage:.2f}% (gate: >= {LINE_GATE}%)")
    print(f"Branch Coverage: {branch_coverage:.2f}% (gate: >= {BRANCH_GATE}%)")
    print("=" * 60)
    
    # Check gates
    line_pass = line_coverage >= LINE_GATE
    branch_pass = branch_coverage >= BRANCH_GATE
    
    if line_pass and branch_pass:
        print("[PASS] All coverage gates PASSED")
        return 0
    
    # Failure report
    print("[FAIL] Coverage gates FAILED:")
    if not line_pass:
        print(f"  - Line coverage: {line_coverage:.2f}% < {LINE_GATE}%")
    if not branch_pass:
        print(f"  - Branch coverage: {branch_coverage:.2f}% < {BRANCH_GATE}%")
    print("=" * 60)
    
    return 1


if __name__ == "__main__":
    sys.exit(main())

