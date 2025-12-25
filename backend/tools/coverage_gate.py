#!/usr/bin/env python3
"""Coverage gate enforcement script.

Reads coverage.json and enforces:
- Global line coverage >= 70%
- Core module branch coverage >= 68%

The branch coverage gate applies ONLY to "core" modules that are:
- Deterministic and unit-testable
- Business-critical logic
- Not tied to optional dependencies or subprocess execution

Integration-heavy modules (app.py, worker.py, tunix_execution.py, etc.)
are excluded from branch gating but still contribute to line coverage.

Exit codes:
    0: All gates passed
    1: At least one gate failed
"""

import json
import sys
from pathlib import Path

# =============================================================================
# CORE MODULES DEFINITION
# =============================================================================
# These prefixes define modules subject to the branch coverage gate.
# Modules not matching these prefixes are excluded from branch gating
# but still count toward global line coverage.

CORE_MODULE_PREFIXES = [
    # Database layer (deterministic, unit-testable)
    "tunix_rt_backend/db/",
    # Schema definitions (pure Pydantic, fully testable)
    "tunix_rt_backend/schemas/",
    # Helper utilities (pure functions)
    "tunix_rt_backend/helpers/",
    # Training renderers/schema (pure logic)
    "tunix_rt_backend/training/",
    # Scoring logic (pure functions)
    "tunix_rt_backend/scoring.py",
    # Metrics definitions (simple module)
    "tunix_rt_backend/metrics.py",
]

# Modules explicitly excluded from branch gating (integration-heavy)
# Listed for documentation; anything not in CORE_MODULE_PREFIXES is excluded.
EXCLUDED_FROM_BRANCH_GATE = [
    "tunix_rt_backend/app.py",  # FastAPI app with many integration branches
    "tunix_rt_backend/worker.py",  # Background worker with subprocess logic
    "tunix_rt_backend/services/tunix_execution.py",  # Subprocess orchestration
    "tunix_rt_backend/services/artifact_storage.py",  # File I/O
    "tunix_rt_backend/integrations/",  # Optional dependency integrations
    "tunix_rt_backend/redi_client.py",  # External client
]


def is_core_module(file_path: str) -> bool:
    """Check if a file path is a core module subject to branch gating."""
    return any(file_path.startswith(prefix) for prefix in CORE_MODULE_PREFIXES)


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
    files = data.get("files", {})

    # ==========================================================================
    # Global Line Coverage (all modules)
    # ==========================================================================
    line_coverage = totals["percent_covered"]

    # ==========================================================================
    # Core Module Branch Coverage (scoped)
    # ==========================================================================
    core_branches_covered = 0
    core_branches_total = 0

    for file_path, file_data in files.items():
        if is_core_module(file_path):
            summary = file_data.get("summary", {})
            core_branches_covered += summary.get("covered_branches", 0)
            core_branches_total += summary.get("num_branches", 0)

    if core_branches_total > 0:
        core_branch_coverage = (core_branches_covered / core_branches_total) * 100
    else:
        core_branch_coverage = 100.0  # No branches = 100% covered

    # Gates
    LINE_GATE = 70.0
    CORE_BRANCH_GATE = 68.0

    # Report
    print("=" * 60)
    print("Coverage Gate Report")
    print("=" * 60)
    print(f"Global Line Coverage:      {line_coverage:.2f}% (gate: >= {LINE_GATE}%)")
    print(f"Core Module Branch Cov:    {core_branch_coverage:.2f}% (gate: >= {CORE_BRANCH_GATE}%)")
    print(f"  (Core branches: {core_branches_covered}/{core_branches_total})")
    print("=" * 60)

    # Check gates
    line_pass = line_coverage >= LINE_GATE
    branch_pass = core_branch_coverage >= CORE_BRANCH_GATE

    if line_pass and branch_pass:
        print("[PASS] All coverage gates PASSED")
        print("=" * 60)
        return 0

    # Failure report
    print("[FAIL] Coverage gates FAILED:")
    if not line_pass:
        print(f"  - Global line coverage: {line_coverage:.2f}% < {LINE_GATE}%")
    if not branch_pass:
        print(f"  - Core branch coverage: {core_branch_coverage:.2f}% < {CORE_BRANCH_GATE}%")
    print("=" * 60)

    return 1


if __name__ == "__main__":
    sys.exit(main())
