#!/usr/bin/env python3
"""Tunix RT - Submission Packaging Tool

This tool creates a single zip archive containing all artifacts needed
for the Google Tunix Hackathon submission.

Usage:
    python backend/tools/package_submission.py

    # With custom output directory
    python backend/tools/package_submission.py --output ./my_submission

    # Include training output artifacts
    python backend/tools/package_submission.py --include-output ./output/final_run

Output:
    Creates a zip file at: submission/tunix_rt_m31_<YYYY-MM-DD>_<shortsha>.zip

The archive contains:
    - notebooks/kaggle_submission.ipynb
    - notebooks/kaggle_submission.py
    - docs/kaggle_submission.md
    - docs/submission_checklist.md
    - docs/submission_freeze.md
    - docs/submission_artifacts.md
    - training/configs/submission_gemma3_1b.yaml
    - training/configs/submission_gemma2_2b.yaml
    - training/evalsets/eval_v1.jsonl
    - backend/datasets/<dataset>/manifest.json (for each dataset)
    - README_SUBMISSION.md (generated)
"""

import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

# ============================================================
# Configuration
# ============================================================

# Files to include in the submission bundle
BUNDLE_FILES = [
    # Notebooks
    "notebooks/kaggle_submission.ipynb",
    "notebooks/kaggle_submission.py",
    # Documentation
    "docs/kaggle_submission.md",
    "docs/submission_checklist.md",
    "docs/submission_freeze.md",
    "docs/submission_artifacts.md",
    "docs/evaluation.md",
    "docs/training_end_to_end.md",
    # Training configs
    "training/configs/submission_gemma3_1b.yaml",
    "training/configs/submission_gemma2_2b.yaml",
    "training/configs/sft_tiny.yaml",
    # Eval sets
    "training/evalsets/eval_v1.jsonl",
]

# Dataset manifests to include
DATASETS = ["golden-v2", "dev-reasoning-v1"]

# Archive naming
ARCHIVE_PREFIX = "tunix_rt_m31"

# ============================================================
# Helper Functions
# ============================================================


def get_git_short_sha() -> str:
    """Get the short SHA of the current git commit.

    Returns:
        Short SHA string, or 'unknown' if git is not available.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_timestamp() -> str:
    """Get current UTC timestamp in YYYY-MM-DD format.

    Returns:
        Formatted date string.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def generate_submission_readme(output_path: Path, sha: str, timestamp: str) -> None:
    """Generate a README file for the submission bundle.

    Args:
        output_path: Path to write the README.
        sha: Git commit SHA.
        timestamp: Build timestamp.
    """
    readme_content = f"""# Tunix RT - Submission Package

**Version:** m31_v1
**Built:** {timestamp}
**Commit:** {sha}

## Quick Start

1. **Open the notebook:**
   - Upload `notebooks/kaggle_submission.ipynb` to Kaggle
   - Or run `python notebooks/kaggle_submission.py` locally

2. **Run smoke test first:**
   - Execute cell 4a (Smoke Run) to validate the pipeline

3. **Run full training:**
   - Execute cell 4b (Full Training Run)

## Contents

| File/Directory | Purpose |
|----------------|---------|
| `notebooks/` | Kaggle submission notebook and script |
| `docs/` | Documentation and checklists |
| `training/configs/` | Training configuration files |
| `training/evalsets/` | Evaluation datasets |
| `datasets/` | Dataset manifests (data not included) |

## Configuration

The default configuration uses:
- **Model:** google/gemma-3-1b-it
- **Dataset:** golden-v2 (100 traces)
- **Steps:** 100

Modify the configuration cell in the notebook to change these.

## Reproducibility

All training uses:
- **Seed:** 42 (deterministic)
- **Pinned dependencies:** See uv.lock in main repo

## Documentation

- [Submission Checklist](docs/submission_checklist.md)
- [Kaggle Submission Guide](docs/kaggle_submission.md)
- [Training End-to-End](docs/training_end_to_end.md)
- [Evaluation Semantics](docs/evaluation.md)

## Competition

Google Tunix Hack - Train a model to show its work
https://www.kaggle.com/competitions/google-tunix-hackathon
"""
    output_path.write_text(readme_content, encoding="utf-8")
    print(f"   [+] Generated {output_path.name}")


def copy_file_to_bundle(src: Path, bundle_dir: Path, relative_path: str | None = None) -> bool:
    """Copy a file to the bundle directory, preserving structure.

    Args:
        src: Source file path.
        bundle_dir: Bundle root directory.
        relative_path: Optional relative path within bundle (defaults to src structure).

    Returns:
        True if successful, False otherwise.
    """
    if not src.exists():
        print(f"   [!] Missing: {src}")
        return False

    if relative_path:
        dest = bundle_dir / relative_path
    else:
        dest = bundle_dir / src

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"   [+] {src}")
    return True


def package_submission(
    output_dir: Path,
    include_output: Path | None = None,
    verbose: bool = True,
) -> Path:
    """Create the submission package.

    Args:
        output_dir: Directory to create the submission in.
        include_output: Optional training output directory to include.
        verbose: Whether to print progress.

    Returns:
        Path to the created zip file.
    """
    # Get metadata
    sha = get_git_short_sha()
    timestamp = get_timestamp()

    # Create archive name
    archive_name = f"{ARCHIVE_PREFIX}_{timestamp}_{sha}"
    bundle_dir = output_dir / archive_name
    archive_path = output_dir / f"{archive_name}.zip"

    print(f"\n{'=' * 60}")
    print("TUNIX RT - SUBMISSION PACKAGING")
    print(f"{'=' * 60}")
    print(f"\nArchive: {archive_name}.zip")
    print(f"Commit:  {sha}")
    print(f"Date:    {timestamp}")
    print(f"{'=' * 60}\n")

    # Clean existing bundle
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    # Copy bundle files
    print("Copying files...")
    missing_files = []
    for file_path in BUNDLE_FILES:
        src = Path(file_path)
        if not copy_file_to_bundle(src, bundle_dir):
            missing_files.append(file_path)

    # Copy dataset manifests
    print("\nCopying dataset manifests...")
    for dataset in DATASETS:
        manifest_path = Path(f"backend/datasets/{dataset}/manifest.json")
        if manifest_path.exists():
            dest_path = f"datasets/{dataset}/manifest.json"
            copy_file_to_bundle(manifest_path, bundle_dir, dest_path)
        else:
            print(f"   [!] Missing: {manifest_path}")

    # Include training output if specified
    if include_output and include_output.exists():
        print(f"\nCopying training output from {include_output}...")
        output_dest = bundle_dir / "output"
        shutil.copytree(include_output, output_dest)
        print(f"   [+] {include_output} -> output/")

    # Generate README
    print("\nGenerating README...")
    generate_submission_readme(bundle_dir / "README_SUBMISSION.md", sha, timestamp)

    # Create zip archive
    print(f"\nCreating archive: {archive_path}")
    with ZipFile(archive_path, "w") as zipf:
        for file in bundle_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(bundle_dir)
                zipf.write(file, arcname)

    # Clean up bundle directory (keep only zip)
    shutil.rmtree(bundle_dir)

    # Summary
    print(f"\n{'=' * 60}")
    print("PACKAGING COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n[OK] Archive created: {archive_path}")
    print(f"   Size: {archive_path.stat().st_size / 1024:.1f} KB")

    if missing_files:
        print(f"\n[!] Warning: {len(missing_files)} file(s) were missing:")
        for f in missing_files:
            print(f"   - {f}")

    print("\nNext steps:")
    print("   1. Verify archive contents: unzip -l <archive>")
    print("   2. Upload notebook to Kaggle")
    print("   3. Record and upload video")
    print("   4. Update docs/submission_freeze.md with archive name")
    print(f"{'=' * 60}\n")

    return archive_path


def main() -> None:
    """Main entry point for the packaging tool."""
    parser = argparse.ArgumentParser(
        description="Package Tunix RT submission for Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic packaging
    python backend/tools/package_submission.py

    # Include training output
    python backend/tools/package_submission.py --include-output ./output/final_run

    # Custom output directory
    python backend/tools/package_submission.py --output ./my_submission
        """,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./submission"),
        help="Output directory for the archive (default: ./submission)",
    )
    parser.add_argument(
        "--include-output",
        type=Path,
        default=None,
        help="Include training output directory in the bundle",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output (default: True)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    # Run packaging
    package_submission(
        output_dir=args.output,
        include_output=args.include_output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
