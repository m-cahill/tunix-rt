"""Dataset helper functions for manifest management.

This module provides utilities for saving and loading dataset manifests
to/from the filesystem.
"""

import json
from pathlib import Path
from typing import Any

from tunix_rt_backend.schemas.dataset import DatasetManifest


def get_datasets_dir() -> Path:
    """Get the datasets directory path.

    Returns:
        Path to backend/datasets directory (creates if doesn't exist)
    """
    # Get backend directory (parent of tunix_rt_backend package)
    backend_dir = Path(__file__).parent.parent.parent
    datasets_dir = backend_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    return datasets_dir


def get_dataset_dir(dataset_key: str) -> Path:
    """Get the directory for a specific dataset.

    Args:
        dataset_key: Dataset key (name-version)

    Returns:
        Path to dataset directory (creates if doesn't exist)
    """
    dataset_dir = get_datasets_dir() / dataset_key
    dataset_dir.mkdir(exist_ok=True, parents=True)
    return dataset_dir


def save_manifest(manifest: DatasetManifest) -> Path:
    """Save dataset manifest to disk.

    Args:
        manifest: DatasetManifest to save

    Returns:
        Path to saved manifest file
    """
    dataset_dir = get_dataset_dir(manifest.dataset_key)
    manifest_path = dataset_dir / "manifest.json"

    # Convert manifest to dict with JSON-serializable types
    manifest_dict = manifest.model_dump(mode="json")

    # Write manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2, ensure_ascii=False)

    return manifest_path


def load_manifest(dataset_key: str) -> DatasetManifest:
    """Load dataset manifest from disk.

    Args:
        dataset_key: Dataset key (name-version)

    Returns:
        DatasetManifest

    Raises:
        FileNotFoundError: If manifest doesn't exist
    """
    dataset_dir = get_dataset_dir(dataset_key)
    manifest_path = dataset_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found for dataset: {dataset_key}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_dict = json.load(f)

    return DatasetManifest(**manifest_dict)


def create_dataset_key(dataset_name: str, dataset_version: str) -> str:
    """Create dataset key from name and version.

    Args:
        dataset_name: Dataset name
        dataset_version: Version string

    Returns:
        Dataset key in format: {name}-{version}
    """
    # Sanitize name and version for filesystem safety
    safe_name = dataset_name.replace(" ", "_").replace("/", "_")
    safe_version = dataset_version.replace(" ", "_").replace("/", "_")
    return f"{safe_name}-{safe_version}"


def compute_dataset_stats(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute statistical summary for a dataset.

    Args:
        traces: List of trace payloads

    Returns:
        Dictionary with statistical summary
    """
    if not traces:
        return {
            "trace_count": 0,
            "avg_step_count": 0.0,
            "min_step_count": 0,
            "max_step_count": 0,
            "avg_total_chars": 0.0,
        }

    step_counts = [len(trace.get("steps", [])) for trace in traces]
    total_chars_list = [
        len(trace.get("prompt", ""))
        + len(trace.get("final_answer", ""))
        + sum(len(step.get("content", "")) for step in trace.get("steps", []))
        for trace in traces
    ]

    return {
        "trace_count": len(traces),
        "avg_step_count": sum(step_counts) / len(step_counts) if step_counts else 0.0,
        "min_step_count": min(step_counts) if step_counts else 0,
        "max_step_count": max(step_counts) if step_counts else 0,
        "avg_total_chars": (
            sum(total_chars_list) / len(total_chars_list) if total_chars_list else 0.0
        ),
    }

