"""Tunix run manifest generation.

This module generates Tunix-compatible training run manifests (YAML configs)
that can be executed by developers on their local machines or TPU VMs.

M12 Design: Mock-first, artifact-based
- Does NOT require Tunix runtime to be installed
- Generates valid YAML that Tunix CLI can consume
- Uses convention-based manifest structure
"""

import yaml

from tunix_rt_backend.schemas.tunix import TunixManifestRequest


def build_sft_manifest(request: TunixManifestRequest, dataset_path: str) -> str:
    """Build a Tunix SFT (Supervised Fine-Tuning) run manifest.

    Args:
        request: Manifest generation request with hyperparameters
        dataset_path: Path to the dataset JSONL file (relative or absolute)

    Returns:
        YAML manifest content as string

    Note:
        This function does NOT validate against Tunix CLI. It generates a
        best-effort YAML config based on common SFT patterns. Future
        milestones may add schema validation when Tunix documentation
        stabilizes.

    Example output:
        ```yaml
        version: "1.0"
        runner: tunix
        mode: sft
        model:
          model_id: google/gemma-2b-it
        dataset:
          format: tunix_sft
          path: ./datasets/ungar_hcd-v1.jsonl
        training:
          learning_rate: 2.0e-05
          num_epochs: 3
          batch_size: 8
          max_seq_length: 2048
        output:
          output_dir: ./output/run_001
        ```
    """
    # Build manifest dictionary
    manifest_dict = {
        "version": "1.0",
        "runner": "tunix",
        "mode": "sft",
        "model": {
            "model_id": request.model_id,
        },
        "dataset": {
            "format": "tunix_sft",
            "path": dataset_path,
        },
        "training": {
            "learning_rate": request.learning_rate,
            "num_epochs": request.num_epochs,
            "batch_size": request.batch_size,
            "max_seq_length": request.max_seq_length,
            "device": request.device,
        },
        "output": {
            "output_dir": request.output_dir,
        },
    }

    # Serialize to YAML
    # Use default_flow_style=False for readable multiline YAML
    # Use sort_keys=False to maintain logical ordering
    yaml_content: str = yaml.dump(
        manifest_dict,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )

    return yaml_content
