# Model Registry

The Model Registry allows you to manage model artifacts and versions, promoting successful Tunix training runs into immutable, versioned models.

## Concepts

### Model Artifact
A **Model Artifact** represents a logical model family or "repository". For example: "Gemma-2B-Finance-SFT".

### Model Version
A **Model Version** is a specific, immutable build of a Model Artifact. It is created by promoting a completed `TunixRun`. Each version contains:
- **Artifacts:** The model weights and config files.
- **Metadata:** Training config, metrics, and provenance (source run, dataset, base model).
- **Version Label:** e.g., `v1`, `v2`, or a custom tag.

## Usage

### Promoting a Run

When a Tunix Run completes successfully, you can promote it to the registry.

1. **Ensure Run is Completed:** Only completed runs can be promoted.
2. **Check Artifacts:** The run output must contain valid model files:
   - **Adapter Track:** `adapter_config.json` AND (`adapter_model.bin` OR `adapter_model.safetensors`)
   - **Full Model Track:** `config.json` AND (`pytorch_model.bin` OR `*.safetensors`)
3. **Promote:** Call the API endpoint:
   `POST /api/models/{artifact_id}/versions/promote`
   Body: `{ "source_run_id": "...", "version_label": "v1" }` (version_label is optional)

The system will:
1. Copy artifacts to the registry storage (content-addressed).
2. Snapshot metrics and config.
3. Create a new `ModelVersion`.

### Downloading

You can download the full artifact bundle as a ZIP file:
`GET /api/models/versions/{version_id}/download`

## Storage

Artifacts are stored locally in the path configured by `MODEL_REGISTRY_PATH` (default: `backend/artifacts/model_registry`). The storage uses a content-addressable scheme based on SHA256 hashes to ensure deduplication and immutability.
