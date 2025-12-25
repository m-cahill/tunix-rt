"""Model Registry Service (M20).

Handles business logic for Model Registry:
- Managing artifacts (families)
- Promoting runs to versions
- Version retrieval
"""

import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models.model_registry import ModelArtifact, ModelVersion
from tunix_rt_backend.db.models.tunix_run import TunixRun
from tunix_rt_backend.schemas.model_registry import (
    ModelArtifactCreate,
    ModelPromotionRequest,
)
from tunix_rt_backend.services.artifact_storage import ArtifactStorageService

logger = logging.getLogger(__name__)


class ModelRegistryService:
    """Service for Model Registry operations."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.storage = ArtifactStorageService()

    async def create_artifact(self, request: ModelArtifactCreate) -> ModelArtifact:
        """Create a new model artifact family."""
        # Check uniqueness
        stmt = select(ModelArtifact).where(ModelArtifact.name == request.name)
        result = await self.db.execute(stmt)
        if result.scalar_one_or_none():
            raise ValueError(f"Model artifact with name '{request.name}' already exists")

        artifact = ModelArtifact(
            name=request.name,
            description=request.description,
            task_type=request.task_type,
        )
        self.db.add(artifact)
        await self.db.commit()
        await self.db.refresh(artifact)
        return artifact

    async def list_artifacts(self) -> list[ModelArtifact]:
        """List all model artifacts."""
        # TODO: Add pagination if needed
        stmt = select(ModelArtifact).order_by(ModelArtifact.updated_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_artifact(self, artifact_id: uuid.UUID) -> ModelArtifact | None:
        """Get artifact by ID."""
        stmt = select(ModelArtifact).where(ModelArtifact.id == artifact_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_version(self, version_id: uuid.UUID) -> ModelVersion | None:
        """Get version by ID."""
        stmt = select(ModelVersion).where(ModelVersion.id == version_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def promote_run(
        self, artifact_id: uuid.UUID, request: ModelPromotionRequest
    ) -> ModelVersion:
        """Promote a TunixRun to a ModelVersion."""
        # 1. Validate Artifact exists
        artifact = await self.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact {artifact_id} not found")

        # 2. Validate Run
        stmt = select(TunixRun).where(TunixRun.run_id == request.source_run_id)
        result = await self.db.execute(stmt)
        run = result.scalar_one_or_none()

        if not run:
            raise ValueError(f"Run {request.source_run_id} not found")

        if run.status != "completed":
            raise ValueError(f"Run must be completed (current status: {run.status})")

        # 3. Locate Output Directory
        output_dir = self._resolve_output_dir(run)
        if not output_dir.exists() or not output_dir.is_dir():
            raise ValueError(f"Run output directory not found: {output_dir}")

        # 4. Validate Files (M20 requirement)
        self._validate_run_artifacts(output_dir)

        # 6. Store Artifacts
        try:
            storage_uri, sha256, size_bytes = await self._store_artifacts(output_dir)
        except Exception as e:
            logger.error(f"Storage failed: {e}")
            raise RuntimeError(f"Failed to store artifacts: {e}")

        # 7. Check idempotency
        # If version was not explicit, check if we already have this content
        if not request.version_label:
            stmt_exist = select(ModelVersion).where(
                ModelVersion.artifact_id == artifact_id, ModelVersion.sha256 == sha256
            )
            existing = (await self.db.execute(stmt_exist)).scalars().first()
            if existing:
                logger.info(
                    f"Returning existing version {existing.version} "
                    f"for identical content (sha256: {sha256})"
                )
                return existing

        # 5. Determine Version Label (delayed to after idempotency check)
        version_label = request.version_label
        if not version_label:
            version_label = await self._generate_next_version(artifact_id)

        # Check uniqueness of version label
        stmt_ver = select(ModelVersion).where(
            ModelVersion.artifact_id == artifact_id, ModelVersion.version == version_label
        )
        if (await self.db.execute(stmt_ver)).scalar_one_or_none():
            raise ValueError(
                f"Version '{version_label}' already exists for artifact {artifact.name}"
            )

        # 8. Create Version
        # Gather metadata
        metrics: dict[str, Any] = {}
        # Try to find metrics from TunixRun (maybe in config if stored, or fetch from Evaluations)
        # For M20, run.config might have some info, or we look for evaluation results
        # Assuming we just store what we have.
        # Run config is in run.config

        # Provenance
        provenance = {
            "source_run_id": str(run.run_id),
            "dataset_key": run.dataset_key,
            "base_model_id": run.model_id,
            "run_started_at": run.started_at.isoformat() if run.started_at else None,
            "run_completed_at": run.completed_at.isoformat() if run.completed_at else None,
        }

        model_version = ModelVersion(
            artifact_id=artifact_id,
            version=version_label,
            source_run_id=run.run_id,
            status="ready",
            metrics_json=metrics,  # TODO: Populate from Evaluations if available
            config_json=run.config,
            provenance_json=provenance,
            storage_uri=storage_uri,
            sha256=sha256,
            size_bytes=size_bytes,
        )

        self.db.add(model_version)

        # Update artifact updated_at
        artifact.updated_at = datetime.now(timezone.utc)

        await self.db.commit()
        await self.db.refresh(model_version)

        return model_version

    def _resolve_output_dir(self, run: TunixRun) -> Path:
        """Resolve output directory from run config."""
        path_str = None
        if run.config and "output_dir" in run.config:
            path_str = run.config["output_dir"]

        if not path_str:
            path_str = f"./output/tunix_run_{str(run.run_id)[:8]}"

        return Path(path_str).resolve()

    def _validate_run_artifacts(self, output_dir: Path) -> None:
        """Validate required files exist (M20 Adapter vs Full track)."""
        files = {p.name for p in output_dir.glob("*") if p.is_file()}

        # Adapter Track
        has_adapter_config = "adapter_config.json" in files
        has_adapter_weights = "adapter_model.bin" in files or "adapter_model.safetensors" in files
        is_adapter = has_adapter_config and has_adapter_weights

        # Full Model Track
        has_config = "config.json" in files
        has_model_weights = (
            "pytorch_model.bin" in files
            or "model.safetensors" in files
            or any(f.endswith(".safetensors") for f in files)  # Generic check for shards
        )
        is_full = has_config and has_model_weights

        if not (is_adapter or is_full):
            found_list = list(files)[:10]
            raise ValueError(
                f"Invalid artifacts. Found: {found_list}. "
                "Required: (adapter_config.json + weights) OR (config.json + weights)."
            )

    async def _generate_next_version(self, artifact_id: uuid.UUID) -> str:
        """Generate next vN version label."""
        # Query all versions
        stmt = select(ModelVersion.version).where(ModelVersion.artifact_id == artifact_id)
        result = await self.db.execute(stmt)
        versions = result.scalars().all()

        max_v = 0
        for v in versions:
            if re.match(r"^v\d+$", v):
                try:
                    num = int(v[1:])
                    max_v = max(max_v, num)
                except ValueError:
                    pass

        return f"v{max_v + 1}"

    async def _store_artifacts(self, src_dir: Path) -> tuple[str, str, int]:
        """Wrap storage call to run in thread if needed (shutil is blocking)."""
        # shutil.copytree is blocking I/O.
        # ArtifactStorageService computes hash (CPU bound) and copies files (I/O bound).
        # Should run in executor.
        import asyncio

        return await asyncio.to_thread(self.storage.put_directory, str(src_dir))
