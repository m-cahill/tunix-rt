"""Tests for Model Registry."""

import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models import TunixRun
from tunix_rt_backend.schemas.model_registry import (
    ModelArtifactCreate,
    ModelPromotionRequest,
)
from tunix_rt_backend.services.model_registry import ModelRegistryService


@pytest.fixture
def registry_root():
    """Create a temporary registry root."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_create_artifact(test_db: AsyncSession) -> None:
    service = ModelRegistryService(test_db)
    req = ModelArtifactCreate(name="test-artifact", description="desc", task_type="sft")

    artifact = await service.create_artifact(req)
    assert artifact.id is not None
    assert artifact.name == "test-artifact"

    # Duplicate check
    with pytest.raises(ValueError, match="already exists"):
        await service.create_artifact(req)


@pytest.mark.asyncio
async def test_promote_run_success(test_db: AsyncSession, registry_root: str) -> None:
    # 1. Create a mock completed run
    run_id = uuid.uuid4()
    output_dir = Path(registry_root) / "run_output"
    output_dir.mkdir()

    # Create required artifacts (Adapter track)
    (output_dir / "adapter_config.json").write_text("{}")
    (output_dir / "adapter_model.bin").write_bytes(b"mock bytes")

    run = TunixRun(
        run_id=run_id,
        dataset_key="ds",
        model_id="m",
        mode="local",
        status="completed",
        config={"output_dir": str(output_dir)},
        created_at=datetime.now(),
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )
    test_db.add(run)
    await test_db.commit()

    # 2. Promote
    with patch("tunix_rt_backend.settings.settings.model_registry_path", registry_root):
        service = ModelRegistryService(test_db)

        # Create artifact
        artifact = await service.create_artifact(ModelArtifactCreate(name="gemma-lora"))

        # Promote
        promote_req = ModelPromotionRequest(source_run_id=run_id)
        version = await service.promote_run(artifact.id, promote_req)

        assert version.version == "v1"
        assert version.status == "ready"
        assert version.sha256 is not None
        assert version.size_bytes > 0

        # Verify files in registry
        # Uri is file://...
        from urllib.parse import urlparse
        from urllib.request import url2pathname

        reg_path = Path(url2pathname(urlparse(version.storage_uri).path))
        assert (reg_path / "adapter_config.json").exists()


@pytest.mark.asyncio
async def test_promote_run_validation_fail(test_db: AsyncSession, registry_root: str) -> None:
    # 1. Create run with missing artifacts
    run_id = uuid.uuid4()
    output_dir = Path(registry_root) / "run_output_bad"
    output_dir.mkdir()

    run = TunixRun(
        run_id=run_id,
        dataset_key="ds",
        model_id="m",
        mode="local",
        status="completed",
        config={"output_dir": str(output_dir)},
        created_at=datetime.now(),
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )
    test_db.add(run)
    await test_db.commit()

    with patch("tunix_rt_backend.settings.settings.model_registry_path", registry_root):
        service = ModelRegistryService(test_db)
        artifact = await service.create_artifact(ModelArtifactCreate(name="bad"))

        with pytest.raises(ValueError, match="Invalid artifacts"):
            await service.promote_run(artifact.id, ModelPromotionRequest(source_run_id=run_id))


@pytest.mark.asyncio
async def test_version_auto_increment(test_db: AsyncSession, registry_root: str) -> None:
    # Setup run and artifact
    run_id = uuid.uuid4()
    output_dir = Path(registry_root) / "run_output_inc"
    output_dir.mkdir()
    (output_dir / "config.json").write_text("{}")
    (output_dir / "pytorch_model.bin").write_bytes(b"bytes")

    run = TunixRun(
        run_id=run_id,
        dataset_key="ds",
        model_id="m",
        mode="local",
        status="completed",
        config={"output_dir": str(output_dir)},
        created_at=datetime.now(),
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )
    test_db.add(run)
    await test_db.commit()

    with patch("tunix_rt_backend.settings.settings.model_registry_path", registry_root):
        service = ModelRegistryService(test_db)
        artifact = await service.create_artifact(ModelArtifactCreate(name="auto-inc"))

        # Promote v1
        v1 = await service.promote_run(artifact.id, ModelPromotionRequest(source_run_id=run_id))
        assert v1.version == "v1"

        # Promote v2 (using same run is allowed for testing logic)
        # We need to change content to avoid idempotency returning v1
        (output_dir / "pytorch_model.bin").write_bytes(b"new bytes")

        v2 = await service.promote_run(artifact.id, ModelPromotionRequest(source_run_id=run_id))
        assert v2.version == "v2"


@pytest.mark.asyncio
async def test_promote_from_failed_run(test_db: AsyncSession, registry_root: str) -> None:
    """Ensure we cannot promote a failed run."""
    run_id = uuid.uuid4()
    run = TunixRun(
        run_id=run_id,
        dataset_key="ds",
        model_id="m",
        status="failed",
        mode="local",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        created_at=datetime.now(),
    )
    test_db.add(run)
    await test_db.commit()

    service = ModelRegistryService(test_db)
    artifact = await service.create_artifact(ModelArtifactCreate(name="fail-test"))

    with pytest.raises(ValueError, match="must be completed"):
        await service.promote_run(artifact.id, ModelPromotionRequest(source_run_id=run_id))


@pytest.mark.asyncio
async def test_idempotency_same_sha(test_db: AsyncSession, registry_root: str) -> None:
    """Ensure promoting identical content returns existing version."""
    run_id = uuid.uuid4()
    output_dir = Path(registry_root) / "run_output_idem"
    output_dir.mkdir()
    (output_dir / "config.json").write_text("{}")
    (output_dir / "pytorch_model.bin").write_bytes(b"identical")

    run = TunixRun(
        run_id=run_id,
        dataset_key="ds",
        model_id="m",
        status="completed",
        mode="local",
        config={"output_dir": str(output_dir)},
        created_at=datetime.now(),
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )
    test_db.add(run)
    await test_db.commit()

    with patch("tunix_rt_backend.settings.settings.model_registry_path", registry_root):
        service = ModelRegistryService(test_db)
        artifact = await service.create_artifact(ModelArtifactCreate(name="idempotent"))

        # First promotion -> v1
        v1 = await service.promote_run(artifact.id, ModelPromotionRequest(source_run_id=run_id))
        assert v1.version == "v1"

        # Second promotion (same content) -> should return v1
        v1_again = await service.promote_run(
            artifact.id, ModelPromotionRequest(source_run_id=run_id)
        )
        assert v1_again.id == v1.id
        assert v1_again.version == "v1"

        # Third promotion with EXPLICIT version -> should fail uniqueness check or return error?
        # Logic says: if explicit version, we check uniqueness of version label.
        # But if content is same, do we create a new version with same content?
        # Yes, if version label is different (e.g. v2 alias).
        # Let's test that behavior:
        v2 = await service.promote_run(
            artifact.id, ModelPromotionRequest(source_run_id=run_id, version_label="v2")
        )
        assert v2.version == "v2"
        assert v2.id != v1.id
        assert v2.sha256 == v1.sha256


@pytest.mark.asyncio
async def test_promote_missing_artifacts_specific(
    test_db: AsyncSession, registry_root: str
) -> None:
    """Test specific error message for missing files."""
    run_id = uuid.uuid4()
    output_dir = Path(registry_root) / "run_empty"
    output_dir.mkdir()
    # Create just one file, but not enough
    (output_dir / "random.txt").write_text("hi")

    run = TunixRun(
        run_id=run_id,
        dataset_key="ds",
        model_id="m",
        status="completed",
        mode="local",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        config={"output_dir": str(output_dir)},
        created_at=datetime.now(),
    )
    test_db.add(run)
    await test_db.commit()

    with patch("tunix_rt_backend.settings.settings.model_registry_path", registry_root):
        service = ModelRegistryService(test_db)
        artifact = await service.create_artifact(ModelArtifactCreate(name="missing"))

        with pytest.raises(ValueError, match="Invalid artifacts"):
            await service.promote_run(artifact.id, ModelPromotionRequest(source_run_id=run_id))
