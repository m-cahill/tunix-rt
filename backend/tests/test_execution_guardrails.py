import uuid
from unittest.mock import AsyncMock, patch

import pytest

from tunix_rt_backend.schemas.tunix import TunixRunRequest, TunixRunResponse
from tunix_rt_backend.services.tunix_execution import execute_dry_run


@pytest.mark.asyncio
async def test_execute_dry_run_accepts_real_export_request():
    """Guardrail: Ensure execute_dry_run uses real Pydantic models for export.

    This test prevents regression where synthetic objects (type(...)) are used
    instead of real models, which causes attribute errors in downstream services.
    """
    # Setup
    run_id = str(uuid.uuid4())
    request = TunixRunRequest(
        dataset_key="test-dataset-v1",
        model_id="test-model",
        output_dir="/tmp/test-output",
        dry_run=True,
    )
    started_at = "2025-01-01T00:00:00+00:00"
    db = AsyncMock()

    # Mocks
    with patch("tunix_rt_backend.services.tunix_execution.load_manifest") as mock_load:
        with patch("tunix_rt_backend.services.tunix_execution.build_sft_manifest") as mock_build:
            with patch(
                "tunix_rt_backend.services.tunix_execution.export_tunix_sft_jsonl"
            ) as mock_export:
                # Configure success path
                mock_load.return_value = {"id": "test-dataset-v1"}
                mock_build.return_value = "manifest: true"

                # The loop inside execute_dry_run iterates over jsonl_lines.splitlines()
                # mock_export should return a string
                mock_export.return_value = '{"prompt": "test"}\n'

                # Execute
                response = await execute_dry_run(
                    run_id, request, request.output_dir, started_at, db
                )

                # Assertions
                assert isinstance(response, TunixRunResponse)

                # Debug if failed
                if response.status != "completed":
                    print(f"FAILED: {response.stderr}")

                assert response.status == "completed"
                assert response.exit_code == 0

                # CRITICAL: Verify export called with correct type
                assert mock_export.called
                call_args = mock_export.call_args
                export_req = call_args[0][0]

                # Check it's not a synthetic type
                assert type(export_req).__name__ == "TunixExportRequest"
                assert export_req.dataset_key == "test-dataset-v1"
