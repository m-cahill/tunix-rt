#!/usr/bin/env python3
"""Tests for sweep_runner module (M34).

These tests verify the SweepRunner class works correctly without requiring
Ray Tune to be installed. They mock the HTTP API calls to test the sweep
workflow logic.

Test scenarios:
- SweepConfig serialization
- Job creation and start
- Polling for completion
- Success and failure handling
- Timeout handling
"""

from unittest.mock import MagicMock

from tunix_rt_backend.tuning import SweepConfig, SweepResult, SweepRunner


class TestSweepConfig:
    """Tests for SweepConfig dataclass."""

    def test_default_config_values(self) -> None:
        """Default config has sensible M34 values."""
        config = SweepConfig()

        assert config.name == "Tuning Sweep"
        assert config.dataset_key == "dev-reasoning-v2"
        assert config.base_model_id == "google/gemma-3-1b-it"
        assert config.metric_name == "answer_correctness"
        assert config.metric_mode == "max"
        assert config.num_samples == 5
        assert config.max_concurrent_trials == 1

    def test_custom_config_values(self) -> None:
        """Custom config values are preserved."""
        config = SweepConfig(
            name="Custom Sweep",
            dataset_key="golden-v2",
            num_samples=10,
        )

        assert config.name == "Custom Sweep"
        assert config.dataset_key == "golden-v2"
        assert config.num_samples == 10

    def test_default_search_space_has_m34_params(self) -> None:
        """Default search space includes M34 parameters."""
        config = SweepConfig()
        space = config.search_space

        assert "learning_rate" in space
        assert "per_device_batch_size" in space
        assert "weight_decay" in space
        assert "warmup_steps" in space

        # Check types
        assert space["learning_rate"]["type"] == "loguniform"
        assert space["per_device_batch_size"]["type"] == "choice"
        assert space["weight_decay"]["type"] == "uniform"
        assert space["warmup_steps"]["type"] == "choice"

    def test_to_api_payload(self) -> None:
        """to_api_payload returns correct structure."""
        config = SweepConfig(
            name="Test Sweep",
            dataset_key="test-v1",
            num_samples=3,
        )
        payload = config.to_api_payload()

        assert payload["name"] == "Test Sweep"
        assert payload["dataset_key"] == "test-v1"
        assert payload["num_samples"] == 3
        assert payload["metric_name"] == "answer_correctness"
        assert "search_space" in payload


class TestSweepResult:
    """Tests for SweepResult dataclass."""

    def test_success_result(self) -> None:
        """Successful result has expected fields."""
        result = SweepResult(
            success=True,
            job_id="abc-123",
            status="completed",
            best_params={"learning_rate": 1e-5},
            best_run_id="run-456",
        )

        assert result.success is True
        assert result.job_id == "abc-123"
        assert result.best_params == {"learning_rate": 1e-5}
        assert result.error is None

    def test_failure_result(self) -> None:
        """Failed result has error message."""
        result = SweepResult(
            success=False,
            error="Connection refused",
        )

        assert result.success is False
        assert result.error == "Connection refused"
        assert result.best_params is None


class TestSweepRunner:
    """Tests for SweepRunner class with mocked HTTP."""

    def test_run_success(self) -> None:
        """Successful sweep returns best params."""
        runner = SweepRunner(api_url="http://test:8000")

        # Mock the HTTP client
        mock_client = MagicMock()
        runner.client = mock_client

        # Mock responses
        mock_client.post.side_effect = [
            # Create job response
            MagicMock(
                status_code=200,
                json=lambda: {"id": "job-123"},
                raise_for_status=lambda: None,
            ),
            # Start job response
            MagicMock(
                status_code=200,
                raise_for_status=lambda: None,
            ),
        ]

        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {
                    "id": "job-123",
                    "status": "completed",
                    "best_params_json": {"learning_rate": 2e-5},
                    "best_run_id": "run-abc",
                }
            ],
            raise_for_status=lambda: None,
        )

        config = SweepConfig(name="Test", num_samples=1)
        result = runner.run(config)

        assert result.success is True
        assert result.job_id == "job-123"
        assert result.best_params == {"learning_rate": 2e-5}
        assert result.best_run_id == "run-abc"

    def test_run_job_creation_fails(self) -> None:
        """Job creation failure returns error result."""
        runner = SweepRunner(api_url="http://test:8000")

        mock_client = MagicMock()
        runner.client = mock_client

        # Mock create failure
        mock_client.post.side_effect = Exception("Connection refused")

        config = SweepConfig(name="Test")
        result = runner.run(config)

        assert result.success is False
        assert "Connection refused" in str(result.error)

    def test_run_job_start_fails_501(self) -> None:
        """Job start with 501 returns Ray not available error."""
        runner = SweepRunner(api_url="http://test:8000")

        mock_client = MagicMock()
        runner.client = mock_client

        # Create succeeds
        mock_create = MagicMock()
        mock_create.status_code = 200
        mock_create.json.return_value = {"id": "job-123"}
        mock_create.raise_for_status = lambda: None

        # Start returns 501
        mock_start = MagicMock()
        mock_start.status_code = 501

        mock_client.post.side_effect = [mock_create, mock_start]

        config = SweepConfig(name="Test")
        result = runner.run(config)

        assert result.success is False
        assert "Ray Tune not installed" in str(result.error)

    def test_run_job_fails(self) -> None:
        """Failed job returns failure result."""
        runner = SweepRunner(api_url="http://test:8000")

        mock_client = MagicMock()
        runner.client = mock_client

        # Create and start succeed
        mock_client.post.side_effect = [
            MagicMock(
                status_code=200,
                json=lambda: {"id": "job-123"},
                raise_for_status=lambda: None,
            ),
            MagicMock(status_code=200, raise_for_status=lambda: None),
        ]

        # Poll returns failed
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"id": "job-123", "status": "failed"}],
            raise_for_status=lambda: None,
        )

        config = SweepConfig(name="Test")
        result = runner.run(config)

        assert result.success is False
        assert result.status == "failed"

    def test_context_manager(self) -> None:
        """SweepRunner works as context manager."""
        with SweepRunner(api_url="http://test:8000") as runner:
            assert runner.api_url == "http://test:8000"

    def test_poll_with_timeout(self) -> None:
        """Polling respects timeout configuration."""
        runner = SweepRunner(api_url="http://test:8000")

        mock_client = MagicMock()
        runner.client = mock_client

        # Create and start succeed
        mock_client.post.side_effect = [
            MagicMock(
                status_code=200,
                json=lambda: {"id": "job-123"},
                raise_for_status=lambda: None,
            ),
            MagicMock(status_code=200, raise_for_status=lambda: None),
        ]

        # Poll always returns running (will timeout)
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"id": "job-123", "status": "running"}],
            raise_for_status=lambda: None,
        )

        # Use short timeout and poll interval
        config = SweepConfig(
            name="Test",
            timeout_seconds=0.1,
            poll_interval=0.01,
        )

        result = runner.run(config)

        assert result.success is False
        assert "timed out" in str(result.error).lower()


class TestSweepRunnerIntegration:
    """Integration tests that verify import and basic structure."""

    def test_imports_work(self) -> None:
        """Module imports work correctly."""
        from tunix_rt_backend.tuning import SweepConfig, SweepResult, SweepRunner

        assert SweepConfig is not None
        assert SweepResult is not None
        assert SweepRunner is not None

    def test_config_isolation(self) -> None:
        """Each config instance has its own search space."""
        config1 = SweepConfig()
        config2 = SweepConfig()

        config1.search_space["new_param"] = {"type": "choice", "values": [1, 2]}

        assert "new_param" not in config2.search_space
