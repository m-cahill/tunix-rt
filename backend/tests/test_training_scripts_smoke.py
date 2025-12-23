"""Smoke tests for training scripts using subprocess.

These tests validate that training scripts can be invoked via subprocess
and that their --dry-run mode works correctly. They don't require actual
training dependencies (Tunix), just JAX and PyYAML.

Test strategy:
- Use subprocess.run() to call scripts
- Check exit codes (0 = success)
- Verify stable stdout markers
- Keep tests fast (<10s total)
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Paths relative to backend/ directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
TRAIN_SCRIPT = TRAINING_DIR / "train_sft_tunix.py"
CONFIG_FILE = TRAINING_DIR / "configs" / "sft_tiny.yaml"


def jax_available() -> bool:
    """Check if JAX is installed (required for training tests)."""
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


class TestTrainingScriptSmoke:
    """Smoke tests for training scripts via subprocess."""

    def test_train_script_exists(self):
        """Verify training script file exists."""
        assert TRAIN_SCRIPT.exists(), f"Training script not found: {TRAIN_SCRIPT}"
        assert CONFIG_FILE.exists(), f"Config file not found: {CONFIG_FILE}"

    def test_train_script_help(self):
        """Test that training script --help works."""
        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
        )

        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "usage:" in result.stdout.lower()
        assert "--dry-run" in result.stdout

    def test_train_script_missing_args_fails(self):
        """Test that script fails gracefully with missing required arguments."""
        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
        )

        assert result.returncode != 0, "Script should fail without required args"
        # Should show usage or error message
        assert "required" in result.stderr.lower() or "usage" in result.stdout.lower()

    @pytest.mark.training
    def test_train_script_dry_run_exits_zero(self, tmp_path):
        """Test that --dry-run validates config and exits 0."""
        if not jax_available():
            pytest.skip("JAX not installed; use: pip install -e '.[training]'")

        # Create a minimal test dataset
        test_data = tmp_path / "test_data.jsonl"
        test_data.write_text('{"prompts": "Test prompt", "response": "Test response"}\n')

        # Run with --dry-run
        result = subprocess.run(
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--config",
                str(CONFIG_FILE),
                "--data",
                str(test_data),
                "--output",
                str(tmp_path / "output"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )

        # Should succeed
        assert result.returncode == 0, (
            f"Dry-run failed with exit code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Should have validation markers in output
        stdout = result.stdout.lower()
        assert "config loaded" in stdout or "✅" in stdout
        assert "dataset" in stdout or "loaded" in stdout
        assert "dry run" in stdout or "dry-run" in stdout

    @pytest.mark.training
    def test_train_script_dry_run_validates_config(self):
        """Test that --dry-run catches missing config file."""
        result = subprocess.run(
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--config",
                "nonexistent.yaml",
                "--data",
                "fake.jsonl",
                "--output",
                "/tmp/output",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
        )

        # Should fail due to missing config
        assert result.returncode != 0
        assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()

    @pytest.mark.training
    def test_train_script_dry_run_validates_data(self, tmp_path):
        """Test that --dry-run catches missing dataset file."""
        result = subprocess.run(
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--config",
                str(CONFIG_FILE),
                "--data",
                "nonexistent.jsonl",
                "--output",
                str(tmp_path / "output"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
        )

        # Should fail due to missing dataset
        assert result.returncode != 0
        assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()

    @pytest.mark.training
    def test_train_script_dry_run_fast(self, tmp_path):
        """Test that --dry-run completes quickly (<10 seconds)."""
        if not jax_available():
            pytest.skip("JAX not installed; use: pip install -e '.[training]'")

        import time

        # Create test dataset
        test_data = tmp_path / "test_data.jsonl"
        test_data.write_text('{"prompts": "Test", "response": "Response"}\n' * 100)

        start_time = time.time()

        result = subprocess.run(
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--config",
                str(CONFIG_FILE),
                "--data",
                str(test_data),
                "--output",
                str(tmp_path / "output"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )

        duration = time.time() - start_time

        assert result.returncode == 0
        assert duration < 10, f"Dry-run took {duration:.2f}s, should be <10s"
