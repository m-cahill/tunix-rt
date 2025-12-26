"""Guardrail test: validate E2E fixture files against Pydantic schemas.

This test ensures fixture JSONL files match expected schemas BEFORE
Playwright E2E tests run, catching schema drift early and fast.
"""

import json
from pathlib import Path

import pytest

from tunix_rt_backend.schemas import ReasoningTrace

# Path to E2E fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "e2e"


class TestE2EFixtureSchemaValidation:
    """Validate E2E fixture files against their expected schemas."""

    def test_e2e_ingest_fixture_validates_against_reasoning_trace(self) -> None:
        """Each line in e2e_ingest.jsonl must be a valid ReasoningTrace."""
        fixture_path = FIXTURES_DIR / "e2e_ingest.jsonl"

        assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

        with open(fixture_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) > 0, "Fixture file is empty"

        for i, line in enumerate(lines, start=1):
            try:
                data = json.loads(line)
                # Validate against ReasoningTrace schema
                trace = ReasoningTrace.model_validate(data)

                # Basic structural assertions
                assert trace.trace_version, f"Line {i}: missing trace_version"
                assert trace.prompt, f"Line {i}: missing prompt"
                assert trace.final_answer, f"Line {i}: missing final_answer"
                assert len(trace.steps) > 0, f"Line {i}: steps must not be empty"

                # Validate step structure
                for step in trace.steps:
                    assert step.i >= 0, f"Line {i}: step index must be non-negative"
                    assert step.type, f"Line {i}: step type must not be empty"
                    assert step.content, f"Line {i}: step content must not be empty"

            except json.JSONDecodeError as e:
                pytest.fail(f"Line {i}: Invalid JSON - {e}")
            except Exception as e:
                pytest.fail(f"Line {i}: Schema validation failed - {e}")

    def test_e2e_ingest_fixture_has_minimum_traces(self) -> None:
        """Ensure fixture has at least 2 traces for meaningful E2E testing."""
        fixture_path = FIXTURES_DIR / "e2e_ingest.jsonl"

        with open(fixture_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) >= 2, f"E2E fixture should have at least 2 traces, found {len(lines)}"
