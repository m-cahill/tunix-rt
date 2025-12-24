"""Tests for Judge implementations."""

from unittest.mock import AsyncMock, Mock

import pytest

from tunix_rt_backend.db.models import TunixRun
from tunix_rt_backend.redi_client import RediClientProtocol
from tunix_rt_backend.services.judges import GemmaJudge, MockJudge


@pytest.mark.asyncio
async def test_mock_judge():
    """Test MockJudge deterministic scoring."""
    judge = MockJudge()

    # Completed run
    run = TunixRun(
        run_id="12345678-1234-5678-1234-567812345678",
        status="completed",
        duration_seconds=10.0,
        stdout="output",
    )

    result = await judge.evaluate(run)
    assert result.verdict in ["pass", "fail"]
    assert 0 <= result.score <= 100
    assert result.judge_info.name == "mock-judge"

    # Failed run
    failed_run = TunixRun(status="failed")
    result_failed = await judge.evaluate(failed_run)
    assert result_failed.score == 0.0
    assert result_failed.verdict == "fail"


@pytest.mark.asyncio
async def test_gemma_judge_success():
    """Test GemmaJudge with successful LLM response."""
    mock_redi = Mock(spec=RediClientProtocol)
    # Mock generate response
    mock_response = """
    ```json
    {
        "score": 85,
        "verdict": "pass",
        "metrics": {
            "correctness": 0.8,
            "safety": 1.0
        }
    }
    ```
    """
    mock_redi.generate = AsyncMock(return_value=mock_response)

    judge = GemmaJudge(mock_redi)
    run = TunixRun(
        run_id="12345678-1234-5678-1234-567812345678",
        status="completed",
        stdout="Execution log...",
        exit_code=0,
    )

    result = await judge.evaluate(run)

    assert result.score == 85.0
    assert result.verdict == "pass"
    assert result.metrics["correctness"] == 0.8
    assert result.judge_info.name == "gemma-judge"

    # Verify RediAI call
    mock_redi.generate.assert_called_once()
    call_args = mock_redi.generate.call_args
    assert call_args.kwargs["model"] == "gemma-judge-v1"
    assert "Execution log..." in call_args.kwargs["prompt"]


@pytest.mark.asyncio
async def test_gemma_judge_malformed_response():
    """Test GemmaJudge handling of invalid JSON."""
    mock_redi = Mock(spec=RediClientProtocol)
    mock_redi.generate = AsyncMock(return_value="Not JSON")

    judge = GemmaJudge(mock_redi)
    run = TunixRun(status="completed", stdout="log")

    with pytest.raises(RuntimeError, match="Judge produced invalid output format"):
        await judge.evaluate(run)


@pytest.mark.asyncio
async def test_gemma_judge_inference_failure():
    """Test GemmaJudge handling of inference error."""
    mock_redi = Mock(spec=RediClientProtocol)
    mock_redi.generate = AsyncMock(side_effect=RuntimeError("RediAI down"))

    judge = GemmaJudge(mock_redi)
    run = TunixRun(status="completed", stdout="log")

    with pytest.raises(RuntimeError, match="Judge evaluation failed"):
        await judge.evaluate(run)
