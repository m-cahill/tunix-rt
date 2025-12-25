import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tunix_rt_backend.db.models import Trace, TunixRun
from tunix_rt_backend.services.judges import AnswerCorrectnessJudge, JudgeFactory, MockJudge


@pytest.mark.unit
def test_compare_normalization():
    """Test string normalization and comparison logic."""
    judge = AnswerCorrectnessJudge()
    assert judge._compare("Answer", "answer")
    assert judge._compare("  Answer  ", "answer")
    assert judge._compare("Answer", "Answer")
    assert not judge._compare("Answer", "Wrong")
    assert judge._compare("Multi\nLine", "multi\nline")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_evaluate_success(test_db, tmp_path):
    """Test full evaluation flow with seeded data and predictions."""
    # Setup
    judge = AnswerCorrectnessJudge(test_db)

    # Create traces in DB
    trace_id_1 = uuid.uuid4()
    trace_id_2 = uuid.uuid4()

    trace1 = Trace(
        id=trace_id_1,
        # dataset_key not in Trace model
        payload={"final_answer": "correct answer 1"},
        trace_version="v1",
        created_at=datetime.now(timezone.utc),
    )
    trace2 = Trace(
        id=trace_id_2,
        # dataset_key not in Trace model
        payload={"final_answer": "correct answer 2"},
        trace_version="v1",
        created_at=datetime.now(timezone.utc),
    )
    test_db.add_all([trace1, trace2])
    await test_db.commit()

    # Create dataset manifest in tmp_path
    dataset_key = "test-dataset"
    dataset_dir = tmp_path / dataset_key
    dataset_dir.mkdir()

    manifest_data = {"trace_ids": [str(trace_id_1), str(trace_id_2)]}
    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)

    # Create predictions in output_dir (also in tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    predictions = [
        {"trace_id": str(trace_id_1), "prediction": "correct answer 1"},  # Correct
        {"trace_id": str(trace_id_2), "prediction": "wrong answer"},  # Incorrect
    ]

    with open(output_dir / "predictions.jsonl", "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    # Mock get_datasets_dir to return tmp_path
    with patch("tunix_rt_backend.services.judges.get_datasets_dir", return_value=tmp_path):
        run = TunixRun(
            run_id=uuid.uuid4(),
            dataset_key=dataset_key,
            status="completed",
            config={"output_dir": str(output_dir)},
        )

        result = await judge.evaluate(run)

        assert result.score == 50.0  # 1 correct out of 2
        assert result.verdict == "fail"  # Assuming strictly 100 is pass
        assert len(result.detailed_metrics) == 2

        # Verify details
        m1 = next(m for m in result.detailed_metrics if m.details["trace_id"] == str(trace_id_1))
        assert m1.score == 1.0
        assert m1.details["correct"] is True

        m2 = next(m for m in result.detailed_metrics if m.details["trace_id"] == str(trace_id_2))
        assert m2.score == 0.0
        assert m2.details["correct"] is False


@pytest.mark.asyncio
@pytest.mark.integration
async def test_evaluate_missing_manifest(test_db, tmp_path):
    """Test error handling when manifest is missing."""
    judge = AnswerCorrectnessJudge(test_db)

    # Mock get_datasets_dir
    with patch("tunix_rt_backend.services.judges.get_datasets_dir", return_value=tmp_path):
        run = TunixRun(
            run_id=uuid.uuid4(),
            dataset_key="nonexistent",
            status="completed",
            config={"output_dir": str(tmp_path)},
        )

        result = await judge.evaluate(run)
        assert result.verdict == "fail"
        assert "Dataset manifest not found" in result.raw_output


@pytest.mark.asyncio
@pytest.mark.integration
async def test_evaluate_missing_predictions(test_db, tmp_path):
    """Test error handling when predictions file is missing."""
    judge = AnswerCorrectnessJudge(test_db)

    dataset_key = "test-dataset"
    dataset_dir = tmp_path / dataset_key
    dataset_dir.mkdir()
    manifest_path = dataset_dir / "manifest.json"
    # Must have trace_ids to pass empty dataset check
    with open(manifest_path, "w") as f:
        json.dump({"trace_ids": [str(uuid.uuid4())]}, f)

    with patch("tunix_rt_backend.services.judges.get_datasets_dir", return_value=tmp_path):
        run = TunixRun(
            run_id=uuid.uuid4(),
            dataset_key=dataset_key,
            status="completed",
            config={"output_dir": str(tmp_path)},  # File doesn't exist here
        )

        result = await judge.evaluate(run)
        assert result.verdict == "fail"
        assert (
            "Run output directory not found" in result.raw_output
            or "Predictions file not found" in result.raw_output
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_evaluate_empty_predictions(test_db, tmp_path):
    """Test error handling when predictions file is empty."""
    judge = AnswerCorrectnessJudge(test_db)

    dataset_key = "test-dataset"
    dataset_dir = tmp_path / dataset_key
    dataset_dir.mkdir()
    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"trace_ids": [str(uuid.uuid4())]}, f)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "predictions.jsonl").touch()  # Empty file

    with patch("tunix_rt_backend.services.judges.get_datasets_dir", return_value=tmp_path):
        run = TunixRun(
            run_id=uuid.uuid4(),
            dataset_key=dataset_key,
            status="completed",
            config={"output_dir": str(output_dir)},
        )

        result = await judge.evaluate(run)
        assert result.verdict == "fail"
        assert (
            "Predictions file is empty" in result.raw_output
            or "contains no valid predictions" in result.raw_output
        )


@pytest.mark.unit
def test_judge_factory(test_db):
    """Test JudgeFactory returns correct instances."""
    redi_mock = MagicMock()
    factory = JudgeFactory(redi_mock, test_db)

    assert isinstance(factory.get_judge("answer_correctness"), AnswerCorrectnessJudge)
    assert isinstance(factory.get_judge("foo"), MockJudge)

    # Verify DB passed
    judge = factory.get_judge("answer_correctness")
    assert judge.db == test_db


@pytest.mark.asyncio
@pytest.mark.unit
async def test_gemma_judge_success():
    """Test GemmaJudge mocked inference."""
    redi_mock = MagicMock()
    # Mock generate method
    redi_mock.generate = AsyncMock(
        return_value=json.dumps(
            {"score": 85, "verdict": "pass", "metrics": {"correctness": 0.85, "safety": 1.0}}
        )
    )

    judge = JudgeFactory(redi_mock, None).get_judge("gemma-judge")
    run = TunixRun(run_id=uuid.uuid4(), status="completed", stdout="Test output")

    result = await judge.evaluate(run)
    assert result.score == 85.0
    assert result.verdict == "pass"
    assert result.metrics["correctness"] == 0.85
