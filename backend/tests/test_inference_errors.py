import pytest

# Skip entire module if optional ML dependencies are not installed
pytest.importorskip("transformers")
pytest.importorskip("torch")


@pytest.mark.asyncio
async def test_generate_predictions_missing_dataset(tmp_path):
    from tunix_rt_backend.services.tunix_execution import generate_predictions

    dataset_path = tmp_path / "nonexistent.jsonl"
    output_dir = tmp_path / "output"

    # Should log warning and return without doing anything (not raise)
    # based on current implementation:
    # if not dataset_path.exists():
    #     logger.warning(...)
    #     return

    await generate_predictions(dataset_path, output_dir)
    assert not (output_dir / "predictions.jsonl").exists()


@pytest.mark.asyncio
async def test_generate_predictions_no_valid_items(tmp_path):
    import json
    from unittest.mock import MagicMock, patch

    dataset_path = tmp_path / "invalid.jsonl"
    output_dir = tmp_path / "output"

    # Write file with invalid JSON or missing fields
    with open(dataset_path, "w") as f:
        f.write("invalid json\n")
        f.write(json.dumps({"no_id": "missing"}) + "\n")

    # We need to mock _run_inference_sync because it runs in a thread
    # But wait, we want to test the logic INSIDE _run_inference_sync that raises if empty.
    # But _run_inference_sync is what we want to test.
    # We can test _run_inference_sync directly for this logic.

    from tunix_rt_backend.services.tunix_execution import _run_inference_sync

    # Mock transformers to be available so it runs real logic, but fails to find valid items
    # We patch at the source (transformers) because it is imported inside the function
    with (
        patch("transformers.AutoTokenizer") as mock_tok,
        patch("transformers.AutoModelForCausalLM") as mock_model,
        patch("torch.device"),
    ):
        # Setup mocks to return something but not crash
        mock_tok.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        with pytest.raises(RuntimeError, match="Inference failed for all items"):
            _run_inference_sync(dataset_path, output_dir)
