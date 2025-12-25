import json
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if optional ML dependencies are not installed
pytest.importorskip("transformers")
pytest.importorskip("torch")

from tunix_rt_backend.services.tunix_execution import _run_inference_sync, generate_predictions


@pytest.mark.asyncio
async def test_generate_predictions_flow(tmp_path):
    """Test the generate_predictions flow (mocking the heavy inference)."""
    dataset_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "output"

    # Create valid dataset
    with open(dataset_path, "w") as f:
        f.write(json.dumps({"id": "123", "prompts": "Test prompt"}) + "\n")

    # Mock to_thread to avoid actual model loading in this unit test
    # We want to test the wrapper logic.
    with patch("asyncio.to_thread") as mock_thread:
        await generate_predictions(dataset_path, output_dir)
        mock_thread.assert_called_once()


def test_inference_sync_logic(tmp_path):
    """Test the synchronous inference logic with a real (tiny) model or mock."""
    dataset_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "output"

    with open(dataset_path, "w") as f:
        f.write(json.dumps({"id": "trace-1", "prompts": "Hello"}) + "\n")

    # We mock transformers to avoid downloading models in unit tests
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tok,
        patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
        patch("torch.device"),
    ):
        # Setup mocks
        mock_tok_instance = MagicMock()
        mock_tok_instance.pad_token = None
        # Mock decode to return a string
        mock_tok_instance.decode.return_value = "Mock prediction"
        mock_tok.return_value = mock_tok_instance

        mock_model_instance = MagicMock()
        # Mock generate return value (needs to be subscriptable tensor-like)
        # We need a tensor that can be sliced.
        # But tokenizer.decode takes the sliced output.
        # Since we mock decode, the input to decode doesn't matter too much,
        # provided code execution reaches decode call.
        # But code does: new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        # So outputs[0] must be sliceable.
        mock_outputs = MagicMock()
        mock_outputs.__getitem__.return_value = MagicMock()  # outputs[0]
        # Slicing returns another mock
        mock_model_instance.generate.return_value = mock_outputs

        mock_model.return_value = mock_model_instance

        # Run sync inference
        _run_inference_sync(dataset_path, output_dir, model_name="test-model")

        # Check output
        output_file = output_dir / "predictions.jsonl"
        assert output_file.exists()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["trace_id"] == "trace-1"
