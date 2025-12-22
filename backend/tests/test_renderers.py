"""Tests for training prompt renderers."""

import pytest

from tunix_rt_backend.training.renderers import render_trace_for_training, render_tunix_sft_prompt


class TestTunixSFTRenderer:
    """Test the Tunix SFT prompt renderer."""

    def test_render_tunix_sft_prompt_basic(self):
        """Test basic Tunix SFT prompt rendering."""
        trace_data = {
            "prompt": "What is 2+2?",
            "trace_steps": ["Add 2 and 2"],
            "final_answer": "4",
        }

        result = render_tunix_sft_prompt(trace_data)

        # Verify basic structure
        assert "<start_of_turn>user" in result
        assert "<end_of_turn>" in result
        assert "<start_of_turn>model" in result
        assert "What is 2+2?" in result
        assert "Answer: 4" in result
        assert "Reasoning:" in result
        assert "1. Add 2 and 2" in result

    def test_render_tunix_sft_prompt_multiple_steps(self):
        """Test rendering with multiple reasoning steps."""
        trace_data = {
            "prompt": "Explain photosynthesis",
            "trace_steps": [
                "Define photosynthesis as a process",
                "Describe light absorption",
                "Explain chemical conversion",
            ],
            "final_answer": "Plants convert light to energy",
        }

        result = render_tunix_sft_prompt(trace_data)

        # Verify all steps are included
        assert "1. Define photosynthesis as a process" in result
        assert "2. Describe light absorption" in result
        assert "3. Explain chemical conversion" in result
        assert "Answer: Plants convert light to energy" in result

    def test_render_tunix_sft_prompt_deterministic(self):
        """Test that rendering is deterministic."""
        trace_data = {
            "prompt": "Test prompt",
            "trace_steps": ["Step 1", "Step 2"],
            "final_answer": "Test answer",
        }

        result1 = render_tunix_sft_prompt(trace_data)
        result2 = render_tunix_sft_prompt(trace_data)

        assert result1 == result2

    def test_render_tunix_sft_prompt_empty_steps(self):
        """Test rendering with no reasoning steps."""
        trace_data = {
            "prompt": "Simple question",
            "trace_steps": [],
            "final_answer": "Simple answer",
        }

        result = render_tunix_sft_prompt(trace_data)

        # Should still have structure but no reasoning section
        assert "<start_of_turn>user" in result
        assert "Simple question" in result
        assert "Answer: Simple answer" in result
        # No "Reasoning:" section
        assert "Reasoning:\n1." not in result

    def test_render_tunix_sft_prompt_preserves_special_chars(self):
        """Test that special characters are preserved."""
        trace_data = {
            "prompt": "What is $100 + €50?",
            "trace_steps": ["Convert € to $ at rate 1.1", "Add $100 + $55"],
            "final_answer": "$155",
        }

        result = render_tunix_sft_prompt(trace_data)

        assert "$100 + €50" in result
        assert "Convert € to $" in result
        assert "$155" in result

    def test_render_tunix_sft_prompt_multiline_content(self):
        """Test rendering with multiline step content."""
        trace_data = {
            "prompt": "Explain markdown",
            "trace_steps": [
                "Markdown is a lightweight markup language.\nIt uses plain text formatting."
            ],
            "final_answer": "A simple formatting syntax",
        }

        result = render_tunix_sft_prompt(trace_data)

        # Multiline content should be preserved
        expected_text = "Markdown is a lightweight markup language.\nIt uses plain text formatting."
        assert expected_text in result


class TestRenderTraceForTraining:
    """Test the dispatcher function for rendering."""

    def test_render_trace_for_training_tunix_sft(self):
        """Test rendering with tunix_sft format."""
        trace_data = {
            "prompt": "Test",
            "trace_steps": ["Step"],
            "final_answer": "Answer",
        }

        result = render_trace_for_training(trace_data, format_type="tunix_sft")

        assert "<start_of_turn>user" in result
        assert "Test" in result
        assert "Answer" in result

    def test_render_trace_for_training_default_format(self):
        """Test that default format is tunix_sft."""
        trace_data = {
            "prompt": "Test",
            "trace_steps": ["Step"],
            "final_answer": "Answer",
        }

        result = render_trace_for_training(trace_data)

        # Default should be tunix_sft
        assert "<start_of_turn>user" in result

    def test_render_trace_for_training_invalid_format(self):
        """Test that invalid format raises ValueError."""
        trace_data = {
            "prompt": "Test",
            "trace_steps": ["Step"],
            "final_answer": "Answer",
        }

        with pytest.raises(ValueError, match="Unsupported format type"):
            render_trace_for_training(trace_data, format_type="invalid_format")

