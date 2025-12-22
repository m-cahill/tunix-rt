"""Tests for training prompt renderers."""

import pytest

from tunix_rt_backend.training.renderers import (
    apply_system_instruction,
    render_gemma_model_turn,
    render_gemma_turn,
    render_gemma_user_turn,
    render_reasoning_steps,
    render_trace_for_training,
    render_tunix_sft_prompt,
)


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


class TestGemmaITHelpers:
    """Test Gemma IT low-level formatting helpers."""

    def test_render_gemma_turn_basic(self):
        """Test basic turn rendering."""
        result = render_gemma_turn("user", "Hello world")

        assert result == "<start_of_turn>user\nHello world<end_of_turn>"

    def test_render_gemma_user_turn(self):
        """Test user turn rendering."""
        result = render_gemma_user_turn("What is AI?")

        assert result == "<start_of_turn>user\nWhat is AI?<end_of_turn>"
        assert "<start_of_turn>user" in result
        assert "What is AI?" in result

    def test_render_gemma_model_turn(self):
        """Test model turn rendering."""
        result = render_gemma_model_turn("AI is artificial intelligence")

        assert result == "<start_of_turn>model\nAI is artificial intelligence<end_of_turn>"
        assert "<start_of_turn>model" in result

    def test_apply_system_instruction_with_instruction(self):
        """Test applying system instruction to user prompt."""
        result = apply_system_instruction(
            user_prompt="What is 2+2?",
            system_instruction="You are a helpful math tutor.",
        )

        assert result == "You are a helpful math tutor.\n\nWhat is 2+2?"
        assert "You are a helpful math tutor" in result
        assert "What is 2+2?" in result

    def test_apply_system_instruction_without_instruction(self):
        """Test that user prompt is returned unchanged when no system instruction."""
        result = apply_system_instruction(
            user_prompt="What is 2+2?",
            system_instruction=None,
        )

        assert result == "What is 2+2?"

    def test_render_reasoning_steps_multiple(self):
        """Test rendering multiple reasoning steps."""
        steps = ["Parse the input", "Perform calculation", "Verify result"]

        result = render_reasoning_steps(steps)

        expected = "Reasoning:\n1. Parse the input\n2. Perform calculation\n3. Verify result\n"
        assert result == expected
        assert "Reasoning:" in result
        assert "1. Parse the input" in result
        assert "2. Perform calculation" in result
        assert "3. Verify result" in result

    def test_render_reasoning_steps_empty(self):
        """Test rendering with no steps."""
        result = render_reasoning_steps([])

        assert result == ""

    def test_render_reasoning_steps_single(self):
        """Test rendering single step."""
        result = render_reasoning_steps(["Only step"])

        assert result == "Reasoning:\n1. Only step\n"


class TestTunixSFTWithSystemInstruction:
    """Test Tunix SFT rendering with system instructions (M09 feature)."""

    def test_render_with_system_instruction(self):
        """Test that system instruction is embedded in user turn."""
        trace_data = {
            "prompt": "What is 2+2?",
            "trace_steps": ["Add the numbers"],
            "final_answer": "4",
        }

        result = render_tunix_sft_prompt(
            trace_data,
            system_instruction="You are a helpful math assistant.",
        )

        # System instruction should appear before the user prompt
        assert "You are a helpful math assistant" in result
        assert "What is 2+2?" in result
        # Both should be in the user turn
        user_turn_start = result.find("<start_of_turn>user")
        user_turn_end = result.find("<end_of_turn>")
        user_turn_content = result[user_turn_start:user_turn_end]

        assert "You are a helpful math assistant" in user_turn_content
        assert "What is 2+2?" in user_turn_content

    def test_render_without_system_instruction(self):
        """Test that rendering works without system instruction."""
        trace_data = {
            "prompt": "What is 2+2?",
            "trace_steps": ["Add"],
            "final_answer": "4",
        }

        result = render_tunix_sft_prompt(trace_data, system_instruction=None)

        # Should just have the user prompt
        assert "What is 2+2?" in result
        assert "<start_of_turn>user" in result


class TestSnapshotStability:
    """Test that output format remains stable (prevents accidental changes)."""

    def test_simple_trace_snapshot(self):
        """Snapshot test for simple trace rendering."""
        trace_data = {
            "prompt": "What is the capital of France?",
            "trace_steps": ["Recall geography knowledge", "Identify France's capital"],
            "final_answer": "Paris",
        }

        result = render_tunix_sft_prompt(trace_data)

        expected = """<start_of_turn>user
What is the capital of France?<end_of_turn>
<start_of_turn>model
Reasoning:
1. Recall geography knowledge
2. Identify France's capital
Answer: Paris<end_of_turn>"""

        assert result == expected

    def test_trace_with_system_snapshot(self):
        """Snapshot test for trace with system instruction."""
        trace_data = {
            "prompt": "Calculate 5*6",
            "trace_steps": ["Multiply 5 by 6"],
            "final_answer": "30",
        }

        result = render_tunix_sft_prompt(
            trace_data,
            system_instruction="You are a calculator.",
        )

        expected = """<start_of_turn>user
You are a calculator.

Calculate 5*6<end_of_turn>
<start_of_turn>model
Reasoning:
1. Multiply 5 by 6
Answer: 30<end_of_turn>"""

        assert result == expected
