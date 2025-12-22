"""Prompt renderers for converting traces to training formats.

This module provides functions to convert reasoning traces into formats
suitable for training with Tunix (SFT, GRPO, etc.).
"""

from typing import Any


def render_tunix_sft_prompt(trace_data: dict[str, Any]) -> str:
    """Render a trace into Tunix SFT-ready prompt format.

    Converts a trace with prompt, steps, and final_answer into a formatted
    prompt string that can be used with Tunix SFT training. The format follows
    the Gemma chat template structure used in Tunix examples.

    Args:
        trace_data: Trace data dict with keys:
            - prompt (str): Original question/task
            - trace_steps (list[str]): List of reasoning step contents
            - final_answer (str): Final answer/conclusion
            - metadata (dict, optional): Additional metadata

    Returns:
        Formatted prompt string with system message, user query, and model response

    Example:
        >>> trace = {
        ...     "prompt": "What is 2+2?",
        ...     "trace_steps": ["Add 2 and 2"],
        ...     "final_answer": "4"
        ... }
        >>> prompt = render_tunix_sft_prompt(trace)
        >>> "<start_of_turn>user" in prompt
        True
    """
    # Extract components
    prompt = trace_data.get("prompt", "")
    trace_steps = trace_data.get("trace_steps", [])
    final_answer = trace_data.get("final_answer", "")

    # Build reasoning section from trace steps
    reasoning_text = ""
    if trace_steps:
        reasoning_text = "Reasoning:\n"
        for i, step in enumerate(trace_steps, 1):
            reasoning_text += f"{i}. {step}\n"

    # Format as Gemma chat template (following Tunix Kaggle examples)
    # This format is compatible with Gemma models and Tunix training
    formatted_prompt = f"""<start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
{reasoning_text}
Answer: {final_answer}<end_of_turn>"""

    return formatted_prompt


def render_trace_for_training(
    trace_data: dict[str, Any],
    format_type: str = "tunix_sft",
) -> str:
    """Render a trace for training using the specified format.

    This is a dispatcher function that routes to the appropriate renderer
    based on the format type. Currently supports only 'tunix_sft' but can
    be extended for other formats in the future.

    Args:
        trace_data: Trace data dictionary
        format_type: Format type (default: 'tunix_sft')

    Returns:
        Formatted training prompt

    Raises:
        ValueError: If format_type is not supported
    """
    if format_type == "tunix_sft":
        return render_tunix_sft_prompt(trace_data)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

