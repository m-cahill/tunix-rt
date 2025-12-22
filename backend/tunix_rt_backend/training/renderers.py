"""Prompt renderers for converting traces to training formats.

This module provides functions to convert reasoning traces into formats
suitable for training with Tunix (SFT, GRPO, etc.).

Key Functions:
- render_tunix_sft_prompt: High-level trace → SFT prompt renderer
- render_gemma_turn: Low-level turn formatting with control tokens
- render_gemma_user_turn: Helper for user turns
- render_gemma_model_turn: Helper for model turns
- apply_system_instruction: Embed system prompt in first user turn (Gemma IT pattern)
"""

from typing import Any

# Gemma IT control tokens (following official specification)
START_OF_TURN = "<start_of_turn>"
END_OF_TURN = "<end_of_turn>"


def render_gemma_turn(role: str, content: str) -> str:
    """Render a single Gemma chat turn with control tokens.

    This is a low-level helper for building Gemma IT formatted turns.
    For most use cases, use render_gemma_user_turn or render_gemma_model_turn.

    Args:
        role: Turn role ('user' or 'model')
        content: Turn content (already formatted)

    Returns:
        Formatted turn with control tokens

    Example:
        >>> render_gemma_turn("user", "Hello!")
        '<start_of_turn>user\\nHello!<end_of_turn>'
    """
    return f"{START_OF_TURN}{role}\n{content}{END_OF_TURN}"


def render_gemma_user_turn(prompt: str) -> str:
    """Render a user turn in Gemma IT format.

    Args:
        prompt: User's question or instruction

    Returns:
        Formatted user turn

    Example:
        >>> render_gemma_user_turn("What is 2+2?")
        '<start_of_turn>user\\nWhat is 2+2?<end_of_turn>'
    """
    return render_gemma_turn("user", prompt)


def render_gemma_model_turn(response: str) -> str:
    """Render a model turn in Gemma IT format.

    Args:
        response: Model's response (reasoning + answer)

    Returns:
        Formatted model turn

    Example:
        >>> render_gemma_model_turn("The answer is 4")
        '<start_of_turn>model\\nThe answer is 4<end_of_turn>'
    """
    return render_gemma_turn("model", response)


def apply_system_instruction(user_prompt: str, system_instruction: str | None = None) -> str:
    """Apply system instruction to user prompt (Gemma IT pattern).

    Gemma IT models do not have a separate 'system' role, so system-like
    instructions must be embedded in the first user turn.

    Args:
        user_prompt: The user's actual question/task
        system_instruction: Optional system-level instruction to prepend

    Returns:
        Combined prompt (system + user if system provided, else just user)

    Example:
        >>> apply_system_instruction(
        ...     "What is 2+2?",
        ...     "You are a helpful math tutor."
        ... )
        'You are a helpful math tutor.\\n\\nWhat is 2+2?'
    """
    if system_instruction:
        return f"{system_instruction}\n\n{user_prompt}"
    return user_prompt


def render_reasoning_steps(steps: list[str]) -> str:
    """Render reasoning steps as a numbered list.

    Args:
        steps: List of reasoning step contents

    Returns:
        Formatted reasoning text (empty string if no steps)

    Example:
        >>> render_reasoning_steps(["Parse input", "Compute"])
        'Reasoning:\\n1. Parse input\\n2. Compute\\n'
    """
    if not steps:
        return ""

    reasoning_text = "Reasoning:\n"
    for i, step in enumerate(steps, 1):
        reasoning_text += f"{i}. {step}\n"

    return reasoning_text


def render_tunix_sft_prompt(
    trace_data: dict[str, Any], system_instruction: str | None = None
) -> str:
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
        system_instruction: Optional system-level instruction (embedded in user turn)

    Returns:
        Formatted prompt string with user query and model response

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
    user_prompt = trace_data.get("prompt", "")
    trace_steps = trace_data.get("trace_steps", [])
    final_answer = trace_data.get("final_answer", "")

    # Apply system instruction if provided (Gemma IT pattern)
    full_prompt = apply_system_instruction(user_prompt, system_instruction)

    # Build reasoning section from trace steps
    reasoning_text = render_reasoning_steps(trace_steps)

    # Build model response
    model_response = f"{reasoning_text}Answer: {final_answer}"

    # Format as Gemma chat template
    user_turn = render_gemma_user_turn(full_prompt)
    model_turn = render_gemma_model_turn(model_response)

    return f"{user_turn}\n{model_turn}"


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
