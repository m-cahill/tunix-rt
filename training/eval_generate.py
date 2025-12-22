#!/usr/bin/env python3
"""Generate evaluation outputs for model comparison.

This script generates model outputs for a fixed evaluation set, enabling
pre/post training comparison.

Usage:
    # Base model (no training)
    python training/eval_generate.py \\
        --model base \\
        --eval-set training/evalsets/eval_v1.jsonl \\
        --output artifacts/training_runs/my_run/eval_before.jsonl

    # Trained model
    python training/eval_generate.py \\
        --model artifacts/training_runs/my_run/checkpoint-final \\
        --eval-set training/evalsets/eval_v1.jsonl \\
        --output artifacts/training_runs/my_run/eval_after.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def load_eval_set(eval_path: Path) -> list[dict]:
    """Load evaluation set from JSONL.

    Args:
        eval_path: Path to eval JSONL

    Returns:
        List of eval examples
    """
    examples = []

    with open(eval_path, "r") as f:
        for i, line in enumerate(f):
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Invalid JSON on line {i+1}: {e}")
                continue

    return examples


def generate_response(prompt: str, model: str, seed: int = 42) -> str:
    """Generate model response for a prompt.

    NOTE: This is a placeholder for actual model inference.
    In a real implementation, this would:
    1. Load the model checkpoint
    2. Tokenize the prompt
    3. Run inference
    4. Decode the response

    For M09, we focus on the eval loop infrastructure.

    Args:
        prompt: Input prompt
        model: Model identifier or checkpoint path
        seed: Random seed

    Returns:
        Generated response
    """
    # Placeholder response
    # Real implementation would call model.generate() here
    response = f"[Generated response for: {prompt[:50]}...]"

    return response


def create_trace_from_eval(
    eval_example: dict,
    generated_response: str,
    model: str,
) -> dict:
    """Create a trace-like record from eval output.

    Args:
        eval_example: Original eval example
        generated_response: Model's generated response
        model: Model used

    Returns:
        Trace-compatible dict for import
    """
    # Parse response to extract reasoning steps and answer
    # For now, treat whole response as answer
    # Real implementation would parse reasoning structure

    trace = {
        "trace_version": "1.0",
        "prompt": eval_example["prompt"],
        "final_answer": generated_response,
        "steps": [
            {
                "i": 0,
                "type": "reasoning",
                "content": "Model-generated reasoning step",
            }
        ],
        "meta": {
            "source": "evaluation",
            "eval_id": eval_example["id"],
            "model": model,
            "category": eval_example.get("category", "unknown"),
        },
    }

    return trace


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation outputs from a model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use ('base' or path to checkpoint)",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        required=True,
        help="Path to eval set JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for generated traces JSONL",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum examples to evaluate (for testing)",
    )

    args = parser.parse_args()

    print("🔬 Tunix RT - Evaluation Generation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Eval set: {args.eval_set}")
    print(f"Seed: {args.seed}")

    # Load eval set
    if not args.eval_set.exists():
        print(f"❌ Eval set not found: {args.eval_set}")
        sys.exit(1)

    eval_examples = load_eval_set(args.eval_set)
    print(f"✅ Loaded {len(eval_examples)} eval examples")

    if args.max_examples:
        eval_examples = eval_examples[:args.max_examples]
        print(f"   Limited to {len(eval_examples)} examples")

    # Generate responses
    print("\n🤖 Generating responses...")

    generated_traces = []

    for i, example in enumerate(eval_examples):
        print(f"   [{i+1}/{len(eval_examples)}] {example['id']}: {example['prompt'][:50]}...")

        response = generate_response(
            prompt=example["prompt"],
            model=args.model,
            seed=args.seed,
        )

        trace = create_trace_from_eval(
            eval_example=example,
            generated_response=response,
            model=args.model,
        )

        generated_traces.append(trace)

    # Save outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for trace in generated_traces:
            f.write(json.dumps(trace) + "\n")

    print(f"\n✅ Saved {len(generated_traces)} traces to: {args.output}")
    print("\nNext steps:")
    print("1. Import traces: POST to /api/traces/batch")
    print("2. Compare with other model: run eval_generate.py with different model")
    print("3. Create delta report: python training/eval_report.py ...")
    print("=" * 70)


if __name__ == "__main__":
    main()

