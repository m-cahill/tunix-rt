#!/usr/bin/env python3
"""Seed dev-reasoning-v2 dataset.

This script generates a deterministic development dataset with 500+ traces:
- 70% reasoning traces (multi-step math, logic, verification)
- 20% synthetic/simple traces (1-2 steps, easy wins)
- 10% golden-v2 style traces (text generation patterns)

Additionally includes ~20 edge-case traces for pipeline hardening.

All traces use strict ReasoningTrace schema with steps: [{i, type, content}, ...]

Usage:
    cd backend
    uv run python tools/seed_dev_reasoning_v2.py

Output:
    Creates backend/datasets/dev-reasoning-v2/dataset.jsonl
    Creates backend/datasets/dev-reasoning-v2/manifest.json
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

SEED = 42
TOTAL_COUNT = 550  # Target: 500-800 traces

# Composition (based on M32 answers: 70/20/10 split)
REASONING_COUNT = 385  # 70%
SYNTHETIC_COUNT = 110  # 20%
GOLDEN_STYLE_COUNT = 35  # ~6% (golden-v2 style patterns)
EDGE_CASE_COUNT = 20  # Edge cases for hardening

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent / "datasets" / "dev-reasoning-v2"
DATASET_FILE = OUTPUT_DIR / "dataset.jsonl"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"

# ============================================================
# Trace Generators
# ============================================================


def generate_reasoning_traces(count: int, seed: int) -> list[dict]:
    """Generate multi-step reasoning traces.

    These are complex traces with 3-5 steps showing explicit reasoning:
    - Math word problems with decomposition
    - Percentage calculations with verification
    - Multi-step arithmetic with carry/borrow

    Args:
        count: Number of traces to generate
        seed: Random seed for reproducibility

    Returns:
        List of trace dictionaries in strict ReasoningTrace format
    """
    random.seed(seed)
    traces = []

    # Template bank for reasoning traces
    templates = [
        # Distance/speed/time problems
        {
            "prompt": "A train travels at {speed} km/h for {hours} hours. How far does it travel?",
            "steps": [
                {"type": "setup", "content": "I need to use the formula: distance = speed × time"},
                {"type": "given", "content": "Given: speed = {speed} km/h, time = {hours} hours"},
                {"type": "calculation", "content": "Distance = {speed} × {hours} = {result} km"},
                {"type": "answer", "content": "The train travels {result} km"},
            ],
            "answer": "{result} km",
            "calc": lambda v: v["speed"] * v["hours"],
        },
        # Percentage problems
        {
            "prompt": "What is {percent}% of {base}?",
            "steps": [
                {
                    "type": "setup",
                    "content": "To find a percentage, multiply the base by (percent/100)",
                },
                {"type": "calculation", "content": "({percent}/100) × {base} = {result}"},
                {
                    "type": "verification",
                    "content": "Check: {result} is approximately {percent}% of {base}",
                },
            ],
            "answer": "{result}",
            "calc": lambda v: round((v["percent"] / 100) * v["base"], 2),
        },
        # Cost per item
        {
            "prompt": "If {items} items cost ${total}, what is the cost per item?",
            "steps": [
                {
                    "type": "setup",
                    "content": "To find cost per item, divide total cost by number of items",
                },
                {"type": "formula", "content": "Cost per item = total cost ÷ number of items"},
                {"type": "calculation", "content": "${total} ÷ {items} = ${result}"},
                {"type": "answer", "content": "Each item costs ${result}"},
            ],
            "answer": "${result}",
            "calc": lambda v: round(v["total"] / v["items"], 2),
        },
        # Addition with decomposition
        {
            "prompt": "Calculate {a} + {b} step by step.",
            "steps": [
                {
                    "type": "decompose",
                    "content": "Break down: {a} = {a_tens}0 + {a_ones}, {b} = {b_tens}0 + {b_ones}",
                },
                {"type": "add_tens", "content": "Add tens: {a_tens}0 + {b_tens}0 = {tens_sum}"},
                {"type": "add_ones", "content": "Add ones: {a_ones} + {b_ones} = {ones_sum}"},
                {"type": "combine", "content": "Combine: {tens_sum} + {ones_sum} = {result}"},
            ],
            "answer": "{result}",
            "calc": lambda v: v["a"] + v["b"],
        },
        # Subtraction with borrowing
        {
            "prompt": "Calculate {a} - {b} showing your work.",
            "steps": [
                {"type": "setup", "content": "Subtracting {b} from {a}"},
                {"type": "ones", "content": "Ones digit: handle {a_ones} - {b_ones}"},
                {"type": "tens", "content": "Tens digit: handle {a_tens} - {b_tens}"},
                {"type": "result", "content": "Final result: {result}"},
            ],
            "answer": "{result}",
            "calc": lambda v: v["a"] - v["b"],
        },
        # Multiplication breakdown
        {
            "prompt": "Multiply {a} × {b} step by step.",
            "steps": [
                {"type": "setup", "content": "Breaking down {a} × {b}"},
                {"type": "partial1", "content": "{a} × {b_ones} (ones) = {partial1}"},
                {"type": "partial2", "content": "{a} × {b_tens}0 (tens) = {partial2}"},
                {"type": "sum", "content": "Add partials: {partial1} + {partial2} = {result}"},
            ],
            "answer": "{result}",
            "calc": lambda v: v["a"] * v["b"],
        },
    ]

    for i in range(count):
        template = templates[i % len(templates)]

        # Generate random values
        values = {
            "speed": random.randint(40, 120),
            "hours": random.randint(1, 8),
            "percent": random.randint(10, 90),
            "base": random.randint(100, 1000),
            "items": random.randint(2, 20),
            "total": random.randint(20, 200),
            "a": random.randint(10, 99),
            "b": random.randint(10, 99),
        }

        # Decomposition values
        values["a_tens"] = values["a"] // 10
        values["a_ones"] = values["a"] % 10
        values["b_tens"] = values["b"] // 10
        values["b_ones"] = values["b"] % 10
        values["tens_sum"] = (values["a_tens"] + values["b_tens"]) * 10
        values["ones_sum"] = values["a_ones"] + values["b_ones"]
        values["partial1"] = values["a"] * values["b_ones"]
        values["partial2"] = values["a"] * values["b_tens"] * 10

        # Calculate result
        values["result"] = template["calc"](values)

        # Build steps with indices
        steps = []
        for step_idx, step_template in enumerate(template["steps"]):
            steps.append(
                {
                    "i": step_idx,
                    "type": step_template["type"],
                    "content": step_template["content"].format(**values),
                }
            )

        trace = {
            "trace_version": "1.0",
            "prompt": template["prompt"].format(**values),
            "final_answer": str(template["answer"].format(**values)),
            "steps": steps,
            "meta": {
                "dataset": "dev-reasoning-v2",
                "generator": "seed_dev_reasoning_v2",
                "seed": SEED,
                "category": "reasoning",
                "template_id": i % len(templates),
                "index": i,
            },
        }
        traces.append(trace)

    return traces


def generate_synthetic_traces(count: int, seed: int) -> list[dict]:
    """Generate simple synthetic traces (1-2 steps).

    These are simpler traces for easy wins:
    - Basic arithmetic (single operation)
    - String reversal
    - String sorting
    - Simple logic

    Args:
        count: Number of traces to generate
        seed: Random seed for reproducibility

    Returns:
        List of trace dictionaries in strict ReasoningTrace format
    """
    random.seed(seed + 1000)  # Offset seed
    traces = []

    for i in range(count):
        task_type = i % 4

        if task_type == 0:
            # Basic arithmetic
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            op = random.choice(["+", "-", "*"])
            op_names = {"+": "addition", "-": "subtraction", "*": "multiplication"}

            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            else:
                result = a * b

            trace = {
                "trace_version": "1.0",
                "prompt": f"Calculate: {a} {op} {b}",
                "final_answer": str(result),
                "steps": [
                    {
                        "i": 0,
                        "type": "operation",
                        "content": f"Perform {op_names[op]}: {a} {op} {b}",
                    },
                    {"i": 1, "type": "result", "content": f"Result: {result}"},
                ],
                "meta": {
                    "dataset": "dev-reasoning-v2",
                    "generator": "seed_dev_reasoning_v2",
                    "seed": SEED,
                    "category": "synthetic",
                    "task_type": "arithmetic",
                    "index": i,
                },
            }

        elif task_type == 1:
            # String reversal
            words = ["hello", "world", "python", "train", "model", "tunix", "reason", "trace"]
            text = random.choice(words)
            reversed_text = text[::-1]

            trace = {
                "trace_version": "1.0",
                "prompt": f"Reverse the string: '{text}'",
                "final_answer": reversed_text,
                "steps": [
                    {
                        "i": 0,
                        "type": "process",
                        "content": f"Take each character from end to start of '{text}'",
                    },
                    {"i": 1, "type": "result", "content": f"Reversed: '{reversed_text}'"},
                ],
                "meta": {
                    "dataset": "dev-reasoning-v2",
                    "generator": "seed_dev_reasoning_v2",
                    "seed": SEED,
                    "category": "synthetic",
                    "task_type": "string_reverse",
                    "index": i,
                },
            }

        elif task_type == 2:
            # String sorting
            length = random.randint(4, 8)
            chars = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=length))
            sorted_chars = "".join(sorted(chars))

            trace = {
                "trace_version": "1.0",
                "prompt": f"Sort these characters alphabetically: {chars}",
                "final_answer": sorted_chars,
                "steps": [
                    {
                        "i": 0,
                        "type": "process",
                        "content": f"Arrange characters of '{chars}' in alphabetical order",
                    },
                    {"i": 1, "type": "result", "content": f"Sorted: {sorted_chars}"},
                ],
                "meta": {
                    "dataset": "dev-reasoning-v2",
                    "generator": "seed_dev_reasoning_v2",
                    "seed": SEED,
                    "category": "synthetic",
                    "task_type": "string_sort",
                    "index": i,
                },
            }

        else:
            # Simple logic
            categories = ["dogs", "cats", "birds", "fish"]
            properties = ["animals", "pets", "living things"]
            items = ["poodle", "siamese", "sparrow", "goldfish"]

            idx = i % len(categories)
            category = categories[idx]
            item = items[idx]
            prop = properties[0]

            prompt = (
                f"If all {category} are {prop}, and this is a {item} "
                f"(which is a {category[:-1]}), is it {prop[:-1]}?"
            )
            deduction = f"A {item} is a type of {category[:-1]}, so it is an {prop[:-1]}"
            trace = {
                "trace_version": "1.0",
                "prompt": prompt,
                "final_answer": "Yes",
                "steps": [
                    {"i": 0, "type": "premise", "content": f"Given: All {category} are {prop}"},
                    {"i": 1, "type": "deduction", "content": deduction},
                ],
                "meta": {
                    "dataset": "dev-reasoning-v2",
                    "generator": "seed_dev_reasoning_v2",
                    "seed": SEED,
                    "category": "synthetic",
                    "task_type": "logic",
                    "index": i,
                },
            }

        traces.append(trace)

    return traces


def generate_golden_style_traces(count: int, seed: int) -> list[dict]:
    """Generate golden-v2 style traces (text generation patterns).

    These mimic the content patterns from golden-v2:
    - Text repetition tasks
    - Word counting
    - Simple transformations

    Still in raw ReasoningTrace format (not pre-rendered SFT).

    Args:
        count: Number of traces to generate
        seed: Random seed for reproducibility

    Returns:
        List of trace dictionaries in strict ReasoningTrace format
    """
    random.seed(seed + 2000)  # Offset seed
    traces = []

    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]

    for i in range(count):
        task_type = i % 3

        if task_type == 0:
            # Text repetition (like golden-v2)
            word = random.choice(words)
            times = random.randint(2, 5)
            result = " ".join([word] * times)

            trace = {
                "trace_version": "1.0",
                "prompt": f"Repeat '{word}' {times} times",
                "final_answer": result,
                "steps": [
                    {"i": 0, "type": "task", "content": f"Repeating '{word}' {times} times"},
                    {"i": 1, "type": "output", "content": result},
                ],
                "meta": {
                    "dataset": "dev-reasoning-v2",
                    "generator": "seed_dev_reasoning_v2",
                    "seed": SEED,
                    "category": "golden_style",
                    "task_type": "text_repeat",
                    "index": i,
                },
            }

        elif task_type == 1:
            # Word counting
            word = random.choice(words)
            sentence = f"The {word} is red and the {word} is also green"
            word_count = sentence.lower().count(word)

            trace = {
                "trace_version": "1.0",
                "prompt": f"How many times does '{word}' appear in: \"{sentence}\"",
                "final_answer": str(word_count),
                "steps": [
                    {
                        "i": 0,
                        "type": "search",
                        "content": f"Searching for '{word}' in the sentence",
                    },
                    {"i": 1, "type": "count", "content": f"Found {word_count} occurrences"},
                ],
                "meta": {
                    "dataset": "dev-reasoning-v2",
                    "generator": "seed_dev_reasoning_v2",
                    "seed": SEED,
                    "category": "golden_style",
                    "task_type": "word_count",
                    "index": i,
                },
            }

        else:
            # Uppercase transformation
            word = random.choice(words)

            trace = {
                "trace_version": "1.0",
                "prompt": f"Convert '{word}' to uppercase",
                "final_answer": word.upper(),
                "steps": [
                    {
                        "i": 0,
                        "type": "transform",
                        "content": f"Converting each character of '{word}' to uppercase",
                    },
                    {"i": 1, "type": "result", "content": word.upper()},
                ],
                "meta": {
                    "dataset": "dev-reasoning-v2",
                    "generator": "seed_dev_reasoning_v2",
                    "seed": SEED,
                    "category": "golden_style",
                    "task_type": "uppercase",
                    "index": i,
                },
            }

        traces.append(trace)

    return traces


def generate_edge_case_traces(count: int, seed: int) -> list[dict]:
    """Generate edge-case traces for pipeline hardening.

    These test edge cases:
    - Long prompts
    - Special characters
    - Whitespace handling
    - Single-step traces (minimum valid)
    - Maximum step traces

    Args:
        count: Number of traces to generate
        seed: Random seed for reproducibility

    Returns:
        List of trace dictionaries in strict ReasoningTrace format
    """
    random.seed(seed + 3000)  # Offset seed
    traces = []

    edge_cases = [
        # Minimal valid trace (1 step)
        {
            "prompt": "What is 1 + 1?",
            "final_answer": "2",
            "steps": [{"i": 0, "type": "answer", "content": "1 + 1 = 2"}],
            "case": "minimal_steps",
        },
        # Long prompt
        {
            "prompt": "This is a very long prompt. " * 50 + "What is 5?",
            "final_answer": "5",
            "steps": [
                {
                    "i": 0,
                    "type": "parse",
                    "content": "Extracting the question from the long prompt",
                },
                {"i": 1, "type": "answer", "content": "The answer is 5"},
            ],
            "case": "long_prompt",
        },
        # Special characters in prompt
        {
            "prompt": "What is the result of 'hello' + \"world\" with <special> & chars?",
            "final_answer": "helloworld",
            "steps": [
                {"i": 0, "type": "parse", "content": "Concatenating 'hello' and 'world'"},
                {"i": 1, "type": "result", "content": "Result: helloworld"},
            ],
            "case": "special_chars",
        },
        # Numbers only
        {
            "prompt": "12345",
            "final_answer": "12345",
            "steps": [{"i": 0, "type": "echo", "content": "Echoing input: 12345"}],
            "case": "numbers_only",
        },
        # Unicode characters
        {
            "prompt": "Translate 'hello' to emoji: 👋",
            "final_answer": "👋",
            "steps": [
                {"i": 0, "type": "translate", "content": "Hello translates to wave emoji"},
                {"i": 1, "type": "output", "content": "👋"},
            ],
            "case": "unicode",
        },
        # Whitespace in answer
        {
            "prompt": "Add spaces between each letter of 'abc'",
            "final_answer": "a b c",
            "steps": [
                {"i": 0, "type": "process", "content": "Inserting space between each character"},
                {"i": 1, "type": "result", "content": "a b c"},
            ],
            "case": "whitespace_answer",
        },
        # Empty-ish prompt (minimal)
        {
            "prompt": "?",
            "final_answer": "Unknown question",
            "steps": [
                {
                    "i": 0,
                    "type": "error",
                    "content": "Cannot parse minimal prompt, returning unknown",
                }
            ],
            "case": "minimal_prompt",
        },
        # Many steps
        {
            "prompt": "Count from 1 to 5",
            "final_answer": "1, 2, 3, 4, 5",
            "steps": [
                {"i": 0, "type": "count", "content": "1"},
                {"i": 1, "type": "count", "content": "2"},
                {"i": 2, "type": "count", "content": "3"},
                {"i": 3, "type": "count", "content": "4"},
                {"i": 4, "type": "count", "content": "5"},
                {"i": 5, "type": "summary", "content": "Counted from 1 to 5"},
            ],
            "case": "many_steps",
        },
        # Punctuation heavy
        {
            "prompt": "What is... 2 + 2???",
            "final_answer": "4!",
            "steps": [
                {"i": 0, "type": "parse", "content": "Removing extra punctuation from '2 + 2'"},
                {"i": 1, "type": "calc", "content": "2 + 2 = 4"},
            ],
            "case": "punctuation",
        },
        # Newlines in content
        {
            "prompt": "Format this as a list: apple, banana, cherry",
            "final_answer": "1. apple\n2. banana\n3. cherry",
            "steps": [
                {
                    "i": 0,
                    "type": "format",
                    "content": "Converting comma-separated values to numbered list",
                },
                {"i": 1, "type": "output", "content": "1. apple\n2. banana\n3. cherry"},
            ],
            "case": "newlines",
        },
    ]

    # Use edge cases, cycling if needed
    for i in range(count):
        case = edge_cases[i % len(edge_cases)]
        trace = {
            "trace_version": "1.0",
            "prompt": case["prompt"],
            "final_answer": case["final_answer"],
            "steps": case["steps"],
            "meta": {
                "dataset": "dev-reasoning-v2",
                "generator": "seed_dev_reasoning_v2",
                "seed": SEED,
                "category": "edge_case",
                "case_type": case["case"],
                "index": i,
            },
        }
        traces.append(trace)

    return traces


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Generate and save dev-reasoning-v2 dataset."""
    print(f"Generating dev-reasoning-v2 dataset (seed={SEED})...")
    print(f"  Reasoning traces: {REASONING_COUNT}")
    print(f"  Synthetic traces: {SYNTHETIC_COUNT}")
    print(f"  Golden-style traces: {GOLDEN_STYLE_COUNT}")
    print(f"  Edge-case traces: {EDGE_CASE_COUNT}")
    print(f"  Total: {REASONING_COUNT + SYNTHETIC_COUNT + GOLDEN_STYLE_COUNT + EDGE_CASE_COUNT}")

    # Generate all traces
    reasoning_traces = generate_reasoning_traces(REASONING_COUNT, SEED)
    synthetic_traces = generate_synthetic_traces(SYNTHETIC_COUNT, SEED)
    golden_traces = generate_golden_style_traces(GOLDEN_STYLE_COUNT, SEED)
    edge_traces = generate_edge_case_traces(EDGE_CASE_COUNT, SEED)

    all_traces = reasoning_traces + synthetic_traces + golden_traces + edge_traces

    # Shuffle with deterministic seed
    random.seed(SEED)
    random.shuffle(all_traces)

    print(f"\nTotal traces generated: {len(all_traces)}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    print(f"\nWriting dataset to {DATASET_FILE}...")
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for trace in all_traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    # Create manifest
    manifest = {
        "dataset_key": "dev-reasoning-v2",
        "build_id": "dev-reasoning-v2-static",
        "dataset_name": "dev-reasoning",
        "dataset_version": "v2",
        "dataset_schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filters": {"type": "mixed"},
        "selection_strategy": "static",
        "seed": SEED,
        "trace_ids": [],  # Static dataset, no DB IDs
        "trace_count": len(all_traces),
        "stats": {
            "trace_count": len(all_traces),
            "composition": {
                "reasoning": REASONING_COUNT,
                "synthetic": SYNTHETIC_COUNT,
                "golden_style": GOLDEN_STYLE_COUNT,
                "edge_case": EDGE_CASE_COUNT,
            },
            "composition_pct": {
                "reasoning": f"{REASONING_COUNT / len(all_traces) * 100:.1f}%",
                "synthetic": f"{SYNTHETIC_COUNT / len(all_traces) * 100:.1f}%",
                "golden_style": f"{GOLDEN_STYLE_COUNT / len(all_traces) * 100:.1f}%",
                "edge_case": f"{EDGE_CASE_COUNT / len(all_traces) * 100:.1f}%",
            },
            "note": "Generated via seed_dev_reasoning_v2.py with strict ReasoningTrace schema",
        },
        "session_id": None,
        "parent_dataset_id": None,
        "training_run_id": None,
        "provenance": {
            "source": "seed_dev_reasoning_v2.py",
            "purpose": "Scaled development dataset (500+ traces) for training/validation",
            "deterministic": True,
            "schema": "ReasoningTrace with steps: [{i, type, content}, ...]",
        },
    }

    print(f"Writing manifest to {MANIFEST_FILE}...")
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\n[OK] dev-reasoning-v2 generation complete!")
    print(f"   Dataset: {DATASET_FILE}")
    print(f"   Manifest: {MANIFEST_FILE}")
    print(f"   Total traces: {len(all_traces)}")


if __name__ == "__main__":
    main()
