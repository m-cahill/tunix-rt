#!/usr/bin/env python3
"""Seed dev-reasoning-v1 dataset.

This script generates a deterministic development dataset with:
- 70% reasoning-trace style items (decomposition, tool-less reasoning, verification)
- 30% structured synthetic tasks (arithmetic, logic, string transforms)

Total: ~200 items for development/validation purposes.
"""

import asyncio
import random
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from tunix_rt_backend.db.base import async_session
from tunix_rt_backend.schemas import ReasoningTrace
from tunix_rt_backend.services.traces_batch import create_traces_batch

SEED = 42
REASONING_COUNT = 140  # 70%
SYNTHETIC_COUNT = 60  # 30%


def generate_reasoning_traces(count: int, seed: int) -> list[dict]:
    """Generate reasoning-trace style items."""
    random.seed(seed)
    traces = []

    reasoning_templates = [
        {
            "prompt": (
                "If a train travels at {speed} km/h for {hours} hours, how far does it travel?"
            ),
            "trace_steps": [
                "I need to use the formula: distance = speed × time",
                "Given: speed = {speed} km/h, time = {hours} hours",
                "Calculation: {speed} × {hours} = {distance} km",
            ],
            "final_answer": "{distance} km",
        },
        {
            "prompt": "What is {num}% of {base}?",
            "trace_steps": [
                "To find a percentage, I multiply the base by the percentage divided by 100",
                "Formula: ({num}/100) × {base}",
                "Calculation: {result}",
            ],
            "final_answer": "{result}",
        },
        {
            "prompt": "If {items} cost ${total}, what is the cost per item?",
            "trace_steps": [
                "To find cost per item, divide total cost by number of items",
                "Formula: total cost ÷ number of items",
                "Calculation: ${total} ÷ {items} = ${per_item}",
            ],
            "final_answer": "${per_item} per item",
        },
    ]

    for i in range(count):
        template = random.choice(reasoning_templates)

        # Generate random values
        values = {
            "speed": random.randint(40, 120),
            "hours": random.randint(1, 8),
            "num": random.randint(10, 90),
            "base": random.randint(100, 1000),
            "items": random.randint(2, 20),
            "total": random.randint(20, 200),
        }

        # Calculate results
        values["distance"] = values["speed"] * values["hours"]
        values["result"] = round((values["num"] / 100) * values["base"], 2)
        values["per_item"] = round(values["total"] / values["items"], 2)

        # Fill template
        prompt = template["prompt"].format(**values)
        trace_steps = [step.format(**values) for step in template["trace_steps"]]
        final_answer = template["final_answer"].format(**values)

        trace = {
            "prompt": prompt,
            "trace_steps": trace_steps,
            "final_answer": final_answer,
            "trace_version": "1.0",
            "meta": {
                "source": "dev-reasoning-v1",
                "category": "reasoning",
                "template_id": f"reasoning_{i % len(reasoning_templates)}",
                "seed": seed + i,
            },
        }
        traces.append(trace)

    return traces


def generate_synthetic_traces(count: int, seed: int) -> list[dict]:
    """Generate structured synthetic tasks."""
    random.seed(seed + 1000)  # Offset seed
    traces = []

    synthetic_tasks = [
        {
            "type": "arithmetic",
            "prompt": "Calculate: {a} {op} {b}",
            "steps": ["Perform {op_name} operation", "Result: {result}"],
            "answer": "{result}",
        },
        {
            "type": "string_transform",
            "prompt": "Reverse the string: '{text}'",
            "steps": ["Take each character from end to start", "Result: '{reversed}'"],
            "answer": "'{reversed}'",
        },
        {
            "type": "logic",
            "prompt": "If all {category} are {property1}, and this is a {item}, is it {property1}?",
            "steps": ["Check if {item} is a {category}", "If yes, then it is {property1}"],
            "answer": "{answer}",
        },
    ]

    for i in range(count):
        task = random.choice(synthetic_tasks)

        if task["type"] == "arithmetic":
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

            values = {"a": a, "b": b, "op": op, "op_name": op_names[op], "result": result}

        elif task["type"] == "string_transform":
            words = ["hello", "world", "python", "data", "model", "train"]
            text = random.choice(words)
            reversed_text = text[::-1]
            values = {"text": text, "reversed": reversed_text}

        else:  # logic
            categories = ["dogs", "cats", "birds"]
            properties = ["animals", "pets", "living things"]
            items = ["poodle", "siamese", "sparrow"]

            idx = random.randint(0, 2)
            values = {
                "category": categories[idx],
                "property1": properties[0],
                "item": items[idx],
                "answer": "Yes",
            }

        prompt = task["prompt"].format(**values)
        trace_steps = [step.format(**values) for step in task["steps"]]
        final_answer = task["answer"].format(**values)

        trace = {
            "prompt": prompt,
            "trace_steps": trace_steps,
            "final_answer": final_answer,
            "trace_version": "1.0",
            "meta": {
                "source": "dev-reasoning-v1",
                "category": "synthetic",
                "task_type": task["type"],
                "seed": seed + 1000 + i,
            },
        }
        traces.append(trace)

    return traces


async def main():
    """Generate and persist dev-reasoning-v1 dataset."""
    print(f"Generating dev-reasoning-v1 dataset (seed={SEED})...")
    print(f"  Reasoning traces: {REASONING_COUNT}")
    print(f"  Synthetic traces: {SYNTHETIC_COUNT}")
    print(f"  Total: {REASONING_COUNT + SYNTHETIC_COUNT}")

    # Generate traces
    reasoning_traces = generate_reasoning_traces(REASONING_COUNT, SEED)
    synthetic_traces = generate_synthetic_traces(SYNTHETIC_COUNT, SEED)
    all_traces_data = reasoning_traces + synthetic_traces

    # Shuffle with deterministic seed
    random.seed(SEED)
    random.shuffle(all_traces_data)

    # Validate with Pydantic
    traces = [ReasoningTrace(**trace_data) for trace_data in all_traces_data]

    print(f"\nPersisting {len(traces)} traces to database...")

    # Persist to DB
    async with async_session() as db:
        result = await create_traces_batch(traces, db)
        print(f"✅ Created {result.created_count} traces")
        print(f"   Trace IDs: {[str(t.id)[:8] for t in result.traces[:5]]}... (showing first 5)")

    print("\nNow build the dataset:")
    print("  POST /api/datasets/build")
    print("  Body: {")
    print('    "dataset_name": "dev-reasoning",')
    print('    "dataset_version": "v1",')
    print('    "filters": {"source": "dev-reasoning-v1"},')
    print('    "limit": 1000,')
    print('    "selection_strategy": "latest"')
    print("  }")

    print("\n✅ dev-reasoning-v1 seed complete!")


if __name__ == "__main__":
    asyncio.run(main())
