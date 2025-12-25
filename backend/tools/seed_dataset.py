import argparse
import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tunix_rt_backend.db.models import Trace
from tunix_rt_backend.schemas.dataset import DatasetBuildRequest
from tunix_rt_backend.services.datasets_builder import build_dataset_manifest
from tunix_rt_backend.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# v1 Golden Traces (original)
GOLDEN_V1_TRACES = [
    {
        "prompt": "What is 2+2?",
        "steps": [{"content": "2+2 is 4"}],
        "final_answer": "4",
        "meta": {"golden": "v1", "case": "basic_math"},
    },
    {
        "prompt": "Capital of France?",
        "steps": [{"content": "France capital is Paris"}],
        "final_answer": "Paris",
        "meta": {"golden": "v1", "case": "knowledge"},
    },
    {
        "prompt": "  Normalize  Whitespace  ",
        "steps": [],
        "final_answer": "normalize whitespace",
        "meta": {"golden": "v1", "case": "normalization"},
    },
    {
        "prompt": "Answer Format Test",
        "steps": [],
        "final_answer": "Answer: Correct",
        "meta": {"golden": "v1", "case": "format"},
    },
    {
        "prompt": "Is this correct?",
        "steps": [],
        "final_answer": "Yes",
        "meta": {"golden": "v1", "case": "boolean"},
    },
]


def generate_golden_v2(count: int = 100) -> list[dict[str, Any]]:
    """Generate deterministic synthetic traces for v2."""
    traces = []

    # Deterministic generation
    rng = random.Random(42)

    # 1. Math sequence
    for i in range(count // 2):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        traces.append(
            {
                "trace_version": "1.0",
                "prompt": f"What is {a} + {b}?",
                "steps": [
                    {"i": 0, "type": "calculation", "content": f"Calculating {a} + {b} = {a + b}"}
                ],
                "final_answer": str(a + b),
                "meta": {"golden": "v2", "case": "math_gen", "idx": i},
            }
        )

    # 2. Text repetition/manipulation
    words = ["apple", "banana", "cherry", "date", "elderberry"]
    for i in range(count // 2, count):
        w = rng.choice(words)
        n = rng.randint(2, 5)
        traces.append(
            {
                "trace_version": "1.0",
                "prompt": f"Repeat '{w}' {n} times",
                "steps": [{"i": 0, "type": "reasoning", "content": f"Repeating {w} {n} times"}],
                "final_answer": " ".join([w] * n),
                "meta": {"golden": "v2", "case": "text_gen", "idx": i},
            }
        )

    return traces


async def seed(dataset_key: str, count: int) -> None:
    dataset_name, dataset_version = dataset_key.split("-", 1)

    # Select Data Source
    traces_data: list[dict[str, Any]]
    if dataset_key == "golden-v1":
        traces_data = GOLDEN_V1_TRACES  # type: ignore[assignment]
        filter_tag = "golden"
        filter_val = "v1"
    elif dataset_key == "golden-v2":
        traces_data = generate_golden_v2(count)
        filter_tag = "golden"
        filter_val = "v2"
    else:
        logger.error(f"Unknown dataset key: {dataset_key}")
        return

    try:
        engine = create_async_engine(settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with async_session() as db:
            logger.info(f"Connecting to database at {settings.database_url}...")
            await db.execute(select(1))

            logger.info(f"Seeding {len(traces_data)} traces for {dataset_key}...")

            # Insert traces
            inserted_count = 0
            for item in traces_data:
                # Add default meta if not present or ensure filter tag exists
                if "meta" not in item:
                    item["meta"] = {}
                item["meta"][filter_tag] = filter_val

                trace = Trace(
                    trace_version="1.0", payload=item, created_at=datetime.now(timezone.utc)
                )
                db.add(trace)
                inserted_count += 1

            await db.commit()
            logger.info(f"Inserted {inserted_count} traces.")

            # Build Manifest
            limit_val = count if count > 0 else 10000
            req = DatasetBuildRequest(
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                selection_strategy="latest",
                limit=limit_val,
                filters={filter_tag: filter_val},
            )

            try:
                key, build_id, final_count, path = await build_dataset_manifest(req, db)
                logger.info(f"✅ Created dataset {key} at {path} with {final_count} traces.")
            except Exception as e:
                logger.error(f"❌ Failed to build manifest: {e}")

        await engine.dispose()

    except Exception as e:
        logger.error(f"Database error: {e}")
        logger.error("Ensure Postgres is running.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed datasets for Tunix RT")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset key (e.g., golden-v2)")
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of traces to generate (for procedural datasets)",
    )

    args = parser.parse_args()

    asyncio.run(seed(args.dataset, args.count))


if __name__ == "__main__":
    main()
