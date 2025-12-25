import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tunix_rt_backend.db.models import Trace
from tunix_rt_backend.schemas.dataset import DatasetBuildRequest
from tunix_rt_backend.services.datasets_builder import build_dataset_manifest
from tunix_rt_backend.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_TRACES = [
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


async def seed() -> None:
    try:
        engine = create_async_engine(settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with async_session() as db:
            logger.info(f"Connecting to database at {settings.database_url}...")
            # Simple check
            await db.execute(select(1))
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        logger.error(
            "Ensure Postgres is running. If using Docker, use port 5433 (DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/postgres)"
        )
        return

    async with async_session() as db:
        logger.info("Seeding golden traces...")

        # Insert traces
        trace_ids = []
        for item in GOLDEN_TRACES:
            trace = Trace(trace_version="1.0", payload=item, created_at=datetime.now(timezone.utc))
            db.add(trace)
            # Flush to get ID if needed, though we rely on filters later
            await db.flush()
            trace_ids.append(trace.id)

        await db.commit()
        logger.info(f"Inserted {len(trace_ids)} traces.")

        # Build Manifest
        req = DatasetBuildRequest(
            dataset_name="golden",
            dataset_version="v1",
            selection_strategy="latest",
            limit=10,
            filters={"golden": "v1"},
        )

        try:
            key, build_id, count, path = await build_dataset_manifest(req, db)
            logger.info(f"Created dataset {key} at {path} with {count} traces.")
        except Exception as e:
            logger.error(f"Failed to build manifest: {e}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())
