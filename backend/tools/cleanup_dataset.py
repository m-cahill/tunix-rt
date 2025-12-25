import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure backend root is in path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tunix_rt_backend.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def cleanup(dataset_key: str) -> None:
    dataset_name, dataset_version = dataset_key.split("-", 1)

    # Logic: delete traces where meta->>golden == version?
    # Or meta->>golden == 'v2' for golden-v2.
    # The seeder sets meta={"golden": "v2" ...}

    if "golden" not in dataset_name:
        logger.error("Only golden datasets supported for cleanup currently.")
        return

    logger.info(f"Cleaning up {dataset_key} (version={dataset_version})...")

    try:
        engine = create_async_engine(settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with async_session() as db:
            # We use raw SQL for JSONB query or sqlalchemy casting.
            # Using text for simplicity or hybrid property if available (it's not).
            # Trace.payload is JSONB.
            # meta is inside payload?
            # seeder: trace = Trace(payload=item...)
            # item["meta"] = ...
            # So payload->'meta'->>'golden' = 'v2'

            from sqlalchemy import text

            stmt = text("DELETE FROM traces WHERE payload->'meta'->>'golden' = :version")
            result = await db.execute(stmt, {"version": dataset_version})
            await db.commit()

            logger.info(f"✅ Deleted {result.rowcount} traces.")

        await engine.dispose()

    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_key", help="Dataset key (e.g. golden-v2)")
    args = parser.parse_args()

    asyncio.run(cleanup(args.dataset_key))


if __name__ == "__main__":
    main()
