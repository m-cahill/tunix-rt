import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure backend root is in path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tunix_rt_backend.schemas import TunixExportRequest
from tunix_rt_backend.services.tunix_export import export_tunix_sft_jsonl
from tunix_rt_backend.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def export(dataset_key: str, output_path: str) -> None:
    logger.info(f"Exporting {dataset_key} to {output_path}...")

    try:
        engine = create_async_engine(settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with async_session() as db:
            req = TunixExportRequest(dataset_key=dataset_key, trace_ids=None)
            jsonl = await export_tunix_sft_jsonl(req, db)

            # Write to file
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(jsonl)

            logger.info(f"✅ Exported {len(jsonl.splitlines())} lines to {path}")

        await engine.dispose()

    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_key", help="Dataset key (e.g. golden-v2)")
    parser.add_argument("output_path", help="Output JSONL path")
    args = parser.parse_args()

    asyncio.run(export(args.dataset_key, args.output_path))


if __name__ == "__main__":
    main()
