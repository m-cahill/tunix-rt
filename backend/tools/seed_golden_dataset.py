import asyncio
import sys
from pathlib import Path

# Ensure backend root is in path
sys.path.append(str(Path(__file__).parent.parent))

from tools.seed_dataset import seed as generic_seed


async def seed() -> None:
    """Legacy wrapper for seeding golden-v1 dataset."""
    await generic_seed("golden-v1", count=5)


if __name__ == "__main__":
    asyncio.run(seed())
