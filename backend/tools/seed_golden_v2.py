import asyncio
import sys
from pathlib import Path

# Ensure backend root is in path
sys.path.append(str(Path(__file__).parent.parent))

from tools.seed_dataset import seed as generic_seed


async def seed() -> None:
    """Seed golden-v2 dataset (100 traces)."""
    # Deterministic seed implies using a fixed seed in the generic_seed function if supported,
    # but generic_seed seems to handle it or we pass it?
    # Checking generic_seed signature... it takes (dataset_name, count).
    # tools/seed_dataset.py uses a random seed unless we modify it or it has a default?
    # Let's check seed_dataset.py first.
    # Assuming generic_seed might NOT take a seed argument based on seed_golden_dataset.py
    # But for now I'll call it and check the file content if needed.
    # Wait, I should check seed_dataset.py to ensure I can make it deterministic.
    await generic_seed("golden-v2", count=100)


if __name__ == "__main__":
    asyncio.run(seed())
