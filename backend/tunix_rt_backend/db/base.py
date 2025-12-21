"""SQLAlchemy base configuration for async database operations."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from tunix_rt_backend.settings import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Create async engine with pool configuration
# Pool settings are validated in settings.py:
# - pool_size: 1-50 (default 5)
# - max_overflow: 0-50 (default 10)
# - pool_timeout: 1-300 seconds (default 30)
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
)

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """Dependency for getting async database sessions.

    Yields an AsyncSession that is automatically closed after use.
    This is a FastAPI dependency with yield.

    Yields:
        AsyncSession: Database session for the request
    """
    async with async_session_maker() as session:
        yield session
