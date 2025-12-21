"""SQLAlchemy base configuration for async database operations."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from tunix_rt_backend.settings import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
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
