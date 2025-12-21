"""FastAPI application with health endpoints."""

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tunix_rt_backend.redi_client import MockRediClient, RediClient, RediClientProtocol
from tunix_rt_backend.settings import settings

app = FastAPI(
    title="Tunix RT Backend",
    description="Reasoning-Trace backend with RediAI integration",
    version="0.1.0",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple TTL cache for RediAI health
_redi_health_cache: dict[str, tuple[dict[str, str], datetime]] = {}


def get_redi_client() -> RediClientProtocol:
    """Dependency provider for RediAI client.

    Returns MockRediClient in mock mode, RediClient in real mode.
    This allows easy testing via dependency_overrides.
    """
    if settings.rediai_mode == "mock":
        return MockRediClient(simulate_healthy=True)
    return RediClient(base_url=settings.rediai_base_url, health_path=settings.rediai_health_path)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Check tunix-rt application health.

    Returns:
        {"status": "healthy"}
    """
    return {"status": "healthy"}


@app.get("/api/redi/health")
async def redi_health(
    redi_client: Annotated[RediClientProtocol, Depends(get_redi_client)],
) -> dict[str, str]:
    """Check RediAI integration health with TTL caching.

    In mock mode: always returns healthy.
    In real mode: probes actual RediAI instance (with 30s cache).

    Args:
        redi_client: Injected RediAI client (real or mock)

    Returns:
        {"status": "healthy"} if RediAI is reachable
        {"status": "down", "error": "..."} if RediAI is unreachable
    """
    # Check cache
    now = datetime.now(timezone.utc)
    cache_ttl = timedelta(seconds=settings.rediai_health_cache_ttl_seconds)

    if "redi_health" in _redi_health_cache:
        cached_result, cached_time = _redi_health_cache["redi_health"]
        if now - cached_time < cache_ttl:
            return cached_result

    # Cache miss or expired - fetch fresh result
    result = await redi_client.health()
    _redi_health_cache["redi_health"] = (result, now)
    return result
