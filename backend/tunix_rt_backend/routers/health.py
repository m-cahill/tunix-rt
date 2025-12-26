"""Health check and metrics endpoints.

Domain: System health and observability

Primary endpoints:
- GET /api/health: Quick health check (returns healthy/unhealthy)
- GET /api/redi/health: RediAI integration status with TTL cache
- GET /metrics: Prometheus metrics (run counts, latency, errors)

Cross-cutting concerns:
- RediAI health cached with 30s TTL to reduce upstream load
- Metrics exposed in Prometheus format for monitoring
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from tunix_rt_backend.dependencies import get_redi_client
from tunix_rt_backend.redi_client import RediClientProtocol
from tunix_rt_backend.settings import settings

router = APIRouter()

# Simple TTL cache for RediAI health
_redi_health_cache: dict[str, tuple[dict[str, str], datetime]] = {}


@router.get("/api/health")
async def health() -> dict[str, str]:
    """Check tunix-rt application health.

    Returns:
        {"status": "healthy"}
    """
    return {"status": "healthy"}


@router.get("/api/redi/health")
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
        # Implicit else: cache expired, fall through to fetch
    # Implicit else: no cache entry, fall through to fetch

    # Cache miss or expired - fetch fresh result
    result = await redi_client.health()
    _redi_health_cache["redi_health"] = (result, now)
    return result


@router.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics (M15)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
