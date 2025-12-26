"""Shared dependencies for API routers."""

from fastapi import HTTPException, Request, status

from tunix_rt_backend.redi_client import MockRediClient, RediClient, RediClientProtocol
from tunix_rt_backend.settings import settings


async def validate_payload_size(request: Request) -> None:
    """Dependency to validate request payload size.

    Raises:
        HTTPException: 413 if payload exceeds TRACE_MAX_BYTES
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.trace_max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Payload size exceeds maximum of {settings.trace_max_bytes} bytes",
        )


def get_redi_client() -> RediClientProtocol:
    """Dependency provider for RediAI client.

    Returns MockRediClient in mock mode, RediClient in real mode.
    This allows easy testing via dependency_overrides.
    """
    if settings.rediai_mode == "mock":
        return MockRediClient(simulate_healthy=True)
    return RediClient(base_url=settings.rediai_base_url, health_path=settings.rediai_health_path)
