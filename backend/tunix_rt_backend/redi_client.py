"""RediAI client for health checks and integration."""

from typing import Protocol

import httpx


class RediClientProtocol(Protocol):
    """Protocol defining RediAI client interface."""

    async def health(self) -> dict[str, str]:
        """Check RediAI health status.
        
        Returns:
            dict with "status" key, optionally "error" key if unhealthy
        """
        ...


class RediClient:
    """Real RediAI client using HTTP requests."""

    def __init__(self, base_url: str, health_path: str = "/health") -> None:
        """Initialize RediAI client.
        
        Args:
            base_url: Base URL of RediAI instance (e.g., "http://localhost:8080")
            health_path: Path to health endpoint (default: "/health")
        """
        self.base_url = base_url.rstrip("/")
        self.health_path = health_path
        self.health_url = f"{self.base_url}{health_path}"

    async def health(self) -> dict[str, str]:
        """Check RediAI health by calling its health endpoint.
        
        Returns:
            {"status": "healthy"} if RediAI is reachable
            {"status": "down", "error": "..."} if unreachable or error
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.health_url)
                response.raise_for_status()
                return {"status": "healthy"}
        except httpx.HTTPError as e:
            return {"status": "down", "error": f"HTTP error: {type(e).__name__}"}
        except Exception as e:
            return {"status": "down", "error": f"Unexpected error: {type(e).__name__}"}


class MockRediClient:
    """Mock RediAI client for testing and CI (no external dependencies)."""

    def __init__(self, simulate_healthy: bool = True) -> None:
        """Initialize mock client.
        
        Args:
            simulate_healthy: If True, returns healthy; if False, returns down
        """
        self.simulate_healthy = simulate_healthy

    async def health(self) -> dict[str, str]:
        """Return mock health status.
        
        Returns:
            {"status": "healthy"} if simulate_healthy is True
            {"status": "down", "error": "..."} if simulate_healthy is False
        """
        if self.simulate_healthy:
            return {"status": "healthy"}
        return {"status": "down", "error": "Mock unhealthy state"}

