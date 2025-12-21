"""Tests for /api/redi/health endpoint with dependency injection."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tunix_rt_backend.app import app, get_redi_client
from tunix_rt_backend.redi_client import MockRediClient, RediClient


@pytest.fixture
def client() -> TestClient:
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_overrides() -> None:
    """Clean up dependency overrides and cache after each test to prevent leakage."""
    from tunix_rt_backend import app as app_module

    # Clear cache before test
    app_module._redi_health_cache.clear()

    yield

    # Clear after test
    app.dependency_overrides.clear()
    app_module._redi_health_cache.clear()


def test_redi_health_with_healthy_mock(client: TestClient) -> None:
    """Test /api/redi/health returns healthy when mock RediAI is healthy."""
    # Override dependency with mock that simulates healthy state
    app.dependency_overrides[get_redi_client] = lambda: MockRediClient(simulate_healthy=True)

    response = client.get("/api/redi/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_redi_health_with_unhealthy_mock(client: TestClient) -> None:
    """Test /api/redi/health returns down when mock RediAI is unhealthy."""
    # Override dependency with mock that simulates unhealthy state
    app.dependency_overrides[get_redi_client] = lambda: MockRediClient(simulate_healthy=False)

    response = client.get("/api/redi/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "down"
    assert "error" in data


def test_redi_health_with_exception_raising_client(client: TestClient) -> None:
    """Test /api/redi/health handles exceptions from RediClient gracefully."""

    # Create a mock that raises an exception
    class FailingRediClient:
        async def health(self) -> dict[str, str]:
            raise RuntimeError("Simulated failure")

    app.dependency_overrides[get_redi_client] = lambda: FailingRediClient()

    # Should handle exception and return error response
    with pytest.raises(RuntimeError):
        client.get("/api/redi/health")


@pytest.mark.asyncio
async def test_real_redi_client_http_error() -> None:
    """Test RediClient handles HTTP errors correctly."""
    from tunix_rt_backend.redi_client import RediClient

    # Use an invalid URL that will cause connection error
    client = RediClient(base_url="http://localhost:9999", health_path="/health")
    result = await client.health()

    assert result["status"] == "down"
    assert "error" in result


@pytest.mark.asyncio
async def test_real_redi_client_constructs_url_correctly() -> None:
    """Test RediClient URL construction."""
    from tunix_rt_backend.redi_client import RediClient

    client = RediClient(base_url="http://example.com/", health_path="/health")
    assert client.health_url == "http://example.com/health"

    client2 = RediClient(base_url="http://example.com", health_path="/api/health")
    assert client2.health_url == "http://example.com/api/health"


@pytest.mark.asyncio
async def test_redi_client_non_2xx_response() -> None:
    """Test RediClient handles non-2xx HTTP status codes."""
    import httpx

    client = RediClient(base_url="http://example.com", health_path="/health")

    # Mock httpx.AsyncClient to return 404
    mock_response = httpx.Response(status_code=404, text="Not Found")

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        result = await client.health()

    assert result["status"] == "down"
    assert result["error"] == "HTTP 404"


@pytest.mark.asyncio
async def test_redi_client_timeout() -> None:
    """Test RediClient handles timeout errors."""
    import httpx

    client = RediClient(base_url="http://example.com", health_path="/health")

    # Mock httpx.AsyncClient to raise TimeoutException
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.TimeoutException("Request timeout")
        result = await client.health()

    assert result["status"] == "down"
    assert "Timeout" in result["error"]


@pytest.mark.asyncio
async def test_redi_client_connection_refused() -> None:
    """Test RediClient handles connection refused errors."""
    import httpx

    client = RediClient(base_url="http://example.com", health_path="/health")

    # Mock httpx.AsyncClient to raise ConnectError
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        result = await client.health()

    assert result["status"] == "down"
    assert "Connection refused" in result["error"]


def test_get_redi_client_returns_mock_in_mock_mode() -> None:
    """Test get_redi_client returns MockRediClient when REDIAI_MODE=mock."""
    from tunix_rt_backend.settings import settings

    # Save original mode
    original_mode = settings.rediai_mode

    try:
        settings.rediai_mode = "mock"
        client = get_redi_client()
        assert isinstance(client, MockRediClient)
    finally:
        settings.rediai_mode = original_mode


def test_get_redi_client_returns_real_in_real_mode() -> None:
    """Test get_redi_client returns RediClient when REDIAI_MODE=real."""
    from tunix_rt_backend.settings import settings

    # Save original mode
    original_mode = settings.rediai_mode

    try:
        settings.rediai_mode = "real"
        client = get_redi_client()
        assert isinstance(client, RediClient)
    finally:
        settings.rediai_mode = original_mode


def test_redi_health_cache_hit(client: TestClient) -> None:
    """Test that RediAI health responses are cached."""
    from tunix_rt_backend import app as app_module

    call_count = 0

    class CountingRediClient:
        async def health(self) -> dict[str, str]:
            nonlocal call_count
            call_count += 1
            return {"status": "healthy"}

    # Override with counting client
    app.dependency_overrides[get_redi_client] = lambda: CountingRediClient()

    # Clear cache
    app_module._redi_health_cache.clear()

    # First call - should hit RediClient
    response1 = client.get("/api/redi/health")
    assert response1.status_code == 200
    assert call_count == 1

    # Second call immediately - should use cache
    response2 = client.get("/api/redi/health")
    assert response2.status_code == 200
    assert call_count == 1  # No additional call
    assert response2.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_redi_health_cache_expiry() -> None:
    """Test that cache expires after TTL."""
    from datetime import datetime, timedelta, timezone

    from tunix_rt_backend import app as app_module
    from tunix_rt_backend.settings import settings

    # Save original TTL
    original_ttl = settings.rediai_health_cache_ttl_seconds

    try:
        # Set very short TTL for testing
        settings.rediai_health_cache_ttl_seconds = 1

        # Clear cache
        app_module._redi_health_cache.clear()

        # Manually add an expired cache entry
        old_time = datetime.now(timezone.utc) - timedelta(seconds=2)
        app_module._redi_health_cache["redi_health"] = ({"status": "healthy"}, old_time)

        # Create a client
        call_count = 0

        class CountingRediClient:
            async def health(self) -> dict[str, str]:
                nonlocal call_count
                call_count += 1
                return {"status": "healthy"}

        # Call the endpoint - should not use expired cache
        from tunix_rt_backend.app import redi_health

        result = await redi_health(CountingRediClient())

        assert result == {"status": "healthy"}
        assert call_count == 1  # Fresh call made
    finally:
        settings.rediai_health_cache_ttl_seconds = original_ttl
