"""Tests for /api/redi/health endpoint with dependency injection."""

import pytest
from fastapi.testclient import TestClient

from tunix_rt_backend.app import app, get_redi_client
from tunix_rt_backend.redi_client import MockRediClient


@pytest.fixture
def client() -> TestClient:
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_overrides() -> None:
    """Clean up dependency overrides after each test to prevent leakage."""
    yield
    app.dependency_overrides.clear()


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

