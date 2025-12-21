"""Tests for /api/health endpoint."""

import pytest
from fastapi.testclient import TestClient

from tunix_rt_backend.app import app


@pytest.fixture
def client() -> TestClient:
    """Create test client for FastAPI app."""
    return TestClient(app)


def test_health_endpoint_returns_200(client: TestClient) -> None:
    """Test that /api/health returns 200 OK."""
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_endpoint_returns_correct_json(client: TestClient) -> None:
    """Test that /api/health returns exact JSON structure."""
    response = client.get("/api/health")
    assert response.json() == {"status": "healthy"}
