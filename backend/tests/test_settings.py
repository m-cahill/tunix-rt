"""Tests for settings validation."""

import pytest
from pydantic import ValidationError

from tunix_rt_backend.settings import Settings


def test_settings_default_values() -> None:
    """Test that settings have correct default values."""
    settings = Settings()
    assert settings.backend_port == 8000
    assert settings.rediai_mode == "mock"
    assert settings.rediai_base_url == "http://localhost:8080"
    assert settings.rediai_health_path == "/health"


def test_settings_invalid_rediai_mode() -> None:
    """Test that invalid rediai_mode raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(rediai_mode="invalid")  # type: ignore

    errors = exc_info.value.errors()
    assert any("rediai_mode" in str(e) for e in errors)


def test_settings_invalid_rediai_base_url() -> None:
    """Test that invalid rediai_base_url raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(rediai_base_url="not-a-url")

    errors = exc_info.value.errors()
    assert any("rediai_base_url" in str(e) for e in errors)


def test_settings_invalid_backend_port_too_low() -> None:
    """Test that backend_port < 1 raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(backend_port=0)

    errors = exc_info.value.errors()
    assert any("backend_port" in str(e) for e in errors)


def test_settings_invalid_backend_port_too_high() -> None:
    """Test that backend_port > 65535 raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(backend_port=99999)

    errors = exc_info.value.errors()
    assert any("backend_port" in str(e) for e in errors)


def test_settings_valid_custom_values() -> None:
    """Test that valid custom values are accepted."""
    settings = Settings(
        backend_port=3000,
        rediai_mode="real",
        rediai_base_url="https://api.example.com:8443",
        rediai_health_path="/api/health",
    )

    assert settings.backend_port == 3000
    assert settings.rediai_mode == "real"
    assert settings.rediai_base_url == "https://api.example.com:8443"
    assert settings.rediai_health_path == "/api/health"


def test_settings_rediai_health_url_property() -> None:
    """Test that rediai_health_url property constructs URL correctly."""
    settings = Settings(
        rediai_base_url="http://example.com",
        rediai_health_path="/status",
    )

    assert settings.rediai_health_url == "http://example.com/status"
