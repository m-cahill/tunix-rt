"""Tests for UNGAR availability checking (no UNGAR installation required).

These tests use sys.modules mocking to test availability logic without
requiring UNGAR to be installed.
"""

import sys
from types import SimpleNamespace
from typing import Any

import pytest


def test_ungar_available_returns_false_when_not_installed():
    """Test ungar_available() returns False when UNGAR is not in sys.modules."""
    # Ensure UNGAR is not in sys.modules
    if "ungar" in sys.modules:
        pytest.skip("UNGAR is installed; this test requires it to be absent")

    from tunix_rt_backend.integrations.ungar.availability import ungar_available

    assert ungar_available() is False


def test_ungar_version_returns_none_when_not_installed():
    """Test ungar_version() returns None when UNGAR is not installed."""
    if "ungar" in sys.modules:
        pytest.skip("UNGAR is installed; this test requires it to be absent")

    from tunix_rt_backend.integrations.ungar.availability import ungar_version

    assert ungar_version() is None


def test_ungar_available_returns_true_when_mocked(monkeypatch: Any):
    """Test ungar_available() returns True when UNGAR is mocked in sys.modules."""
    # Create a fake UNGAR module
    fake_ungar = SimpleNamespace(__version__="1.0.0-mock")

    # Temporarily inject into sys.modules
    monkeypatch.setitem(sys.modules, "ungar", fake_ungar)

    # Import after mocking
    from tunix_rt_backend.integrations.ungar.availability import ungar_available

    assert ungar_available() is True


def test_ungar_version_returns_version_when_mocked(monkeypatch: Any):
    """Test ungar_version() returns version string when UNGAR is mocked."""
    # Create a fake UNGAR module with __version__
    fake_ungar = SimpleNamespace(__version__="1.0.0-mock")

    # Temporarily inject into sys.modules
    monkeypatch.setitem(sys.modules, "ungar", fake_ungar)

    # Import after mocking
    from tunix_rt_backend.integrations.ungar.availability import ungar_version

    assert ungar_version() == "1.0.0-mock"


def test_ungar_version_returns_unknown_when_no_version_attr(monkeypatch: Any):
    """Test ungar_version() returns 'unknown' when UNGAR has no __version__."""
    # Create a fake UNGAR module without __version__
    fake_ungar = SimpleNamespace()  # No __version__ attribute

    # Temporarily inject into sys.modules
    monkeypatch.setitem(sys.modules, "ungar", fake_ungar)

    # Import after mocking
    from tunix_rt_backend.integrations.ungar.availability import ungar_version

    # Should return "unknown" as fallback
    assert ungar_version() == "unknown"
