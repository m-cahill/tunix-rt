"""Tunix availability checking utilities.

This module provides functions to check if Tunix is installed and available,
without importing Tunix at module load time. This ensures the application can
run without Tunix being installed as an optional dependency.

M12 Design: Mock-first artifact generation (no runtime dependency)
M13 Design: Optional runtime integration following UNGAR pattern
"""

import shutil
import subprocess
from typing import Any


def tunix_available() -> bool:
    """Check if Tunix is available (installed and CLI accessible).

    Returns:
        True if Tunix can be imported AND CLI is accessible, False otherwise.

    Note:
        M13 implementation checks both:
        - Python package importability (tunix module)
        - CLI availability (tunix command in PATH)

        This function does not import Tunix at the module level,
        allowing the application to run without Tunix installed.
    """
    try:
        # Check 1: Can we import the tunix package?
        import tunix  # type: ignore[import-not-found]  # noqa: F401

        # Note: optional dependency, not available to mypy in default environment

        # Check 2: Is the tunix CLI accessible?
        if not shutil.which("tunix"):
            return False

        return True
    except ImportError:
        return False


def tunix_version() -> str | None:
    """Get the Tunix version string if available.

    Returns:
        Version string if Tunix is installed, None otherwise.

    Note:
        Returns None if Tunix is not installed or doesn't have a __version__ attribute.
        Also attempts to get CLI version if package version unavailable.
    """
    if not tunix_available():
        return None

    try:
        import tunix

        # Try package __version__ first
        version: str | None = getattr(tunix, "__version__", None)
        if version:
            return version

        # Fallback: Try CLI version
        result = subprocess.run(
            ["tunix", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip()

        return "unknown"
    except (ImportError, AttributeError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def check_tunix_cli() -> dict[str, Any]:
    """Check if Tunix CLI is functional and accessible.

    Returns:
        Dictionary with CLI status information:
        - accessible: bool (CLI found in PATH)
        - version: str | None (CLI version if available)
        - error: str | None (error message if check failed)

    Note:
        This function attempts to run 'tunix --version' to verify
        the CLI is not just installed, but functional.
    """
    if not shutil.which("tunix"):
        return {
            "accessible": False,
            "version": None,
            "error": "tunix command not found in PATH",
        }

    try:
        result = subprocess.run(
            ["tunix", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            version = result.stdout.strip() if result.stdout else "unknown"
            return {
                "accessible": True,
                "version": version,
                "error": None,
            }
        else:
            return {
                "accessible": False,
                "version": None,
                "error": f"tunix --version failed with exit code {result.returncode}",
            }
    except subprocess.TimeoutExpired:
        return {
            "accessible": False,
            "version": None,
            "error": "tunix --version timed out after 5 seconds",
        }
    except FileNotFoundError:
        return {
            "accessible": False,
            "version": None,
            "error": "tunix command not found",
        }
    except Exception as e:
        return {
            "accessible": False,
            "version": None,
            "error": f"Unexpected error: {str(e)}",
        }
