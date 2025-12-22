"""UNGAR availability checking utilities.

This module provides functions to check if UNGAR is installed and available,
without importing UNGAR at module load time. This ensures the application can
run without UNGAR being installed as an optional dependency.
"""


def ungar_available() -> bool:
    """Check if UNGAR is available (installed).

    Returns:
        True if UNGAR can be imported, False otherwise.

    Note:
        This function does not import UNGAR at the module level,
        allowing the application to run without UNGAR installed.
    """
    try:
        import ungar  # type: ignore[import-not-found]  # noqa: F401

        # Note: optional dependency, not available to mypy in default environment
        return True
    except ImportError:
        return False


def ungar_version() -> str | None:
    """Get the UNGAR version string if available.

    Returns:
        Version string if UNGAR is installed, None otherwise.

    Note:
        Returns None if UNGAR is not installed or doesn't have a __version__ attribute.
    """
    if not ungar_available():
        return None

    try:
        import ungar

        # UNGAR may not have __version__ yet; return a placeholder
        return getattr(ungar, "__version__", "unknown")
    except (ImportError, AttributeError):
        return None
