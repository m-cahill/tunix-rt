"""Tunix availability checking utilities.

This module provides functions to check if Tunix is installed and available.
Unlike UNGAR integration, M12 Tunix integration is **mock-first** and does NOT
require Tunix runtime to be installed for core functionality.

M12 Design Decision:
- Tunix is treated as an external consumer of artifacts (JSONL + manifests)
- Export and manifest generation work WITHOUT Tunix installed
- This module exists for future expansion (M13+) when real Tunix runtime
  integration may be added
"""


def tunix_available() -> bool:
    """Check if Tunix is available (installed).

    Returns:
        True if Tunix can be imported, False otherwise.

    Note:
        M12 implementation always returns False. This is intentional.
        Export and manifest generation do NOT require Tunix runtime.
        Future milestones (M13+) may implement actual Tunix runtime checks.
    """
    # M12: Mock-first implementation
    # Do NOT attempt to import Tunix runtime
    return False


def tunix_version() -> str | None:
    """Get the Tunix version string if available.

    Returns:
        Version string if Tunix is installed, None otherwise.

    Note:
        M12 implementation always returns None (no runtime dependency).
    """
    # M12: No runtime dependency
    return None


def tunix_runtime_required() -> bool:
    """Check if Tunix runtime is required for current operations.

    Returns:
        False for M12 (all operations work without Tunix).

    Note:
        This function documents M12's design decision: artifacts can be
        generated and consumed by Tunix WITHOUT installing Tunix in tunix-rt.
    """
    # M12: All operations are artifact-based (no runtime needed)
    return False
