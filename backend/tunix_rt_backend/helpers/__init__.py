"""Helper functions for tunix-rt backend.

This module provides reusable helper functions for common operations
across the application, such as fetching entities with automatic 404 handling.
"""

from tunix_rt_backend.helpers.traces import get_trace_or_404

__all__ = ["get_trace_or_404"]
