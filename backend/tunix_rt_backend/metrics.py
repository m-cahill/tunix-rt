"""Prometheus metrics for Tunix RT."""

from prometheus_client import Counter, Histogram  # type: ignore[import-not-found]

TUNIX_RUNS_TOTAL = Counter(
    "tunix_runs_total",
    "Total number of Tunix runs",
    ["status", "mode"],
)

TUNIX_RUN_DURATION_SECONDS = Histogram(
    "tunix_runs_duration_seconds",
    "Duration of Tunix runs in seconds",
    ["mode", "status"],
)

TUNIX_DB_WRITE_LATENCY_MS = Histogram(
    "tunix_db_write_latency_ms",
    "Database write latency in milliseconds",
    ["operation"],
    buckets=(10, 50, 100, 200, 500, 1000, 2000, 5000, float("inf")),
)
