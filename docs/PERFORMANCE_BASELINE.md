## M15: Performance Baseline

### Load Testing
We recommend using `ab` (Apache Bench) or `k6` to establish a performance baseline for the new async execution flow.

**Target:** `POST /api/tunix/run?mode=async`

**Command:**
```bash
ab -n 100 -c 10 -p post_data.json -T application/json http://localhost:8000/api/tunix/run?mode=async
```

**Thresholds:**
- **P95 Latency:** < 200ms (for enqueue operation)
- **Throughput:** > 50 req/sec

### Metrics
We have exposed standard Prometheus metrics at `/metrics`.

**Key Metrics:**
- `tunix_runs_total`: Counter of total runs by status/mode.
- `tunix_runs_duration_seconds`: Histogram of run duration.
- `tunix_db_write_latency_ms`: Histogram of DB write latency.

### SLOs
- **Availability:** 99.9% uptime for API
- **Async Enqueue Latency:** 99% of requests < 500ms
