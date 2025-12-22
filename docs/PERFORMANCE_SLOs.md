# Performance SLOs (Service Level Objectives)

**Last Updated:** 2025-12-21 (M11)  
**Purpose:** Define performance targets and measurement strategy for tunix-rt  
**Status:** Baseline targets (to be validated with profiling in future milestones)

---

## Overview

This document defines **P95 latency targets** for all tunix-rt API endpoints and describes how to measure and validate performance.

**P95 Latency:** 95% of requests must complete within the target time (excluding network transit).

---

## API Endpoint SLOs

### Health & System Endpoints

| Endpoint | P95 Latency | Rationale |
|----------|-------------|-----------|
| `GET /api/health` | <50ms | Simple status check, no DB |
| `GET /api/redi/health` | <100ms (uncached)<br><1ms (cached) | External HTTP call (uncached)<br>In-memory cache (cached) |

**Cache Behavior:** RediAI health cached for 30s (configurable via `REDIAI_HEALTH_CACHE_TTL_SECONDS`).

---

### Trace Management Endpoints

| Endpoint | P95 Latency | Notes |
|----------|-------------|-------|
| `POST /api/traces` (single) | <150ms | Single DB INSERT + refresh |
| `POST /api/traces/batch` (100 traces) | <500ms | Optimized bulk INSERT (M10) |
| `POST /api/traces/batch` (1000 traces) | <3s | Max batch size |
| `GET /api/traces/{id}` | <100ms | Single DB SELECT by PK |
| `GET /api/traces` (paginated, 20 items) | <200ms | Indexed query on `created_at` |
| `POST /api/traces/{id}/score` | <250ms | Compute score + DB INSERT |
| `GET /api/traces/compare` | <300ms | 2x trace fetch + scoring |

**Indexing:** `ix_traces_created_at` improves list pagination (M3).

---

### Dataset Management Endpoints

| Endpoint | P95 Latency | Notes |
|----------|-------------|-------|
| `POST /api/datasets/build` (100 traces) | <1s | Manifest creation + file write |
| `POST /api/datasets/build` (1000 traces) | <5s | Python-level filtering + stats |
| `GET /api/datasets/{key}/export.jsonl` (100 traces) | <2s | Bulk SELECT + JSONL streaming |
| `GET /api/datasets/{key}/export.jsonl` (1000 traces) | <10s | Large export with formatting |

**Note:** Dataset build uses Python-level filtering (compatible with SQLite/Postgres). Future optimization: DB-specific JSON queries.

---

### UNGAR Generator Endpoints (Optional Dependency)

| Endpoint | P95 Latency | Notes |
|----------|-------------|-------|
| `GET /api/ungar/status` | <50ms | Availability check (no UNGAR import) |
| `POST /api/ungar/high-card-duel/generate` (10 traces) | <2s | Episode generation + DB persist |
| `POST /api/ungar/high-card-duel/generate` (100 traces) | <15s | Bulk generation |
| `GET /api/ungar/high-card-duel/export.jsonl` (100 traces) | <3s | Filter + JSONL export |

**501 Behavior:** Endpoints return 501 Not Implemented if UNGAR not installed (graceful degradation).

---

## Database Connection Pool Targets

**Configuration (M3):**
- `DB_POOL_SIZE=5` (default)
- `DB_MAX_OVERFLOW=10` (default)
- `DB_POOL_TIMEOUT=30` seconds

**Concurrent Request Support:**
- ~50 concurrent requests with default settings
- Scale `pool_size` based on CPU cores for production
- Scale `max_overflow` for request spikes (2x expected concurrency)

**Monitoring:**
- Watch for `QueuePool limit exceeded` errors in logs
- If errors occur: increase `max_overflow` or `pool_size`

---

## Frontend Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to Interactive (TTI) | <2s | Lighthouse audit |
| First Contentful Paint (FCP) | <1s | Chrome DevTools |
| Bundle size (production) | <500 KB | Vite build output |
| Test suite duration | <10s | `npm run test` |

**Current Status:** No baseline measurements (to be added in M12+).

---

## E2E Test Performance

| Test Suite | Target | Current (M11 Baseline) |
|------------|--------|------------------------|
| Playwright tests (5 tests) | <60s | ~30s (including setup) |
| Backend CI (132 tests) | <30s | ~6s |
| Frontend CI (11 tests) | <10s | ~4s |

---

## Load Testing Strategy (Future - M12+)

### Tools
- **Backend:** [Locust](https://locust.io/) or [k6](https://k6.io/)
- **Frontend:** [Lighthouse CI](https://github.com/GoogleChrome/lighthouse-ci)

### Test Scenarios

#### Scenario 1: Normal Load
- **Users:** 10 concurrent
- **Duration:** 5 minutes
- **Pattern:** 50% GET traces, 30% POST single trace, 20% batch operations
- **Expected:** All requests meet P95 targets

#### Scenario 2: Spike Test
- **Users:** 0 → 100 in 10 seconds → sustained 100 for 2 minutes
- **Purpose:** Validate connection pool overflow handling
- **Expected:** No 500 errors, P95 degrades gracefully (< 2x normal)

#### Scenario 3: Soak Test
- **Users:** 20 concurrent
- **Duration:** 1 hour
- **Purpose:** Detect memory leaks, connection leaks
- **Expected:** No performance degradation over time

---

## Profiling Plan (M12+)

### Step 1: Baseline Measurement (2 hours)

```bash
# Install profiling tools
pip install py-spy locust

# Start backend
uvicorn tunix_rt_backend.app:app --host 127.0.0.1 --port 8000

# Profile hot paths with py-spy
py-spy record -o profile.svg -- python -m pytest tests/test_traces_batch.py -k test_batch_1000

# Load test with locust
locust -f locustfile.py --host=http://127.0.0.1:8000 --users=50 --spawn-rate=10 --run-time=5m
```

### Step 2: Identify Bottlenecks

Analyze flamegraph (`profile.svg`) for:
- Slow database queries (N+1 patterns)
- JSON serialization overhead (Pydantic)
- Connection pool exhaustion

### Step 3: Optimize

Common optimizations:
1. **Database:** Add indexes, use bulk operations, optimize queries
2. **Serialization:** Cache Pydantic model_dump results
3. **Connection Pool:** Tune `pool_size` and `max_overflow`
4. **Caching:** Add TTL caches for expensive computations

---

## Measurement Tools

### Backend Profiling

```python
# Add to critical endpoints for ad-hoc profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... endpoint logic ...
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(10)  # Top 10 slowest functions
```

### Production Monitoring (Future)

```python
# OpenTelemetry instrumentation
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

tracer = trace.get_tracer(__name__)
FastAPIInstrumentor.instrument_app(app)

@app.post("/api/traces")
async def create_trace(...):
    with tracer.start_as_current_span("create_trace") as span:
        span.set_attribute("trace.size", len(trace.model_dump_json()))
        # ... endpoint logic ...
```

**Export to:** Jaeger, Prometheus, Grafana

---

## Performance Budget vs Reality

| Budget | Current (M10) | Status |
|--------|---------------|--------|
| Batch 1000 traces | <3s | **1.2s ✅** (10x improvement) |
| Health (uncached) | <100ms | ~10-50ms ✅ |
| Single trace create | <150ms | Not measured yet ⚠️ |
| Dataset export (1000) | <10s | Not measured yet ⚠️ |

**M10 Optimization:** Batch endpoint improved from 12s → 1.2s via bulk refresh (see M10_SUMMARY.md).

---

## Degradation Alerts (Production - Future)

Set up alerts when:
- P95 latency exceeds **1.5x target** for 5 consecutive minutes
- P99 latency exceeds **2x target**
- Error rate (5xx) exceeds **1%**
- Database connection pool utilization exceeds **80%**

---

## Continuous Improvement

**Monthly Review:**
1. Analyze P95/P99 latency from production metrics
2. Identify slowest 3 endpoints
3. Profile and optimize
4. Update SLOs if targets unrealistic

**Quarterly Goals:**
- Q1 2026: Establish baseline measurements for all endpoints
- Q2 2026: Implement OpenTelemetry instrumentation
- Q3 2026: Add automated load testing in CI
- Q4 2026: Achieve <1s P95 for all read endpoints

---

## References

- [M10 Summary](M10_SUMMARY.md) - Batch endpoint optimization
- [M03 Baseline](M03_TRACE_SYSTEM_HARDENING.md) - Database indexing
- [Backend README](../README.md) - DB pool configuration
- [tunix-rt.md](../tunix-rt.md) - API documentation

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-21 | Initial SLOs defined (M11) |

---

**Next Review:** M12 (after profiling and load testing implementation)

