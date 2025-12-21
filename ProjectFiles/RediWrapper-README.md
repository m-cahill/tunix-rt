# RediWrapper

Repository scaffold per `ROADMAP.md` **Phase I — M31 Complete ✓: Audit Closeout + Integration Polish**

## Quick Links

📚 **Documentation:**
- 🚀 **[Quick Start Guide (15 minutes)](docs/ONBOARDING.md)** - Get running with Malmo in 15 minutes
- 🎨 **[Frontend Guide](docs/FRONTEND_GUIDE.md)** - React dashboard development (M39)
- 📖 **[Architecture Overview](RediWrapper.md)** - System design and patterns
- 🗺️ **[Roadmap](ROADMAP.md)** - Development timeline and milestones
- 🏗️ **[ADR 001: Multi-Agent State Format](docs/ADR_001_multi_agent_state_format.md)** - Design decisions
- ⚡ **[Performance Guide](docs/PERFORMANCE.md)** - Benchmarks and optimization
- 🔍 **[Profiling Guide](docs/PROFILING.md)** - Performance profiling with py-spy and cProfile
- 🔌 **[Adding Providers](docs/ADDING_PROVIDERS.md)** - Extend with new game engines
- ☁️ **[GCP Smoke Test Setup](docs/smoke-gcp.md)** - Configure live GCP testing
- 🔧 **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- 📦 **[Optional Dependencies](docs/OPTIONAL_DEPENDENCIES.md)** - Installation guide for extras
- 🔄 **[CI Architecture](docs/CI_ARCHITECTURE.md)** - CI/CD job structure, local reproduction, and benchmark strategies (M38 enhanced)
- 🤖 **[Tunix Integration Guide](docs/TUNIX_INTEGRATION.md)** - ML training with Tunix (M35 dry-run + M36 real execution)
- 📊 **[Trace Dashboard API](docs/TRACE_DASHBOARD.md)** - Trace visibility and comparison (M37, M38 validation)
- 🔍 **[Trace Regression Detection](docs/TRACE_REGRESSION.md)** - Automated quality control for traces (M37, M38 SQL baseline + cache)

## Project Status & Metrics

**Status:** ✅ **M39 In Progress** - Trace Dashboard Frontend + Observability. React-based UI for trace visualization with list/detail views, filters, and CI integration. M38 complete: CI Documentation, Code Polish & Quality Improvements.

| Metric | Value | Notes |
|--------|-------|-------|
| **Tests** | 2,020+ total passing | Comprehensive coverage across 8 CI jobs + M38 enhancements (70+ new tests) |
| **Coverage** | ≥85% (M37+M38 code) | Unit + integration tests, isolated environments |
| **Game Engines** | 9 supported | Ludii, Malmo, Poker (Antematter), Poker Stub, Chess (Stockfish + UCI framework for Lc0) |
| **API Endpoints** | 18 core | Sessions, training export, chess analysis, **reasoning traces**, **Tunix job orchestration + real training**, **trace dashboard (list/detail/compare)** - all with OpenAPI/Swagger docs |
| **Security Gates** | ✅ All | Cosign signing, SLSA provenance, dependency scanning, signed images |
| **Performance** | ✅ Sub-10ms | Enrichment overhead with intelligent caching |
| **Architecture** | ✅ Clean | Hexagonal pattern, strategy/registry patterns, service layer, chess engine registry |
| **Documentation** | ✅ Enterprise | Deployment guides, chess integration guide, training examples, game templates, **reasoning API guide**, **Tunix real execution guide**, **trace dashboard + regression docs** |

## Deployed Services

- **Backend:** `https://rediwrapper.onrender.com` (Render)
- **Frontend:** Netlify (proxies `/api/*` to backend)
- **CI/CD:** GitHub Actions (lint, type-check, test with coverage gates, image signing)

## Future Game Integrations

RediWrapper is architected for easy extension to new game domains. Enterprise-grade templates and plans are ready:

### 🎯 Chess Suite (M30 - Complete ✓)
**UCI Protocol Integration** - Connect with world-class chess engines:
- **Stockfish** - Objective tactical analysis ✅
- **Leela Chess Zero (Lc0)** - Neural network evaluation (framework ready)
- **Maia** - Human-like positional understanding (planned)

**Implemented Features (M30):**
- ✅ Chess position analysis API (`POST /api/chess/analyze`)
- ✅ Multi-engine support framework with registry pattern
- ✅ UCI adapter with `bestmove_for_fen()` method
- ✅ Engine configuration via environment variables (`STOCKFISH_PATH`, `LC0_PATH`)
- ✅ Comprehensive error handling (503/422/500 status codes)
- ✅ Unit tests with mocked UCI flow + optional real engine tests

**Planned Features:**
- Position analysis with multi-engine fusion
- Real-time coaching during games
- Tournament automation
- Training data export for ML

*📋 [Implementation Guide](docs/CHESS_ENGINE_INTEGRATION.md) | [Detailed Plan](docs/CHESS_SUITE_PLAN.md)*

### 🃏 Poker Suite (R-M1 - Complete ✓)
**Antematter Integration** - Competitive poker AI:
- **No-limit Texas Hold'em** support ✅
- **Omaha High/Lo** variants ✅
- **Seven Card Stud** ✅
- **Training data export** for poker ML research (via RediWrapper sessions)

**Implemented Features (R-M1):**
- ✅ Antematter v0.3.0 integration (`poker.antematter` provider)
- ✅ Malmo-shaped state format for consistency
- ✅ Terminal reward model (stack change)
- ✅ Smoke tests and error handling

**Planned Features:**
- Multi-seat control
- Advanced reward models
- Additional variants (Razz, Triple Draw, etc.)

**Powered by:** Antematter (CFR-based poker AI, multiple competition winner)

*📋 [Integration Guide](docs/integrations/antematter_poker.md) | [Detailed Plan](docs/POKER_SUITE_PLAN.md)*

### 🔧 Extension Framework
Ready-to-use templates for new game suites:
- **[Game Suite Template](docs/GAME_SUITE_TEMPLATE.md)** - Complete integration guide
- **[Engine Adapter Template](docs/ENGINE_ADAPTER_TEMPLATE.md)** - Protocol implementation patterns

**Supported Patterns:**
- Strategy pattern for state retrieval
- Adapter pattern for engine protocols
- Service layer for business logic
- Registry pattern for provider management

## API Documentation

- **Swagger UI:** `http://localhost:8000/docs` (when running locally)
- **ReDoc:** `http://localhost:8000/redoc` (when running locally)
- Deployed: `https://rediwrapper.onrender.com/docs`

## Tunix Integration (M35)

RediWrapper now supports **Tunix Job Orchestration** for ML training workflows:

**Quick Example:**
```bash
# 1. Create job spec (see examples/tunix_job_spec_example.json)
# 2. Run dry-run
python scripts/tunix_run.py --spec examples/tunix_job_spec_example.json

# 3. Validate artifacts
python scripts/tunix_run.py --validate example-experiment-001
```

**Key Features:**
- Job contracts (TunixJobSpec v0.1) with metadata
- Standard artifact layout with SHA-256 checksums
- Dry-run execution (generates stub artifacts)
- Checkpoint/resume support
- Comprehensive validation

📚 **[Full Documentation](docs/TUNIX_INTEGRATION.md)** - Complete guide with examples

---

## Reasoning Sessions API (M33/M34)

RediWrapper supports **Reasoning Trace Management** for LLM reasoning workflows:

**Quick Example:**
```bash
# 1. Create session
curl -X POST http://localhost:8000/api/reasoning/sessions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?", "metadata": {"run_id": "exp-001"}}'

# 2. Append reasoning steps
curl -X POST http://localhost:8000/api/reasoning/sessions/{id}/steps \
  -H "Idempotency-Key: step-001" \
  -d '{"phase": "think", "content": "Need to add 2 and 2"}'

# 3. Mark complete
curl -X POST http://localhost:8000/api/reasoning/sessions/{id}/complete \
  -d '{"final_answer": "4"}'

# 4. Export training data
curl http://localhost:8000/api/reasoning/sessions/{id}/export?mode=trace > trace.jsonl
```

**Key Features:**
- Append-only step log with idempotency (201/200/409)
- Snapshot-aware replay (O(tail) performance)
- Deterministic JSONL export for ML training
- Tunix-shaped metadata support (run_id, model_id, dataset_id)
- Schema versioning (v0.1 with documented evolution path)

**Endpoints:**
- `POST /api/reasoning/sessions` - Create session
- `POST /api/reasoning/sessions/{id}/steps` - Append step (idempotent)
- `POST /api/reasoning/sessions/{id}/complete` - Mark complete
- `GET /api/reasoning/sessions/{id}` - Get state
- `GET /api/reasoning/sessions/{id}/history` - Get history
- `GET /api/reasoning/sessions/{id}/export` - Export JSONL (mode=trace|steps)

📚 **[Full Documentation](docs/REASONING_SESSIONS.md)** - Complete API reference with examples

## Game Sessions API (M9/M10)

Endpoints:

- `POST /api/game/sessions` → `{sessionId}` (body: `{rulesId}`)
- `GET /api/game/sessions/{sessionId}` → `{sessionId, state}`
  - M10: Uses snapshots when available, otherwise linear replay
- `GET /api/game/sessions/{sessionId}/history?limit&offset` → `[{seq, moveIndex, label, createdAt}]`
- `POST /api/game/sessions/{sessionId}/moves` → `{appliedIndex, seq}`
- `GET /api/game/sessions/{sessionId}/export?limit=N` → JSONL training data stream (**M17C**)

### Training Data Export (M17C)

Export session data as **JSON Lines (JSONL)** for training ML models:

**cURL Example:**
```bash
# Export first 10 moves for inspection
curl http://localhost:8000/api/game/sessions/{sessionId}/export | head -n 10

# Download full training dataset
curl http://localhost:8000/api/game/sessions/{sessionId}/export > training.jsonl
```

**Python Example (httpx streaming):**
```python
import httpx

session_id = "abc123"
async with httpx.AsyncClient() as client:
    async with client.stream("GET", f"http://localhost:8000/api/game/sessions/{session_id}/export") as resp:
        async for line in resp.aiter_lines():
            record = json.loads(line)
            # Process each training example...
```

**Response Format:** `Content-Type: application/x-ndjson` ([NDJSON](http://ndjson.org/))  
*Note:* NDJSON doesn't have an official IANA MIME type; `application/x-ndjson` is the widely-used convention.

**JSONL Format** (one JSON object per line):
```json
{"timestamp": "2025-11-04T12:00:00Z", "session_id": "...", "agent_id": "0", "step": 0, "action": 0, "reward": 0.0, "cumulative_reward": 0.0, "done": false, "obs_hash": "a1b2c3d4", "mission_id": "basic_nav"}
```

**Query Parameters:**
- `?limit=N` - Maximum moves to export (default: 100k, max: 100k)
- `?agentId=X` - Filter by agent ID (M19C)
- `?sinceStep=N` - Export moves since step N (M19C)

**Error Responses:**
- `413 Content Too Large` - Session exceeds limit
- `404 Not Found` - Invalid or non-existent session

**Use Cases:**
- Training reinforcement learning agents
- Offline dataset analysis
- Session replay and debugging

### Idempotency:

- Include `Idempotency-Key` (or `X-Idempotency-Key`) on move POSTs.
- 201 on first apply, 200 on exact duplicate, 409 if the same key is reused with a different payload.

### Multi-Agent Batch Moves (M18):

Submit atomic batches for multiple agents per tick:

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/game/sessions/{sessionId}/moves \
  -H "Content-Type: application/json" \
  -H "X-Idempotency-Key: batch-001" \
  -d '{
    "moves": [
      {"agentId": "0", "moveIndex": 2},
      {"agentId": "1", "moveIndex": 0}
    ]
  }'
```

**Response:**
```json
{
  "seq": 1,
  "moves": [
    {"agentId": "0", "appliedIndex": 2},
    {"agentId": "1", "appliedIndex": 0}
  ]
}
```

**Features:**
- Atomic coordination: All agents move together or none
- Consecutive seq allocation: Each agent gets unique seq (N, N+1, N+2...)
- Per-batch idempotency with `X-Idempotency-Key` header
- Backward compatible: Single-agent moves still use `{"moveIndex": 2}` format

### Replay snapshots (M10)

- Snapshots stored in `session_snapshots` every `SNAPSHOT_EVERY_MOVES` (default `50`).
- Replay starts from the latest snapshot and applies only tail moves.
- Best‑effort snapshot creation after move commit; failures don’t affect the move response.

### Configuration (M10)

- `SNAPSHOT_EVERY_MOVES` (default `50`)
- `DB_POOL_SIZE` (default `10`), `DB_MAX_OVERFLOW` (default `20`), `DB_POOL_RECYCLE` (default `1800`)

## Getting started

**Backend Quick Setup:**
```bash
# One-command setup (Linux/macOS)
./scripts/dev-setup.sh

# Or manually:
pip install -e '.[dev]'
docker compose up -d postgres
alembic upgrade head
uvicorn backend.app.main:app --reload
```

**Frontend Quick Setup (M39):**
```bash
# Install dependencies
cd frontend
npm install

# Start dev server (requires backend running)
npm run dev

# Open http://localhost:5173
```

📚 **See [Frontend Guide](docs/FRONTEND_GUIDE.md)** for detailed setup and development instructions.

**Prerequisites:**
- Python 3.10+
- Docker and docker-compose (for PostgreSQL)
- See [docs/OPTIONAL_DEPENDENCIES.md](docs/OPTIONAL_DEPENDENCIES.md) for optional extras

**Common Commands:**
```bash
make install    # Install dependencies
make test       # Run tests
make dev        # Start development server
make migrate    # Run database migrations
```

**Optional Extras:**
- For Azure adapters: `pip install -e '.[dev,azure]'`
- For observability (Azure Monitor): `pip install -e '.[dev,azure,observability]'`
- See [docs/OPTIONAL_DEPENDENCIES.md](docs/OPTIONAL_DEPENDENCIES.md) for full list

**Troubleshooting:** See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues.

### Local Postgres with Docker Compose

```bash
docker compose up -d postgres
set DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres  # Windows
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres # Linux/macOS
alembic upgrade head
```

## Observability

RediWrapper includes comprehensive observability features:

- **Structured Logging**: JSON-formatted logs with request correlation (structlog)
- **Request Tracing**: Unique request IDs (`X-Request-ID` header) flow through all logs
- **Adapter Metrics**: Operation duration, size, error tracking for all Azure operations
- **Azure Monitor Integration** (optional): Exports traces, metrics, and logs to Application Insights
- **GCP Cloud Trace (optional)**: Set `GCP_TRACE_EXPORT=1` and install `.[observability]` to export traces
- **Health Endpoints**: `/api/health` (basic), `/api/health/azure`, `/api/health/gcp`, `/api/health/ludii`

See [OBSERVABILITY.md](OBSERVABILITY.md) for detailed configuration and usage.

## Malmo Platform Integration (M11, M15, M16, M19)

RediWrapper supports Microsoft's **Malmo Platform** for Minecraft-based RL environments with full **multi-agent** and **continuous action** support.

### Mission Pack (M19D)

**📦 Curated example missions** are available in [`missions/examples/`](missions/README.md):
- **pack_discrete_simple:** Simple discrete navigation with grid observations
- **pack_continuous_nav:** Continuous movement with smooth turn/move/pitch control
- **pack_two_agent_coop:** Two-agent cooperative continuous navigation
- **pack_inventory_basic:** Inventory management with discrete movement and item collection
- **pack_chat_basic:** Chat commands with discrete movement for communication tests

See [`missions/README.md`](missions/README.md) for full documentation, usage examples, and handler allowlists.

### Quick Start

```bash
# 1. Start Malmo client (Docker)
docker compose -f docker-compose.malmo.yml up -d

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Run backend
uvicorn backend.app.main:app --reload

# 4. Open http://localhost:8000/docs
```

See **[Quick Start Guide](docs/ONBOARDING.md)** for the full 15-minute tutorial.

---

## Development

### Quick Commands (Makefile)

```bash
make help          # Show all available commands
make test          # Run all tests (excluding live tests)
make test-fast     # Quick unit tests only  
make bench         # Run benchmarks with autosave
make lint          # Run ruff linter
make type-check    # Run mypy
make format        # Format code with ruff
```

### Manual Testing

```bash
# Backend unit tests only (no DB, no perf)
pytest tests/ --ignore=tests/perf -m "not require_postgres"

# Backend integration tests (requires PostgreSQL)
pytest tests/ -m "require_postgres"

# Backend with coverage
pytest --cov=backend --cov=core --cov-report=term

# Specific test file
pytest tests/test_sessions_api.py -v

# Frontend unit tests (Vitest)
cd frontend && npm run test

# Frontend E2E tests (Playwright) - M41
cd frontend && npm run test:e2e
```

**E2E Test Notes (M41):**
- E2E tests require backend running on port 8000
- Playwright automatically starts the frontend dev server
- Install Playwright browsers first: `cd frontend && npx playwright install chromium`
- See [docs/ONBOARDING.md](docs/ONBOARDING.md) for detailed E2E setup

### Installing Optional Extras

```bash
# Azure cloud support
make extras-azure
# or: pip install -e ".[azure]"

# GCP cloud support  
make extras-gcp
# or: pip install -e ".[gcp]"

# Observability (OpenTelemetry)
make extras-observability
# or: pip install -e ".[observability]"

# All extras
make extras-all
# or: pip install -e ".[dev,azure,gcp,observability]"
```

See **[Optional Dependencies Guide](docs/OPTIONAL_DEPENDENCIES.md)** for details.

---

# 2. Configure backend
export PROVIDER=game-malmo
export MALMO_HOST=localhost
export MALMO_PORT=10000

# 3. Run backend
uvicorn backend.app.main:app --port 8000

# 4. Create session and execute moves (see docs/ONBOARDING.md)
```

**📖 Full Guide:** [docs/ONBOARDING.md](docs/ONBOARDING.md) (15-minute walkthrough)

### Action Modes (M16)

Malmo supports two action modes per mission:

#### Discrete Actions (Default)
Integer move indices (0-4) for `DiscreteMovementCommands`:
```json
{"moveIndex": 0}  // 0=forward, 1=back, 2=left, 3=right, 4=jump
```

#### Continuous Actions (M16B)
Float vectors for smooth control with `ContinuousMovementCommands`:
```json
{"action": {"turn": 0.5, "move": 1.0, "pitch": 0.0}}
// turn: -1.0 (left) to 1.0 (right)
// move: -1.0 (back) to 1.0 (forward)
// pitch: -1.0 (down) to 1.0 (up)
```

**Action mode** is configured per-mission in `missions/registry.yaml`:
```yaml
my_mission:
  action_mode: continuous  # or discrete
  allowlist:
    movement: [ContinuousMovementCommands]
```

### Multi-Agent Support (M15C)

Run missions with 2+ agents using batch move format:
```json
{
  "moves": [
    {"agentId": 0, "moveIndex": 2},
    {"agentId": 1, "moveIndex": 0}
  ]
}
```

**State format** (nested dict by agent ID, M20 observation enrichment):
```json
{
  "agents": {
    "0": {
      "observation": {
        // Base Malmo observation keys
        "XPos": 0.5, "YPos": 5.0, "ZPos": 0.5, "Yaw": 0, ...

        // M20: Enriched fields (mission-scoped)
        "position": {"x": 0.5, "y": 5.0, "z": 0.5},  // if ObservationFromFullStats
        "yaw": 0.0,                                  // if ObservationFromFullStats
        "grid": {"shape": [3, 3], "data": "..."}     // if ObservationFromGrid
      },
      "reward": 0.0,
      "done": false,
      "info": {}
    },
    "1": {"observation": {...}, "reward": 0.5, "done": false, "info": {}}
  },
  "global": {"done": false}
}
```

**Observation Enrichment (M20):**
- **Mission-scoped**: `position`/`yaw` added when `ObservationFromFullStats` allowlisted
- **Grid data**: Structured `grid` field when `ObservationFromGrid` allowlisted
- **Additive only**: Backward compatible, existing clients unaffected

**📖 See:** [docs/ADR_001_multi_agent_state_format.md](docs/ADR_001_multi_agent_state_format.md) for design rationale.

### Mission Registry & Security (M15B)

All missions must be registered in `missions/registry.yaml` (security boundary):
```yaml
basic_nav:
  path: missions/basic_nav.xml
  action_mode: discrete
  allowlist:
    movement: [DiscreteMovementCommands]
    observation: [ObservationFromGrid]
  description: "Basic navigation with discrete movement"
  max_agents: 1
```

**Security:**
- Unregistered missions → HTTP 404
- Path traversal attempts → HTTP 400
- Allowlist enforcement is trust-based (registry is security boundary)

**📖 Handler Reference:** [Malmo MissionHandlers](https://microsoft.github.io/malmo/0.30.0/Schemas/MissionHandlers.html)

### Configuration

- `PROVIDER=game-malmo` - Enable Malmo adapter
- `MALMO_HOST=localhost` - Malmo client hostname
- `MALMO_PORT=10000` - Malmo client port (searches 10000-11000)

### Health Check

```bash
curl http://localhost:8000/api/health/malmo
# {"status": "healthy"}
```

### Live Integration Testing (M17B)

Run smoke tests against a live Malmo client using GitHub Actions:

```bash
# Manually trigger from GitHub Actions tab:
# .github/workflows/malmo-live.yml (workflow_dispatch)

# Or run locally:
docker compose -f docker-compose.malmo.yml up -d
timeout 60 bash -c 'until nc -z 127.0.0.1 10000; do sleep 2; done'
pytest -m malmo_local -v
docker compose -f docker-compose.malmo.yml down -v
```

**Workflow Features:**
- Manual trigger (`workflow_dispatch`)
- Auto-waits for MCP readiness (TCP healthcheck on port 10000)
- Runs `pytest -m malmo_local` tests
- Uploads logs on failure
- 15-minute timeout

---

## Performance

RediWrapper includes comprehensive performance optimizations and benchmarks (M21A):

### Observation Enrichment Caching (M21A)

Malmo observations are automatically enriched with structured fields when missions allowlist `ObservationFromFullStats` or `ObservationFromGrid`:

- **position**: `{"x": float, "y": float, "z": float}` (from XPos/YPos/ZPos)
- **yaw**: `float` (from Yaw)
- **grid**: `{"shape": [int, int], "data": str|list}` (from floor3x3)

**Performance Optimizations:**
- **TTL LRU Cache**: Grid parsing cached per `(mission_id, grid_hash)` with 1s TTL and 1024 entry limit
- **Opt-out Control**: Add `?enrich=0` to GET `/api/game/sessions/{id}` to skip enrichment entirely
- **Environment Override**: Set `MALMO_ENRICH_OBS=0` as default (request param takes precedence)

**Benchmark Target**: p50 enrichment overhead <10% (measured via `pytest-benchmark`)

### Benchmark Artifacts

CI publishes benchmark results as build artifacts:

1. **GitHub Actions**: Go to Actions → Latest run → "enrichment-benchmarks" artifact
2. **Contents**:
   - `enrichment-bench.json`: Raw pytest-benchmark data
   - `enrichment-summary.md`: Human-readable performance report
3. **Reproduce locally**: `pytest tests/perf/test_malmo_enrichment_bench.py --benchmark-only --benchmark-json=local-bench.json`

### Benchmarks

- **Enrichment**: `pytest tests/perf/test_malmo_enrichment_bench.py --benchmark-only`
- **Strategies**: `pytest tests/perf/ --benchmark-only` (state retrieval patterns)
- **Baselines**: Autosaved to `.benchmarks/` (gitignored)
- **Compare runs**: `pytest tests/perf/ --benchmark-compare`

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for baseline metrics, optimization tips, and regression tracking.

**Quick Start (Azure Monitor):**
```bash
# Install observability extras
pip install -e '.[observability]'

# Set connection string
export AZURE_MONITOR_CONNECTION_STRING="InstrumentationKey=<key>;IngestionEndpoint=https://<region>.in.applicationinsights.azure.com/"

# Restart app - telemetry will be exported automatically
uvicorn backend.app.main:app
```

## Provider Configuration

RediWrapper supports multiple adapter backends via the `PROVIDER` environment variable:

**📘 For developers:** See [docs/ADDING_PROVIDERS.md](docs/ADDING_PROVIDERS.md) for a complete guide on adding new game providers.

**Null Provider (default):**
```bash
# No configuration needed - uses in-memory adapters
PROVIDER=null uvicorn backend.app.main:app
```

**Azure Provider:**
```bash
# Required environment variables:
PROVIDER=azure
AZURE_STORAGE_ACCOUNT_URL=https://myaccount.blob.core.windows.net
AZURE_STORAGE_CONTAINER=my-container
KEY_VAULT_NAME=my-keyvault
SERVICE_BUS_FQDN=mynamespace.servicebus.windows.net
SERVICE_BUS_QUEUE=my-queue

# Authentication: uses DefaultAzureCredential
# Local dev: az login
# Azure: Managed Identity
```

See [Azure Setup](#azure-setup) for detailed configuration instructions.

**GCP Provider:**
```bash
# Required environment variables:
PROVIDER=gcp
GCP_PROJECT=your-project-id
GCS_BUCKET=your-bucket
GCP_PUBSUB_TOPIC=your-topic

# Authentication (ADC):
# Local dev: gcloud auth application-default login
# CI/Prod: Service Account JSON (GOOGLE_APPLICATION_CREDENTIALS) or Workload Identity Federation
```

**Ludii Provider (Sidecar):**
```bash
# Required environment variables:
PROVIDER=game-ludii
LUDII_URL=http://localhost:8080  # default

# Requests propagate X-Request-ID to the sidecar for correlation
```

**Malmo Provider (M11, M15):**

RediWrapper now supports Microsoft's Malmo Platform for Minecraft-based RL environments with **multi-agent support** (M15).

```bash
# Install Malmo optional dependency
pip install -e '.[malmo]'

# Start Malmo Platform client (headless with Xvfb)
docker compose -f docker-compose.malmo.yml up

# Run backend with Malmo provider
PROVIDER=game-malmo \
MALMO_HOST=localhost \
MALMO_PORT=10000 \
uvicorn backend.app.main:app --reload
```

**Docker Compose Options:**
- **Default**: Xvfb headless mode (port 10000)
- **Optional**: Uncomment `novnc` service in `docker-compose.malmo.yml` for browser-based VNC debugging
- **Port range**: Malmo searches 10000-11000 if port busy

**Poker Provider (Antematter) (R-M1):**

RediWrapper integrates with Antematter v0.3.0 for poker game environments.

```bash
# Install poker optional dependency
pip install -e '.[poker]'

# Run backend with Antematter poker provider
PROVIDER=poker.antematter uvicorn backend.app.main:app --reload
```

**Supported Variants:**
- Texas Hold'em (No-Limit, Pot-Limit, Fixed-Limit)
- Omaha High
- Omaha Hi/Lo (PLO8)
- Seven Card Stud

**Usage:**
```bash
# Create session with Texas Hold'em
POST /api/game/sessions
{"rulesId": "texas_holdem"}

# Extended config: variant:seats:stack:sb/bb
{"rulesId": "texas_holdem:6max:2000:10/20"}
```

*📋 [Integration Guide](docs/integrations/antematter_poker.md)*

**Mission Registry (M15B):**
- All missions must be registered in `missions/registry.yaml` (security boundary)
- Mission IDs reference registry entries (no file paths in API)
- Registry includes allowlists for movement/observation handlers
- Example: `POST /api/game/sessions` with `{"rulesId": "basic_nav"}` (not `malmo://...`)

**Single-Agent Missions:**
```bash
# Create session with basic_nav mission (1 agent)
POST /api/game/sessions
{"rulesId": "basic_nav"}

# Execute move (discrete actions 0-4: forward, back, turn_left, turn_right, jump)
POST /api/game/sessions/{sessionId}/moves
{"moveIndex": 2}

# Get state (M15C multi-agent format)
GET /api/game/sessions/{sessionId}
{
  "agents": {
    "0": {
      "observation": {"x": 0.5, "y": 5.0, "z": 0.5, "yaw": 0, ...},
      "reward": 0.0,
      "done": false,
      "info": {}
    }
  },
  "global": {"done": false}
}
```

**Multi-Agent Missions (M15C):**
```bash
# Create session with two_agent_nav mission (2 agents)
POST /api/game/sessions
{"rulesId": "two_agent_nav"}

# Execute batch move (atomic step for all agents)
POST /api/game/sessions/{sessionId}/moves
{
  "moves": [
    {"agentId": 0, "moveIndex": 2},
    {"agentId": 1, "moveIndex": 0}
  ]
}

# Get state (nested dict by agent ID)
GET /api/game/sessions/{sessionId}
{
  "agents": {
    "0": {"observation": {...}, "reward": 1.0, "done": false, "info": {}},
    "1": {"observation": {...}, "reward": 0.5, "done": false, "info": {}}
  },
  "global": {"done": false}  # Episode ends if any agent done
}
```

**Action Space:**
- Discrete actions 0-4 map to movement commands (forward, back, turn left, turn right, jump)
- See [MissionHandlers docs](https://microsoft.github.io/malmo/0.30.0/Schemas/MissionHandlers.html) for available handlers

**Multi-Agent Coordination (M15C):**
- Agent count parsed from mission XML (`<AgentSection>` count)
- MalmoEnv role 0 coordinates multi-agent rendezvous
- Batch moves ensure atomic step across all agents

**Testing:**
```bash
# Unit tests (hermetic, mocked malmoenv)
pytest tests/test_mission_registry.py
pytest tests/test_malmo_multi_agent.py

# Integration tests (requires live Malmo client)
MALMO_LOCAL=1 pytest -v -m malmo_local
```

**Note:** Malmo sessions use `last_state` storage (non-deterministic replay) instead of move history replay.

## Live Azure Testing (Optional)

Run manual smoke tests against real Azure resources:

- Trigger the `Azure Live Smoke Tests` workflow in GitHub Actions (workflow_dispatch)
- Required repository secrets:
  - `AZURE_STORAGE_ACCOUNT_URL`, `AZURE_STORAGE_CONTAINER`
  - `KEY_VAULT_NAME`
  - `SERVICE_BUS_FQDN`, `SERVICE_BUS_QUEUE`
- Sets `AZURE_LIVE=1` to enable tests in `tests_live/`
- Tests are read-only and safe to run (no queue publish required)

Local run:
```bash
export AZURE_LIVE=1
pytest tests_live/ -m azure_live -v
```

## Supply Chain Security

- **Dependency Review**: Blocks PRs introducing high/critical vulnerabilities
- **pip-audit**: Informational job in CI with human-readable output (see [docs/SECURITY.md](docs/SECURITY.md))
- **SBOM (CycloneDX)**: Generated on each CI run and uploaded as an artifact
- **Image Signing**: All Docker images signed with Cosign (keyless via GitHub OIDC)
- **SLSA Provenance**: Build attestations for supply chain transparency

See [docs/SECURITY.md](docs/SECURITY.md) for dependency upgrade procedures and security best practices.

## Tests

Run tests:
```bash
pytest -q
```

## Live GCP Testing (Optional)

Run manual smoke tests against real GCP resources:

- Trigger the `GCP Live Smoke Tests` workflow in GitHub Actions (workflow_dispatch)
- Required repository secrets:
  - `GCP_PROJECT`, `GCS_BUCKET`, `GCP_PUBSUB_TOPIC`
  - `GCP_SA_KEY_JSON` (used to set GOOGLE_APPLICATION_CREDENTIALS)
- Sets `GCP_LIVE=1` to enable tests in `tests_live_gcp/`

Local run:
```bash
export GCP_LIVE=1
pytest tests_live_gcp/ -m gcp_live -v
```

Run tests with coverage (70% required):
```bash
pytest --cov=backend --cov=core --cov-report=term --cov-fail-under=70
```

Coverage reports are generated as `coverage.xml` and uploaded as CI artifacts.

## Ludii Sidecar Smoke (Optional)

Manual workflow builds and runs the Java sidecar and performs a short playout:

- Trigger `Ludii Sidecar Smoke` (workflow_dispatch) in GitHub Actions
- No secrets required; uses local Docker build

## CI / Smoke Testing

### Malmo Image Management

The project intentionally uses `andkram/malmo:latest` for smoke testing to ensure compatibility with the latest Malmo Platform releases. This provides:

- **Latest features**: Access to newest Malmo capabilities and bug fixes
- **CI stability**: No manual intervention needed for Malmo updates
- **Reproducibility option**: Can pin by digest for production deployments

**To pin by digest for reproducible builds:**

```bash
# Get current digest
docker inspect andkram/malmo:latest --format='{{index .RepoDigests 0}}'
# Example output: andkram/malmo@sha256:abc123...

# Update docker-compose.malmo.yml:
image: andkram/malmo@sha256:abc123...
```

### Smoke Test Workflows

- **Malmo Live**: Tests full Minecraft+Malmo integration (manual trigger, weekly cron)
- **Ludii Sidecar**: Tests Java sidecar integration (manual trigger)

See [Performance](#performance) section for benchmark artifacts and regression tracking.

## Local CI Dry Run

Before pushing, run the full CI suite locally:

**Windows:**
```cmd
scripts\local-ci.bat
```

**Linux/macOS:**
```bash
bash scripts/local-ci.sh
```

This mirrors `.github/workflows/ci.yml` and catches issues early.

## Schemathesis (Contract Testing)

- The `Schemathesis Contract Tests` workflow runs against the sidecar OpenAPI (`/v3/api-docs`).
- Reproduce locally with a known seed:
```bash
schemathesis run http://localhost:8080/v3/api-docs \
  --checks all \
  --wait-for-schema 60 \
  --generation-maximize response_time \
  --hypothesis-seed <seed> \
  --max-workers 1 \
  --hypothesis-max-examples 1000 \
  --show-errors-traceback
```
- CI publishes `schemathesis-report.json` and `sidecar.log` as artifacts on failure.

## Structure

- `backend/` — FastAPI app with health, info, and adapter demo endpoints
- `backend/app/settings.py` — Provider configuration and Azure settings
- `backend/app/dependencies.py` — Dependency injection for adapters
- `backend/app/logging_config.py` — Structured logging configuration (structlog)
- `backend/app/middleware.py` — Request logging and correlation middleware
- `backend/app/telemetry.py` — Azure Monitor OpenTelemetry integration (optional)
- `core/ports/` — Protocol interfaces for adapters (Storage, Secrets, Queue, GameEnv)
- `core/adapters/` — Adapter implementations:
  - `null.py` — In-memory adapters for local development
  - `azure_storage.py` — Azure Blob Storage adapter (with logging)
  - `azure_secrets.py` — Azure Key Vault adapter (with logging)
  - `azure_queue.py` — Azure Service Bus adapter (with logging)
  - `cached_secrets.py` — TTL cache wrapper for secrets
- `frontend/` — Minimal static page with diagnostics panel
- `.github/workflows/ci.yml` — CI pipeline (lint, type-check, tests with 70% coverage gate, security audit)
- `scripts/` — Local CI dry run scripts
- `Dockerfile` & `netlify.toml` — Deployment configs
- `OBSERVABILITY.md` — Detailed observability guide

## Architecture

RediWrapper uses **Ports & Adapters (Hexagonal Architecture)**:
- **Ports** define interfaces in `core/ports/`
- **Adapters** implement ports for specific providers:
  - **Null adapters** (in-memory) for local development
  - **Azure adapters** for cloud-backed storage, secrets, and queues
- **Dependency injection** via FastAPI `Depends()` allows swapping adapters
- **Provider switch** via `PROVIDER` environment variable

## Azure Setup

### Prerequisites

1. Azure subscription
2. Azure CLI: `az login`
3. Install Azure SDKs: `pip install -e '.[azure]'`

### Resource Creation

```bash
# Set variables
RESOURCE_GROUP="rediwrapper-rg"
LOCATION="eastus"
STORAGE_ACCOUNT="rediwrapperstorage"  # Must be globally unique
KEY_VAULT="rediwrapper-kv"             # Must be globally unique
SERVICE_BUS="rediwrapper-sb"           # Must be globally unique

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Storage Account
az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

# Create container
az storage container create \
  --account-name $STORAGE_ACCOUNT \
  --name rediwrapper-container \
  --auth-mode login

# Create Key Vault
az keyvault create \
  --name $KEY_VAULT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Create Service Bus
az servicebus namespace create \
  --name $SERVICE_BUS \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard

az servicebus queue create \
  --namespace-name $SERVICE_BUS \
  --resource-group $RESOURCE_GROUP \
  --name rediwrapper-queue
```

### RBAC Role Assignments

Assign appropriate roles to your user or managed identity:

```bash
# Get your user's object ID
USER_ID=$(az ad signed-in-user show --query id -o tsv)

# Storage: Blob Data Contributor
az role assignment create \
  --role "Storage Blob Data Contributor" \
  --assignee $USER_ID \
  --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$STORAGE_ACCOUNT"

# Key Vault: Use RBAC permission model (recommended)
az keyvault update \
  --name $KEY_VAULT \
  --enable-rbac-authorization true

az role assignment create \
  --role "Key Vault Secrets Officer" \
  --assignee $USER_ID \
  --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$KEY_VAULT"

# Service Bus: Data Sender
az role assignment create \
  --role "Azure Service Bus Data Sender" \
  --assignee $USER_ID \
  --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ServiceBus/namespaces/$SERVICE_BUS"
```

### Environment Variables

```bash
export PROVIDER=azure
export AZURE_STORAGE_ACCOUNT_URL=https://$STORAGE_ACCOUNT.blob.core.windows.net
export AZURE_STORAGE_CONTAINER=rediwrapper-container
export KEY_VAULT_NAME=$KEY_VAULT
export SERVICE_BUS_FQDN=$SERVICE_BUS.servicebus.windows.net
export SERVICE_BUS_QUEUE=rediwrapper-queue
```

### Troubleshooting

**Authentication issues:**
- Ensure `az login` is successful
- Check RBAC roles: `az role assignment list --assignee $USER_ID`
- Wait 5-10 minutes after role assignment for propagation

**SDK logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

See `docs/AZURE_SETUP.md` for more details.

See `ROADMAP.md` for phases and milestones.
