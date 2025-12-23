# M07: UNGAR Integration Documentation

**Status:** ✅ Complete  
**UNGAR Version:** Pinned to commit `0e29e104aa1b13542b193515e3895ee87122c1cb`  
**Integration Type:** Optional bridge (no core coupling)

## Overview

M07 adds **UNGAR (Universal Neural Grid for Analysis and Research)** as an optional data source for tunix-rt. UNGAR can generate reasoning traces from High Card Duel game episodes, providing structured training data for Tunix workflows.

### Key Features

1. **Optional Dependency:** UNGAR is not required for core tunix-rt functionality
2. **High Card Duel Generator:** Converts game episodes into reasoning traces
3. **JSONL Export:** Tunix-friendly format for training pipelines
4. **Frontend Integration:** Minimal UI panel for trace generation
5. **Stable Testing:** Default tests pass without UNGAR; optional tests validate integration

## Installation

### Standard Installation (No UNGAR)

```bash
cd backend
pip install -e ".[dev]"
```

Tunix-rt works fully without UNGAR. The `/api/ungar/*` endpoints will return appropriate 501 status codes.

### UNGAR Integration Installation

```bash
cd backend
pip install -e ".[dev,ungar]"
```

This installs UNGAR from the pinned GitHub commit for reproducible builds.

#### Local Development with UNGAR

If you have UNGAR cloned as a sibling directory:

```bash
cd backend
pip install -e ".[dev]"
pip install -e ../../ungar  # Adjust path as needed
```

---

## Quick Start Example (Happy Path)

Complete workflow from installation to JSONL export:

```bash
# 1. Install with UNGAR
cd backend
pip install -e ".[dev,ungar]"

# 2. Start server (new terminal)
uvicorn tunix_rt_backend.app:app --reload

# 3. Verify UNGAR available (in another terminal)
curl http://localhost:8000/api/ungar/status
# Expected: {"available": true, "version": "unknown"}

# 4. Generate 5 traces
curl -X POST http://localhost:8000/api/ungar/high-card-duel/generate \
  -H "Content-Type: application/json" \
  -d '{"count": 5, "seed": 42}' | jq '.'

# 5. Export to JSONL
curl "http://localhost:8000/api/ungar/high-card-duel/export.jsonl?limit=10" > traces.jsonl

# 6. Verify format
head -1 traces.jsonl | jq '.'
# Expected: {"id": "...", "prompts": "...", "trace_steps": [...], "final_answer": "...", "metadata": {...}}
```

## API Endpoints

### 1. UNGAR Status

**Endpoint:** `GET /api/ungar/status`

**Description:** Check if UNGAR is installed and available.

**Response (UNGAR not installed):**
```json
{
  "available": false,
  "version": null
}
```

**Response (UNGAR installed):**
```json
{
  "available": true,
  "version": "unknown"
}
```

**Status Code:** Always `200 OK`

**Example:**
```bash
curl http://localhost:8000/api/ungar/status
```

### 2. Generate High Card Duel Traces

**Endpoint:** `POST /api/ungar/high-card-duel/generate`

**Description:** Generate N High Card Duel episodes and convert them to traces.

**Request Body:**
```json
{
  "count": 10,
  "seed": 42,
  "persist": true
}
```

**Parameters:**
- `count` (required): Number of episodes to generate (1-100)
- `seed` (optional): Random seed for reproducibility
- `persist` (optional, default `true`): Whether to save traces to database

**Response (Success):**
```json
{
  "trace_ids": [
    "550e8400-e29b-41d4-a716-446655440000",
    "660f9500-f39c-52e5-b827-557766551111"
  ],
  "preview": [
    {
      "trace_id": "550e8400-e29b-41d4-a716-446655440000",
      "game": "high_card_duel",
      "result": "win",
      "my_card": "AS"
    }
  ]
}
```

**Status Codes:**
- `201 Created`: Traces generated successfully
- `422 Unprocessable Entity`: Invalid count (not 1-100)
- `501 Not Implemented`: UNGAR not installed

**Example:**
```bash
curl -X POST http://localhost:8000/api/ungar/high-card-duel/generate \
  -H "Content-Type: application/json" \
  -d '{"count": 5, "seed": 42, "persist": true}'
```

### 3. Export Traces as JSONL

**Endpoint:** `GET /api/ungar/high-card-duel/export.jsonl`

**Description:** Export UNGAR-generated traces in Tunix-friendly JSONL format.

**Query Parameters:**
- `limit` (optional, default 100): Maximum traces to export
- `trace_ids` (optional): Comma-separated UUIDs to export specific traces

**Response:** `application/x-ndjson` (one JSON object per line)

**JSONL Format:**
```jsonl
{"id": "550e8400-e29b-41d4-a716-446655440000", "prompts": "High Card Duel: You have 1 hidden card. Action: reveal.", "trace_steps": ["Legal moves: [reveal]", "My hand: AS", "Unseen cards: 51", "Action chosen: reveal"], "final_answer": "reveal", "metadata": {"created_at": "2025-12-21T10:30:00Z", "source": "ungar", "game": "high_card_duel", "seed": 42, "episode_index": 0, "my_card": "AS", "opponent_card": "KH", "result": "win", "trace_version": "1.0"}}
```

**Example:**
```bash
# Export latest 10 UNGAR traces
curl "http://localhost:8000/api/ungar/high-card-duel/export.jsonl?limit=10"

# Export specific traces
curl "http://localhost:8000/api/ungar/high-card-duel/export.jsonl?trace_ids=550e8400-e29b-41d4-a716-446655440000"
```

## Trace Format

### High Card Duel Trace Structure

**Prompt:**
```
High Card Duel: You have 1 hidden card. Action: reveal.
```

**Steps:**
1. **Legal Moves:** Lists available actions
2. **Observation:** Card in hand (e.g., "My hand: AS")
3. **Unseen Cards:** Count of unseen cards
4. **Decision:** Action chosen

**Final Answer:**
```
reveal
```

**Metadata:**
```json
{
  "source": "ungar",
  "game": "high_card_duel",
  "episode_index": 0,
  "my_card": "AS",
  "opponent_card": "KH",
  "result": "win",
  "seed": 42
}
```

## Coverage Strategy

### Core vs Optional Coverage

M07 uses a **two-tier coverage strategy** to maintain high quality standards for core code while supporting optional integrations:

**Default CI (Core Coverage):**
- Measures coverage with `.coveragerc` configuration
- Omits `integrations/ungar/high_card_duel.py` (requires UNGAR installed)
- Tests all "UNGAR not installed" code paths (501 responses)
- **Gate:** ≥70% coverage (currently ~84%)
- **Purpose:** Ensure core runtime quality

**Optional UNGAR Workflow (Full Coverage):**
- Measures coverage with `.coveragerc.full` configuration
- Includes all UNGAR code (generator, conversion logic)
- Tests all UNGAR functionality end-to-end
- **Report-only** (non-blocking)
- **Purpose:** Validate optional integration quality

**Rationale:** Optional integration code that requires external dependencies should not dilute core coverage metrics or block CI when dependencies are absent.

## Testing

### Default Tests (No UNGAR)

Run tests without UNGAR installed:

```bash
cd backend
pytest tests/test_ungar.py -v -m "not ungar"
```

**Tests:**
- ✅ Status endpoint returns `available=false`
- ✅ Generate endpoint returns `501 Not Implemented`
- ✅ Export endpoint returns empty JSONL

### Integration Tests (UNGAR Required)

Run tests with UNGAR installed:

```bash
cd backend
pip install -e ".[dev,ungar]"
pytest tests/test_ungar.py -v -m ungar
```

**Tests:**
- ✅ Status endpoint returns `available=true`
- ✅ Generate creates traces successfully
- ✅ Export returns valid JSONL
- ✅ End-to-end: generate → persist → export

### CI/CD

**Default CI:** Runs without UNGAR (fast, stable)

**Optional UNGAR CI:** `.github/workflows/ungar-integration.yml`
- Trigger: Manual dispatch or nightly
- Non-blocking (continues even if fails)
- Installs `backend[ungar]` and runs `pytest -m ungar`

## Frontend Usage

The frontend includes a minimal UNGAR panel (visible when status loads):

1. **Status Display:** Shows "✅ Available" or "❌ Not Installed"
2. **Generator Form:**
   - Trace Count (1-100)
   - Random Seed (optional)
   - Generate button
3. **Results Display:**
   - List of generated trace IDs
   - Preview of first 3 traces (game, result, card)

**Test IDs:**
- `ungar:status` - Status text
- `ungar:generate-count` - Count input
- `ungar:generate-seed` - Seed input
- `ungar:generate-btn` - Generate button
- `ungar:results` - Results container

## Architecture Notes

### Design Principles

1. **Optional by Design:** Core runtime never imports UNGAR
2. **Lazy Loading:** UNGAR imported only inside endpoint functions
3. **Graceful Degradation:** 501 responses when UNGAR unavailable
4. **Bridge Pattern:** UNGAR integration isolated in `tunix_rt_backend/integrations/ungar/`

### File Structure

```
backend/tunix_rt_backend/integrations/ungar/
├── __init__.py
├── availability.py         # UNGAR availability checks
└── high_card_duel.py       # Episode → trace conversion
```

### Conversion Logic

**Episode → Trace Flow:**
1. UNGAR runs N episodes with `play_random_episode()`
2. Each episode result contains:
   - Initial state (my hand, opponent hand)
   - Final state (revealed cards)
   - Returns (win/loss/tie)
3. Converter creates minimal, deterministic natural language:
   - Prompt: Fixed template
   - Steps: Legal moves, observation, unseen count, decision
   - Metadata: Game info, cards, result

## Limitations

1. **Single Game:** Only High Card Duel supported (M07 scope)
2. **Simple NLG:** Minimal natural language (no narrative)
3. **No Training Loop:** Export only; Tunix training integration is future work (M08)
4. **SQLite/PostgreSQL Differences:** Export uses Python-level filtering for compatibility

## Next Steps (M08+)

- **M08:** Multi-game support (Mini Spades, Gin Rummy)
- **M09:** Richer trace schemas with reasoning explanations
- **M10:** Tunix SFT training workflow integration
- **M11:** Bulk export + evaluation loop

## Troubleshooting

### UNGAR Not Installing

**Error:** `Could not find a version that satisfies the requirement ungar`

**Solution:** Check network connection and Git access to `github.com/m-cahill/ungar`

### Tests Failing with UNGAR Installed

**Error:** Conversion functions fail

**Solution:** Ensure UNGAR version matches pinned commit:
```bash
cd ../ungar
git log -1 --format="%H"
# Should output: 0e29e104aa1b13542b193515e3895ee87122c1cb
```

### Frontend Shows "Not Installed" Despite Backend Working

**Solution:** Check UNGAR status endpoint directly:
```bash
curl http://localhost:8000/api/ungar/status
```

If available is `true` but frontend shows "Not Installed", clear browser cache.

## References

- **UNGAR Repository:** https://github.com/m-cahill/ungar
- **UNGAR Documentation:** See `ProjectFiles/ungar-README.md`
- **Tunix Library:** https://github.com/google/tunix
- **Tunix Docs:** https://tunix.readthedocs.io/
