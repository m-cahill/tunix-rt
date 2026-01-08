# Tunix RT Demo Guide

This guide explains how Tunix RT works and how to run a demonstration.

## Overview

Tunix RT is a **reasoning-trace framework** that:
1. Collects and stores reasoning traces (prompt → steps → final answer)
2. Trains models to improve reasoning quality
3. Evaluates and compares model performance via leaderboards

---

## Demo Flow (What the Application Does)

### 1. Health Dashboard
When the app loads, it displays the health status of:
- **API**: The Tunix RT backend
- **RediAI**: Integration with the RediAI framework
- **UNGAR**: Optional game engine for synthetic data generation
- **Tunix**: Training runtime status

### 2. Reasoning Traces
Users can:
- **Upload** reasoning traces in JSON format
- **Fetch** existing traces by ID
- **Compare** two traces side-by-side with computed scores

### 3. UNGAR Generator (Optional)
If UNGAR is installed, users can generate synthetic reasoning traces from game simulations (High Card Duel).

### 4. Tunix Integration
The Tunix section allows:
- **Export JSONL**: Download training data in SFT format
- **Generate Manifest**: Create training configuration YAML
- **Run Training**: Execute training runs (dry-run or local execution)
- **View Run History**: Track past training runs with status, duration, and metrics

### 5. Leaderboard
Navigate to the Leaderboard tab to see:
- Ranked evaluation results by primary score
- Filtering by dataset, model, or date
- Scorecard details (items scored, standard deviation)

### 6. Tuning (M19)
The Tuning tab shows hyperparameter optimization jobs:
- Search space configuration
- Trial results with metrics
- "Promote Best" workflow to register winning models

### 7. Model Registry (M20)
The Registry tab displays:
- Registered model artifacts
- Version history with provenance
- Promotion status and metrics

---

## Running a Demo Locally

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ with `uv` installed
- Node.js 18+

### Quick Start

#### 1. Start Database
```powershell
docker compose up -d postgres
```

#### 2. Apply Migrations
```powershell
cd backend
uv sync --all-extras
uv run alembic upgrade head
```

#### 3. Start Backend
```powershell
uv run uvicorn tunix_rt_backend.app:app --reload --port 8000
```

#### 4. Start Frontend (in a new terminal)
```powershell
cd frontend
npm install
npm run dev
```

#### 5. Open the App
Navigate to: **http://localhost:5173**

---

## Demo Script (Video Recording Reference)

### Scene 1: App Overview (~30 sec)
- Show the main dashboard with health indicators
- Point out API: healthy, RediAI status

### Scene 2: Trace Upload (~45 sec)
- Click "Load Example" to populate the trace JSON
- Click "Upload" to store the trace
- Click "Fetch" to retrieve and display it

### Scene 3: Trace Comparison (~30 sec)
- Enter two trace IDs
- Click "Fetch & Compare"
- Show side-by-side with scores

### Scene 4: Training Run (~60 sec)
- Enter a dataset key (e.g., `dev-reasoning-v2`)
- Click "Run (Dry-run)" to validate
- Expand Run History to see the recorded run
- Click "View" to see run details and stdout/stderr

### Scene 5: Leaderboard (~30 sec)
- Navigate to Leaderboard tab
- Show ranked results by primary score
- Apply a filter (e.g., by dataset)

### Scene 6: Model Registry (~15 sec)
- Navigate to Registry tab
- Show a registered model artifact with version history

---

## Evidence Locations

After running training, evidence is stored in:
```
submission_runs/mXX_v1/
├── run_manifest.json    # Run metadata (dataset, model, config)
├── eval_summary.json    # Evaluation results with scorecard
├── kaggle_output_log.txt # Training stdout/stderr
└── screenshots/         # UI screenshots (if captured)
```

---

## GPU Training (RTX 5090)

For local GPU training on RTX 5090 (Blackwell architecture), see the **GPU Development** section in `CONTRIBUTING.md`. Key points:
- Requires separate `.venv-gpu` environment
- Uses PyTorch nightly cu128 wheels
- Validated at 31.7 samples/second

---

## Troubleshooting

### Backend not starting?
- Check PostgreSQL is running: `docker compose ps`
- Verify migrations applied: `uv run alembic current`

### Frontend not loading?
- Ensure backend is running on port 8000
- Check browser console for errors

### Training run fails?
- Check dataset exists: `GET /api/datasets/{key}/export.jsonl`
- For GPU runs, ensure `.venv-gpu` is activated
