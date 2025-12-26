# M29 Clarifying Questions

After analyzing the codebase, I have the following questions before proceeding:

---

## Phase 0 — Baseline Gate

1. **Branch Creation**: Should I create the branch `milestone/M29-competition-data-and-routers` from `main`, or is there a different base branch you prefer?

---

## Phase 1 — `app.py` Modularization

2. **Router Granularity**: The plan mentions these routers:
   - `routers/health.py`
   - `routers/traces.py`
   - `routers/datasets.py`
   - `routers/tunix_runs.py`
   - `routers/tuning.py`
   
   Looking at `app.py` (~1,885 lines), I see these route groups:
   - Health endpoints (`/api/health`, `/api/redi/health`, `/metrics`)
   - Traces CRUD (`/api/traces`, `/api/traces/{id}`, etc.)
   - UNGAR integration (`/api/ungar/*`)
   - Datasets (`/api/datasets/*`)
   - Tunix integration (`/api/tunix/status`, `/api/tunix/sft/*`)
   - Tunix runs (`/api/tunix/run`, `/api/tunix/runs/*`)
   - Evaluation (`/api/tunix/runs/{id}/evaluate`, `/api/tunix/evaluations`)
   - Regression (`/api/regression/*`)
   - Tuning (`/api/tuning/*`)
   - Model Registry (`/api/models/*`)

   **Question**: Should I create separate routers for each group (10+ files), or consolidate some? For example:
   - Option A: Keep as suggested in plan (5-6 routers, consolidate related functionality)
   - Option B: Create fine-grained routers matching each endpoint group (10+ files)
   
   My recommendation: **Option A** with these groupings:
   - `health.py` — health, metrics
   - `traces.py` — traces CRUD + compare
   - `datasets.py` — dataset build/export
   - `ungar.py` — UNGAR generation/export
   - `tunix.py` — tunix status, sft export/manifest, run execution
   - `tunix_runs.py` — run list/details, logs, artifacts, metrics, cancellation
   - `evaluation.py` — evaluation trigger/get, leaderboard
   - `regression.py` — baselines + checks
   - `tuning.py` — tuning jobs CRUD
   - `models.py` — model registry + versions

   **Does this grouping work for you, or do you want fewer/different splits?**

---

## Phase 2 — TODO/FIXME Resolution

3. **TODO Disposition**: I found 3 TODO markers:
   - `model_registry.py:57` — "Add pagination if needed"
   - `model_registry.py:159` — "Populate from Evaluations if available"
   - `regression.py:123` — "Support lower is better configuration"

   **Options:**
   - A. Implement them now (adds scope)
   - B. Convert to GitHub issues and remove from code
   - C. Keep as documented exceptions with rationale
   
   **Which approach do you prefer?**

---

## Phase 3 — Dataset Scale-Up Pipeline

4. **Dataset Builder Endpoint**: The plan mentions `POST /api/datasets/build`. This endpoint already exists in `app.py` (lines 570-616). Do you want me to:
   - A. Enhance the existing endpoint with provenance metadata
   - B. Create a new ingestor endpoint (e.g., `POST /api/datasets/ingest` for JSONL import)
   - C. Both

5. **"Bigger-than-golden" Dataset**: The plan mentions creating `dev-reasoning-v1` with "a few hundred items."
   - Should this be generated from UNGAR episodes, or should I create a static seed script?
   - Is there a specific format/content type you want (High Card Duel, math reasoning, other)?

---

## Phase 4 — Kaggle Submission Path

6. **Notebook Location**: The plan mentions creating a `notebooks/` folder. Should this be:
   - A. At repo root (`notebooks/kaggle_submission.ipynb`)
   - B. Inside training directory (`training/notebooks/`)
   - C. Other location

7. **Notebook Format**: Kaggle uses `.ipynb` format. Do you want:
   - A. A full Jupyter notebook (`.ipynb`)
   - B. A Python script that can be converted to notebook (`kaggle_submission.py`)
   - C. Both

8. **Training Backend**: Should the Kaggle notebook use:
   - A. JAX/Flax path (existing `train_jax.py`)
   - B. PyTorch path (existing `train_torch.py`)
   - C. Tunix orchestrator (`train_sft_tunix.py` with backend selection)

---

## Phase 5 — CI Nightly Tier

9. **Nightly Scope**: The plan marks this as "optional but recommended." Do you want me to:
   - A. Skip for M29, defer to M30
   - B. Implement a basic nightly workflow (just runs all tests on schedule)
   - C. Implement full nightly with extended dataset validation + smoke train

---

## General

10. **Priority Order**: If time is constrained, which phases are must-haves vs nice-to-haves?
    - My understanding of priority: Phase 1 (routers) > Phase 3 (data pipeline) > Phase 4 (Kaggle) > Phase 2 (TODOs) > Phase 5 (nightly)

11. **E2E Test Updates**: If any frontend API client paths change due to router extraction, should I update the frontend client and E2E tests in the same PR, or defer to a follow-up?

---

Please respond to these questions, and I'll finalize the todo list and begin implementation!
