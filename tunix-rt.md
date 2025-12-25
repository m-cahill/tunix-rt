# Tunix RT - Reasoning-Trace Framework

**Milestone M22 Complete** ✅  
**Coverage:** 67% Backend Line (new code), 45 Frontend Tests | **Evaluation:** answer_correctness@v1 frozen | **Dataset:** golden-v1 seeded | **Training:** Readiness checklist complete

## Overview

Tunix RT is a full-stack application for managing reasoning traces and integrating with the RediAI framework for the Tunix Hackathon. The system provides health monitoring, RediAI integration, and a foundation for trace-quality optimization workflows.

**M1 Enhancements:** Enterprise-grade testing (90% branch coverage), security scanning (pip-audit, npm audit, gitleaks), configuration validation, TTL caching, and developer experience tools.

**M2 Enhancements:** Database integration (async SQLAlchemy + PostgreSQL), Alembic migrations, trace CRUD API (create/retrieve/list), frontend trace UI, comprehensive validation, and payload size limits.

**M3 Enhancements:** Trace system hardening - DB connection pool settings applied, created_at index for list performance, frontend trace UI unit tests (8 total), frontend coverage artifact generation confirmed, Alembic auto-ID migration policy documented, curl API examples, and DB troubleshooting guide.

**M4 Enhancements:** Deterministic E2E testing - Postgres service container in CI, `skip-locked` job distribution logic, explicit `DATABASE_URL` configuration, reliable health checks, and standardized port bindings.

**M5 Enhancements:** Enterprise-grade code coverage - Enforced 80% line / 68% branch gates, dedicated coverage reports, `coverage_gate.py` tool, and removal of synthetic coverage hacks.

**M6 Enhancements:** Validation Refactoring - Centralized validation logic into `helpers/`, eliminated duplication across endpoints, improved error consistency, and boosted maintainability while increasing test coverage.

**M7 Enhancements:** UNGAR Integration - Optional integration with the UNGAR game engine for synthetic data generation (High Card Duel), JSONL export, and graceful degradation when dependencies are missing.

**M8 Enhancements:** Dataset Pipeline - Versioned dataset creation (`POST /api/datasets/build`), manifest-based reproducibility, Tunix SFT prompt rendering, and training smoke tests.

**M9 Enhancements:** Reproducible Training Loop - Training manifests, deterministic evaluation loop, static eval sets, batch trace import, and comprehensive documentation for the full trace-to-training lifecycle.

**M10 Enhancements:** App Layer Refactor - Service layer extraction, thin controllers, typed export formats, performance optimization for batch operations, and architectural guardrails documentation.

**M11 Enhancements:** Stabilization & Hardening - Complete service extraction (<600 line app.py), security hardening (SHA-pinned CI, SBOM, pre-commit), production documentation (SLOs, training guides), and Windows compatibility fixes.

**M12 Enhancements:** Tunix Integration Skeleton - Mock-first Tunix integration, portable JSONL/YAML artifact generation, frontend integration panel, and 14 new tests without runtime dependencies.

**M13 Enhancements:** Tunix Runtime Execution - Optional local execution of training runs, dry-run validation, subprocess output capture, graceful 501 degradation, and separate CI workflow for runtime tests.

**M14 Enhancements:** Tunix Run Registry - persistent storage with `tunix_runs` table (UUID PK, indexed columns), Alembic migration, immediate run persistence (create with status="running", update on completion), graceful DB failure handling, `GET /api/tunix/runs` with pagination/filtering, `GET /api/tunix/runs/{run_id}` for details, frontend Run History panel (collapsible, manual refresh), stdout/stderr truncation (10KB), 12 new backend tests + 7 frontend tests (all dry-run, no Tunix dependency).

**M15 Enhancements:** Async Execution Engine - `POST /api/tunix/run?mode=async` for non-blocking enqueue, dedicated worker process (`worker.py`) using Postgres `SKIP LOCKED` for robust job claiming, status polling endpoint, frontend "Run Async" toggle with auto-refresh, Prometheus metrics (`/metrics`) for run counts/duration/latency, `config` column migration for deferred execution parameters.

**M16 Enhancements:** Operational UX - Real-time log streaming via SSE (`/api/tunix/runs/{id}/logs`), Run cancellation (`POST /cancel`) with worker termination, Artifacts/Checkpoints management (`/artifacts` list + download), Hardening (pinned dependencies, optimized trace batch insertion), `tunix_run_log_chunks` table for streaming persistence.

**M17 Enhancements:** Evaluation & Quality Loop - `tunix_run_evaluations` table for persisting scores, "Mock Judge" for deterministic evaluation, Auto-trigger on run completion (async/sync), Leaderboard UI (`/leaderboard`), Manual re-evaluation API (`POST /evaluate`), Dry-run exclusion.

**M18 Enhancements:** Judge Abstraction & Regression - `GemmaJudge` implementation (using RediAI), `regression_baselines` table and endpoints (`POST /api/regression/baselines`, `/check`), Leaderboard pagination (API + UI), Pluggable Judge interface.

**M19 Enhancements:** Hyperparameter Tuning - Ray Tune integration for optimization sweeps (`/api/tuning/jobs`), `TunixTuningJob` and `TunixTuningTrial` tables, Search Space validation, Automated best-run selection, Frontend Tuning UI.

**M20 Enhancements:** Model Registry - `ModelArtifact` and `ModelVersion` tables, content-addressed storage for artifacts, promotion from TunixRun, API endpoints for artifact management and download, Frontend Registry UI.

**M21 Enhancements:** Security & Reliability Hardening - Frontend dependency audit clean (Vite/esbuild patched), `setuptools` CVE patched, Dependencies pinned (`save-exact=true`), E2E failure observability (traces retained, stderr attached), "Promote Best Trial" UI workflow, Repo hygiene (`ProjectFiles/Workflows` ignored).

**M22 Enhancements:** Training Readiness - Backend coverage 67% (new code for judges/evaluation), `answer_correctness@v1` evaluator (AnswerCorrectnessJudge with deterministic normalization), Golden Dataset (`golden-v1`) seed script with 5 curated test cases, Empty dataset guardrails, Metrics visibility in UI (Registry/History/Tuning), Tuning guardrails against undefined metrics, Idempotency tests for model promotion, Frontend tests for "Promote Best" workflow (Tuning.test.tsx), Frozen evaluation semantics documentation (docs/evaluation.md), Training readiness checklist (docs/training_readiness.md).


## Database Schema

### `traces`
Core table for reasoning traces.
- `trace_id` (UUID, PK)
- `trace_version` (String)
- `payload` (JSON)
- `created_at` (DateTime)

### `scores`
Scores associated with traces.
- `score_id` (UUID, PK)
- `trace_id` (UUID, FK)
- `score` (Float)
- `details` (JSON)
- `created_at` (DateTime)

### `tunix_runs`
Training run execution records.
- `run_id` (UUID, PK)
- `dataset_key` (String)
- `model_id` (String)
- `status` (String)
- `metrics` (JSON)
- `config` (JSON)
- `created_at` (DateTime)

### `tunix_tuning_jobs` (M19)
Hyperparameter tuning jobs.
- `id` (UUID, PK)
- `name` (String)
- `search_space_json` (JSON)
- `best_run_id` (UUID, FK)
- `status` (String)

### `tunix_tuning_trials` (M19)
Individual trials within a tuning job.
- `id` (String, PK)
- `tuning_job_id` (UUID, FK)
- `run_id` (UUID, FK)
- `params_json` (JSON)
- `metric_value` (Float)

### `model_artifacts` (M20)
Logical model families.
- `id` (UUID, PK)
- `name` (String, Unique)
- `task_type` (String)
- `created_at`, `updated_at` (DateTime)

### `model_versions` (M20)
Immutable versions of artifacts.
- `id` (UUID, PK)
- `artifact_id` (UUID, FK)
- `version` (String)
- `source_run_id` (UUID, FK)
- `status` (String)
- `metrics_json` (JSON)
- `config_json` (JSON)
- `provenance_json` (JSON)
- `storage_uri` (String)
- `sha256` (String)
- `created_at` (DateTime)
