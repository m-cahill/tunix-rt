# Tunix RT - Reasoning-Trace Framework

**Milestone M36 Complete** ✅  
**Coverage:** >70% Backend Line (380+ tests) | **Training:** End-to-End Loop + Evaluation | **Ops:** JAX/Flax Pipeline | **Architecture:** Router-based (10 modules) | **Data:** dev-reasoning-v2 (550 traces) | **Eval:** eval_v2.jsonl (100 items) | **Tuning:** Ray Tune Sweep Runner | **Kaggle:** Evidence Lock v2

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

**M23 Enhancements:** Evaluation Engine Completion - Real scoring via `AnswerCorrectnessJudge` with `predictions.jsonl` contract, generation step in execution pipeline, Strict `LOCKED_METRICS` enforcement (`answer_correctness`), Backend coverage restored to 70%+, Frontend test stability fixes (`act()` warnings eliminated), Judge validation tests.

**M24 Enhancements:** Deterministic Builds & Inference - `uv.lock` for reproducible dependencies, Real Inference integration using Transformers fallback (replacing placeholders), Test suite stabilization, and audit fixes.

**M25 Enhancements:** Real Tunix/JAX Training Path - JAX/Flax/Optax SFT implementation (`train_jax.py`), Split training orchestrator (`train_sft_tunix.py`) supporting Torch and JAX backends, Coverage restored to >70%, Inference fragility fixes, Metadata recording (`predictions_meta.json`), Device selection (`auto/cpu/cuda`).

**M26 Enhancements:** Training Readiness Complete - GPU selection (`--device gpu`), Provenance tracking, Orbax Checkpointing/Resume, Throughput Benchmarking (`bench_jax.py`), Scaled dataset (`golden-v2` with 100 traces), Metrics Visualization (UI + API), and `CONTRIBUTING.md`.

**M27 Enhancements:** End-to-End Training Validation & Evaluation Loop - Full training convergence on `golden-v2` (loss 2.26→0.08), Real inference using `FlaxAutoModelForCausalLM`, `--dataset` CLI argument for training scripts, `--eval_after_train` flag for offline evaluation, Auto-detect JAX/PyTorch backend in `eval_generate.py`, Dataset tooling (`seed_golden_v2.py`, `export_dataset.py`, `cleanup_dataset.py`), Training config `train_golden_v2.yaml`, End-to-end documentation (`docs/training_end_to_end.md`).

**M28 Enhancements:** Hyperparameter Tuning & Leaderboard - Ray Tune integration for optimization sweeps (`scripts/run_m28_sweep.py`), Run Comparison UI (`RunComparison.tsx`) with loss curve overlay and deep-link support, `answer_correctness` metric wired to leaderboard, UNGAR Episode API fix (`high_card_duel.py`), CI fixes (`cyclonedx-py` flag update, `ruff format`).

**M29 Enhancements:** Competition-Ready Data + App Router Modularization - Router refactor (10 modules, `app.py` reduced to 56 lines), TODO resolution (`lower_is_better` configuration for regression baselines), Dataset pipeline enhancements (provenance metadata, `POST /api/datasets/ingest` endpoint), `dev-reasoning-v1` seed script (200 traces: 70% reasoning + 30% synthetic), Kaggle submission path (`notebooks/kaggle_submission.ipynb` + docs), Nightly CI workflow, Frontend client exports test.

**M30 Enhancements:** Polish & Submission Prep - Type ignore cleanup (narrowed with error codes, rationale comments), HTTP 422 deprecation fix (`HTTP_422_UNPROCESSABLE_CONTENT`), Router docstrings expanded (summary-level, 3-8 lines per module), Dataset ingest E2E test (`e2e/tests/datasets_ingest.spec.ts`), `dev-reasoning-v1` standardized to versioned folder format, Kaggle dry-run verified, Submission checklist (`docs/submission_checklist.md`) with video requirements (3 min, YouTube, Kaggle Media Gallery).

**M31 Enhancements:** Final Submission Package - Submission freeze document (`docs/submission_freeze.md`) for reproducibility, Gemma submission configs (`submission_gemma3_1b.yaml`, `submission_gemma2_2b.yaml`), Kaggle notebook refactored with smoke/full run modes and subprocess-based execution, One-command packaging tool (`backend/tools/package_submission.py`), Submission artifacts documentation (`docs/submission_artifacts.md`), Video script and shot list (`docs/video_script_m31.md`, `docs/video_shotlist_m31.md`), Updated submission checklist with video URL placeholders.

**M32 Enhancements:** Data Scale-Up & Coverage Uplift - Submission execution runbook (`docs/submission_execution_m32.md`), Evidence capture folder (`submission_runs/m32_v1/`), Scaled dataset (`dev-reasoning-v2` with 550 traces in strict ReasoningTrace schema), Dataset seeder (`seed_dev_reasoning_v2.py` with 70/20/10 composition), 8 schema validation tests, 9 new datasets_ingest unit tests (0%→full coverage), 7 new worker edge case tests, Updated packaging tool with evidence files, 260 total backend tests.

**M33 Enhancements:** Kaggle Submission Rehearsal v1 + Evidence Lock - Notebook updated to m33_v1 with `dev-reasoning-v2` default dataset, Evidence folder (`submission_runs/m33_v1/`) with `run_manifest.json`, `eval_summary.json`, `kaggle_output_log.txt`, Packaging tool enhanced with `--run-dir` argument for evidence bundling, 13 new evidence schema validation tests (`test_evidence_files.py`), Submission checklist strengthened with explicit ≤3 min YouTube requirement, Local CPU/smoke rehearsal captured, Archive prefix updated to m33, 273 total backend tests.

**M34 Enhancements:** Optimization Loop 1 — Primary Score + Ray Tune Sweep Infrastructure:
- **Primary Score Module** (`scoring.py`): `compute_primary_score()` function for canonical 0-1 metric aggregation (prefers `answer_correctness`, fallback to normalized `score/100`), 19 unit tests.
- **Evaluation API Enhancement**: `primary_score` field added to `EvaluationResponse` and `LeaderboardItem` schemas, automatically computed from metrics.
- **Sweep Runner Module** (`tunix_rt_backend/tuning/`): Reusable `SweepRunner` class with `SweepConfig` dataclass for Ray Tune sweeps via API, context manager support, timeout handling, 14 unit tests.
- **M34 Sweep Script** (`backend/tools/run_tune_m34.py`): One-command sweep with M34 defaults (dev-reasoning-v2, gemma-3-1b-it, 5-20 trials, learning_rate/batch_size/weight_decay/warmup_steps search space).
- **Best Config Template** (`training/configs/m34_best.yaml`): Template for promoted hyperparameters with provenance section.
- **Evidence Folder** (`submission_runs/m34_v1/`): `run_manifest.json` (with tuning_job_id/trial_id fields), `eval_summary.json`, `best_params.json`, `kaggle_output_log.txt`.
- **M34 Evidence Tests**: 15 new tests validating M34 schema including `best_params.json`.
- Updated M28 sweep script to use shared `SweepRunner`, backward compatible.
- Archive prefix updated to m34, 310+ total backend tests.

**M35 Enhancements:** Quality Loop 1 — Eval Signal Hardening + Leaderboard Fidelity + Regression Guardrails:
- **Eval Set v2** (`training/evalsets/eval_v2.jsonl`): 100 items with 60/25/15 split (core/trace_sensitive/edge_case), purpose-built for scoring with section/category/difficulty labels.
- **Eval Set Validator** (`backend/tools/validate_evalset.py`): CLI tool to validate eval set schema, print summary stats, exits non-zero on errors.
- **Scorecard Aggregator** (`scoring.py`): New `Scorecard` dataclass and `compute_scorecard()` function providing n_items, n_scored, n_skipped, stddev, per-section/category/difficulty breakdowns.
- **Leaderboard Filtering** (API): `GET /api/tunix/evaluations` now supports `dataset_key`, `model_id`, `config_path`, `date_from`, `date_to` query params with AND logic.
- **Leaderboard UI** (`Leaderboard.tsx`): Inline filter inputs, scorecard summary (items/scored), primary_score display, dataset column.
- **Run Comparison** (`RunComparison.tsx`): New collapsible per-item diff table showing expected vs predicted correctness.
- **Regression Baseline Enhancements** (`regression.py`): `primary_score` as default metric, `eval_set`/`dataset_key` fields for multi-baseline scoping.
- **Determinism Check** (`backend/tools/check_determinism.py`): Standalone script verifying evaluation pipeline produces identical results on repeated runs.
- **Evidence Folder** (`submission_runs/m35_v1/`): `run_manifest.json` (with eval_set field), `eval_summary.json` (with scorecard), `kaggle_output_log.txt`.
- **Schema Tests**: 12 new M35 evidence tests, 23 eval set validator tests, 15 scorecard tests, 14 regression tests.
- Archive prefix updated to m35, 370+ total backend tests.

**M36 Enhancements:** Real Kaggle Execution + Evidence Lock v2 + Quick-Win Audit Uplift:
- **Model Selection**: Gemma 2 2B (Flax-compatible). Note: Gemma 3 1B is NOT supported by Flax; use Gemma 2 2B for JAX/Flax training.
- **Kaggle Evidence Schema**: `run_manifest.json` now includes `kaggle_notebook_url`, `kaggle_notebook_version`, `kaggle_run_id` fields for evidence capture.
- **Notebook Updates**: Version m36_v5, uses absolute paths for clone (prevents nesting), config-based model selection (`submission_gemma2_2b.yaml`), RESULT SUMMARY block for evidence capture.
- **Training Script Fix**: `train_jax.py` now supports both `model.model_id` and `model.name` config keys.
- **Kaggle Execution Runbook** (`docs/M36_KAGGLE_RUN.md`): Step-by-step instructions for Kaggle execution, evidence field mapping, troubleshooting guide.
- **Frontend Test Coverage**: 10 new `Leaderboard.test.tsx` tests (loading/empty/error states, filters, pagination, scorecard), 5 new `LiveLogs.test.tsx` tests (SSE events, connection status).
- **Per-Item Predictions Documentation** (`docs/evaluation.md`): Current state, Run Comparison limitations, planned M37 artifact storage.
- **Evidence Folder** (`submission_runs/m36_v1/`): Templates for real Kaggle run evidence with scorecard structure.
- **M36 Evidence Tests**: 12 new tests validating M36 schema including Kaggle-specific fields.
- Archive prefix updated to m36, 380+ total backend tests, 75 frontend tests.


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

### `regression_baselines` (M18, updated M29)
Baseline configurations for regression testing.
- `id` (UUID, PK)
- `name` (String, Unique)
- `run_id` (UUID, FK)
- `metric` (String)
- `lower_is_better` (Boolean, nullable) — Added M29
- `created_at` (DateTime)

### `tunix_run_evaluations` (M17)
Evaluation results for training runs.
- `id` (UUID, PK)
- `run_id` (UUID, FK)
- `score` (Float)
- `verdict` (String)
- `details` (JSON)
- `judge_name` (String)
- `judge_version` (String)
- `evaluated_at` (DateTime)

### `tunix_run_log_chunks` (M16)
Streaming log storage for training runs.
- `id` (UUID, PK)
- `run_id` (UUID, FK)
- `chunk_index` (Integer)
- `content` (Text)
- `stream` (String) — 'stdout' or 'stderr'
- `created_at` (DateTime)
