# Tunix RT - Reasoning-Trace Framework

**Milestone M42 Complete** ✅  
**Coverage:** >70% Backend Line | **Training:** PyTorch Local (GPU) + JAX (TPU) | **Hardware:** RTX 5090 (sm_120 Active) | **Frontend:** 75 tests passing | **Status:** SUBMISSION PACKAGE READY — `tunix_rt_m42_2026-01-08_e54267b.zip`

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
- **Model Selection**: Gemma 2B (Flax, via `revision="flax"`). Gemma 2 2B / Gemma 3 are NOT natively supported by `FlaxAutoModelForCausalLM`.
- **Training Modes** (`docs/TRAINING_MODES.md`): GPU Smoke (tiny model) vs TPU Full (Gemma) separation. Gemma 2B OOMs on T4 GPUs even with aggressive optimizations.
- **Smoke Testing**: `training/configs/smoke_tiny.yaml` using `sshleifer/tiny-gpt2`, `--smoke_config` CLI flag, validated on Kaggle GPU.
- **HuggingFace Auth**: Notebook uses Kaggle Secrets for gated model access.
- **Kaggle Evidence Schema**: `run_manifest.json` now includes `kaggle_notebook_url`, `kaggle_notebook_version`, `kaggle_run_id` fields for evidence capture.
- **Notebook Updates**: Version m36_v8, config-based model selection (`submission_gemma_flax.yaml`), smoke/full separation, TPU recommendation.
- **Training Script Fix**: `train_jax.py` supports `revision` param, bfloat16 loading, Adafactor optimizer, GPU preflight warning.
- **Launcher Script**: `training/run_train_jax.py` sets XLA env vars before JAX import to prevent preallocation OOM.
- **Transformers Pin**: `transformers>=4.40,<5` in `pyproject.toml` and notebook (v5 removes Flax support).
- **Kaggle Execution Runbook** (`docs/M36_KAGGLE_RUN.md`): Hardware requirements table, TPU recommendation, evidence field mapping.
- **Frontend Test Coverage**: 10 new `Leaderboard.test.tsx` tests, 5 new `LiveLogs.test.tsx` tests.
- **Per-Item Predictions Documentation** (`docs/evaluation.md`): Current state, Run Comparison limitations, planned M37 artifact storage.
- **Evidence Folder** (`submission_runs/m36_v1/`): Templates for real Kaggle run evidence with scorecard structure.
- **M36 Evidence Tests**: 12 new tests validating M36 schema including Kaggle-specific fields.
- **CI Fix**: `uv.lock` regenerated to exclude private `ungar` git dependency (prevents auth failures).
- Archive prefix updated to m36, 380+ total backend tests, 75 frontend tests.

**M37 Enhancements:** TPU Training for Submission — Production-ready TPU execution path:
- **TPU Device Support**: Added `--device tpu` option to `train_jax.py` with explicit device validation and clear error messages if TPU unavailable.
- **Enhanced Device Logging**: Platform, device count, and all devices printed at startup for auditability.
- **GPU Hard Block**: Upgraded GPU guardrail from warning to hard block — training exits with error if Gemma model attempted on GPU (prevents OOM waste).
- **TPU Config**: New `training/configs/submission_tpu.yaml` with TPU-optimized settings (batch_size=8, 200 steps, max_seq_length=512).
- **Notebook Update**: Version m37_v1 with explicit `--device tpu` flag, pre-flight TPU detection, and clear instructions for Kaggle TPU v3-8 setup.
- **Evidence Folder**: `submission_runs/m37_v1/` with `run_manifest.json` (TPU hardware fields), `eval_summary.json`, `kaggle_output_log.txt` templates.
- **Model Lock**: Gemma 1 2B (`google/gemma-2b` with `revision="flax"`) confirmed as submission model — Gemma 2/3 NOT supported by FlaxAutoModelForCausalLM.
- Archive prefix updated to m37, ready for TPU production training.

**M38 Enhancements:** TPU HBM OOM Fix + Evidence Population — Comprehensive TPU memory optimization:
- **HBM OOM Root Cause**: Gemma's 256K vocabulary creates massive logits tensor: `[batch, seq_len, 256000]` exceeds Kaggle TPU v5e-8's 16GB HBM during XLA compilation.
- **Memory-Safe Config** (`submission_tpu.yaml`): Reduced `max_seq_length` from 512 to 64, `batch_size` from 8 to 1, added `gradient_accumulation_steps: 8` (effective batch still 8).
- **Code-Level Overrides**: `train_jax.py` forces seq_len=64, batch=1, bfloat16 for all TPU runs regardless of config — defense in depth.
- **bfloat16 for TPU**: Training automatically uses bfloat16 when `--device tpu` is specified — native TPU support.
- **Adafactor Optimizer**: Switched from AdamW to Adafactor for memory savings.
- **Notebook Execution Fix**: Version m38_v1 uses `runpy.run_path()` with proper `SystemExit` handling — no more false "Training completed!" banners.
- **Cross-Platform Fixes**: UTF-8 encoding for YAML, Python path fix for runpy compatibility.
- **Sanity Check**: Added pre-compile diagnostic output showing actual batch/seq shapes and expected logits size.
- **TPU Constraint Documented**: Full fine-tuning Gemma 2B on free Kaggle TPU (16GB HBM) is not feasible — pivoting to local GPU (RTX 5090) in M39.
- Archive prefix updated to m38, 384 backend tests, 75 frontend tests.

**M39 Enhancements:** Local Execution (RTX 5090) Pivot + PyTorch Migration — Established parallel PyTorch training path (`training_pt/`), validated local CPU execution pipeline (with mock GPT-2), documented RTX 5090 compatibility blocker (sm_120 requires CUDA 12.8+, current public PyTorch wheels max out at 12.6/sm_90), evidence captured (`submission_runs/m39_v1`).

**M40 Enhancements:** Native RTX 5090 GPU Acceleration — Resolved M39 blocker by installing PyTorch nightly `cu128` wheels (torch 2.11.0.dev+cu128), confirmed CUDA 12.8 support for Blackwell sm_120 compute capability, dedicated GPU virtual environment (`.venv-gpu`), GPU smoke test validated (31.7 samples/sec throughput on GPT-2), evidence captured (`submission_runs/m40_v1`), training infrastructure ready for production runs.

**M41 Enhancements:** Frontend Polish, DX Cleanup, and Submission Readiness:
- **Frontend Test Hygiene**: Suppressed React `act()` warnings with documented rationale (concurrent health fetches, 30-second polling), intercepted interval timers during tests, zero test output pollution.
- **Key Prop Fixes**: Fixed "unique key prop" warnings in `Tuning.tsx` and `App.tsx` by using `React.Fragment` with explicit keys.
- **Demo Documentation**: Created `docs/DEMO.md` with demo flow guide for judges (trace management, training pipeline, leaderboard).
- **Video Checklist**: Created `docs/submission/VIDEO_CHECKLIST.md` with detailed recording checklist (≤3 min, content coverage, talking points).
- **Evidence Capture**: `submission_runs/m41_v1/` with clean test output, README summary.
- **No Backend/Training Changes**: Zero changes to backend logic, training scripts, or database schema.
- 75 frontend tests passing, clean output.

**M42 Enhancements:** Final Submission Package — Documentation and packaging only, no code changes:
- **VIDEO_CHECKLIST.md**: Added source-of-truth statement, references `docs/DEMO.md` as authoritative guide.
- **README Polish**: Added "Why Tunix RT?" section, Demo Flow, Training Paths, Evidence & Reproducibility, video URL placeholder.
- **GPU Fragility Docs**: Added nightly PyTorch fragility warning to `CONTRIBUTING.md` with version pinning guidance.
- **Packaging Script**: Updated `package_submission.py` prefix from m36 to m42.
- **Final Test Run**: Backend 384 passed (75.79% coverage), Frontend 75 passed, outputs captured to `submission_runs/m42_v1/test_run_outputs/`.
- **Evidence Index**: Created `submission_runs/m42_v1/evidence_index.md` documenting what each folder proves.
- **Dependency Snapshot**: `pip_freeze_backend.txt` captured for reproducibility.
- **Submission ZIP**: `tunix_rt_m42_2026-01-08_e54267b.zip` (104.8 KB) containing notebooks, configs, evalsets, manifests, and README.

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
