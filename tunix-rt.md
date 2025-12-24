# Tunix RT - Reasoning-Trace Framework

**Milestone M19 Complete** ✅  
**Coverage:** 82% Backend Line, 77% Frontend Line | **Security:** SHA-Pinned CI, SBOM Enabled, Pre-commit Hooks | **Architecture:** Ray Tune Integration + Job Persistence | **Tests:** 209 backend + 28 frontend tests

## Overview

Tunix RT is a full-stack application for managing reasoning traces and integrating with the RediAI framework for the Tunix Hackathon. The system provides health monitoring, RediAI integration, and a foundation for trace-quality optimization workflows.

**M1 Enhancements:** Enterprise-grade testing (90% branch coverage), security scanning (pip-audit, npm audit, gitleaks), configuration validation, TTL caching, and developer experience tools.

**M2 Enhancements:** Database integration (async SQLAlchemy + PostgreSQL), Alembic migrations, trace CRUD API (create/retrieve/list), frontend trace UI, comprehensive validation, and payload size limits.

**M3 Enhancements:** Trace system hardening - DB connection pool settings applied, created_at index for list performance, frontend trace UI unit tests (8 total), frontend coverage artifact generation confirmed, Alembic auto-ID migration policy documented, curl API examples, and DB troubleshooting guide.

**M14 Enhancements:** Tunix Run Registry - persistent storage with `tunix_runs` table (UUID PK, indexed columns), Alembic migration, immediate run persistence (create with status="running", update on completion), graceful DB failure handling, `GET /api/tunix/runs` with pagination/filtering, `GET /api/tunix/runs/{run_id}` for details, frontend Run History panel (collapsible, manual refresh), stdout/stderr truncation (10KB), 12 new backend tests + 7 frontend tests (all dry-run, no Tunix dependency).

**M15 Enhancements:** Async Execution Engine - `POST /api/tunix/run?mode=async` for non-blocking enqueue, dedicated worker process (`worker.py`) using Postgres `SKIP LOCKED` for robust job claiming, status polling endpoint, frontend "Run Async" toggle with auto-refresh, Prometheus metrics (`/metrics`) for run counts/duration/latency, `config` column migration for deferred execution parameters.

**M16 Enhancements:** Operational UX - Real-time log streaming via SSE (`/api/tunix/runs/{id}/logs`), Run cancellation (`POST /cancel`) with worker termination, Artifacts/Checkpoints management (`/artifacts` list + download), Hardening (pinned dependencies, optimized trace batch insertion), `tunix_run_log_chunks` table for streaming persistence.

**M17 Enhancements:** Evaluation & Quality Loop - `tunix_run_evaluations` table for persisting scores, "Mock Judge" for deterministic evaluation, Auto-trigger on run completion (async/sync), Leaderboard UI (`/leaderboard`), Manual re-evaluation API (`POST /evaluate`), Dry-run exclusion.

**M18 Enhancements:** Judge Abstraction & Regression - `GemmaJudge` implementation (using RediAI), `regression_baselines` table and endpoints (`POST /api/regression/baselines`, `/check`), Leaderboard pagination (API + UI), Pluggable Judge interface.

**M19 Enhancements:** Hyperparameter Tuning - Ray Tune integration for optimization sweeps (`/api/tuning/jobs`), `TunixTuningJob` and `TunixTuningTrial` tables, Search Space validation, Automated best-run selection, Frontend Tuning UI.

## System Architecture
