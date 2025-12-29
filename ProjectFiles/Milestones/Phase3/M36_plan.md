Act as a Staff ML/Platform Engineer shipping “competition-grade evidence + bounded audit-score improvements.”
You are implementing Milestone M36 on top of a green M35 baseline.

## Milestone: M36 — Real Kaggle Execution + Evidence Lock v2 + Quick-Win Audit Uplift
Branch: `milestone/M36-kaggle-evidence-v2`
Base: `main` (post-M35)

### North Star
1) Produce a real Kaggle GPU/TPU run with eval_v2 and populate evidence files with real values.
2) Deliver 2–3 small, high-leverage audit wins (frontend test coverage + remove act warnings + docs touch-up).
3) Keep scope tight: no new product features besides evidence/polish.

---

## Phase 0 — Baseline Gate (required)
- Create branch from `main`.
- Verify locally:
  - backend: `uv run ruff format --check . && uv run ruff check . && uv run pytest -q`
  - frontend: `npm ci && npm test && npm run build`
  - e2e: run standard Playwright command used in repo
- CI must remain green throughout.

Acceptance:
- CI green on branch.

---

## Phase 1 — Kaggle “Real Run” Evidence v2 (required)
Goal: prepare everything so a human can run Kaggle notebook, then commit “real-run evidence.”

### 1.1 Evidence folder
- Create `submission_runs/m36_v1/` with:
  - `run_manifest.json` (include: run_version, model_id, dataset, config_path, command, commit_sha, timestamp, plus Kaggle notebook URL or run identifier fields)
  - `eval_summary.json` (include: run_version, eval_set=eval_v2, metrics, scorecard object, primary_score NON-NULL, evaluated_at)
  - `kaggle_output_log.txt` (captured real Kaggle console output, not local smoke)

### 1.2 Notebook readiness
- Ensure `notebooks/kaggle_submission.ipynb` supports:
  - selecting model id (Gemma)
  - selecting dataset (dev-reasoning-v2 or golden-v2)
  - selecting eval set (eval_v2)
  - producing outputs in a predictable folder and printing a final “RESULT SUMMARY” block (primary_score + key scorecard fields)
- Add a short doc: `docs/M36_KAGGLE_RUN.md`:
  - step-by-step Kaggle instructions
  - exact cells to run
  - where to copy logs from
  - how to paste results into evidence files
  - how to re-package zip after evidence is filled

### 1.3 Packaging update
- Update `backend/tools/package_submission.py` so that `--run-dir submission_runs/m36_v1` bundles:
  - evidence files
  - `training/evalsets/eval_v2.jsonl`
  - any required config(s) used for the run
- Produce: `submission/tunix_rt_m36_<YYYY-MM-DD>_<shortsha>.zip` (do not commit large artifacts; zip is OK if already established practice).

Acceptance:
- Evidence schema tests updated/added so M36 evidence is validated in CI (primary_score must be non-null for M36).
- `python backend/tools/package_submission.py --run-dir submission_runs/m36_v1` succeeds.

---

## Phase 2 — Quick Win Audit Uplift (required)
Keep this bounded and directly aligned with M35 audit “Opportunities.”

### 2.1 Frontend tests to raise coverage meaningfully
- Add 5–10 tests for `frontend/src/components/Leaderboard.tsx` and 3–5 tests for `LiveLogs.tsx`.
- Prefer user-facing behavior tests (filters, rendering scorecard, empty state, error state).
- Target: Leaderboard coverage > 50% (or at least a large jump from ~2%).
- Ensure `npm run build` and `npm test` pass.

### 2.2 Fix React `act()` warnings
- Identify warnings in current test output.
- Wrap async updates in `act()` or use `await` + proper testing-library patterns.
- Goal: test output clean (or materially reduced warnings).

### 2.3 Docs touch-up
- Update `docs/evaluation.md` with a section:
  - “Per-item predictions: current state + limitation + planned M37 artifact storage”
  - Clarify what RunComparison can/can’t show today.

Acceptance:
- `npm run build` passes
- frontend tests pass with warnings resolved/reduced
- docs updated
- CI green

---

## Phase 3 — Stop Line & Deliverables (required)
Create a short `docs/M36_SUMMARY.md` including:
- what changed
- commands to reproduce
- evidence contents and where they came from
- remaining known limitations (push M37/M38 to backlog)

Hard stop:
- Do not start per-item artifact storage implementation (that is M37).

---

## Output format
- Implement directly in repo with clean commits.
- Keep PR description crisp with acceptance checklist.
