You are Cursor acting on the tunix-rt repo.

Goal (M32):
1) Make the competition submission “execution” reproducible by adding a lightweight evidence-capture runbook + artifacts folder.
2) Scale dataset generation from dev-reasoning-v1 (200) to a new 500+ trace dataset (dev-reasoning-v2), validated against ReasoningTrace schema.
3) Add unit tests to lift coverage for datasets_ingest.py (currently called out as 0% coverage) and improve worker coverage where feasible.
4) Keep CI green (ruff/mypy/pytest/frontend/e2e).

Branch:
- Create branch: milestone/M32-data-scale-up

Phase 0 — Baseline Gate
- From repo root:
  - backend: `cd backend && uv run ruff format --check . && uv run ruff check . && uv run mypy tunix_rt_backend && uv run pytest`
  - frontend: `cd ../frontend && npm test`
  - e2e: run existing Playwright suite (whatever the repo’s standard command is)
- Do not begin if baseline is not green.

Phase 1 — “Submission execution evidence” (code + docs; manual steps are documented only)
Create a small, versioned evidence capture kit:
1) Create `docs/submission_execution_m32.md`:
   - Step-by-step: (a) run Kaggle notebook (b) export outputs (c) paste final URLs (video + notebook) (d) package artifacts
   - Include placeholders for: YouTube URL, Kaggle notebook URL, submission timestamp, model_id, dataset, max_steps, eval_score
2) Create `submission_runs/m32_v1/` (gitignored if it contains large artifacts) with:
   - `run_manifest.json` (model_id, dataset, config path, commit sha, timestamp, env notes)
   - `eval_summary.json` (whatever the notebook prints: eval score + key metrics)
   - `kaggle_output_log.txt` (pasted console output)
3) Update/extend `backend/tools/package_submission.py` to optionally include:
   - `docs/submission_execution_m32.md`
   - `submission_runs/m32_v1/run_manifest.json` + `eval_summary.json` (but NOT large checkpoints)
   - Ensure archive naming stays consistent with existing naming scheme (tunix_rt_*), and keep size small.
4) Verify: `python backend/tools/package_submission.py` creates a zip successfully.

Phase 2 — Data Scale-Up: dev-reasoning-v2 (500+ traces)
1) Add new seeder: `backend/tools/seed_dev_reasoning_v2.py`
   - Deterministic seed (42)
   - Target 500–800 traces total
   - Must emit ReasoningTrace JSONL that matches the Pydantic schema (steps[] with {i,type,content}, etc.)
   - Mix: math word problems, verification steps, simple logic/string transforms, and a small subset that mimics golden-v2 style.
   - Add meta tags: {"dataset":"dev-reasoning-v2","generator":"seed_dev_reasoning_v2","seed":42,"category":...}
2) Add dataset folder format to match the training expectation:
   - `backend/datasets/dev-reasoning-v2/manifest.json`
   - `backend/datasets/dev-reasoning-v2/dataset.jsonl` (or whatever the repo standard is)
3) Add/extend a schema guardrail test:
   - Similar to the existing fixture schema test, validate at least N sampled lines from dev-reasoning-v2 against ReasoningTrace.
4) Verify locally:
   - Generate the dataset
   - Run a smoke training run (CPU) using existing sft_tiny.yaml against dev-reasoning-v2 (smoke_steps=2)
   - Ensure output artifacts (metrics.jsonl + checkpoint folder) appear.

Phase 3 — Coverage uplift: datasets_ingest.py + worker.py
1) Add unit tests for `tunix_rt_backend/services/datasets_ingest.py`:
   - Happy path: ingest JSONL with 2–3 valid traces -> inserted count matches
   - Skip invalid line(s): one invalid trace should be skipped and reported
   - File-not-found / bad path -> returns 422/400 as appropriate
   - Mock DB session + trace creation function; do NOT require Postgres
2) Add targeted unit test(s) for `tunix_rt_backend/worker.py`:
   - Mock job claiming / state transitions at a “unit seam”
   - If too heavy, cover at least the decision branches around “no jobs / job locked / job claimed”
3) Ensure tests are fast and deterministic.

Phase 4 — CI and hygiene
- Run:
  - `cd backend && uv run ruff format . && uv run ruff check . && uv run mypy tunix_rt_backend && uv run pytest`
  - `cd ../frontend && npm test`
  - E2E suite
- Confirm coverage gates still pass.
- Update milestone docs:
  - Add `M32_summary.md` and `M32_audit.md` with what changed, verification commands, and remaining gaps.

Acceptance Criteria
- dev-reasoning-v2 exists and seeds 500+ valid traces (schema-validated).
- Smoke training on dev-reasoning-v2 succeeds.
- New unit tests raise coverage meaningfully for datasets_ingest.py; worker coverage improved or clearly documented if blocked.
- Packaging tool can bundle the new execution runbook + small evidence artifacts.
- CI is green.

Non-goals (do not do in M32)
- No major refactors
- No new UI features
- No tuning sweeps (that’s M33)

Deliverables checklist in PR description
- [ ] Evidence runbook added (docs/submission_execution_m32.md)
- [ ] dev-reasoning-v2 seeder + dataset artifacts
- [ ] Schema guardrail test(s)
- [ ] datasets_ingest unit tests
- [ ] worker unit tests (or documented partial)
- [ ] package_submission updated + verified
- [ ] CI green
