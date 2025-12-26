# M32 Clarifying Questions

## Phase 1 — Submission Execution Evidence

**Q1: submission_runs/ folder tracking**
The plan mentions `submission_runs/m32_v1/` with "gitignored if it contains large artifacts." Should I:
- (a) Create the folder structure and add it to `.gitignore` entirely (keep artifacts local only)?
- (b) Track the folder but `.gitignore` specific large files (e.g., `*.ckpt`, `*.safetensors`)?
- (c) Track the small manifest/summary files (`run_manifest.json`, `eval_summary.json`, `kaggle_output_log.txt`) and gitignore nothing?

Leaning toward (c) since the plan says to optionally include these files in `package_submission.py` — they should be in version control.

---

## Phase 2 — Data Scale-Up

**Q2: Dataset format mismatch**
I noticed a format discrepancy:
- **`ReasoningTrace` Pydantic schema** expects: `steps: list[TraceStep]` where `TraceStep = {i: int, type: str, content: str}`
- **`dev-reasoning-v1/dataset.jsonl`** uses: `trace_steps: list[str]` (flat string list, no `i`/`type` structure)

The existing seeder (`seed_dev_reasoning_v1.py`) generates `trace_steps` (strings), but passes through `ReasoningTrace(**trace_data)` validation — meaning it must be converting somewhere, or there's a flexible schema.

For `dev-reasoning-v2`, should I:
- (a) Follow `dev-reasoning-v1` pattern exactly (use `trace_steps` as flat strings, let existing pipeline handle conversion)?
- (b) Generate strict `ReasoningTrace` format with `steps: [{i, type, content}, ...]`?

**Q3: What is "golden-v2 style"?**
The plan says "a small subset that mimics golden-v2 style." Looking at `golden-v2/dataset.jsonl`, it's in **Tunix SFT format** (already rendered chat prompts), not raw ReasoningTrace format.

Should the "golden-v2 style" subset in dev-reasoning-v2:
- (a) Be raw ReasoningTrace format but with similar content patterns (text_gen / repeat-N-times tasks)?
- (b) Already be pre-rendered in Tunix SFT format like `golden-v2/dataset.jsonl`?

I assume (a) since training/export handles rendering.

**Q4: Dataset generation target**
The plan says "500–800 traces total." Any preference on the exact composition?
- Suggested: 350 reasoning (70%), 100 synthetic (20%), 50 golden-v2-style (10%)?
- Or should I maintain the 70%/30% split from v1 and just scale up?

---

## Phase 3 — Coverage Uplift

**Q5: Worker test scope**
The plan says worker coverage is difficult due to Postgres SKIP LOCKED semantics. The existing `test_worker.py` has 3 tests (2 for `process_run_safely`, 1 that skips non-Postgres). 

Should I:
- (a) Add more unit tests around `process_run_safely` edge cases (e.g., missing config, auto-evaluation paths)?
- (b) Document that `claim_pending_run` is inherently Postgres-only and add a skip note?
- (c) Both?

---

## Phase 4 — CI Verification

**Q6: E2E suite command**
The plan mentions "run existing Playwright suite (whatever the repo's standard command is)." I see `e2e/` folder with Playwright config. Is the standard command:
```bash
cd e2e && npx playwright test
```
Or is there a Makefile target I should use?

---

## General

**Q7: Branch creation**
Should I create the branch `milestone/M32-data-scale-up` from `main` or from the current M31 branch (`milestone/M31-final-submission-package`)?
