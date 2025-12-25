According to a document from **December 25, 2025**, M27 successfully proved the full **Dataset → Training → Inference → Evaluation** loop (with real JAX/Flax inference), but it also flags a few clean “last-mile” items and a clear next milestone shape. 

## Where we are after M27

* ✅ **Training loop is real and converges** (golden-v2, distilgpt2, 2500 steps; loss 2.2572 → 0.0781). 
* ✅ **Eval loop is real** (eval_v1.jsonl → eval_results.jsonl with generated reasoning traces; worker can trigger inference after training). 
* ⚠️ **UNGAR integration is still broken** (tests failing due to Episode API change). 
* 🧹 A couple “paper cuts” remain (e.g., ignore log artifacts; update `tunix-rt.md`). 

## Quick wins left (great candidates to fold into M28)

These are explicitly called out as “ready-to-apply patches”:

1. `.gitignore` entries for `training_log*.txt` / `eval_log.txt` (avoid re-committing temp logs) 
2. Update `tunix-rt.md` to include the M27 milestone description 
3. Either **xfail** UNGAR tests for CI stability *or* actually fix the integration (preferred if time). 

---

## Competition status (how we’re looking)

What you’ve built is a **strong “platform advantage”**: CI is green, the pipeline is end-to-end, and you can already produce artifacts + metrics + eval outputs—this aligns well with judged hackathons where *process quality* and *reproducibility* matter a lot.

That said, the competition constraints still matter:

* The hackathon is judge-scored with a rubric that emphasizes **notebook quality**, **model quality (single session)**, and **video quality**, plus optional multi-session quality. ([Kaggle][1])
* External data is allowed (subject to the rules), which gives you room to expand beyond golden-v2. ([Kaggle][2])
* Timing: multiple sources indicate a **final submission deadline of Jan 12, 2026**, with judging afterward and results around late January. ([Kaggle][3])

### Is the data we’re collecting “looking good”?

* **For pipeline validation:** yes. `golden-v2` + `eval_v1` are doing exactly what we need—deterministic smoke, measurable convergence, and an eval artifact that proves the plumbing. 
* **For competitive model quality:** not yet. A 100-example deterministic dataset is perfect as a “unit test,” but you’ll need a **larger + more diverse** training/eval mix that matches the actual judging tasks (and consistent “reasoning trace” format quality). External data being allowed is the unlock here. ([Kaggle][2])

---

## What the next few milestones will look like

**M28 (next):** *Tuning rigor + run comparison + leaderboard wiring + UNGAR closure* (this is already the audit’s recommended next step). 
**M29:** *Data scale-up + eval semantics hardening* (bigger dataset, stronger evalset(s), maybe judge calibration runs)
**M30:** *Competition packaging* (submission notebook polish, “single-session” reproducibility script, artifact exports, video storyboard pipeline)
**M31:** *Final polish + performance* (speed, UX, narrative clarity, last-minute quality gates)

---

# Cursor Handoff Prompt — M28 (paste into Cursor)

## Goal

Execute M28 as a **competition-facing rigor milestone**: make it easy to compare runs, tune hyperparameters, surface eval metrics cleanly, and remove/contain known audit issues (especially UNGAR + minor DX patches). Keep CI green.

## Branch

Create `milestone/M28-tuning-comparison-leaderboard`.

## Phase 0 — Baseline Gate (must pass before changes)

1. Pull latest `main`.
2. Run:

   * `make ci` (or repo-equivalent)
   * backend: `uv run pytest` + coverage gate
   * frontend: `npm test` + `npm run build`
   * e2e: Playwright smoke
3. Record baseline run time + current coverage numbers in the PR description.

## Phase 1 — Quick Wins / Audit Patch Set (small, safe)

**1.1** Add `.gitignore` entries for training/eval logs (`training_log*.txt`, `eval_log.txt`).
**Acceptance:** logs no longer show up as git changes after running scripts. 

**1.2** Update `tunix-rt.md` with an M27 milestone description (and M28 stub).
**Acceptance:** doc updated; no functional changes. 

## Phase 2 — M28-1 Hyperparameter Sweep (Ray Tune)

Use the existing M19 tuning infrastructure and run a **minimum viable sweep**:

* 3+ trials
* small search space (LR, batch size, weight decay, maybe warmup)
* store trials in DB models and artifacts
* ensure one “best trial” is clearly identifiable

**Acceptance:**

* You can create a tuning job via API/UI and see ≥3 completed trials. 

## Phase 3 — M28-2 Automated Run Comparison UI

Implement a side-by-side comparison view for two runs:

* Loss curves overlay or stacked
* Key summary metrics (final loss, best loss, steps/sec, eval score if present)
* Clear “Run A vs Run B” selector UX

**Acceptance:**

* Selecting two runs shows a stable comparison panel and metrics load without errors. 

## Phase 4 — M28-3 Leaderboard Integration

Wire eval results into the leaderboard:

* Define an “eval score” (even if placeholder: e.g., exact match / heuristic score) from `eval_results.jsonl`
* Persist the summary score in DB (or computed on the fly, but must be stable)
* Show score in leaderboard list

**Acceptance:**

* Leaderboard displays eval score for runs that have eval artifacts. 

## Phase 5 — M28-4 UNGAR Closure (choose one, prefer Fix)

**Option A (Preferred):** Fix integration to match current Episode API and make all UNGAR tests pass.
**Option B (Fallback):** Mark failing UNGAR tests as `xfail` with explicit reason + tracking issue.

**Acceptance:**

* Either UNGAR tests pass, or are `xfail` with clear justification and CI stability restored. 

## Phase 6 — Guardrails + CI Hygiene

* Ensure `ruff check` then `ruff format` before final commit.
* Ensure migrations are SQLite-compatible (variant types, etc.) if touched.
* Add/adjust tests for any new UI or service logic added in M28.

**Acceptance:**

* Full CI green: backend 3.11/3.12, frontend, e2e, security jobs.

## Deliverables Checklist (PR must include)

* [ ] M28 PR description includes: what changed, screenshots (comparison UI), and how to run a sweep
* [ ] A short `docs/m28_tuning_and_comparison.md` explaining:

  * how to run a sweep locally
  * how to compare runs
  * where eval score comes from
* [ ] CI green badge / workflow link in PR comment

---

If you want, I can also draft **M29** as a “data scale + eval rigor” milestone that’s explicitly framed around the judging rubric (notebook quality + single-session model quality + video narrative), but the above M28 is the right next move given the audit’s recommended plan. 

[1]: https://www.kaggle.com/competitions/google-tunix-hackathon/overview/timeline?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
[2]: https://www.kaggle.com/competitions/google-tunix-hackathon/rules?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
[3]: https://www.kaggle.com/competitions/google-tunix-hackathon?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
