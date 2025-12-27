Here are my answers/decisions for **M35** (so you can finalize the todo list and implement without ambiguity). 

---

## 1) Eval Set v2 composition

**(a) Create a fresh `eval_v2.jsonl`, keep schema backward-compatible.**
Do **not** extend `eval_v1.jsonl` in-place. Make `eval_v2.jsonl` a new file, but keep the existing core fields (`id`, `prompt`, `expected`, `category`, etc.) so tooling doesn’t fork. You *may* add new optional fields (see Q2). Using a “golden/regression” style eval set that stays stable over time is standard practice. ([The Turing Way][1])

**(b) Split:** **60% core / 25% trace-sensitive / 15% edge cases** is good.
(If you want a simpler first cut: 70/20/10 is also fine, but your proposal is balanced.)

**(c) Trace sensitivity items:** **No ground-truth traces.**
Keep them as problems that *require* multi-step reasoning, but don’t add “expected trace” fields. Your current judges are answer-focused; adding ground-truth traces creates a second evaluation problem (trace grading) you’re not set up to do yet. (You can still label these items `section="trace_sensitive"`.)

---

## 2) Scorecard breakdown categories

**(a) Keep existing `category`, add a new primary label `section`.**

* `section`: `core | trace_sensitive | edge_case` (primary slice)
* `category`: keep the existing taxonomy (arithmetic, geometry, etc.) (secondary slice)

This lets you show a 3-way “quality signal” view *and* preserve continuity with v1.

**(b) Difficulty:** Yes—add `difficulty: easy|medium|hard` (optional but recommended).
Difficulty slices are helpful for diagnosing “it fails only on hard reasoning” vs “it fails on formatting.” ([The Turing Way][1])

---

## 3) Leaderboard filtering scope

**(a) Implement all four filters in M35** (they’re low-risk and high-UX):

* dataset (required)
* model_id (required)
* config (required)
* date range (nice but cheap; do it now)

**(b) Filtering logic:** **Additive AND only.**
No OR logic in M35—keeps API predictable and UI simple.

**(c) Config filtering:** **Match by config identity (file name + hash), not deep field matching.**
Do: `config_path` exact/contains + `config_hash` exact if available.
Avoid “learning_rate=…” deep queries for now (fragile and invites schema coupling).

---

## 4) Run comparison per-item diff table

**(a) Yes, you need per-item predictions persisted somewhere.**
Your current `detailed_metrics` is not enough if it lacks the raw predicted text.

**(b) Persist *full per-item comparison* as an artifact, store only pointers + summary in DB.**
Recommended structure:

* DB: store `primary_score`, aggregate metrics, and `eval_artifact_key` (or similar)
* Artifact (JSONL or JSON): per-item rows: `{item_id, expected, predicted, correctness, notes}`

This is aligned with common experiment tracking practice: keep heavyweight outputs as artifacts and store lightweight metadata/aggregates in the DB. ([Ray][2])

**(c) MockJudge behavior:** **Show the table if you can populate expected/predicted; otherwise hide with a clear note.**
If MockJudge doesn’t produce meaningful `predicted`, hide the table and show: “Per-item diff unavailable for MockJudge.” Don’t render “N/A” rows—too noisy.

---

## 5) Regression baseline lock mechanism

**(a) M35 should include *enhancements + tests*, not just documentation.**
Specifically:

* allow multiple baselines (keyed by `eval_set` + `dataset` + maybe `model_id`)
* support “promote best run to baseline” (manual action via API is fine)

**(b) CI integration:** **Not as a hard gate in PR CI.**
Do one of:

* a **manual** CI job (`workflow_dispatch`) that runs regression checks, or
* a **nightly** job
  Hard-gating PRs on regression in this repo will be flaky until Kaggle/GPU runs are stable.

**(c) Compare against `primary_score` by default**, with optional override to metric name.
Defaulting to one canonical scalar is how you keep “leaderboard fidelity” coherent.

---

## 6) Determinism check implementation

**(a) Focus on end-to-end determinism for the evaluation pipeline** (same inputs → same outputs).
Pure function determinism is necessary but insufficient; you want to catch ordering/serialization nondeterminism too.

**(b) Implement as:**

* **Standalone script** `backend/tools/check_determinism.py` (runs eval twice, compares)
* **Unit tests** for the aggregator (`compute_primary_score`) + schema invariants
  Avoid making it a required CI step for the full pipeline (too easy to become flaky).

**(c) Use a fixed snapshot dataset + fixed ordering.**
Pin evaluation item ordering (sort by `item_id`) and ensure DB queries are ordered. That makes AnswerCorrectnessJudge deterministic given the same inputs.

---

## 7) Evidence folder and packaging

**(a) Populate with real values from a local run.**
Same philosophy as M33: “proof, not template.” Evidence should be reviewable without manual fill-in. Recording run metadata is part of reproducible practice. ([Ray][2])

**(b) Yes: M35 evidence must show evaluation against `eval_v2.jsonl`.**
That’s the point of “eval signal hardening”—the evidence should demonstrate it.

---

## 8) Frontend priorities

**(a) Add the per-item diff table as a new section** (don’t replace existing content).
Keep existing loss/config panels, but make the diff table collapsible (default collapsed) if large.

**(b) Leaderboard scorecard summary:** **Inline** (compact), with optional expand.
Inline: `items/scored/skipped` + `primary_score`. Expand row for per-category breakdown.

**(c) Filters UI:** **Inline inputs above the table** (no sidebar).
A sidebar is overkill and slows iteration.

---

If you want, I can also paste a **Cursor-ready M35 implementation prompt** that encodes these decisions as acceptance criteria + file-by-file task list.

[1]: https://book.the-turing-way.org/reproducible-research/renv/ "Reproducible Environments - The Turing Way"
[2]: https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html "A Guide To Callbacks & Metrics in Tune — Ray 2.53.0"
