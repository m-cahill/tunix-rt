## M33 Answers (decisions)

### 1) Kaggle execution scope

**Answer: (c)** — **Both**: prepare everything so *you* can run on Kaggle **and** add a local CPU/smoke “rehearsal” path that produces the same evidence files.

* Rationale: Cursor can’t click-run Kaggle, but we still want (1) confidence locally and (2) a Kaggle-ready “no edits needed” notebook/script.

---

### 2) Model selection

**Answer:** Stick with **`google/gemma-3-1b-it`** for M33 rehearsal.

* It is explicitly within the competition’s allowed starting models (Gemma2 2B or Gemma3 1B). ([Kaggle][1])
* We can do a later “final pass” run with Gemma2 2B once the rehearsal loop is smooth.

---

### 3) Dataset for the run (dev-reasoning-v2 vs golden-v2)

**Answer: (a)** — Update the notebook default to **`dev-reasoning-v2`**, but keep **`golden-v2`** as a documented “quick sanity” option.

* M33 is a rehearsal of the full end-to-end path; using the larger dev set better approximates reality while still being small enough to iterate.

---

### 4) Evidence folder naming

**Answer: (a)** — Create a **new** `submission_runs/m33_v1/` (do **not** rename/update m32 artifacts).

* Keeps a clean audit trail of milestone rehearsals.

---

### 5) Packaging tool enhancement (`--run-dir`)

**Answer:** **Yes** — add `--run-dir submission_runs/m33_v1` to `package_submission.py`, specifically for bundling *evidence folder contents*.

* Keep `--include-output` as-is (backward compatible).
* `--run-dir` should include the small tracked files (manifest/summary/log), but not huge artifacts.

---

### 6) CI test schema (required keys)

Your proposed required keys are good. I’d keep them minimal + stable.

**Answer: Approved, with 2 small additions.**

**`run_manifest.json` required:**

* `run_version`
* `model_id`
* `dataset`
* `commit_sha`
* `timestamp`
* **`config_path`** (or `config_name`) ← needed to reproduce exactly
* **`command`** (string or list) ← the literal invocation used

**`eval_summary.json` required:**

* `run_version`
* `eval_set`
* `metrics`
* `evaluated_at`
* **`primary_score`** (number) ← whatever you display on leaderboard (even if duplicated inside `metrics`)

Everything else (Kaggle URLs, TPU type, runtime, notes) can be **optional**.

---

If you want one extra “competition guardrail” in M33: Kaggle’s timeline page specifies the video must be on YouTube and **3 minutes or less**—worth putting into the checklist template so it can’t be missed. ([Kaggle][2])

[1]: https://www.kaggle.com/competitions/google-tunix-hackathon?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
[2]: https://www.kaggle.com/competitions/google-tunix-hackathon/overview/timeline?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
