Here are my calls on each M32 question (with the “why” and the concrete implementation choice).

---

## Q1 — `submission_runs/` tracking

**Answer: (b)** Track the folder + **commit the small evidence files**, but **gitignore large artifacts**.

Why: you get reproducibility + reviewability (manifests/summaries in git), without risking repo bloat from checkpoints/log dumps. This is also the “intended” use of `.gitignore`: keep non-source artifacts out of history. ([Atlassian][1])

**Do this:**

* Commit:

  * `submission_runs/**/run_manifest.json`
  * `submission_runs/**/eval_summary.json`
  * (optional) `submission_runs/**/kaggle_output_log.txt` *if it’s small + scrubbed*
* Ignore:

  * `submission_runs/**/checkpoints/`
  * `submission_runs/**/*.ckpt`
  * `submission_runs/**/*.safetensors`
  * `submission_runs/**/*.pt`
  * `submission_runs/**/*.bin`
  * `submission_runs/**/*.zip` / `*.tar*`

---

## Q2 — dev dataset format (`trace_steps: list[str]` vs `steps: [{i,type,content}]`)

**Answer: (b)** Generate **strict `ReasoningTrace` format** with `steps: [{i,type,content}, …]`.

Why: you’ve already seen CI/E2E reject invalid traces when the schema expects `steps` objects. If we’re scaling data, we should scale the *canonical* format and avoid relying on “mystery conversion” paths.

**Guardrail:** if you still need to support legacy `trace_steps` (v1), add a tiny compatibility converter in the seeder or ingest path—but **v2 should be strict**.

---

## Q3 — What does “golden-v2 style” mean for dev-v2?

**Answer: (a)** Keep dev-v2 in **raw ReasoningTrace format**, but mimic **content patterns** (task types, wording, step style). Do **not** pre-render to Tunix SFT format in dev-v2.

Why: rendering to SFT is the job of the export/builder pipeline; keeping a single canonical raw format makes everything (ingest, validate, transform, audit) cleaner.

---

## Q4 — Dataset composition target (500–800)

Your proposal is solid.

**Answer:** go with **~70/20/10** as you suggested:

* **70% reasoning** (multi-step, explicit steps, arithmetic/logic/string transforms)
* **20% “synthetic/simple”** (shorter, 1–2 steps, easy wins)
* **10% “golden-v2-style content patterns”** (but still raw trace format)

Also add a tiny “edge-case” slice inside the 70% bucket (like 20 items): empty-ish prompts, whitespace, tricky punctuation, and off-by-one style problems—these harden the pipeline.

---

## Q5 — Worker test scope (SKIP LOCKED / Postgres-only)

**Answer: (c) Both.**

* **Add more unit tests** around `process_run_safely` edge cases (pure Python logic, state transitions, error handling).
* **Document + skip** `claim_pending_run` as Postgres-only where appropriate.

This matches reality: `SKIP LOCKED` semantics are inherently DB-specific; unit tests should focus on deterministic logic, while DB-specific behavior is covered via integration/E2E (which you already have passing). (You’ve already seen this pattern in your existing test suite.)

---

## Q6 — E2E suite command

**Answer:** default to:

```bash
cd e2e
npx playwright test
```

That’s Playwright’s standard CLI invocation. ([Playwright][2])

**Guardrail:** if the repo has a `package.json` script (e.g., `npm test`) or a Makefile target, prefer that *for consistency*, but the canonical fallback is `npx playwright test`.

---

## Q7 — Branch base

**Answer:** create `milestone/M32-data-scale-up` from **`main`**.

Reason: M31 is “complete,” and you want each milestone to start from the latest green, merged baseline. Only base off the M31 branch if M31 wasn’t merged yet.

---

If you want, paste the first ~30 lines of `backend/datasets/dev-reasoning-v1/dataset.jsonl` and the `ReasoningTrace` schema file path, and I’ll tell you exactly where the v1 “trace_steps → steps” conversion is happening (so M32 doesn’t accidentally fork formats).

[1]: https://www.atlassian.com/git/tutorials/saving-changes/gitignore?utm_source=chatgpt.com ".gitignore file - ignoring files in Git | Atlassian Git Tutorial"
[2]: https://playwright.dev/docs/test-cli?utm_source=chatgpt.com "Command line"
