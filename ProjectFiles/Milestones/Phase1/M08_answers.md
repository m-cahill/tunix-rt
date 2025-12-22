Here are my answers to **Q1–Q15** (with concrete decisions you can implement immediately). 

## Phase 0 & 1

**Q1 (M07 paper cuts):**
Do **all three** in M08 **Phase 1** (not optional). They’re tiny, low-risk, and they tighten the “enterprise polish” loop:

* ✅ Add explanatory comments for `type: ignore`
* ✅ Add “Quick Start Happy Path” to `M07_UNGAR_INTEGRATION.md`
* ✅ Add minimal logging on `"??"` defensive fallbacks (warning-level, no crashes)

---

## Phase 2: Dataset Export v1

**Q2 (storage location):**
Use **`backend/datasets/`**. Your recommendation is correct: it keeps dataset artifacts co-located with the backend that produces them and keeps the repo portable.

**Q3 (manifest persistence):**
Start **file-based manifests on disk** (no DB table in M08).
Add DB persistence later only if/when you need querying, user ownership, or multi-tenant dataset catalogs.

**Q4 (dataset ID generation):**
Use **two identifiers**:

* `dataset_key`: **`{dataset_name}-{dataset_version}`** (human + reproducible)
* `build_id`: **UUID** per build (uniqueness + provenance)

This avoids collisions while preserving reproducibility.

**Q5 (selection strategies):**
Support **only**:

* `latest`
* `random` (seeded)

Defer `stratified` to M09.

**Q6 (schema versioning):**
Yes — track a separate **`dataset_schema_version`** in the manifest (independent from trace schema). This is the correct long-term move.

---

## Phase 3: Tunix Prompt Renderer

**Q7 (Tunix SFT prompt format):**
Do both:

1. **Reference the Kaggle Tunix example** for the *chat template framing* (it formats prompts in the same structure used for training, including tokens like `<start_of_turn>` and role blocks). ([Kaggle][1])
2. Implement a **simple deterministic template renderer** that can be swapped later.

Concretely in M08:

* Export a `prompts` string that uses a consistent **system + user** wrapper and appends trace steps in a stable format. (No LLM rewriting.)

**Q8 (rendering strategy):**
Use a **simple function-based renderer** (no Jinja2 in M08).
Design it like:

* `render_tunix_sft_prompt(example) -> str`
* allow branching on `metadata.source` (e.g., UNGAR vs generic) if needed, but keep it minimal.

---

## Phase 4: Training Smoke Harness

**Q9 (Tunix/JAX deps):**
Yes: add optional extra **`backend[training]`** that includes Tunix + JAX (and any minimal helpers). Tunix is explicitly designed as a JAX-native post-training library. ([Google Developers Blog][2])
But keep the smoke harness **tiny** and CPU-default.

**Q10 (training script location):**
Put it in **`backend/training/sft_smoke.py`** (repo-level alongside backend), not inside the runtime package.

**Q11 (CI strategy):**

* Add a **blocking “dataset validation”** step to main CI (fast): schema check + renderer output check.
* Add **training smoke** as **separate workflow**: `workflow_dispatch + nightly`, **non-blocking**.

---

## Phase 5: E2E Coverage

**Q12 (E2E timing):**
Add the UNGAR panel E2E test in **Phase 1** (early). It closes the M07 audit gap immediately and locks stability before dataset work expands the surface area.

---

## Cross-cutting

**Q13 (multi-session compatibility):**
Yes — design manifests with multi-session in mind, but don’t implement multi-session behavior yet.
Add optional fields now:

* `session_id` (nullable)
* `parent_dataset_id` (nullable)
* `training_run_id` (nullable)

**Q14 (dataset UI in M08):**
Keep **backend-only** in M08. Add a dataset browser UI in **M09** (or later) once the pipeline is proven.

**Q15 (backwards compatibility):**
Coexist in M08:

* keep `/api/ungar/high-card-duel/export.jsonl`
* add new dataset endpoints in parallel
  Deprecate the UNGAR-specific export in **M09** only after you’ve proven datasets are the preferred path.

---

## One extra guardrail (because you just lived this once)

For coverage strategy: keep using **coverage config omit patterns** for truly-optional modules rather than sprinkling pragmas. Coverage.py explicitly supports excluding code in config “without littering your code with exclusion pragmas,” and `omit` is a first-class mechanism. ([Coverage][3])
If you need pytest-cov to respect it consistently, use `--cov-config` in CI. ([pytest-cov.readthedocs.io][4])

If you want, I can now turn these decisions into a **single M08 Cursor handoff prompt** with: exact files, endpoint shapes, manifest schema JSON, and test checklist (baseline → dataset build → export → validation → optional smoke).

[1]: https://www.kaggle.com/code/yekahaaagayeham/guide-gemma-3-fine-tuning-with-tunix?utm_source=chatgpt.com "Guide : Gemma 3 fine tuning with Tunix"
[2]: https://developers.googleblog.com/introducing-tunix-a-jax-native-library-for-llm-post-training/?utm_source=chatgpt.com "Introducing Tunix: A JAX-Native Library for LLM Post-Training"
[3]: https://coverage.readthedocs.io/en/latest/excluding.html?utm_source=chatgpt.com "Excluding code from coverage.py - Read the Docs"
[4]: https://pytest-cov.readthedocs.io/en/latest/config.html?utm_source=chatgpt.com "Configuration - pytest-cov 7.0.0 documentation"
