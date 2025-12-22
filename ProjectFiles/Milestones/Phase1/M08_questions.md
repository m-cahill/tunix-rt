# M08 Clarifying Questions

Based on the M08 plan, I need the following clarifications before creating the todo list and beginning implementation:

---

## Phase 0 & 1: Baseline & Paper Cuts

**Q1:** Should I proceed with all the M07 "paper cuts" listed in Phase 1, or are any of these optional/deferred?
- Add explanatory comments for `type: ignore` in UNGAR availability.py
- Add "Quick Start Happy Path" to M07_UNGAR_INTEGRATION.md
- Add minimal logging on defensive fallback (`"??"`) paths in high_card_duel.py

---

## Phase 2: Dataset Export v1

**Q2: Dataset Storage Location**
- The plan says "repo root or `backend/datasets/`" - which should we use?
- **Recommendation:** I suggest `backend/datasets/` to keep datasets co-located with the backend code that generates them, making the backend directory more portable.

**Q3: Dataset Manifest Persistence**
- Should manifests be stored in the database (new `datasets` table) OR as JSON files on disk?
- If database: need migration, schema design
- If file-based: simpler, version-controllable, but harder to query
- **Recommendation:** Start with file-based (simpler, aligns with "show your work" narrative), add DB persistence in M09 if needed.

**Q4: Dataset ID Generation**
- Should dataset IDs be:
  - UUIDs (like traces)?
  - Name-based: `{dataset_name}-{dataset_version}` (e.g., `ungar_hcd_baseline-v1`)?
  - Timestamp-based: `{dataset_name}-{timestamp}` (e.g., `ungar_hcd_baseline-20251221_103000`)?
- **Recommendation:** Name + version for reproducibility and human-readability.

**Q5: Selection Strategy Details**
- The plan mentions "selection strategy" in the build endpoint - what options should we support?
  - `latest` - most recent N traces
  - `random` - random N traces (with seed for reproducibility)
  - `all` - all matching filters
  - `stratified` - balanced by metadata field (e.g., win/loss/tie ratio)
- **Recommendation:** Start with `latest` and `random` only; add stratified in M09 if needed.

**Q6: Schema Versioning**
- Should we track dataset schema versions separately from trace schema versions?
- Example: trace schema might stay at "1.0" but dataset export format could evolve to "1.1"
- **Recommendation:** Yes, separate versioning (`dataset_schema_version` in manifest).

---

## Phase 3: Tunix Prompt Renderer

**Q7: Tunix SFT Prompt Format**
- What specific format does Tunix SFT expect for the `prompts` field?
- Should I reference the Kaggle example (https://www.kaggle.com/code/yekahaaagayeham/guide-gemma-3-fine-tuning-with-tunix) to understand the format?
- Or should we create a simple template-based format that's Tunix-compatible?
- **Recommendation:** Review Tunix/Kaggle examples and implement a simple template that mirrors their format.

**Q8: Rendering Strategy**
- Should the renderer be:
  - Template-based (Jinja2 or simple string formatting)?
  - Custom function per trace type (high_card_duel vs future games)?
  - Configurable via manifest (manifest specifies template)?
- **Recommendation:** Start with simple function-based approach, make it configurable in M09.

---

## Phase 4: Training Smoke Harness

**Q9: Tunix/JAX Dependencies**
- Should we add Tunix and JAX as optional dependencies (`backend[training]`)?
- Or should the smoke test be mock-based (no actual training, just validates dataset loading)?
- **Recommendation:** Add as optional `backend[training]` extra to match the UNGAR pattern, but make smoke test very lightweight (5-10 steps on CPU).

**Q10: Training Script Location**
- Should the smoke SFT script live in:
  - `backend/training/sft_smoke.py` (new `training/` directory)?
  - `backend/tunix_rt_backend/training/sft_smoke.py` (inside package)?
  - `notebooks/sft_smoke.ipynb` (notebook format)?
- **Recommendation:** `backend/training/` at repo level (not inside package) to keep training code separate from runtime backend.

**Q11: CI Strategy for Training Validation**
- Should the "training-data validation" job be a new step in the existing CI workflow, or a separate workflow?
- Should the "training smoke" job be:
  - Manual dispatch only?
  - Manual dispatch + nightly (like UNGAR)?
  - Always run but non-blocking (continue-on-error)?
- **Recommendation:** Add validation to existing CI (fast, blocking). Add smoke as separate workflow (manual + nightly, non-blocking).

---

## Phase 5: E2E Coverage

**Q12: E2E Test Timing**
- Should the UNGAR panel E2E test be added:
  - In Phase 1 (paper cuts)?
  - In Phase 5 (as planned)?
  - As part of the first PR (to establish baseline)?
- **Recommendation:** Phase 1 to close the M07 audit gap early, then Phase 5 can add dataset UI E2E if we build one.

---

## Cross-Cutting Concerns

**Q13: Tunix Multi-Session Compatibility**
- The VISION doc mentions multi-session training - should datasets be designed with this in mind from the start?
- Should manifests include fields like `session_id`, `parent_dataset_id`, or `training_run_id`?
- **Recommendation:** Add basic metadata fields (`session_id`, `parent_dataset_id`) to manifest schema but don't build multi-session logic yet.

**Q14: Dataset UI**
- Should M08 include any frontend UI for datasets, or keep it backend-only?
- The plan doesn't mention frontend changes beyond E2E for UNGAR panel.
- **Recommendation:** Backend-only for M08; add minimal dataset browser in M09.

**Q15: Backwards Compatibility**
- The existing `/api/ungar/high-card-duel/export.jsonl` endpoint returns JSONL in Tunix-friendly format.
- Should the new dataset endpoints:
  - Replace this endpoint (breaking change)?
  - Coexist with it (keep UNGAR-specific endpoint for backwards compat)?
  - Deprecate it with a migration path?
- **Recommendation:** Coexist for M08; deprecate UNGAR-specific endpoint in M09 once datasets are proven.

---

## Summary of Recommendations (Pending Your Approval)

If you approve the recommendations above, I'll proceed with:

1. **Dataset storage:** `backend/datasets/` (file-based manifests)
2. **Dataset IDs:** `{name}-{version}` format
3. **Selection strategies:** `latest` and `random` only
4. **Prompt format:** Template-based, Tunix-compatible (review Kaggle examples)
5. **Training deps:** Optional `backend[training]` extra with Tunix/JAX
6. **Training script:** `backend/training/sft_smoke.py` (repo-level, not in package)
7. **CI:** Validation in main CI (blocking), smoke as separate workflow (non-blocking)
8. **E2E:** Add UNGAR panel test in Phase 1
9. **Frontend:** Backend-only for M08
10. **Backwards compat:** Keep existing UNGAR endpoint, add new dataset endpoints

Please confirm, modify, or provide additional guidance for any of these questions before I proceed.

