# M09 Clarifying Questions

## Phase Structure & Overlap with M8

**Q1: Schema Overlap**
M8 already has:
- `schemas/dataset.py` with `DatasetBuildRequest`, `DatasetManifest`
- `schemas/trace.py` with `ReasoningTrace`, `TraceStep`

M09 Phase 1.1 proposes creating `training/schema.py` with `TrainingExample`.

**Question:** Should we:
- A) Create a new `training/schema.py` with `TrainingExample` as a separate abstraction?
- B) Extend existing `schemas/dataset.py` to include training-specific schemas?
- C) Reuse existing trace schemas and add training metadata fields?

---

**Q2: Gemma IT Formatter Overlap**
M8 already has `training/renderers.py` with `render_tunix_sft_prompt()` that implements Gemma chat template formatting with `<start_of_turn>` tokens.

M09 Phase 1.2 proposes creating `training/gemma_format.py`.

**Question:** Should we:
- A) Extend/refactor existing `renderers.py` to add more Gemma IT formatting utilities?
- B) Create new `gemma_format.py` and deprecate parts of `renderers.py`?
- C) Keep both (renderers.py for full traces, gemma_format.py for low-level token formatting)?

---

**Q3: Export Functionality Overlap**
M8 already has:
- `POST /api/datasets/build` - builds datasets with filters and selection strategies
- `GET /api/datasets/{dataset_key}/export.jsonl?format=trace|tunix_sft` - exports with formatting

M09 Phase 2.1-2.2 proposes creating `training/export_traces.py` and `training/export_ungar.py`.

**Question:** Should we:
- A) Create standalone CLI scripts in `training/` that call existing API endpoints?
- B) Extend the existing dataset build/export system with new features?
- C) Create completely new export pipeline (refactor M8 work)?

---

## Phase 3: Training Runner Structure

**Q4: Training Folder Location**
The plan mentions creating a top-level `training/` folder, but there's already `backend/training/` with `sft_smoke.py`.

**Question:** Should we:
- A) Create top-level `training/` (peer to `backend/`, `frontend/`, `e2e/`)?
- B) Continue using `backend/training/` and organize better?
- C) Use `backend/training/` for library code, top-level `training/` for scripts?

**Recommendation context:** Repo rules emphasize "modules should not be needlessly spread out in different directories" and prefer "portable, well-structured directory."

---

**Q5: Tunix Installation Source**
M09 Phase 3.1 mentions "Tunix install options are documented (PyPI, GitHub)."

**Question:** 
- What is the actual Tunix installation method we should document?
- Should we pin to a specific version/commit like UNGAR (which uses commit `0e29e104`)?
- Is Tunix actually available on PyPI yet, or only GitHub?

---

## Phase 4: Evaluation & Import

**Q6: Post-Training Trace Import**
M09 Phase 4 mentions "Convert outputs into tunix-rt trace objects and POST them into your API (or write import.jsonl used by an import endpoint if you already have one)."

**Question:** Should we:
- A) Use existing `POST /api/traces` to import eval outputs one-by-one?
- B) Create a new bulk import endpoint `POST /api/traces/batch`?
- C) Create an import-from-file endpoint `POST /api/traces/import` that reads JSONL?

---

**Q7: Evaluation Storage Strategy**
The plan mentions creating `eval_before.jsonl`, `eval_after.jsonl`, and `delta_report.md` in `artifacts/training_runs/<run_id>/`.

**Question:** Should we:
- A) Store eval results ONLY as files (no database)?
- B) Also persist eval results in database (new `evaluations` table)?
- C) Store files first, add DB persistence in M10?

---

## Recipe & Format Details

**Q8: Training Recipe Definition**
Phase 2.1 mentions "Recipe v1 (simple): user prompt = original question + 'show your work as steps' instruction."

**Question:**
- Should the "show your work as steps" instruction be:
  - A) Hardcoded string in exporter?
  - B) Configurable template in config file?
  - C) Stored in dataset manifest as `recipe_version`?

---

**Q9: System Role Handling**
Phase 1.2 mentions Gemma IT doesn't support `system` role, so instructions should be "embedded in the initial user turn."

**Question:**
- Where should system-like instructions be defined?
  - A) Hardcoded in formatter (e.g., "You are a helpful math tutor...")?
  - B) Configurable via `configs/sft_tiny.yaml`?
  - C) Pulled from trace metadata?

---

## Determinism & Reproducibility

**Q10: Seed Management**
The plan emphasizes "deterministic seed, recorded in manifest" (Phase 3).

**Question:**
- Should we:
  - A) Require user to provide seed explicitly?
  - B) Auto-generate seed if not provided, then record it?
  - C) Use timestamp-based seed by default?

---

## Scope Confirmation

**Q11: Actual Training in M09**
Phase 3.1 mentions "runs a very small number of steps (like 10-50) just to verify end-to-end."

**Question:**
- Should M09 include ACTUAL Tunix SFT training (even tiny), or just:
  - A) Setup the runner script that COULD run training (but we don't execute in M09)?
  - B) Run 10-50 actual training steps with a tiny model?
  - C) Mock the training run but validate all inputs/outputs?

**Context:** M8 audit recommends "M09: Run actual Tunix SFT training" but the plan says "optional dependency" which might conflict.

---

**Q12: Eval Set Source**
Phase 4 mentions "Pick a fixed eval set (e.g., 25 prompts) from: a static file `training/evalsets/eval_v1.jsonl`, or deterministic DB query."

**Question:**
- Should we:
  - A) Create static `eval_v1.jsonl` file with handcrafted examples?
  - B) Use deterministic DB query (e.g., "first 25 traces created")?
  - C) Sample from existing test traces in database?

---

## CI/CD Strategy

**Q13: CI Coverage Impact**
The plan emphasizes "Keep main CI GREEN and coverage gates stable" and "Tunix + UNGAR remain OPTIONAL deps."

**Question:**
- Should we add `.coveragerc` omit patterns for new training code to prevent coverage dilution (like M8 did), or:
  - A) Add omit patterns for `training/` folder?
  - B) Write enough tests to maintain 70% gate naturally?
  - C) Accept slight coverage dip as long as >70%?

---

## Summary of Key Decision Points

1. **Reuse vs. Refactor:** Should M09 extend M8 components or create parallel implementations?
2. **Folder Structure:** Top-level `training/` vs. `backend/training/`?
3. **Actual Training:** Run real Tunix SFT in M09, or defer to M10?
4. **Import Strategy:** Extend existing endpoints vs. create new ones?
5. **Configuration:** How much should be configurable vs. hardcoded for v1?

---

**Ready for answers!** Please clarify these questions so I can create an accurate TODO list and implementation plan.
