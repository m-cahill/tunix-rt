# M12 Clarifying Questions

## Project Analysis Complete ✅

I've analyzed your project and understand:
- Current architecture (M11 complete, service layer pattern established)
- Optional dependency pattern (UNGAR integration as reference)
- Existing dataset export formats (trace, tunix_sft, training_example)
- Coverage strategy (.coveragerc with omit patterns for optional dependencies)
- Test markers (@pytest.mark.ungar, @pytest.mark.training)

## Questions Before Implementation

### 1. Tunix Installation Source

**Q:** The plan mentions "Prefer pinning to a specific version (use `google-tunix` from PyPI unless you already standardized on Git SHA pins)."

- Have you confirmed `google-tunix` exists on PyPI? 
- Should I check PyPI first, or do you prefer a specific Git commit from the Tunix repository?
- If PyPI, what's your preferred version number (or should I use the latest)?

**My Recommendation:** Let me search PyPI and the Tunix GitHub repo to determine the best installation method, similar to how UNGAR was pinned to a specific commit.

---

### 2. Tunix "Run Manifest" Structure

**Q:** What exactly is a "Tunix run manifest"? 

From the plan: "generate a Tunix run manifest (YAML/JSON config) that a developer can execute locally on their own machine/TPU VM."

- Is this a config file that Tunix CLI accepts (like `tunix train --config manifest.yaml`)?
- What fields are required? (e.g., model_id, dataset_path, learning_rate, num_epochs, etc.)
- Do you have example Tunix configs I should reference, or should I infer from Tunix documentation?

**My Recommendation:** I'll search for Tunix documentation/examples to understand the expected manifest format.

---

### 3. JSONL Export Format - New or Reuse?

**Q:** Should the Tunix export use existing formats or create a new one?

Currently you have:
- `trace` - raw trace data
- `tunix_sft` - Gemma chat template with reasoning steps
- `training_example` - prompt/response pairs

The plan says: "Use your existing 'training example' / 'tunix_sft' record shape from M09/M10 if present."

**Should I:**
- A) Reuse `tunix_sft` format for M12 Tunix export?
- B) Create a new `tunix_export` format that's optimized for the Tunix manifest workflow?
- C) Default to `training_example` format?

**My Recommendation:** Reuse `tunix_sft` format since it's already Tunix-optimized with Gemma templates.

---

### 4. Frontend Panel Scope

**Q:** How minimal should the frontend panel be?

The plan says "minimal" but lists:
- Tunix availability status
- Trace IDs or dataset selector
- Model ID input
- Export JSONL button
- Generate Manifest button

**Should I include:**
- A simple text input for trace IDs (comma-separated)?
- A dropdown for existing datasets (from `/api/datasets/*` endpoints)?
- Basic hyperparameter inputs (learning rate, epochs, batch size)?
- Or just bare minimum: status + two buttons that use hardcoded defaults?

**My Recommendation:** Match the UNGAR panel complexity - status display, simple form with 2-3 inputs, action buttons.

---

### 5. Hyperparameters for Manifest Generation

**Q:** What hyperparameters should the manifest builder accept?

The plan mentions: "inputs: dataset export reference + model_id + output_dir + basic hyperparams"

**Which hyperparams?**
- Learning rate?
- Number of epochs?
- Batch size?
- Max sequence length?
- Optimizer type?

**My Recommendation:** Start with the most common SFT params: `learning_rate`, `num_epochs`, `batch_size`, `max_seq_length` - all with sensible defaults.

---

### 6. CI Workflow Timing

**Q:** When should I create the `.github/workflows/tunix-integration.yml`?

The plan has this in "Phase 5 — Optional CI workflow + docs" but also says "Default CI must remain green without Tunix installed."

**Should I:**
- A) Create the workflow file from the start (Phase 1) but marked non-blocking?
- B) Wait until Phase 5 after all backend/frontend work is done?

**My Recommendation:** Phase 1 - add the workflow early so we can validate the integration throughout development, but with `continue-on-error: true`.

---

### 7. Completion Expectation

**Q:** Should M12 be completed in this single session?

The plan is substantial:
- Backend: 6 deliverables (availability, export, manifest, endpoints, tests)
- Frontend: Tunix panel
- CI: Optional workflow
- Docs: M12_TUNIX_INTEGRATION.md

**Estimated effort:** 3-5 hours of implementation + testing

**Should I:**
- A) Complete all phases in this session (I can handle it with context window support)?
- B) Stop at specific checkpoints for your review?
- C) Prioritize certain phases over others?

**My Recommendation:** Complete Phases 0-4 in this session (backend + frontend), then pause for your review before adding CI workflow and final docs.

---

### 8. Tunix Import Strategy

**Q:** Given Tunix might be Google-internal or have installation complexity, should I implement a mock-first approach?

**Option A: Real-first (like UNGAR)**
- Try to install real Tunix from PyPI/GitHub
- Implement real export logic
- Fall back to 501 when not installed

**Option B: Mock-first (safer for M12 Phase 1)**
- Implement availability shim + manifest builder with no Tunix dependency
- Export uses existing `tunix_sft` format from M09
- Manifest builder creates valid YAML without actually calling Tunix APIs
- Defer real Tunix client integration to M13

**My Recommendation:** Option B (mock-first) aligns with plan's "Phase 1: manifest generation + export only, no real training execution" - we just need to emit valid configs and exports that someone with Tunix installed can use.

---

## Summary of Recommendations

Based on my analysis, here's what I propose:

1. **Tunix Installation:** Search PyPI + GitHub, pin to stable version/commit
2. **Manifest Format:** Infer from Tunix docs (will search for examples)
3. **JSONL Format:** Reuse existing `tunix_sft` format
4. **Frontend Panel:** Match UNGAR panel complexity (status + simple form)
5. **Hyperparameters:** learning_rate, num_epochs, batch_size, max_seq_length (with defaults)
6. **CI Workflow:** Add in Phase 1, non-blocking
7. **Completion:** Phases 0-4 this session, pause before CI/docs finalization
8. **Strategy:** Mock-first for Phase 1 (no actual Tunix runtime calls, just manifest/export generation)

**Please confirm or correct these assumptions, and I'll create the TODO list and begin implementation.**
