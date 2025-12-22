According to the current public Tunix + Gemma docs, the most leverageful “next step” after M8 being green is to make **an end-to-end, reproducible SFT training + eval loop** that turns your stored traces (plus optional UNGAR-generated episodes) into **Gemma-formatted JSONL**, runs a **tiny Tunix SFT job**, then **re-imports outputs as traces** so your existing scoring/compare UI can evaluate deltas. ([GitHub][1])

Below is a **Cursor handoff plan for M09** that stays small, keeps CI stable, and is “enterprise-grade” in the sense that everything is reproducible, validated, and doesn’t blow up coverage gates.

---

## M09 Goal

**M09 = “Reproducible Training Loop v1 (SFT)”:**
Create a **deterministic dataset exporter + minimal Tunix SFT runner + post-train evaluation harness** that produces artifacts you can compare inside tunix-rt.

Why this now:

* Tunix explicitly supports **Supervised Fine-Tuning** (and later RL), and provides recipes/examples you can align with. ([GitHub][1])
* Gemma IT models expect a specific **turn-token formatting** at inference time; your exporter should enforce it so training/inference match. ([Google AI for Developers][2])
* The Kaggle hackathon framing rewards **reproducibility + “show your work”**; this turns tunix-rt into a real pipeline, not just a viewer. ([AI Competition Hub][3])

---

## Phase 0 — Baseline gate (no code changes)

**Objective:** prove M8 remains stable as the base.

**Tasks**

* Checkout `main`, pull latest.
* Run full suite locally:

  * `backend`: ruff/mypy/pytest + coverage
  * `frontend`: tests/build
  * `e2e`: playwright

**Acceptance**

* CI-equivalent local run is green.
* Record commit hash in `docs/M09_BASELINE.md`.

**Guardrail**

* If anything is flaky, fix first—M09 assumes a stable base.

---

## Phase 1 — Dataset contract (the “golden pipe”)

**Objective:** define one stable JSONL format you can use for Tunix SFT, and test it hard.

### 1.1 Add a “training dataset” schema (pure-Python, testable)

**Files**

* `backend/tunix_rt_backend/training/schema.py` (new)
* `backend/tests/test_training_schema.py` (new)

**What to implement**

* A `TrainingExample` structure:

  * `id` (uuid)
  * `prompt` (string)
  * `response` (string)  *(this is your “final answer” or full trace style output depending on recipe)*
  * `metadata` (source trace id, created_at, tags, etc.)

**Acceptance**

* Unit tests validate serialization, required fields, stable ordering, and round-trips.

### 1.2 Implement **Gemma IT formatter** (critical)

**Files**

* `backend/tunix_rt_backend/training/gemma_format.py` (new)
* `backend/tests/test_gemma_format.py` (new)

**Implementation detail**

* Implement helpers that build the actual training text with control tokens:

  * `<start_of_turn>user … <end_of_turn>`
  * `<start_of_turn>model … <end_of_turn>`
* Follow Gemma’s documented control tokens and the “prompt prefix” behavior. ([Google AI for Developers][2])

**Acceptance**

* Snapshot tests for:

  * single-turn prompt
  * “system-like” instructions embedded in the initial user turn (since Gemma IT doesn’t support `system` role). ([Google AI for Developers][2])
* Formatter is deterministic and newline-stable.

**Guardrail**

* If the formatter changes, tests must change *explicitly* (snapshots). No silent drift.

---

## Phase 2 — Exporters (Traces → JSONL)

**Objective:** export (a) tunix-rt traces and (b) UNGAR episodes into the same dataset contract.

### 2.1 Trace exporter

**Files**

* `backend/tunix_rt_backend/training/export_traces.py` (new)
* `backend/tests/test_export_traces.py` (new)

**What it does**

* Pull traces from DB (existing tables).
* Convert each trace into a `TrainingExample` using a **single recipe**:

  * **Recipe v1 (simple):**
    user prompt = original question + “show your work as steps” instruction
    model response = `steps + final_answer` (in your existing trace JSON rendering style)

**Acceptance**

* Can export N traces deterministically with:

  * `--limit`
  * `--since`
  * `--out path.jsonl`
* Tests validate stable output count, schema, and formatting.

### 2.2 UNGAR exporter (optional dependency, no CI penalty)

**Files**

* `backend/tunix_rt_backend/training/export_ungar.py` (new)
* Reuse your existing UNGAR optional integration patterns from M7/M8.

**What it does**

* If UNGAR is available: generate High Card Duel episodes and convert to examples.
* If UNGAR not available: cleanly fail with a clear message.

**Acceptance**

* In default CI: only tests the “UNGAR unavailable” path (no coverage dilution).
* In optional UNGAR workflow: real generation test runs.

**Guardrail**

* Keep all UNGAR-only code behind lazy imports so default CI doesn’t pull it in.

---

## Phase 3 — Minimal Tunix SFT runner (reproducible “toy train”)

**Objective:** run a **tiny** SFT job locally (GPU if available, CPU fallback) mainly to prove the pipe.

### 3.1 Add a training runner script (optional dependency)

**Files**

* `training/` (new top-level folder)

  * `training/train_sft_tunix.py`
  * `training/configs/sft_tiny.yaml`
  * `training/README.md`

**Implementation approach**

* Treat Tunix as an optional extra install:

  * Tunix install options are documented (PyPI, GitHub). ([GitHub][1])
* The runner:

  * loads exported JSONL
  * tokenizes according to the Gemma recipe you chose
  * runs a very small number of steps (like 10–50) just to verify end-to-end
  * writes artifacts to `artifacts/training_runs/<run_id>/`

**Acceptance**

* Running locally produces:

  * `run_manifest.json` (config, git sha, seed, dataset hash)
  * `metrics.jsonl` (loss over steps)
  * checkpoint output location (whatever Tunix emits)

**Guardrails**

* No checkpoints committed to git.
* Deterministic seed, recorded in manifest.
* If Tunix isn’t installed, script exits with a helpful message and docs link.

---

## Phase 4 — Post-train evaluation harness (model → traces → compare)

**Objective:** automatically generate a “before vs after” report that your UI can consume.

**Files**

* `training/eval_generate.py` (new)
* `training/eval_report.py` (new)
* `docs/M09_EVAL_LOOP.md` (new)

**What it does**

* Pick a fixed eval set (e.g., 25 prompts) from:

  * a static file `training/evalsets/eval_v1.jsonl`, or
  * deterministic DB query.
* Run inference using:

  * base model (no tuning)
  * tuned checkpoint
* Convert outputs into tunix-rt trace objects and POST them into your API (or write `import.jsonl` used by an import endpoint if you already have one).

**Acceptance**

* Produces:

  * `artifacts/training_runs/<run_id>/eval_before.jsonl`
  * `.../eval_after.jsonl`
  * `.../delta_report.md` (simple: score deltas, examples)

**Guardrail**

* Keep evaluation deterministic; always record exact prompts + seeds.

---

## Phase 5 — CI guardrails (keep it green)

**Objective:** ensure M09 doesn’t reintroduce “optional code breaks coverage” issues.

**Tasks**

* Add a **non-blocking** workflow `training-smoke.yml` that:

  * installs minimal deps (no Tunix)
  * runs formatter + exporter unit tests only
* Keep your optional UNGAR workflow as-is (runs real UNGAR tests separately).

**Acceptance**

* Default CI remains fast and coverage-stable.
* Optional workflows provide depth without blocking merges.

---

## Deliverables checklist (Definition of Done)

* [ ] `TrainingExample` schema + tests
* [ ] Gemma IT formatter + snapshot tests (control tokens, no “system” role) ([Google AI for Developers][2])
* [ ] Trace exporter → JSONL + tests
* [ ] Optional UNGAR exporter (lazy import)
* [ ] Tunix SFT “tiny runner” script (optional Tunix install) ([GitHub][1])
* [ ] Eval harness that generates before/after traces and a delta report
* [ ] Docs: baseline, dataset format, training quickstart, eval loop
* [ ] CI: new smoke workflow that doesn’t affect coverage gates

---

## Cursor handoff prompt (copy/paste)

Implement **M09: Reproducible Training Loop v1 (SFT)** for tunix-rt.

**Constraints**

* Keep main CI GREEN and coverage gates stable.
* Tunix + UNGAR remain OPTIONAL deps; default CI must not require them.
* Everything must be deterministic (seeded) with run manifests.
* No large artifacts committed.

**Phases**

1. Baseline gate doc: add `docs/M09_BASELINE.md` with commit hash and local run results.
2. Add `backend/tunix_rt_backend/training/schema.py` with `TrainingExample` + tests.
3. Add `backend/tunix_rt_backend/training/gemma_format.py` implementing Gemma IT turn-token formatting; include snapshot tests.
4. Add `backend/tunix_rt_backend/training/export_traces.py` to export DB traces → JSONL dataset; include tests.
5. Add optional `backend/tunix_rt_backend/training/export_ungar.py` with lazy imports; default tests only verify “UNGAR unavailable” behavior.
6. Create `training/` folder with:

   * `train_sft_tunix.py` (optional Tunix runner; exits gracefully if Tunix not installed)
   * `configs/sft_tiny.yaml`
   * `README.md` quickstart
   * outputs to `artifacts/training_runs/<run_id>/` including `run_manifest.json`, `metrics.jsonl`
7. Add `training/eval_generate.py` + `training/eval_report.py` that produce before/after eval JSONL and a markdown delta report; deterministic evalset.
8. Add CI workflow `training-smoke.yml` (non-blocking) to run formatter/exporter unit tests only, with no Tunix/UNGAR install.
9. Add docs: `docs/M09_DATASET_FORMAT.md`, `docs/M09_TRAINING_QUICKSTART.md`, `docs/M09_EVAL_LOOP.md`.

**Acceptance**

* `pytest` passes locally and in CI.
* Exporter produces Gemma-formatted text with correct `<start_of_turn>` / `<end_of_turn>` structure.
* Training runner produces a run manifest + metrics in artifacts folder.
* Eval harness produces before/after traces + delta report.

---

If you want M10 to be the “real” Kaggle-ready notebook + TPU recipe, M09 is the clean, low-risk substrate that makes that inevitable rather than aspirational.

[1]: https://github.com/google/tunix "GitHub - google/tunix: A Lightweight LLM Post-Training Library"
[2]: https://ai.google.dev/gemma/docs/core/prompt-structure "Gemma formatting and system instructions  |  Google AI for Developers"
[3]: https://www.competehub.dev/en/competitions/kagglegoogle-tunix-hackathon "Google Tunix Hack - Train a model to show its work - Comp..."
