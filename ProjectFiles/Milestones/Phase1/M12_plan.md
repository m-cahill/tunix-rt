According to a document from **December 22, 2025**, **M11 is closed** and the intended next step is to move beyond “trace CRUD/export” and start building the **trace → dataset → run/eval loop** that can feed Tunix workflows (with clean abstractions/guardrails).

Below is a **Cursor-handoff prompt for M12** that keeps core CI stable, treats Tunix as an **optional integration**, and delivers a concrete “artifact” you can run locally: a **Tunix-ready dataset + generated run manifest** (without trying to actually train inside CI).

---

## Cursor Prompt — M12: Tunix Integration Skeleton + Run Manifest Pipeline (Phase 1)

### Goal (M12 North Star)

Add an **optional Tunix integration** that can:

1. **export traces into Tunix-friendly JSONL** (SFT-focused to start), and
2. generate a **Tunix run manifest** (YAML/JSON config) that a developer can execute locally on their own machine/TPU VM.

Tunix context: it’s a lightweight, JAX-native LLM post-training library supporting **SFT + multiple RL methods**, with install paths via PyPI and GitHub.

### Hard Constraints

* **Default CI must remain green** without Tunix installed.
* Tunix code paths must **fail gracefully** (501/503 + clear message) when Tunix isn’t installed.
* Optional integration code must **not dilute coverage gates** (use the same strategy you used for optional UNGAR integration).
* Keep this milestone “Phase 1”: **manifest generation + export only**, no real training execution in backend.

### Deliverables

#### Backend

1. `backend[tunix]` optional extra:

   * Add a `tunix` extra in `backend/pyproject.toml` (or project root, depending on current layout) that installs Tunix.
   * Prefer pinning to a specific version (use `google-tunix` from PyPI unless you already standardized on Git SHA pins).

2. Tunix availability shim (pattern-matched to UNGAR):

   * `tunix_rt_backend/integrations/tunix/availability.py`

     * `is_tunix_available() -> bool`
     * `require_tunix() -> None` raising a clear ImportError message.

3. Tunix dataset export service (SFT only, Phase 1):

   * New service module: `tunix_rt_backend/services/tunix_export.py`
   * Function: `export_tunix_sft_jsonl(traces: list[Trace], ...) -> str | Path`
   * Use your existing “training example” / “tunix_sft” record shape from M09/M10 if present; otherwise define:

     * `{"prompt": ..., "trace_steps": [...], "final_answer": ..., "metadata": {...}}`
   * Keep it intentionally minimal and deterministic.

4. Tunix run-manifest builder (no execution):

   * `tunix_rt_backend/integrations/tunix/manifest.py`
   * `build_sft_manifest(dataset_path: str, model_id: str, output_dir: str, hyperparams: ...) -> dict[str, Any]`
   * Output should be easy to run from CLI / notebook.
   * Note in docs that Tunix supports SFT and various RL approaches; we’re only emitting an SFT manifest in M12.

5. API endpoints (graceful when Tunix missing):

   * `GET /api/tunix/status`

     * returns `{ available: bool, version?: str }`
   * `POST /api/tunix/sft/export`

     * inputs: trace IDs (or dataset filters if already supported), plus output naming
     * outputs: JSONL file (or content + filename)
   * `POST /api/tunix/sft/manifest`

     * inputs: dataset export reference + model_id + output_dir + basic hyperparams
     * outputs: manifest JSON + downloadable YAML/JSON

6. Tests

   * Default (no Tunix installed):

     * status endpoint returns `available=false`
     * export/manifest endpoints return `501` (or `503`) with clear message
   * Optional (when `backend[tunix]` installed):

     * mark tests `@pytest.mark.tunix`
     * validate manifest schema fields, and JSONL line format
   * Ensure mypy passes (no bare `dict`).

#### Frontend (minimal)

7. Add a “Tunix” panel (mirrors UNGAR style):

   * Show Tunix availability (from `/api/tunix/status`)
   * Provide a small form:

     * Trace IDs or dataset selector
     * Model ID
     * Buttons: **Export JSONL** and **Generate Manifest**
   * Add `data-testid` attributes for anything touched by E2E.

#### CI (optional job)

8. Add `.github/workflows/tunix-integration.yml` (non-blocking):

   * Installs `backend[tunix]`
   * Runs `pytest -m tunix`
   * Uses `continue-on-error: true` (or separate “optional checks” lane)
   * Upload manifest/JSONL artifacts for debugging (if helpful)

#### Docs

9. `docs/M12_TUNIX_INTEGRATION.md`

   * How to install optional extra
   * What endpoints exist
   * JSONL schema emitted
   * How to run locally (example CLI snippet)
   * Mention Tunix install commands and that it supports SFT + RL; this milestone is SFT export + manifest only.

---

## Implementation Plan (Phased, with checkpoints)

### Phase 0 — Baseline Gate

* Pull latest main, run:

  * backend: `ruff`, `mypy`, `pytest`, coverage gate
  * frontend: tests/build
  * e2e: full suite
* Save `docs/M12_BASELINE.md` (short: commit hash + green checks).

### Phase 1 — Backend scaffolding (availability + wiring)

* Create `integrations/tunix/*` and `services/tunix_export.py`
* Add endpoints returning 501/503 when Tunix absent
* Add default tests for “Tunix missing” behavior
* Keep coverage stable via the established optional-integration approach

### Phase 2 — JSONL export (SFT)

* Implement deterministic JSONL writer
* Add optional `@pytest.mark.tunix` tests validating real export formatting

### Phase 3 — Manifest generation

* Implement manifest builder + endpoint
* Add tests for schema correctness
* Ensure mypy types are complete (`dict[str, Any]`, TypedDict where appropriate)

### Phase 4 — Frontend panel + light E2E hardening

* Add UI panel + testids
* Add one E2E test that:

  * verifies status shows “not installed” in CI (or skips if panel hidden)
  * (optional) only runs manifest path if Tunix is available (probably skip in default CI)

### Phase 5 — Optional CI workflow + docs

* Add `tunix-integration.yml` as non-blocking
* Write `docs/M12_TUNIX_INTEGRATION.md`
* Ensure main CI still green

---

## Acceptance Criteria (Must-Haves)

* Default CI: **GREEN** (no Tunix installed)
* `/api/tunix/status` works in all environments
* Tunix endpoints fail gracefully when unavailable (501/503 + clear message)
* Optional workflow can be run to validate real exports/manifests
* Docs exist and a dev can generate:

  * Tunix SFT JSONL
  * Tunix run manifest

---

## Out of Scope (explicit)

* Running Tunix training inside the backend service
* TPU orchestration / long-running job management
* RL pipelines (GRPO/PPO/DPO etc.) beyond a doc mention
* LLM-as-judge scoring expansion (park for M13+; if needed later, we can align to known “LLM judge” evaluation patterns)

---

If you want, M13 can be the natural follow-on: **Run registry + job execution adapter** (local subprocess / remote runner) + “judge” scoring upgrades—while keeping the same optional-integration discipline.
