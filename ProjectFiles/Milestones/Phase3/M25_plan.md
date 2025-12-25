M24 got you **real inference + reproducible backend deps (uv.lock)**, but the audit flags the immediate next priorities as: **stabilize the new inference tests (currently brittle), recover coverage, and start the “real” Tunix/JAX training path**.  

Below is the **Cursor handoff prompt for M25**.

---

## Cursor Handoff Prompt — M25: CI Stabilization + Coverage Recovery + Real Tunix/JAX Training Path

### Context

* M24: `uv.lock` enforced deterministic builds + real inference via `transformers` fallback (`distilgpt2`) + baseline experiment loop. 
* M24 audit flags:

  * **Tests failing**: `test_generate_predictions_success` hitting `JSONDecodeError` (likely due to optional `transformers` availability or mock path producing empty/invalid output). 
  * **Coverage** dipped to ~69%. 
* Goal: restore green, restore coverage gate, then implement the **real Tunix/JAX** training pipeline (Torch fallback stays as dev/CI smoke).

---

# Phase 0 — Baseline Gate (mandatory)

1. Pull `main`.
2. Run the failing tests locally and confirm repro:

   * `cd backend && pytest -q -k m24_inference -vv`
3. Identify whether `transformers` is installed in the minimal test env or absent.

**Exit criteria:** root cause of `JSONDecodeError` reproduced and understood.

---

# Phase 1 — Fix inference test fragility (highest ROI, smallest diff)

## 1.1 Make tests deterministic regardless of optional ML deps

**Preferred approach:** fully mock the inference engine in the test so it never depends on transformers/torch.

* In `backend/tests/test_m24_inference.py`:

  * Patch `_run_inference_sync` (or whatever writes predictions) to write a valid `predictions.jsonl` with a couple lines.
  * Assert that `generate_predictions()` returns expected path(s) and that JSONL parsing works.

**Alternative (only if needed):** skip tests cleanly when optional deps are missing using `pytest.importorskip`, which is explicitly supported for optional imports. ([pytest][1])

> Use mocking for unit tests; use `importorskip` only for an optional “smoke” test that actually exercises transformers.

## 1.2 Guardrail: never write invalid JSONL

If the service can produce an empty file on failure, harden it:

* On exception, either:

  * raise and **do not create** the predictions file, or
  * write a valid JSONL “error record” line (but then the judge must ignore it)

**Recommendation:** raise + no file. This keeps contracts strict and debuggable.

**Exit criteria:** `pytest -q` passes; no JSON decode failures.

---

# Phase 2 — Restore coverage gate (keep it honest)

## 2.1 Add unit tests for inference error paths

Add targeted tests for:

* missing dataset manifest → raises with clear message
* empty dataset → raises with clear message
* missing model artifacts → raises

## 2.2 Reinstate a coverage gate

* Bring backend coverage back to **≥70% immediately**, target **≥75%** if it’s within reach in this milestone.
* Add/adjust CI step to enforce `--cov-fail-under=<threshold>`.

**Exit criteria:** coverage is back above gate and non-regressing.

---

# Phase 3 — Real Tunix/JAX training path (the “real training” milestone)

M24’s `training/train_sft_tunix.py` hybrid is useful, but it’s getting large; M25 should start splitting responsibilities. 

## 3.1 Split training script for maintainability

Refactor:

* `training/train_sft_tunix.py` → orchestrator
* `training/train_torch.py` → the existing Transformers fallback
* `training/train_jax.py` → **real Tunix/JAX implementation**

Keep the subprocess boundary the same (service calls script) to preserve isolation.

## 3.2 Implement “real” JAX/Tunix training entrypoint

Implement:

* dataset ingestion from the exported JSONL
* training loop that emits:

  * model artifacts
  * training logs/metrics
  * whatever Tunix expects

**Guardrail:** if JAX/Tunix deps aren’t installed, fail early with a crisp error, and keep Torch fallback as the default dev path.

---

# Phase 4 — Hardware selector + reproducibility metadata

## 4.1 Add a device selector to TunixRunRequest

Add `device: "cpu" | "cuda" | "auto"` (and store it in run config/provenance).

* Default: `auto` for local
* CI: `cpu` explicitly

## 4.2 Record inference determinism parameters

Your inference uses greedy decode (`do_sample=False`, `num_beams=1`). That is the documented greedy decoding mode for HF `generate()`. ([Hugging Face][2])
Persist the generation config in `predictions_meta.json` alongside:

* model id / registry version
* dataset key
* judge version

---

# Phase 5 — End-to-end “first real JAX run” smoke

Add a **very small** smoke config:

* 10–50 steps
* tiny batch
* `golden-v1`
* run: Train → Inference → Eval → Promote

**Exit criteria:** one successful JAX run produces non-trivial metrics and can be promoted.

---

# CI / DX Guardrails (keep M24 wins)

* Keep `uv sync --locked` in CI; prefer official `setup-uv` integration and caching patterns. ([Astral Docs][3])
* Ensure optional ML libs do not break core CI:

  * mock by default
  * tag “ML smoke” tests and run only when extras are installed

---

## Definition of Done (M25)

* [ ] `test_generate_predictions_success` stabilized (no JSONDecodeError). 
* [ ] Backend coverage back above gate (≥70% minimum; ≥75% preferred). 
* [ ] Training scripts split (`train_torch.py`, `train_jax.py`) to stop the monolith from growing. 
* [ ] Real Tunix/JAX training path implemented (Torch remains fallback). 
* [ ] Device selector stored in run config/provenance.
* [ ] One tiny “real JAX” E2E training smoke run completes and can be promoted.

---

If you want M26 queued immediately after M25, the natural follow-on is: **GPU acceleration + throughput tuning + “real” dataset scale-up (beyond golden-v1)**—but M25 is the clean step that turns your system from “loop proven” into “training actually happening.”

[1]: https://docs.pytest.org/en/stable/how-to/skipping.html?utm_source=chatgpt.com "How to use skip and xfail to deal with tests that cannot ..."
[2]: https://huggingface.co/docs/transformers/en/main_classes/text_generation?utm_source=chatgpt.com "Generation"
[3]: https://docs.astral.sh/uv/guides/integration/github/?utm_source=chatgpt.com "Using uv in GitHub Actions - Astral Docs"
