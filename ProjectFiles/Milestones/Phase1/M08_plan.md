Here’s how I’d proceed now that **M07 is green and complete**: lock in the remaining **low-risk hardening items** called out in the M07 audit, then move to a focused next milestone that turns “UNGAR traces exist” into a **repeatable learning dataset pipeline** for Tunix—without prematurely taking on full-blown TPU training in CI.

Below is a **Cursor-ready plan** you can hand off as **M08** (with small phases + end-to-end verification).

(References: your M07 audit + summary.  )

---

# M08 Plan — Dataset & Training Bridge v1 (Tunix-Ready), + UNGAR Hardening

## Milestone intent

Make tunix-rt produce **training-ready, versioned datasets** from stored traces (including UNGAR traces) and add a **smoke training harness** that validates Tunix compatibility—without making model training a CI blocker.

This directly supports Tunix’s purpose as a **post-training library** (SFT/RL/etc.) and the “show your work” Kaggle framing. ([Google Developers Blog][1])

---

## Phase 0 — Baseline Gate (no risk)

**Goal:** prove M07 baseline is still stable before touching anything.

**Tasks**

* Create `docs/M08_BASELINE.md` with:

  * commit SHA
  * backend/frontend/e2e pass counts
  * core coverage numbers
* Run standard local commands (same as M07 baseline process).

**Exit criteria**

* Baseline doc committed; CI remains green.

---

## Phase 1 — Close the M07 “paper cuts” (fast, stabilizing)

These are explicitly called out as low-priority improvements in the audit; knock them out early. 

### 1. Add explanatory comments for optional dependency `type: ignore`

**Files**

* `backend/tunix_rt_backend/integrations/ungar/availability.py`
* any other UNGAR import sites

**Exit criteria**

* mypy still clean; no functional changes.

### 2. Add a “Quick Start Happy Path” block to UNGAR docs

**File**

* `docs/M07_UNGAR_INTEGRATION.md`

**Exit criteria**

* One copy/pasteable sequence: install extra → run server → generate → export JSONL.

### 3. Add minimal logging on “defensive fallback” (`"??"`) paths

**Files**

* `backend/tunix_rt_backend/integrations/ungar/high_card_duel.py`

**Guardrail**

* Use `logger.warning(...)` but **never** crash; keep fallback behavior.

**Exit criteria**

* Optional UNGAR workflow shows helpful logs if extraction fails.

---

## Phase 2 — Canonical Dataset Export v1 (DB → dataset manifest + JSONL)

Right now you can export JSONL, but M08 should make it **reproducible**, **versioned**, and **Tunix-consumable**.

### 2.1 Add “dataset manifests”

**New folder**

* `datasets/` (repo root or `backend/datasets/`—pick one convention and stick to it)

**New artifact**

* `datasets/<dataset_name>/manifest.json`

  * `dataset_name`, `dataset_version`
  * export timestamp
  * source filters (e.g., `source=ungar`, `game=high_card_duel`)
  * trace_ids included
  * schema version
  * stats: count, avg step count, etc.

### 2.2 Add export endpoint for datasets (not just raw traces)

**New endpoints**

* `POST /api/datasets/build`

  * request: filters (source/game), limit, seed, selection strategy
  * response: manifest + dataset_id
* `GET /api/datasets/{dataset_id}/export.jsonl`

  * streams JSONL based on manifest

**Key: schema stability**

* Output should keep the already-established fields (`prompts`, `trace_steps`, `final_answer`, `metadata`) from M07. 

**Exit criteria**

* Can create a dataset from UNGAR traces and export it deterministically later.

---

## Phase 3 — Tunix Prompt Renderer (dataset → SFT-ready “prompts”)

This is the bridge from “trace JSONL” to “what Tunix notebooks expect”.

### 3.1 Add a renderer module

**New module**

* `backend/tunix_rt_backend/training/renderers/tunix_sft.py`

**Function**

* `render_tunix_prompt(example) -> str`

  * Takes `{prompt, trace_steps, final_answer}`
  * Produces a single “prompt string” formatted the same way training expects.

**Notes**

* Tunix examples commonly create a “prompts” field used for training/inference formatting. ([Kaggle][2])
* Keep it deterministic; no LLM-based rewriting.

### 3.2 Add export format switch

* `GET /api/datasets/{dataset_id}/export.jsonl?format=trace|tunix_sft`

  * `trace`: current format
  * `tunix_sft`: includes:

    * `prompts` (rendered)
    * `final_answer`
    * `metadata`

**Exit criteria**

* You can export a dataset in a format that a Tunix SFT script/notebook can consume with minimal glue.

---

## Phase 4 — Training Smoke Harness (non-blocking, but real)

We’re not doing full training in CI, but we **must** verify end-to-end that:

* dataset export loads,
* tokenization / shaping works,
* Tunix training loop can run a *tiny* dry-run.

Tunix is designed for scalable post-training (SFT/RL) in JAX. ([Google Developers Blog][1])

### 4.1 Add a local “smoke SFT” script

**New**

* `training/sft_smoke.py` (or `notebooks/` if you prefer)

**Behavior**

* Loads exported `tunix_sft` JSONL (small N like 32)
* Runs minimal steps (e.g., 5–10) with tiny batch size
* Prints loss and exits successfully

**Guardrails**

* Default to CPU (so it runs everywhere)
* Allow GPU/TPU via env flags (but not required)

### 4.2 Add CI: “training-data validation” (blocking) and “training smoke” (optional)

* Blocking CI job: validates dataset schema + renderer output only (fast)
* Optional/manual job: runs the tiny training script (can be slow / env-sensitive)

**Exit criteria**

* Every PR proves dataset exports are valid + renderable.
* Training smoke can be run when needed without destabilizing main CI.

---

## Phase 5 — E2E coverage (close the gap)

The M07 audit notes there’s **no Playwright E2E** for the UNGAR panel. Add a minimal one now. 

**E2E test**

* Asserts UNGAR section renders
* Asserts status shows “Not Installed” in default env
* (Optional) dataset UI not required unless you add it in M08

**Exit criteria**

* E2E protects against silent frontend regressions around optional integrations.

---

# Definition of Done

* ✅ CI green
* ✅ Dataset manifests exist + deterministic export from manifest
* ✅ Tunix SFT prompt renderer implemented + tested
* ✅ JSONL export supports `format=tunix_sft`
* ✅ Training smoke harness exists (optional/manual)
* ✅ E2E includes UNGAR panel smoke
* ✅ Docs updated (M07 quick start + M08 docs)

---

# What comes after M08

**M09 (recommended):** “Evaluation Loop v2 + Tunix Fine-tune Run”

* Actually run a small SFT (and later GRPO/GRPO-style) pipeline using Tunix, then use your existing trace scoring/compare to evaluate output quality. Tunix supports multiple post-training algorithms (SFT + RL variants) so this becomes the natural next escalation. ([Google Developers Blog][1])

---

If you want, I can also generate the **exact file-by-file checklist** Cursor should follow (with test names and acceptance checks per PR) in the same style as your earlier M05–M07 handoffs.

[1]: https://developers.googleblog.com/introducing-tunix-a-jax-native-library-for-llm-post-training/?utm_source=chatgpt.com "Introducing Tunix: A JAX-Native Library for LLM Post-Training"
[2]: https://www.kaggle.com/code/yekahaaagayeham/guide-gemma-3-fine-tuning-with-tunix?utm_source=chatgpt.com "Guide : Gemma 3 fine tuning with Tunix"
