According to a document from **September 30, 2025**, **Tunix** is a JAX-native post-training library that supports SFT + RL at TPU scale and is explicitly designed to make post-training loops customizable (“white-box” design). ([Google Developers Blog][1])
Given that **M6 is now stable**, M7 should be the **first UNGAR-facing milestone**, but done in a way that **does not couple UNGAR into the core runtime** (so we preserve the CI stability you just earned).

Below is a **Cursor-ready M7 prompt** that’s small, end-to-end verifiable, and sets up UNGAR as a *data / task source* for Tunix trace training—without prematurely “merging projects.”

---

## Prompt to hand off to Cursor — M07: UNGAR Dataset Bridge + Tunix Trace Export (Phase 1)

### Context

* We have **tunix-rt** (FastAPI + React + Playwright) with stable CI after M6.
* We now want to **incorporate UNGAR** into the learning system by using UNGAR as a **structured generator of training examples** (not by embedding UNGAR into the API core).
* UNGAR’s architectural split is **Core vs Bridge**: core stays lightweight; bridge is where external integration belongs. 
* UNGAR’s canonical representation is a **4×14×n card tensor** (4×13×n base slice). 

### M7 North Star

Deliver an **UNGAR → Tunix-RT trace data bridge**:

1. We can **generate/import UNGAR-derived traces** into tunix-rt’s `traces` table for inspection/comparison/scoring.
2. We can **export tunix-rt traces** to a **Tunix-friendly JSONL** format for training (“show your work” style traces). (This aligns with the Kaggle “train a model to show its work” framing.) ([Kaggle][2])
3. All of this is **optional** at runtime: tunix-rt must still run **without UNGAR installed**.

### Non-goals (strict)

* No new “UNGAR UI” in the frontend.
* No deep training loop integration inside tunix-rt yet (that’s later).
* No changes to UNGAR repo or its contracts; we only consume it.

---

## Phase 0 — Baseline Gate (must pass before coding)

**Goal:** Prove M6 baseline is truly stable.

* Pull latest `main`.
* Run:

  * `backend` tests (py3.11 + py3.12)
  * `frontend` tests
  * `e2e`
* Save the CI outputs in `docs/M07_BASELINE.md` (short: date, commit, pass/fail, coverage numbers).

**Acceptance:** everything green locally.

---

## Phase 1 — Define the “Training Trace Export” Contract (small, explicit)

### 1.1 Create a minimal export schema (new module)

Add `backend/tunix_rt_backend/export/tunix_jsonl.py` with:

* `export_trace_to_jsonl_record(trace: Trace, *, include_score: bool = True) -> dict`
* The record should be:

  * `id`
  * `prompt` (or `question`)
  * `trace_steps` (array of step strings)
  * `final_answer` (if present)
  * `metadata` (created_at, source, tags)
  * `scores` (optional: latest baseline score)

**Guardrails**

* Function must be deterministic.
* No network calls.
* No dependency on UNGAR.

### 1.2 Add API endpoint: export trace

Add:

* `GET /api/traces/{trace_id}/export?tunix=jsonl`

  * returns JSONL **text** (single record) with `Content-Type: application/x-ndjson`
  * If `tunix` missing or not `jsonl`, return 400 with helpful message.

**Acceptance Tests**

* New backend tests:

  * 200 success: correct content-type + JSONL parseable
  * 404 if trace not found
  * 400 for unsupported format

---

## Phase 2 — UNGAR “Generator/Importer” (Phase 1, minimal)

We will treat UNGAR as a **data source** that yields `(prompt, steps, answer, metadata)`.

### 2.1 Add optional dependency wiring

* Add a backend extra: `.[ungar]`
* In code: never import UNGAR at module import time unless within a guarded function:

  * `try: import ungar ... except ImportError: ...`

**Acceptance**

* Running backend without UNGAR installed still works and all tests pass.

### 2.2 Implement UNGAR sample generator (thin slice)

Create `backend/tunix_rt_backend/datasets/ungar_generate.py`:

* `generate_ungar_traces(count: int, seed: int) -> list[TraceCreate]`
* Start with **High Card Duel** only (smallest state/action space).
* Convert each UNGAR episode into a tunix-rt trace:

  * `prompt`: “Given game state…, choose action …”
  * `steps`: a short, consistent reasoning trace derived from:

    * legal moves
    * simple tensor summary (“my_hand contains …”, “unseen count …”)
  * `final_answer`: chosen move
  * `metadata.source = "ungar.high_card_duel"`

**Important:** keep the transformation simple; do *not* attempt perfect natural language.

**Grounding to UNGAR contracts**

* UNGAR core includes a simulation runner and a 4×14×N tensor representation. 

### 2.3 Add API endpoint: generate & insert UNGAR traces

Add:

* `POST /api/datasets/ungar/generate`

  * body: `{ "count": 25, "seed": 42 }`
  * behavior:

    * if UNGAR not installed: 501 with message “Install backend with [ungar] extra”
    * else: generate traces and insert into DB using existing create-trace flow
  * response: `{ inserted: n, trace_ids: [...] }`

**Acceptance Tests**

* With UNGAR installed in CI job (or at least in a dedicated optional job):

  * returns 200 and inserts n traces
* Without UNGAR installed:

  * returns 501, does not crash app

**CI Strategy (important)**

* Do **not** break default CI.
* Add a **separate** GitHub Actions job (non-blocking OR nightly) that installs `.[ungar]` and runs only the UNGAR dataset tests.

---

## Phase 3 — End-to-End Verification

### 3.1 Add one E2E test (small)

Test flow:

1. Create a normal trace (existing behavior).
2. Export it via `/export?tunix=jsonl`.
3. Validate that exported JSONL includes `trace_steps` and is parseable.

*(We do NOT E2E the UNGAR generator yet to avoid CI flakiness from optional deps.)*

### 3.2 Add one “integration smoke” backend test

* Start app, hit `/health` (or existing status endpoint), ensure it works with UNGAR absent.

---

## Phase 4 — Docs (short, but real)

Add:

* `docs/M07_UNGAR_BRIDGE.md`:

  * how to install: `pip install -e ".[ungar]"`
  * how to generate traces: curl example
  * how to export JSONL: curl example
  * explicit statement: “UNGAR is an optional dependency; core runtime does not require it.”
* Update main README with a tiny “UNGAR dataset bridge (optional)” section.

---

## Definition of Done

* ✅ Default CI remains green (no mandatory UNGAR deps).
* ✅ Export endpoint works + is tested.
* ✅ UNGAR generator endpoint works when extras installed + has tests in a dedicated job.
* ✅ Docs added.
* ✅ No frontend changes required.

---

## Implementation Notes / Guardrails

* Keep UNGAR imports *inside* functions to prevent import-time failures.
* Keep endpoints small and deterministic.
* Prefer adding `data-testid` only if UI changes occur (none planned in M7).
* Any new CI job must be pinned and use caching like existing jobs.

---

If you want, after M7 lands we can make **M8** the “real learning loop” milestone: producing bulk JSONL exports, wiring a Tunix SFT script, and closing the loop with “train → evaluate → compare traces” using Tunix’s post-training design. ([GitHub][3])

[1]: https://developers.googleblog.com/introducing-tunix-a-jax-native-library-for-llm-post-training/?utm_source=chatgpt.com "Introducing Tunix: A JAX-Native Library for LLM Post-Training"
[2]: https://www.kaggle.com/competitions/google-tunix-hackathon?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
[3]: https://github.com/google/tunix?utm_source=chatgpt.com "google/tunix: A Lightweight LLM Post-Training Library"
