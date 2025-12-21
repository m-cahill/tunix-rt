Below is a **Cursor handoff prompt for M07** that **answers the clarifying questions by making explicit decisions** and then gives an **end-to-end, phased implementation plan** (small milestones, CI-safe, enterprise-grade guardrails).

---

## Cursor Prompt — M07: UNGAR → Tunix-RT Trace Generator + JSONL Export (High Card Duel v1)

### Context

M06 is complete and stable. M07 is the **first, minimal integration** of **UNGAR** into **tunix-rt** so we can generate **Tunix-style reasoning traces** from a simple UNGAR environment (**High Card Duel**) and export them as **JSONL** for later training/eval loops.

### Decisions (answering M07 clarifying questions)

1. **UNGAR repo location / integration strategy**

* Use UNGAR as an **optional backend extra**.
* Local dev: support `pip install -e ../ungar` (sibling repo) for fast iteration.
* Repo URL is: `https://github.com/m-cahill/ungar.git` (from UNGAR docs you’ve been given).
* In tunix-rt backend, define an extra like: `backend[ungar]` that installs UNGAR from git.

2. **UNGAR version target**

* **Pin to a specific commit SHA** (immutable) for reproducible builds.
* Implementation step: resolve HEAD SHA once and pin it in `pyproject.toml` (or in a single constants file that feeds the dependency string).
* Rationale: pip explicitly supports VCS installs pinned to a commit hash and recommends using the full hash. ([Pip Documentation][1])

3. **UNGAR → trace conversion detail level**

* Use **Option A (minimal, deterministic)**. No narrative fluff. Goal: reliable scaffolding, not perfect NLG.

4. **CI strategy for UNGAR tests**

* Add a **separate workflow/job that is non-blocking** (`continue-on-error: true`) and/or **manual dispatch**.
* Main CI stays fast and stable; UNGAR is an optional integration surface.

5. **JSONL export “Tunix-friendly” format**

* Use a structure that aligns with Tunix examples which build training prompts from dataset fields like `question/answer` into `prompts`. ([tunix.readthedocs.io][2])
* We will export JSONL with at least:

  * `id`
  * `prompts` (string)
  * `final_answer` (string)
  * `trace_steps` (string[])
  * `metadata` (object)
  * optional `scores` (object)

6. **UNGAR import strategy**

* Use **lazy imports inside functions** and a single availability probe (Option B) so the app boots without UNGAR installed.

7. **Testing without UNGAR installed**

* Default suite: **Option A** — verify UNGAR endpoints return **501** with a clear message when UNGAR is not installed.
* Optional suite (only when `[ungar]` is installed): run real integration tests.

8. **Phase 0 baseline doc**

* Use **Option A (minimal baseline verification doc)**.

---

## Goals (M07 deliverables)

### Backend

1. Optional UNGAR dependency (`backend[ungar]`) pinned to commit SHA.
2. New UNGAR integration module that can:

   * run N episodes of **High Card Duel**
   * convert each episode into a tunix-rt `Trace` (prompt + steps + answer + metadata)
3. New endpoints:

   * `GET /api/ungar/status` → `{ available: bool, version?: str }`
   * `POST /api/ungar/high-card-duel/generate` → generates traces (persist to DB) and returns created trace IDs + a small preview
   * `GET /api/ungar/high-card-duel/export.jsonl?limit=...` → JSONL export of generated traces (or accept a list of trace_ids)

### Frontend (minimal)

4. Add a small “UNGAR (optional)” panel:

   * shows availability (`available` / “not installed”)
   * inputs: count, seed
   * button: “Generate High Card Duel Traces”
   * shows resulting trace IDs and allows quick navigation to view the traces

### Tests + CI

5. Tests (default suite, no UNGAR installed):

   * `/api/ungar/status` returns `available=false`
   * `/api/ungar/high-card-duel/generate` returns 501 with message
6. Tests (optional suite with UNGAR installed):

   * generate 1–3 traces successfully
   * verify trace schema fields exist and are stable
7. CI:

   * Add optional workflow or job that installs `backend[ungar]` and runs only integration tests (non-blocking + manual dispatch recommended)

### Docs

8. `docs/M07_BASELINE.md` (baseline verification)
9. `docs/M07_UNGAR_INTEGRATION.md` (how to install extra, run generator, export JSONL, and what the trace format is)
10. Update main README with a short “Optional: UNGAR generator” section.

---

## Implementation Plan (phased, CI-safe)

### Phase 0 — Baseline Gate (must stay green)

* Create `docs/M07_BASELINE.md` with commit SHA and “all green” checks.
* Run locally: backend tests, frontend tests, e2e (whatever standard commands exist).

**Exit criteria:** baseline doc exists; no changes yet that affect behavior.

---

### Phase 1 — Optional Dependency Wiring

**Backend:**

* In backend `pyproject.toml`, add:

  * `[project.optional-dependencies]`
  * `ungar = ["ungar @ git+https://github.com/m-cahill/ungar.git@<FULL_SHA>"]`
* Add a tiny module `tunix_rt_backend/integrations/ungar/availability.py`:

  * `def ungar_available() -> bool`
  * `def ungar_version() -> str | None`

**Guardrails:**

* No unconditional imports of `ungar` at module import time except inside availability checks.
* App must boot and all existing endpoints must work without UNGAR installed.

**Exit criteria:** default tests still pass.

---

### Phase 2 — UNGAR Episode → Trace Conversion (High Card Duel v1)

Create `tunix_rt_backend/integrations/ungar/high_card_duel.py`:

* `generate_high_card_duel_traces(count: int, seed: int | None) -> list[TraceCreate]`

**Trace mapping (minimal, deterministic):**

* `prompt`: `"High Card Duel: You have 1 hidden card. Action: reveal."`
* `steps`:

  * `"Legal moves: [reveal]"`
  * `"My hand: <rank><suit>"`
  * `"Unseen cards: 51"`
  * `"Action chosen: reveal"`
* `final_answer`: `"reveal"`
* `metadata`:

  * `source: "ungar"`
  * `game: "high_card_duel"`
  * `seed`, `episode_index`, `my_card`, `result` (win/loss/tie)

**Important:** do **not** include opponent hidden card in steps (keep it “reasoning-compatible”).

**Exit criteria:** conversion function unit-tested (pure logic), but integration execution may be behind optional tests.

---

### Phase 3 — API Endpoints

In FastAPI app:

* `GET /api/ungar/status`
* `POST /api/ungar/high-card-duel/generate`

  * request: `{ count: int (1..100), seed?: int, persist?: bool=true }`
  * behavior:

    * if UNGAR not installed: 501 + message: “UNGAR extra not installed”
    * else:

      * generate traces
      * persist using existing DB Trace create path
      * return `{ trace_ids: [...], preview: [...] }`
* `GET /api/ungar/high-card-duel/export.jsonl`

  * query: `limit`, `trace_ids` optional
  * response: `application/x-ndjson`

**Exit criteria:** default tests pass; endpoint returns 501 when ungar missing.

---

### Phase 4 — Frontend Panel + testids

Add a minimal panel with stable selectors:

* `data-testid="ungar-status"`
* `data-testid="ungar-generate-count"`
* `data-testid="ungar-generate-seed"`
* `data-testid="ungar-generate-btn"`
* `data-testid="ungar-results"`

**Exit criteria:** frontend unit tests updated/added; Playwright not impacted.

---

### Phase 5 — Tests & Optional CI Job

**Default backend tests (no UNGAR):**

* `test_ungar_status_unavailable`
* `test_ungar_generate_returns_501_without_extra`

**Optional tests (requires UNGAR):**

* mark as `@pytest.mark.ungar`
* skip if not installed
* `test_ungar_generate_creates_traces` (count=2)
* `test_ungar_export_jsonl` (basic shape)

**CI:**

* Add `.github/workflows/ungar-integration.yml`:

  * triggers: `workflow_dispatch` + nightly schedule (optional)
  * job installs backend with `[ungar]`
  * runs `pytest -m ungar`
  * non-blocking preferred (`continue-on-error: true`) to protect core CI

**Exit criteria:** core CI unaffected; optional job works when run.

---

### Phase 6 — Docs

* `docs/M07_UNGAR_INTEGRATION.md` includes:

  * how to install: `pip install -e backend[ungar]`
  * how to run: curl examples
  * JSONL format definition (fields + examples)
  * limitations (only High Card Duel v1)
* README section: “Optional: UNGAR trace generator”.

---

## Acceptance Criteria (Definition of Done)

* ✅ Existing CI remains green (core workflows unchanged in stability).
* ✅ Backend runs without UNGAR installed.
* ✅ `/api/ungar/status` works in both cases (installed/uninstalled).
* ✅ Generator endpoint produces persisted traces when UNGAR installed; otherwise returns 501.
* ✅ JSONL export returns NDJSON with required fields (`id`, `prompts`, `trace_steps`, `final_answer`, `metadata`).
* ✅ Minimal frontend panel present with `data-testid`s.
* ✅ Default test suite passes; optional UNGAR integration tests can run in optional CI workflow.
* ✅ Docs added.

---

## Notes for Tunix compatibility

Tunix examples commonly build training prompts from dataset fields (e.g., converting `question/answer` into `prompts`). Our export includes `prompts` explicitly so it can be consumed in a similar preprocessing path. ([tunix.readthedocs.io][2])

---

If you want, after M07 lands we can plan **M08** as “UNGAR multi-game expansion + richer trace-step schemas + evaluator hooks,” but M07 should stay intentionally small: **one game, one generator, one export path, no training loop changes yet.**

[1]: https://pip.pypa.io/en/stable/topics/vcs-support/?highlight=git%2B&utm_source=chatgpt.com "VCS Support - pip documentation v25.2"
[2]: https://tunix.readthedocs.io/en/latest/_collections/examples/grpo_gemma.html "GRPO Demo — Tunix  documentation"
