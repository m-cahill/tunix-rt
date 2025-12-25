Below is a Cursor handoff prompt for **M26** (next milestone). It’s based on the M25 audit’s “Next Milestone Plan” plus the full codebase audit’s top opportunities (performance benchmarking, training readiness polish, DX/docs).   

---

## Cursor Handoff Prompt — M26: Training Readiness, GPU Path, Benchmarking, and Metrics

### Context

* **CI is green** and **M25 is closed**.
* M25 established stable CI, optional-dep handling, scoped coverage gates, and migration portability. 
* The audits recommend M26 focus on: **GPU device selection**, **throughput benchmarking**, **dataset scale-up**, **training metrics dashboard**, **checkpoint resumption**.  

### Goal (M26 North Star)

Make training “real and repeatable” on local GPU (RTX 4080 mobile), with:

1. explicit device selection + provenance,
2. checkpoint/resume,
3. benchmark capture (throughput + basic profiling hooks),
4. dataset scale-up beyond the tiny golden set,
5. minimal UI visibility into training metrics.

---

# Phase 0 — Baseline Gate (must stay green)

**Acceptance**

* `cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy tunix_rt_backend && uv run pytest -q`
* `cd frontend && npm test && npm run build`
* `cd e2e && npm test` (Playwright)

**Guardrail**

* Any new training functionality must be behind an **explicit CLI flag** or “optional deps installed” check; default CI must not require torch/transformers/jax.

---

# Phase 1 — GPU Device Selection + Provenance (JAX-first)

### Why

We need deterministic, user-controlled device selection and auditability. JAX supports controlling default device placement (e.g., `jax.default_device`). ([JAX Documentation][1])

### Tasks

1. **Add device selector to `train_jax.py`**

   * CLI args:

     * `--device` in `{cpu,gpu}` (default: `gpu` if available else `cpu`)
     * `--device_index` (default 0)
   * Use:

     * `jax.devices("gpu")[idx]` / `jax.devices("cpu")[0]`
     * Wrap training step(s) with `jax.default_device(device)` when appropriate. ([JAX Documentation][1])
   * If `--device=gpu` requested but no GPU present: fail with clear message.

2. **Provenance**

   * Record into your existing run metadata/provenance:

     * selected device kind + index
     * `jax.devices()` summary (names / platform)
     * CUDA visibility (if set)
   * Store in DB as part of the training run record (or as attached artifact JSON next to checkpoints).

3. **“Tiny real JAX run” smoke mode**

   * Add `--smoke_steps N` (e.g., 5–20) that runs a minimal loop and writes one metrics entry + checkpoint.

**Acceptance**

* Local:

  * `uv run python training/train_jax.py --smoke_steps 10 --device gpu --device_index 0 ...` runs to completion on GPU when available.
* CI:

  * Existing training smoke tests remain **skipped** when JAX not installed; no new failures.

---

# Phase 2 — Checkpointing + Resume (Orbax)

### Why

Reliable resume is a prerequisite for long runs and for later tuning. Flax recommends **Orbax** for checkpointing and supports async saving/versioning. ([flax.readthedocs.io][2])

### Tasks

1. Implement checkpoint save/restore using **Orbax CheckpointManager** for the JAX TrainState. ([flax.readthedocs.io][2])
2. CLI:

   * `--checkpoint_dir` (default: `artifacts/checkpoints/<run_id>/`)
   * `--save_every_steps`
   * `--resume_from` (path or “latest”)
3. Persist minimal metadata alongside checkpoints:

   * step, loss, learning rate, device info, dataset key/version

**Tests**

* Add a **pure-CPU** unit/integration test that:

  * creates a tiny TrainState
  * saves a checkpoint
  * restores it
  * asserts parameters match (or step increments correctly)
* Mark it `@pytest.mark.slow` only if needed; keep runtime minimal.

**Acceptance**

* Can interrupt and resume a run without changing final metrics trajectory (within tolerance).
* Checkpoint directory layout is stable and documented.

---

# Phase 3 — Throughput Benchmark + Optional Profiling Hook

### Why

The audit calls out lack of automated perf regression tests; we should at least capture a baseline now. 
JAX provides profiling trace capture via `jax.profiler.trace` and Perfetto/TensorBoard workflows. ([JAX Documentation][3])

### Tasks

1. Add `training/bench_jax.py` (or `--benchmark` mode in `train_jax.py`) that logs:

   * steps/sec, examples/sec, tokens/sec (if applicable)
   * compile time vs steady-state time
   * device info
   * batch size, seq len, precision
2. Add optional profiling flag:

   * `--profile_dir artifacts/profiles/<run_id>/`
   * Wrap a short window with `jax.profiler.trace(log_dir=...)` ([JAX Documentation][4])
3. Store benchmark results as an artifact JSON (and optionally DB row if you already have a runs table suitable for it).

**CI Guardrail**

* Do **not** run benchmarks in default CI.
* Add a `workflow_dispatch` / manual trigger job for benchmarks later (or just document local usage).

**Acceptance**

* One command produces a benchmark JSON + (optional) profile trace directory.

---

# Phase 4 — Dataset Scale-Up Beyond Golden-v1

### Why

Training signal and benchmark realism require a larger “canonical” dataset (even if still synthetic/curated).

### Tasks

1. Create a new dataset key, e.g. `golden-v2` with **≥100 traces**
2. Add a deterministic “seed” pathway so CI/dev can reproduce:

   * `uv run python tools/seed_dataset.py --dataset golden-v2 --count 100`
3. Ensure dataset manifests and DB seeding stay consistent (avoid the earlier “manifest exists but DB empty” trap).

**Acceptance**

* `POST /api/datasets/build` can build `golden-v2` reliably in a clean DB.
* E2E remains stable and does not depend on “magic” local-only files.

---

# Phase 5 — Minimal Training Metrics Visibility (UI + API)

### Goal

A minimal UI that proves training is happening and measurable (loss curve + throughput).

### Tasks

1. Backend:

   * Add an endpoint like `GET /api/training/runs/{id}/metrics` (or reuse existing artifacts API) returning time-series points.
2. Frontend:

   * Add a simple “Training” view:

     * run metadata (device, dataset, steps)
     * line chart: loss over steps
     * table: latest throughput/step time
3. Keep the UI minimal; no styling bikeshedding.

**Acceptance**

* After a smoke run, UI can display the run and its metrics.

---

# Phase 6 — Quick Wins From Audit (DX + Supply Chain)

(These are fast, high-ROI audit score boosters.)

### A) Add `CONTRIBUTING.md`

Include:

* `uv sync` workflow
* ruff ordering rule (`ruff check --fix` then `ruff format`)
* optional deps extras (`backend[training]`, etc.)
* local DB + migrations + E2E notes

### B) SBOM Generation (optional but strong)

CycloneDX Python supports SBOM from the installed environment via `cyclonedx-py environment`. ([CycloneDX BOM Tool][5])
Add a CI artifact step (non-blocking at first) or a `make sbom` doc command.

---

## Definition of Done (M26)

* ✅ JAX training supports **explicit device selection** and records device provenance. ([JAX Documentation][1])
* ✅ JAX training supports **checkpoint save + resume** using Orbax. ([flax.readthedocs.io][2])
* ✅ A benchmark command outputs throughput metrics; optional trace capture works. ([JAX Documentation][4])
* ✅ `golden-v2` dataset (≥100 traces) build/seed path is deterministic.
* ✅ Minimal training metrics can be viewed via API + simple UI.
* ✅ CI stays green; no new default dependency requirements.

---

## Suggested Branch / PR Strategy

* One PR per phase (1–2 max per day), each with a green CI merge.
* Keep Phase 1–2 separate (device selection vs checkpointing) to reduce risk.

---

If you want M26 to be even tighter: drop Phase 5 (UI) and do it as M27, but keep **device selection + resume + benchmark + dataset scale-up** together—those are the real “training readiness” unlocks.

[1]: https://docs.jax.dev/en/latest/_autosummary/jax.default_device.html?utm_source=chatgpt.com "jax.default_device"
[2]: https://flax.readthedocs.io/en/v0.6.11/guides/use_checkpointing.html?utm_source=chatgpt.com "Save and load checkpoints - Flax - Read the Docs"
[3]: https://docs.jax.dev/en/latest/profiling.html?utm_source=chatgpt.com "Profiling computation"
[4]: https://docs.jax.dev/en/latest/_autosummary/jax.profiler.trace.html?utm_source=chatgpt.com "jax.profiler.trace"
[5]: https://cyclonedx-bom-tool.readthedocs.io/en/latest/usage.html?utm_source=chatgpt.com "Usage — CycloneDX Python 7.2.1 documentation"
