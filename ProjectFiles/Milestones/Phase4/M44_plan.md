Below is a **Cursor-ready, copy/paste handoff prompt** for **M44**.
It is written as an **execution contract**, not a brainstorming doc, and reflects everything we just learned (Windows + PyTorch + 5090 reality).

---

```md
# Cursor Handoff Prompt — M44: Full Production Training (PyTorch / RTX 5090)

## Context (Locked State)

- **Project:** Tunix RT (Submission Track)
- **Submission Status:** ✅ M42 COMPLETE, CI GREEN, submission ZIP frozen
- **Previous Milestone:** M43 CLOSED (Exploratory)
  - Reason: JAX/Flax CUDA unsupported on Windows (JAX ≥0.5)
  - No code or dependency changes made
  - Outcome documented; submission unaffected

M44 is a **new milestone**, not a retry.

---

## M44 Purpose (Very Important)

Run a **single, authoritative full training run** using:
- **PyTorch**
- **Gemma 2B**
- **RTX 5090 (Windows native)**

This milestone exists to:
- Close the emotional / technical loop (“it really ran”)
- Capture high-quality training + evaluation evidence
- Produce **optional post-submission artifacts**

⚠️ M44 **must not** modify submission artifacts or CI state.

---

## Non-Negotiable Guardrails

These are hard rules:

1. ❌ No code changes
2. ❌ No dependency changes
3. ❌ No CI changes
4. ❌ No edits to M42 artifacts or ZIP
5. ❌ No retries, tuning, or parameter sweeps
6. ✅ All outputs go ONLY to `submission_runs/m44_v1/`

If anything fails:
- Log it
- Stop
- Do not fix

---

## Environment Assumptions

- OS: **Windows**
- GPU: **RTX 5090 (sm_120)**
- Backend: **PyTorch**
- Virtual env: existing `.venv-gpu`
- PyTorch: nightly cu128 (already validated in M40)
- HuggingFace auth: assumed present; if missing → log + stop

---

## Training Configuration

### Model
- **Gemma 2B (PyTorch weights)**

### Dataset
- Training: `dev-reasoning-v2` (≈550 samples)
- Evaluation: `eval_v2.jsonl` (100 samples)

### Run Shape
- **Single run only**
- Target duration: **300–500 steps** (or 1 epoch if epoch-based)
- No early stopping unless failure occurs

This is not a benchmark race.  
Completion + evidence > optimal convergence.

---

## M44 Execution Phases

### Phase M44.0 — Pre-Flight Snapshot

Create:
```

submission_runs/m44_v1/

```

Capture:
- Commit SHA
- `python --version`
- `torch.__version__`
- `torch.version.cuda`
- `nvidia-smi`

Artifacts:
- `env_snapshot.txt`
- `gpu_snapshot.txt`

---

### Phase M44.1 — Full Training Run (Main Event)

Run the **best known stable PyTorch Gemma training config**.

Requirements:
- GPU must be active
- No CPU fallback
- No retries

Artifacts:
- `training_log.txt`
- `training_metrics.json`
- Loss / throughput stats
- Checkpoints (if produced)

If training fails:
- Stop immediately
- Create `failure_analysis.md`

---

### Phase M44.2 — Evaluation Pass

If training completes:
- Run eval on `eval_v2.jsonl`
- Generate metrics summary

If training fails:
- Skip eval
- Document why

Artifacts:
- `evaluation_results.json` (if applicable)
- `evaluation_summary.md`

---

### Phase M44.3 — Evidence Consolidation

Ensure the following exist:

```

submission_runs/m44_v1/
├── env_snapshot.txt
├── gpu_snapshot.txt
├── training_log.txt
├── training_metrics.json
├── evaluation_results.json (optional)
├── evaluation_summary.md (optional)
├── failure_analysis.md (optional)
└── m44_summary.md

```

---

## Required Summary (m44_summary.md)

Must clearly state:
- What was attempted
- On what hardware
- What completed / failed
- Why M44 does **not** affect submission validity
- Explicit statement:

> “M44 is post-submission, evidence-only.  
> M42 remains the authoritative submission artifact.”

---

## Definition of Done (M44)

M44 is considered **complete** when:
- Training completes OR fails cleanly with documented cause
- All artifacts are captured under `submission_runs/m44_v1/`
- No repository state outside that folder is modified

No CI run is required.

---

## Scope Exclusions (Explicit)

- No JAX / Flax work
- No multi-run experiments
- No tuning
- No README edits
- No submission ZIP updates

---

## Mental Model (Read This)

This is a **GPU burn for confidence**, not a gate.
A failed run is still a successful milestone if documented.

Proceed calmly and deliberately.
```

---

If you want, next I can:

* Write the **exact PyTorch Gemma command** Cursor should run
* Draft a **one-page M43 closeout summary** (for narrative cleanliness)
* Or sanity-check the **PyTorch Gemma config** you plan to use

You handled this pivot perfectly. M44 is the victory lap.
