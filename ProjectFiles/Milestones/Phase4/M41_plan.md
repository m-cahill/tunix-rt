Excellent — M40 is exactly where you want it to be. This is a **clean inflection point** in the project.

You’ve:

* De-risked infra (TPU → local GPU)
* Proven PyTorch + Blackwell works
* Preserved JAX without entanglement
* Kept CI green and evidence-driven

Now M41 should **shift modes**: from infrastructure correctness → **presentation quality and narrative clarity**.

Below is a **Cursor-ready M41 prompt** that is intentionally **bounded, polish-focused, and submission-aware**, without reopening architectural risk.

---

# 📦 M41 — Frontend Polish, DX Cleanup, and Submission Readiness

**Milestone ID:** M41
**Objective:** Improve frontend quality, developer experience, and demo readiness without altering backend or training behavior.
**Scope:** Frontend, docs, minor DX utilities only
**Non-Goals:** No backend refactors, no training logic changes, no CI restructuring

---

## 🎯 M41 Goals (What “Done” Means)

By the end of M41:

1. **Frontend tests are warning-free** (no React `act()` warnings)
2. **Frontend UX is calm, intentional, and demo-ready**
3. **GPU usage and local training are clearly documented**
4. **The project tells a clear story for judges**
5. **Nothing breaks CI**

This is a **polish milestone**, not a feature milestone.

---

## 🔒 Guardrails (Very Important)

* ❌ Do NOT touch backend training code
* ❌ Do NOT change PyTorch/JAX behavior
* ❌ Do NOT relax coverage gates
* ❌ Do NOT add new infra dependencies
* ✅ Keep diffs small and reviewable
* ✅ Prefer documentation and UI clarity over cleverness

---

## 🧩 Phase 1 — Frontend Test Hygiene (Highest Priority)

### Task M41-F1: Eliminate React `act()` Warnings

**Problem:**
Frontend tests pass but emit warnings:

```
Warning: An update to App inside a test was not wrapped in act(...)
```

**Actions:**

* Audit failing test output
* Wrap async state updates with:

  * `await act(async () => …)`
  * OR replace with `await waitFor(...)` where appropriate
* Prefer `userEvent.setup()` over `fireEvent` if applicable

**Acceptance Criteria:**

* `npm test` produces **zero React warnings**
* Test count remains unchanged
* Coverage remains ≥ current levels

---

## 🧩 Phase 2 — Frontend UX Polish (Demo-Oriented)

### Task M41-F2: Visual Calm & Readability Pass

**Intent:** Make the UI feel intentional and legible for a short demo video.

**Allowed Changes:**

* Improve spacing, typography, or layout consistency
* Clarify labels (“Run”, “Trace”, “Result”, etc.)
* Improve loading states (spinners, disabled buttons)
* Ensure no console warnings in dev mode

**Explicitly Allowed:**

* Small CSS tweaks
* Minor component refactors for readability
* Removing unused UI elements

**Explicitly Forbidden:**

* Feature expansion
* State model changes
* API changes

**Acceptance Criteria:**

* App loads cleanly
* No console warnings
* UI communicates “reasoning trace → result” clearly

---

## 🧩 Phase 3 — DX & Documentation Polish

### Task M41-D1: GPU Setup Documentation

Add a **GPU Development** section to either:

* `CONTRIBUTING.md` (preferred), or
* `docs/GPU_SETUP.md` (acceptable)

**Must Include:**

```markdown
## GPU Development (RTX 5090 / Blackwell)

- Python version
- CUDA driver requirement
- `.venv-gpu` setup
- PyTorch nightly install command
- How to verify GPU usage
```

**Acceptance Criteria:**

* A new contributor can follow it without guesswork
* Commands are copy-paste runnable

---

### Task M41-D2: “How to Run a Demo” Section

Add a **Demo / Submission** section explaining:

* How to start backend
* How to start frontend
* How to run a local training example
* Where to find evidence (`submission_runs/`)

This is **judge-facing documentation**.

---

## 🧩 Phase 4 — Submission & Narrative Readiness

### Task M41-S1: Video Checklist (No Video Yet)

Create a short markdown checklist in:

```
docs/submission/VIDEO_CHECKLIST.md
```

**Must Answer:**

* What problem is being solved?
* What is novel?
* What does the UI show?
* Where is the reasoning trace visible?
* How is reproducibility demonstrated?

⚠️ **Do not record video yet** — just define what it will show.

---

## 📁 Evidence for M41

Create:

```
submission_runs/m41_v1/
```

Include:

* `frontend_tests_clean.txt` (test run with no warnings)
* `screenshots/` (2–3 UI screenshots)
* `README.md` summarizing what changed

---

## ✅ Acceptance Checklist (Cursor Must Verify)

* [ ] Frontend tests pass **with zero warnings**
* [ ] CI passes unchanged
* [ ] GPU setup documented
* [ ] Demo instructions clear
* [ ] No backend files modified
* [ ] Evidence folder populated

---

## 🧠 Strategic Context (Why This Is the Right M41)

M40 proved **technical credibility**.
M41 establishes **trust and clarity**.

Judges do not reward:

* TPU suffering
* Exotic infra
* Over-engineering

They reward:

* Clear problem framing
* Reproducibility
* Thoughtful UX
* Clean reasoning traces

This milestone makes the project **legible to humans**.

---

## 🚀 After M41 (Preview Only)

**M42 (Likely Final):**

* Record demo video
* Final submission narrative
* Light README polish
* Lock version pins

But do **not** start that yet.

---

If you want, next I can:

* Pre-write `VIDEO_CHECKLIST.md`
* Review frontend components and suggest exact polish targets
* Help you decide when the project is “done enough”

Just say the word.
