# M41 Clarifying Questions

I've analyzed the project and have the following questions before finalizing the task plan:

---

## 1. Frontend `act()` Warnings — Current State

The M40 audit mentions "75 tests pass but emit React `act()` warnings." I see existing mitigation attempts in the test files:
- `App.test.tsx` has `flushPendingUpdates()` and uses `waitFor()`/`userEvent.setup()`
- `Leaderboard.test.tsx` wraps some operations in `act(async () => ...)`
- A TODO comment on line 78-81 of `App.test.tsx` notes the challenge

**Question:** Should I run the tests first to capture the actual warning output, or do you have a log/screenshot of the current warnings you'd like me to target?

---

## 2. GPU Documentation — Already Exists?

I see `CONTRIBUTING.md` already has a **GPU Development (RTX 5090 / Blackwell)** section (lines 105-149) covering:
- `.venv-gpu` setup
- PyTorch nightly cu128 install command
- GPU verification
- Training command examples
- Version notes (torch 2.11.0.dev+cu128, CUDA 12.8+ driver)

**Question:** Is this existing documentation sufficient, or does M41-D1 require expansion/relocation to a separate `docs/GPU_SETUP.md` file?

---

## 3. Demo Instructions — Audience & Depth

M41-D2 asks for "How to Run a Demo" section explaining backend/frontend startup, local training, and evidence location.

**Question:** 
- Is this for **judges running the project locally** (requires setup instructions) or for **video recording reference** (just a checklist)?
- Should this go in `CONTRIBUTING.md`, `README.md`, or a new `docs/DEMO.md` file?

---

## 4. Video Checklist Location

The plan specifies `docs/submission/VIDEO_CHECKLIST.md`, but the `docs/submission/` directory doesn't exist.

**Question:** Should I create the new `docs/submission/` subdirectory, or place the file elsewhere (e.g., `docs/VIDEO_CHECKLIST.md`)?

---

## 5. Frontend UX Polish — Specific Targets?

Phase 2 (M41-F2) allows "small CSS tweaks, minor component refactors, removing unused UI elements."

**Question:** Do you have specific UI areas you'd like improved, or should I:
- Review the frontend for obvious cleanup opportunities?
- Focus mainly on loading states, button labels, and visual consistency?
- Take screenshots for the evidence folder without major changes?

---

## 6. Priority Order

Given time constraints, if we can't complete everything, which phases are most critical?

**Suggested priority:**
1. Phase 1 (act() warnings) — highest impact on test hygiene
2. Phase 3 (documentation) — judge-facing value
3. Phase 4 (video checklist) — submission prep
4. Phase 2 (UX polish) — nice-to-have

**Question:** Does this priority order align with your goals, or should I reorder?

---

## 7. CI Verification

The plan requires "CI passes unchanged." 

**Question:** After making changes, should I:
- Run `npm test` locally and capture the clean output?
- Include the test output in `submission_runs/m41_v1/frontend_tests_clean.txt`?
- Verify backend tests still pass (quick `pytest` run)?

---

**Ready to proceed once you confirm these points!**
