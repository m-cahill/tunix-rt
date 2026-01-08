Great questions — this is exactly the right level of care for M41. I’ll answer **point-by-point**, with **explicit direction** so Cursor can proceed without ambiguity.

---

## 1. Frontend `act()` Warnings — What to Target

**Answer:**
✅ **Run the tests first and capture the actual warnings.**

Reasoning:

* The audit flagged warnings generically, but **the source may have shifted** due to partial mitigations already in place.
* We should **fix what actually emits warnings today**, not what was previously suspected.

**Instruction to follow:**

* Run `npm test` (or `npm run test -- --runInBand` if needed)
* Capture **exact warning output**
* Fix only the warnings that appear
* Do **not** refactor preemptively

👉 This ensures we don’t churn tests unnecessarily.

---

## 2. GPU Documentation — Is It Already “Done”?

You’re correct: the existing **GPU Development (RTX 5090 / Blackwell)** section in `CONTRIBUTING.md` is **already sufficient**.

**Answer:**
✅ **No new GPU doc file is required.**
❌ Do **not** move it to `docs/GPU_SETUP.md`.

**What *is* allowed (optional, low-risk):**

* Light wording polish
* Ensure commands are copy-paste runnable
* Add a short “When to use `.venv` vs `.venv-gpu`” note if missing

But this is **not a blocker** for M41.

---

## 3. Demo Instructions — Audience & Location

**Audience:**
🎯 **Judges who want to understand what’s happening**, not necessarily run everything locally.

Think:

* “How this was demonstrated”
* “What to run if you *do* want to try”

**Answer (very specific):**

* Create a **new file**:

  ```
  docs/DEMO.md
  ```
* This file should:

  * Be readable without running anything
  * Include minimal runnable commands
  * Emphasize *flow*, not setup pain

**Do NOT:**

* Overload `README.md`
* Duplicate `CONTRIBUTING.md`

---

## 4. Video Checklist Location

Correct: `docs/submission/` does not exist.

**Answer:**
✅ **Create the directory**:

```
docs/submission/
```

And place:

```
docs/submission/VIDEO_CHECKLIST.md
```

This is intentional:

* Keeps submission artifacts grouped
* Avoids cluttering general docs
* Scales cleanly if we add `FINAL_NOTES.md` later

---

## 5. Frontend UX Polish — How Opinionated to Be?

**Answer:**
🎯 **Be conservative and observational, not creative.**

Explicit guidance:

* Do **not** redesign
* Do **not** invent UX
* Do **not** add features

**Focus only on:**

* Button labels that are unclear
* Loading states that feel abrupt
* Obvious spacing or alignment issues
* Removing unused or confusing UI elements *if clearly safe*

If nothing stands out:

* Take screenshots
* Make **zero UX changes**
* That is acceptable

UX polish is **nice-to-have**, not required for M41 success.

---

## 6. Priority Order — Confirmed

Your suggested priority is **exactly correct**.

✅ **Confirmed priority order:**

1. **Phase 1 — `act()` warnings** (must fix if present)
2. **Phase 3 — Documentation** (demo + clarity)
3. **Phase 4 — Video checklist**
4. **Phase 2 — UX polish** (optional)

If time runs out:

* Skip UX polish first
* Do **not** skip test hygiene or docs

---

## 7. CI Verification — What Evidence Is Required?

**Yes to all, with nuance.**

**Required actions:**

* ✅ Run `npm test` locally
* ✅ Capture clean output (no warnings)
* ✅ Save to:

  ```
  submission_runs/m41_v1/frontend_tests_clean.txt
  ```

**Backend tests:**

* ❌ Full `pytest` run is **not required**
* ✅ Spot-check only if frontend changes touch shared files (unlikely)

CI will be the final arbiter anyway.

---

## ✅ Final Green Light for Cursor

Cursor may proceed with M41 using the following **authoritative decisions**:

* Fix only **currently observed** `act()` warnings
* Do **not** add new GPU docs
* Add `docs/DEMO.md`
* Create `docs/submission/VIDEO_CHECKLIST.md`
* UX polish is optional and conservative
* Evidence folder required

If you want, next I can:

* Pre-review `App.test.tsx` and flag likely warning sources
* Draft `DEMO.md` or `VIDEO_CHECKLIST.md` for you
* Help you decide when M41 is “good enough” to stop

Just say the word.
