# M21 Questions

1. **Frontend Audit (`npm audit` failure):**
   - `npm audit` currently fails with `ENOLOCK` (missing or invalid lockfile), even though `package-lock.json` exists.
   - **Proposed Fix:** Should I regenerate `package-lock.json` (`npm i --package-lock-only`) before running the audit fix?

2. **Phase 4.2 (CI Artifacts):**
   - The `ci.yml` already has `if: always()` for uploading `playwright-report`.
   - **Question:** Do you want to change the Playwright config `trace` setting from `'on-first-retry'` to `'retain-on-failure'` (or `'on'`) to ensure traces are available for the *first* failure in CI?

3. **Phase 5.1 (Promote Best Trial UI):**
   - I see `Tuning.tsx` lists jobs and their trials.
   - **Question:** Where should the "Promote Best Trial" button go?
     - Option A: A "Promote Best" button in the **Job List** (actions column).
     - Option B: A "Promote" button next to each **Trial** in the Job Details view.
     - Option C: Both?
