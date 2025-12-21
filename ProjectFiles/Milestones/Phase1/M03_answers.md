Here are the decisions for each M03 clarifying question, aligned with the “small + tested + no churn” intent. 

---

## 1) Alembic auto-generated IDs policy

**Go with Option A (document the policy).**
Also include a **tiny one-line snippet** (basically Option B-lite) in the doc, but **no enforcement tooling** in M3.

* Policy: “Don’t manually set revision IDs; use Alembic-generated IDs going forward.”
* Snippet: `alembic revision -m "add created_at index"`
  Alembic supports overriding via `--rev-id`, but we’re explicitly choosing not to. ([alembic.sqlalchemy.org][1])

✅ **Decision:** A (+ 1 command line as an example)

---

## 2) created_at index migration approach

**Use a manual migration (not autogenerate).**
Create a new revision with an auto-generated ID, and in the migration file call `op.create_index(...)` / `op.drop_index(...)`. Alembic documents `op.create_index` explicitly. ([alembic.sqlalchemy.org][2])

Reason: index autogenerate behavior can be inconsistent / noisy across environments; a manual migration is deterministic and reviewable (and this change is trivially expressed).

✅ **Decision:** `alembic revision -m "add traces created_at index"` (manual), then use `op.create_index` in upgrade.

---

## 3) Frontend coverage not generating

**Option C: fix what’s broken and document.** Your instinct is right.

The most common root causes in Vitest are:

* missing the provider package (often `@vitest/coverage-v8`), or
* provider/version mismatch with `vitest`. ([Stack Overflow][3])

Also note: Vitest’s default coverage output directory is `./coverage` unless `coverage.reportsDirectory` overrides it. ([vitest.dev][4])

✅ **Decision:** Diagnose + fix. The first things to check:

1. `package.json` includes `@vitest/coverage-v8` and version matches `vitest`. ([Stack Overflow][3])
2. Vitest config is under the **`test:` block** (not just Vite config), and CI uploads the correct `reportsDirectory`. ([vitest.dev][4])

---

## 4) Frontend trace UI test scope

Your plan is correct.

* **Yes**: mock `fetch` for `/api/traces` endpoints (like existing tests).
* **No**: don’t spend time on button enabled/disabled states unless it’s already implemented and stable.
* **Optional (nice if cheap)**: add **one** error rendering test only if the UI already has a clear error state (it’s a good guardrail, but not required for M3).

✅ **Decision:** 3 tests = Load Example, Upload success, Fetch success (mocked, deterministic)

---

## 5) README curl examples location

✅ **Option A**: inline examples under the existing “API Endpoints” section.
That’s the fastest path to “copy/paste usable.”

---

## 6) DB troubleshooting section location

✅ **Option B**: extend the existing README troubleshooting section with a “DB Troubleshooting” subsection.

---

## 7) Coverage thresholds

✅ **Confirm: do not change thresholds.**
Keep exactly as-is:

* Backend: 80% line / 68% branch gate
* Frontend: 60% line / 50% branch

If new tests shift % slightly, we fix by adding/adjusting tests—not by relaxing gates. (Your new trace UI tests should *increase* frontend coverage anyway.)

---

## 8) Testing strategy for the new migration

✅ **Yes: do all 3 locally before committing** (this is the right “enterprise-grade but practical” bar):

1. `alembic upgrade head` on SQLite (CI parity)
2. Run upgrade on local Postgres via compose
3. Verify the index exists via SQL

The `op.create_index` approach makes #3 unambiguous. ([alembic.sqlalchemy.org][2])

---

## One extra tiny guardrail I recommend (still M3-friendly)

When you add the created_at index migration, use a **stable explicit index name** (e.g. `ix_traces_created_at`) rather than relying on autogen naming. This prevents cross-backend naming surprises.

---

You’re cleared to proceed exactly as you outlined. If you want to save Cursor time, have it implement in this order:

1. Frontend coverage fix (so CI artifacts become reliable early)
2. Trace UI tests
3. DB pool wiring
4. created_at index migration + verification
5. docs updates (policy + curl + troubleshooting)

[1]: https://alembic.sqlalchemy.org/en/latest/api/commands.html?utm_source=chatgpt.com "Commands — Alembic 1.17.2 documentation"
[2]: https://alembic.sqlalchemy.org/en/latest/ops.html?utm_source=chatgpt.com "Operation Reference — Alembic 1.17.2 documentation"
[3]: https://stackoverflow.com/questions/77971541/coverage-command-with-vitest-wont-return-the-coverage-report-why?utm_source=chatgpt.com "Coverage command with Vitest won't return the coverage report. Why?"
[4]: https://vitest.dev/config/coverage?utm_source=chatgpt.com "coverage | Config - Vitest"
