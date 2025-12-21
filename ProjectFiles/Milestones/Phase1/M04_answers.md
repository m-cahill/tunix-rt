Here are **clear decisions** for each M04 question, with the **key decision** called out, so Cursor can implement without guessing. 

---

## 1) Database setup in CI

### Q1.1 Postgres service container?

✅ **Yes — use a GitHub Actions Postgres service container with a healthcheck.**
This is the most deterministic pattern and is officially documented by GitHub. ([GitHub Docs][1])

### Q1.2 SQLite instead?

❌ **No for M4.**
M4’s goal is to harden E2E for the “real” stack; SQLite would mask Postgres/asyncpg/migration issues and is likely to reintroduce pain later.

**Decision:** Postgres service container in CI.

---

## 2) Migration strategy for E2E

### Q2.1 Separate CI step?

✅ **Yes — run migrations as a separate CI step before Playwright starts.**
This yields better errors (DB readiness vs migration failure vs app failure) and keeps webServer commands clean.

### Q2.2 Include migrations in backend webServer?

❌ **No** (unless we’re forced to).
Keep separation.

**Decision:** Separate CI step: `alembic upgrade head` (after Postgres is healthy).

---

## 3) IPv6 vs IPv4 (localhost vs 127.0.0.1)

### Q3.1 Standardize on IPv4 everywhere?

✅ **Yes — standardize on `127.0.0.1` everywhere.**
This directly targets the observed `ECONNREFUSED ::1:8000` class of failures. Also, loopback semantics are well understood (`127.0.0.1` = local host loopback). ([Stack Overflow][2])

### Q3.2 Frontend API client?

You already use **relative URLs**, which is correct. Keep that.
But you **must** align the dev proxy / server bind addresses to `127.0.0.1` (see Q11).

**Decision:** `127.0.0.1` for Playwright, Vite host, Uvicorn host, and proxy target.

---

## 4) Frontend server for E2E: dev vs preview

### Q4.1 Use preview in CI?

Normally, **yes** (production-like). And importantly: **Vite preview *can* proxy** — it supports `preview.proxy` (defaulting to `server.proxy`) per official docs. ([vitejs][3])

### Q4.2 Keep dev server?

✅ **For M4, pick Vite dev mode** (bind to `127.0.0.1`).
Reason: M4’s success criterion is **deterministic green E2E** with minimal risk. You already have a working dev proxy setup; preview introduces one more moving part, and your team observed preview/proxy mismatch in practice.

**Key decision for M4:** **Option A (Dev mode) for E2E**
**Follow-up (M5+ optional):** revisit preview mode once infra is stable; if needed, explicitly set `preview.proxy` or add `VITE_API_BASE_URL`. ([vitejs][3])

---

## 5) Port configuration

### Q5.1 Keep current ports?

✅ **Keep them.**

* Frontend dev: **5173**
* Frontend preview (future): **4173**
* Backend: **8000**

### Q5.2 Env vars?

✅ **Yes — add optional env vars** (`FRONTEND_PORT`, `BACKEND_PORT`) in Playwright config for flexibility, but keep defaults.

---

## 6) Smoke check script

### Q6.1 Script vs inline?

✅ **Inline in CI** (simpler than adding a new script file).

### Q6.2 What should it check?

* `curl http://127.0.0.1:8000/api/health` (or whichever health endpoint exists)
* `curl http://127.0.0.1:5173/` (expect 200)

### Q6.3 Redundant with Playwright webServer waiting?

Playwright `webServer.url` waiting is good, and Playwright explicitly supports running multiple servers and waiting on URLs. ([Playwright][4])
But the **inline curl is still worth it** because when it fails, the logs are **much more actionable** than a generic Playwright timeout.

**Decision:** Inline curl smoke checks for better diagnostics.

---

## 7) Local E2E target

### Q7.1 Full lifecycle?

✅ Yes:

* start postgres (compose)
* run migrations
* run Playwright (which starts backend/frontend via webServer)

Leave Postgres running by default (best for iteration).

### Q7.2 Separate targets?

✅ Add two targets:

* `make e2e` → setup + run tests (leave DB running)
* `make e2e-down` → teardown

(Keep it minimal; no need for three targets unless you feel strongly.)

---

## 8) Retries and stability

### Q8.1 Retries?

✅ Set CI retries to **1** (not 2).
That aligns with “don’t rely on retries,” but still acknowledges that E2E can have rare timing blips.

### Q8.2 How many reruns?

✅ **3 consecutive successful runs** is the right bar for “stable” in M4.

---

## 9) DATABASE_URL for E2E

### Q9.1 Set explicitly?

✅ **Yes — set `DATABASE_URL` explicitly in CI** for clarity and to avoid “implicit default drift.”

### Q9.2 Rely on defaults?

❌ Not in CI. Defaults are fine locally, but CI should be explicit.

**Decision:** Explicit `DATABASE_URL` in CI (migrations step + backend webServer env).

---

## 10) CORS configuration

### Q10.1 Add all 4 origins?

✅ **Yes — simplest and correct for dev/testing.**
Add:

* `http://localhost:5173`
* `http://127.0.0.1:5173`
* (Optionally if you ever use preview later)
* `http://localhost:4173`
* `http://127.0.0.1:4173`

### Q10.2 Environment-based?

Not needed in M4.

**Decision:** hardcode the four origins for M4.

---

## 11) Frontend API client URL / proxy

### Q11.1 Update Vite proxy to 127.0.0.1?

✅ **Yes.** Change proxy target to `http://127.0.0.1:8000`.

### Q11.2 Preview proxy support?

Official Vite docs indicate preview supports proxy configuration (`preview.proxy`, defaulting to `server.proxy`). ([vitejs][3])
But for M4 we’re choosing dev mode anyway (see Q4/Q12).

### Q11.3 Env-configurable proxy?

Optional. If you add it, keep default to `127.0.0.1:8000`. Otherwise skip in M4.

---

## 12) Vite preview mode limitation (critical)

Your doc claims preview doesn’t support proxy, but current Vite docs say it does via `preview.proxy` / `server.proxy`. ([vitejs][3])
Regardless, **M4 should not hinge on this**.

✅ **Decision for M4:** **Use Vite dev mode** for E2E (Option A).
Then later (M5+), if you want production-like, either:

* validate preview proxy with your pinned Vite version, or
* implement `VITE_API_BASE_URL` (Option B) in a controlled milestone.

---

## 13) Stability metrics

### Q13.1 What counts as stable?

✅ **3 consecutive successful CI runs** with **retries=1** is stable.

### Q13.2 Add flakiness metrics/logging?

❌ Not in M4.
M4 should only add minimal diagnostics (curl checks + better logs) and eliminate the root cause.

---

# Summary decisions (what Cursor should implement)

* **Postgres service container** in E2E CI job + healthcheck. ([GitHub Docs][1])
* **Migrations as separate CI step** before Playwright.
* **Standardize on `127.0.0.1`** for baseURL, proxy target, and uvicorn host. ([uvicorn.dev][5])
* **E2E uses Vite dev mode** in M4 (production-like preview is deferred).
* **Add CORS origins** for both localhost and 127.0.0.1 on ports 5173/4173.
* **Inline curl smoke checks** + improved diagnostics.
* **Local `make e2e` + `make e2e-down`**.
* **Retries=1** and **3 consecutive successful CI runs** as stability bar.
* **Explicit DATABASE_URL** in CI.

If you want, I can now translate these decisions into a **single M4 Cursor prompt** (like prior milestones) with phased steps + exact file touch list.

[1]: https://docs.github.com/actions/guides/creating-postgresql-service-containers?utm_source=chatgpt.com "Creating PostgreSQL service containers"
[2]: https://stackoverflow.com/questions/20778771/what-is-the-difference-between-0-0-0-0-127-0-0-1-and-localhost?utm_source=chatgpt.com "What is the difference between 0.0.0.0, 127.0.0.1 and ..."
[3]: https://vite.dev/config/preview-options?utm_source=chatgpt.com "Preview Options"
[4]: https://playwright.dev/docs/test-webserver?utm_source=chatgpt.com "Web server"
[5]: https://uvicorn.dev/settings/?utm_source=chatgpt.com "Settings"
