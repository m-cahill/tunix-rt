# M04 Clarifying Questions - E2E Infrastructure Hardening

**Context:** M04 focuses on making E2E deterministic and green by fixing the `ECONNREFUSED ::1:8000` error and ensuring all services start properly in CI.

---

## 1. Database Setup in CI

**Current State:**
- The E2E CI job does NOT start a postgres service
- `playwright.config.ts` starts backend via `webServer` but backend expects DB to be available
- The trace upload E2E test fails because backend can't connect to DB

**Questions:**
Q1.1: Should the E2E CI job start postgres as a **service container** in GitHub Actions?
```yaml
services:
  postgres:
    image: postgres:15-alpine
    env:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: postgres
    options: >-
      --health-cmd pg_isready
      --health-interval 10s
      --health-timeout 5s
      --health-retries 5
```

Q1.2: OR should we use SQLite for E2E tests to avoid the postgres dependency?
- Pro: Faster, simpler CI
- Con: Different DB than production (PostgreSQL)

**Recommended:** Service container (matches production better)

---

## 2. Migration Strategy for E2E

**Current State:**
- Backend starts in `webServer` but migrations haven't been run
- No DB exists when backend tries to query traces table

**Questions:**
Q2.1: Should migrations run as a **separate CI step** before Playwright runs?
```yaml
- name: Run migrations
  run: cd backend && alembic upgrade head
  env:
    DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost:5432/postgres
```

Q2.2: OR should the backend `webServer` command include migrations?
```typescript
command: 'cd ../backend && alembic upgrade head && uvicorn ...'
```

**Recommended:** Separate CI step (cleaner separation, better error messages)

---

## 3. IPv6 vs IPv4 (localhost vs 127.0.0.1)

**Current State:**
- Playwright config uses `baseURL: 'http://localhost:5173'`
- Backend webServer starts with `uvicorn ... --port 8000` (no explicit host)
- Error shows `ECONNREFUSED ::1:8000` (IPv6)

**Questions:**
Q3.1: Should we standardize on IPv4 (127.0.0.1) everywhere?
- Playwright baseURL: `http://127.0.0.1:5173`
- Backend webServer: `uvicorn ... --host 127.0.0.1 --port 8000`
- Frontend dev: `npm run dev -- --host 127.0.0.1`

Q3.2: Should the frontend API client also use 127.0.0.1 instead of localhost?
- Current: Likely uses `http://localhost:8000` (need to verify in `src/api/client.ts`)
- Change to: `http://127.0.0.1:8000` or use relative URLs

**Recommended:** Full IPv4 (127.0.0.1) standardization

---

## 4. Frontend Server: Dev vs Preview

**Current State:**
- Current webServer command: `cd ../frontend && npm run dev`
- This uses Vite dev server (hot reload, not production-like)

**Questions:**
Q4.1: Should E2E use **Vite preview** (production build)?
```typescript
command: 'cd ../frontend && npm run build && npm run preview -- --host 127.0.0.1 --port 4173'
```
- Pro: Tests production build
- Con: Slower startup, no source maps

Q4.2: OR keep **Vite dev** but bind to 127.0.0.1?
```typescript
command: 'cd ../frontend && npm run dev -- --host 127.0.0.1 --port 5173'
```
- Pro: Faster, source maps available
- Con: Doesn't test production build

**Recommended:** Vite preview for CI (production-like), Vite dev for local (faster iteration)

---

## 5. Port Configuration

**Current State:**
- Frontend: 5173 (Vite default dev), 4173 (Vite default preview)
- Backend: 8000

**Questions:**
Q5.1: Should we keep these ports or make them configurable?
- Current plan seems to use 5173 for dev, 4173 for preview
- Is this acceptable or do you want consistency?

Q5.2: Should the Playwright config ports be environment variables?
```typescript
const FRONTEND_PORT = process.env.FRONTEND_PORT || '4173';
const BACKEND_PORT = process.env.BACKEND_PORT || '8000';
```

**Recommended:** Keep current ports, add env vars for flexibility

---

## 6. Smoke Check Script

**M04 Plan mentions:**
> Add a CI step (or script) that curls backend and frontend health endpoints before Playwright runs

**Questions:**
Q6.1: Should this be a **bash script** (`scripts/e2e_sanity_check.sh`) or **inline in CI**?

Q6.2: What should it check?
- `curl http://127.0.0.1:8000/api/health` → expect `{"status": "healthy"}`
- `curl http://127.0.0.1:4173` → expect HTTP 200
- Print docker/service logs on failure?

Q6.3: Is this needed if Playwright's `webServer.url` already waits for readiness?
- Playwright waits for `url` to return 200 before running tests
- Smoke check would be redundant unless we want better diagnostics

**Recommended:** Skip dedicated smoke check script, rely on Playwright's built-in waiting + good logging

---

## 7. Local E2E Target

**M04 Plan mentions:**
> Add `make e2e` or `npm run e2e:local`

**Questions:**
Q7.1: Should this target:
- Start postgres (`docker compose up -d postgres`)
- Run migrations (`make db-upgrade`)
- Run Playwright tests (relies on `webServer` to start frontend/backend)
- Stop postgres (`docker compose down`) or leave running?

Q7.2: Should there be separate targets?
- `make e2e-setup` → Start postgres + run migrations
- `make e2e` → Run tests (assumes setup done)
- `make e2e-teardown` → Stop postgres

**Recommended:** Single `make e2e` that handles full lifecycle (setup → test → leave postgres running for iteration)

---

## 8. Retries and Stability

**Current State:**
- Playwright config has `retries: process.env.CI ? 2 : 0`

**Questions:**
Q8.1: Should we reduce retries to 0 or 1 after fixing the infrastructure?
- M04 plan says: "No reliance on 'retries' to mask failures"
- But some flakiness is normal in E2E (network timing, etc.)

Q8.2: How many times should we rerun the full CI workflow to confirm stability?
- Plan says "at least 2 additional times"
- Is 2-3 reruns sufficient or do you want more?

**Recommended:** Keep retries=1 (one retry is reasonable for E2E), rerun workflow 3 times total

---

## 9. DATABASE_URL for E2E

**Current State:**
- Backend webServer command doesn't set DATABASE_URL
- Default from settings.py is `postgresql+asyncpg://postgres:postgres@localhost:5432/postgres`

**Questions:**
Q9.1: Should the webServer command explicitly set DATABASE_URL?
```typescript
command: 'cd ../backend && DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres uvicorn ...'
```

Q9.2: OR rely on the default from settings.py (which matches our postgres setup)?

**Recommended:** Rely on default (already correct), but document it

---

## 10. CORS Configuration

**Current State:**
- Backend CORS allows `http://localhost:5173` only
- If we switch to 127.0.0.1, CORS will block requests

**Questions:**
Q10.1: Should we update CORS to allow both?
```python
allow_origins=[
    "http://localhost:5173",      # Dev (DNS)
    "http://127.0.0.1:5173",      # Dev (IPv4)
    "http://localhost:4173",      # Preview (DNS)
    "http://127.0.0.1:4173",      # Preview (IPv4)
]
```

Q10.2: OR should CORS be environment-based?
```python
allow_origins=settings.cors_origins.split(",")  # New setting
```

**Recommended:** Add all 4 origins for local dev/testing (simple, no new config needed)

---

## 11. Frontend API Client URL

**Current State (Verified):**
- Frontend uses **relative URLs** (`/api/health`, `/api/traces`) in `src/api/client.ts`
- Vite dev server has a **proxy** in `vite.config.ts`:
  ```typescript
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  }
  ```
- This proxy target uses `localhost:8000` (not 127.0.0.1)

**Questions:**
Q11.1: Should we update the Vite proxy to use 127.0.0.1?
```typescript
proxy: {
  '/api': {
    target: 'http://127.0.0.1:8000',  // Changed from localhost
    changeOrigin: true,
  },
}
```

Q11.2: Does `vite preview` support the same proxy configuration?
- Need to verify if preview mode uses `vite.config.ts` proxy settings
- If not, we may need a different approach for E2E with preview mode

Q11.3: Should the proxy target be environment-configurable?
```typescript
target: process.env.VITE_API_URL || 'http://127.0.0.1:8000',
```

**Recommended:** Update proxy to 127.0.0.1 (matches IPv4 standardization)

---

## 12. Vite Preview Mode Limitation (CRITICAL)

**Issue Discovered:**
- Vite **preview mode does NOT support proxy configuration**
- The proxy in `vite.config.ts` only works in dev mode
- If E2E uses preview mode (4173), relative URLs like `/api/health` will fail
- Frontend will try `http://127.0.0.1:4173/api/health` instead of `http://127.0.0.1:8000/api/health`

**Questions:**
Q12.1: How should we handle this for E2E with preview mode?

**Option A:** Use Vite dev mode for E2E (keeps proxy working)
- Pro: Proxy works out of the box
- Con: Not testing production build

**Option B:** Update frontend to use absolute URLs in production
- Add environment variable `VITE_API_BASE_URL`
- Update client.ts to use `${import.meta.env.VITE_API_BASE_URL}/api/health`
- Set in Playwright webServer: `VITE_API_BASE_URL=http://127.0.0.1:8000`
- Con: Requires code changes

**Option C:** Add a reverse proxy for E2E
- Use nginx or caddy to serve frontend and proxy /api to backend
- Con: Complex setup, additional service

**Option D:** Use same-origin setup (both on same port)
- Not feasible without reverse proxy

**Recommended:** Option A (use dev mode for E2E) OR Option B (add env var for API base URL)
- Option A is simplest, no code changes needed
- Option B is more "production-like" but requires small refactor

---

## 13. Stability Metrics

**Questions:**
Q12.1: What constitutes "stable" for you?
- All E2E tests pass 3 times in a row?
- All E2E tests pass 5 times in a row?
- All E2E tests pass 10 times with no failures?

Q12.2: Should we add any metrics/logging to track flakiness over time?

**Recommended:** 3 consecutive successful runs (as per plan)

---

## Summary of Recommendations

Based on the questions above, here's my recommended approach:

### Core Infrastructure
1. **Database:** Add postgres service container in CI with healthcheck
2. **Migrations:** Run as separate CI step before Playwright starts
3. **IPs:** Standardize on 127.0.0.1 everywhere (Playwright, uvicorn, Vite)

### Frontend/Backend Communication (CRITICAL DECISION NEEDED)
4. **Frontend Mode:** Two options:
   - **Option A (Simpler):** Use Vite **dev mode** for E2E (keeps proxy working, no code changes)
   - **Option B (Production-like):** Add `VITE_API_BASE_URL` env var + update client.ts to support absolute URLs
   
   **I recommend Option A for M04** (faster, simpler, less risky), then consider Option B for M5+ if needed.

### Testing & Validation
5. **Smoke checks:** Skip dedicated script (Playwright's built-in waiting is sufficient)
6. **Local target:** Single `make e2e` that handles full lifecycle (start DB, migrate, test)
7. **Retries:** Keep at 1 (reasonable for E2E), verify with 3 consecutive CI runs
8. **Stability metric:** 3 consecutive successful CI runs = stable

### Configuration Updates
9. **CORS:** Add all 4 origins (localhost + 127.0.0.1, ports 5173 + 4173) for dev/testing
10. **Vite proxy:** Update to `http://127.0.0.1:8000` (consistent with IPv4 standardization)
11. **Playwright baseURL:** Use `http://127.0.0.1:5173` (if using dev mode)
12. **Backend host:** Explicitly bind to `--host 127.0.0.1`

### Files to Modify (Estimated)
- `.github/workflows/ci.yml` - Add postgres service, migrations step, update E2E job
- `e2e/playwright.config.ts` - Update baseURL, webServer configs, explicit host bindings
- `backend/tunix_rt_backend/app.py` - Update CORS origins list
- `frontend/vite.config.ts` - Update proxy target to 127.0.0.1
- `Makefile` - Add `make e2e` target
- `README.md` - Document running E2E locally (small section)

**Please confirm or adjust these recommendations before I proceed with implementation.**

**KEY DECISION REQUIRED:** Should we use Vite dev mode (Option A) or add API base URL env var (Option B) for E2E?
