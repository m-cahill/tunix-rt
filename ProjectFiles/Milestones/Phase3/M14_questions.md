# M14 Clarifying Questions

## Database Schema Questions

1. **Stdout/Stderr Truncation Strategy:**
   - M13 truncates subprocess output to 10KB for ephemeral responses
   - For persistent storage, should we maintain 10KB limit or use a different limit?
   - Suggested: Keep TEXT fields but document expected truncation at capture time (10KB from M13)

2. **Status State Machine:**
   - M13 uses: `pending | running | completed | failed | timeout`
   - M14 plan specifies same status values
   - **Question:** Should `pending` be used for async execution (M15), or should we skip it in M14 since execution is still synchronous?
   - Suggested: Include `pending` in schema for forward compatibility, but M14 writes `running` → `completed/failed/timeout` immediately

3. **Dataset Key Indexing:**
   - Plan specifies `dataset_key (string, indexed)`
   - **Question:** Should we also index `created_at` for pagination/filtering recent runs?
   - Suggested: Yes, add `ix_tunix_runs_created_at` index for list performance

4. **Mode Field:**
   - Plan specifies: `mode (dry_run | local)`
   - M13 uses: `mode: ExecutionMode = Literal["dry-run", "local"]` (with hyphen)
   - **Question:** Use `dry-run` or `dry_run` in the database?
   - Suggested: Use `dry-run` to match M13 response schema (consistency)

5. **Nullable Fields Clarification:**
   - `exit_code (nullable int)`: Null for dry-run and timeout cases?
   - `completed_at (nullable datetime)`: Null only if run never completes (system crash)?
   - `duration_seconds (nullable float)`: Should this ever be null if we have started_at?
   - Suggested: Document null cases clearly in migration

## Service Layer Questions

6. **When to Create Run Record:**
   - Plan says: "Create a run record immediately when execution starts"
   - M14 execution is still synchronous (blocking)
   - **Question:** Should we create the record with `status='running'` at the start of `execute_tunix_run()`, or create with final status at the end?
   - Suggested: Create immediately with `status='running'`, update on completion (even though it's synchronous, this prepares for M15 async)

7. **Dry-Run Persistence:**
   - Should dry-run executions be persisted to the database?
   - They have `exit_code=0` and minimal stdout
   - **Question:** Store all dry-runs, or only local executions?
   - Suggested: Store all runs (including dry-run) for complete audit trail

8. **Failure Handling:**
   - Plan says: "Ensure failures (timeouts, subprocess errors) are persisted"
   - **Question:** What happens if database commit fails after successful Tunix execution?
   - Suggested: Wrap persistence in try/except, log error, but still return response to user

9. **Output Truncation in Database:**
   - M13 already truncates stdout/stderr to 10KB
   - **Question:** Should we apply additional truncation at persistence time, or rely on M13 truncation?
   - Suggested: Rely on M13 truncation (10KB), store as-is

## API Design Questions

10. **List Endpoint Pagination:**
    - Plan specifies: `GET /api/tunix/runs` with pagination
    - **Question:** Default limit/offset values? Follow traces pattern (limit=20, max=100)?
    - Suggested: Yes, reuse pagination pattern from `/api/traces`

11. **List Endpoint Filtering:**
    - Plan specifies filterable by: status, dataset_key, mode
    - **Question:** Should filters be query params like `?status=completed&dataset_key=my_dataset-v1&mode=local`?
    - **Question:** Should filtering be AND or OR logic? (likely AND)
    - Suggested: Query params with AND logic, all filters optional

12. **Detail Endpoint Response:**
    - Plan says: "Returns full run details"
    - **Question:** Should this match `TunixRunResponse` schema from M13, or create a new `TunixRunDetail` schema?
    - Suggested: Reuse `TunixRunResponse` schema for consistency

13. **Run ID Format:**
    - Plan specifies: `run_id (UUID, primary key)`
    - M13 generates: `run_id = str(uuid.uuid4())`
    - **Question:** Should the database use UUID type or VARCHAR(36)?
    - Suggested: Use UUID type (postgres supports it) for efficiency

## Frontend Questions

14. **Run History Panel Placement:**
    - **Question:** Should this be a new tab/section, or integrated into existing Tunix panel?
    - Suggested: New collapsible section below existing "Run with Tunix" buttons

15. **Polling vs Static Fetch:**
    - Plan says: "No live updates yet (polling is OK or static fetch)"
    - **Question:** Should the list auto-refresh on a timer (e.g., every 10s), or manual refresh only?
    - Suggested: Manual refresh button for M14, defer auto-refresh to M15

16. **Run History List Contents:**
    - **Question:** Show only current dataset's runs, or all runs across datasets?
    - Suggested: Show all runs with dataset_key column, allow filtering by current dataset

17. **Click to View Details:**
    - Plan says: "Click to view stdout/stderr"
    - **Question:** Expand inline, or navigate to detail page?
    - Suggested: Expand inline with collapsible sections (similar to M13 result display)

## Migration Questions

18. **Migration Naming:**
    - Existing migrations use Alembic auto-generated IDs (e.g., `f3cc010ca8a6_add_scores_table.py`)
    - **Question:** Confirm migration command: `alembic revision -m "add_tunix_runs_table"`
    - Suggested: Yes, let Alembic auto-generate the revision ID

19. **Downgrade Strategy:**
    - Plan requires: "downgrade with no breaking changes"
    - **Question:** Should downgrade drop the table, or just remove indexes?
    - Suggested: Drop entire `tunix_runs` table (no FK dependencies on it)

20. **Foreign Keys:**
    - **Question:** Should `dataset_key` have a FK to a datasets table?
    - Current state: Datasets are file-based manifests, not database records
    - Suggested: No FK constraint (dataset_key is just a string reference)

## Testing Questions

21. **Test Scope:**
    - Plan says: "No Tunix runtime required"
    - **Question:** Should tests mock the execution service, or use real execution with in-memory DB?
    - Suggested: Use real execution with dry-run mode (no Tunix required)

22. **Coverage Impact:**
    - Plan says: "Coverage does not regress below gates"
    - Current gates: Backend 80% line, 68% branch
    - **Question:** Should new persistence code be excluded from coverage if it causes regression?
    - Suggested: No exclusions, write comprehensive tests to maintain gates

## Documentation Questions

23. **M14 Document Structure:**
    - Plan requires: `docs/M14_RUN_REGISTRY.md`, `M14_BASELINE.md`, `M14_SUMMARY.md`
    - **Question:** Should `M14_RUN_REGISTRY.md` cover both backend and frontend, or separate docs?
    - Suggested: Single comprehensive doc covering full feature (backend + frontend + API)

24. **README Updates:**
    - Plan says: Add "Run Persistence (M14)" section
    - **Question:** Should this go in API Endpoints section, or new "Tunix Features" section?
    - Suggested: Extend existing "Tunix Integration (M12/M13)" section with M14 updates

## Scope Boundary Questions

25. **Run Deletion:**
    - Plan explicitly excludes: "Run deletion UI"
    - **Question:** Should backend support DELETE endpoint even if UI doesn't expose it?
    - Suggested: No, defer to M15+ (YAGNI principle)

26. **Run Retry/Rerun:**
    - **Question:** Should M14 support re-running a previous configuration?
    - Suggested: No, users can re-submit via POST /api/tunix/run with same params

27. **Run Metadata Updates:**
    - **Question:** Should M14 support PATCH to update run metadata (e.g., adding notes)?
    - Suggested: No, runs are immutable audit records

## Implementation Order Questions

28. **Development Sequence:**
    - **Question:** Preferred order: (1) Schema → Service → API → Frontend → Tests, or different?
    - Suggested: Schema → Migration → Service → API → Tests → Frontend → E2E

29. **Stop Criteria:**
    - **Question:** What's the minimal viable M14? 
    - Suggested:
      - ✅ Runs persist across server restarts
      - ✅ GET /api/tunix/runs returns list
      - ✅ GET /api/tunix/runs/{run_id} returns detail
      - ✅ Frontend shows run history (basic table)
      - ✅ CI green with no coverage regression

---

## Summary of Key Decisions Needed

1. **Status field:** Include `pending` in schema even though M14 doesn't use it?
2. **Dry-run persistence:** Store dry-run executions or only local runs?
3. **Database failure handling:** Return response to user even if DB commit fails?
4. **Frontend placement:** New section or integrate into existing Tunix panel?
5. **Frontend refresh:** Manual button or auto-polling?
6. **Run record creation timing:** Create immediately with `running` status, or only at completion?

Please review and provide guidance on these questions before I begin implementation. 🙏
