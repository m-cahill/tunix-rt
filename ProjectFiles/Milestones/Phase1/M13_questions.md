# M13 Clarifying Questions

## Architecture & Design

### 1. Tunix Execution Modes
The plan mentions "dry-run (default)" and "local" modes. Can you clarify:
- **Dry-run mode**: Should this validate that the manifest/dataset are valid but NOT actually call Tunix CLI?
- **Local mode**: Should this execute `tunix train --config manifest.yaml` as a subprocess on the local machine?
- Should we support any other modes (e.g., remote, TPU)?

### 2. Tunix Runtime Dependency
Following the UNGAR pattern:
- Should we add `backend[tunix]` as an optional dependency extra in pyproject.toml?
- If Tunix is not installed and user calls `/api/tunix/run` with `dry_run=false`, we return 501 "Not Implemented" (like UNGAR does)?
- Should we update `tunix_available()` to actually check for Tunix imports (like `ungar_available()` does)?

### 3. Execution Metadata Storage
The plan says "Store execution metadata (no result parsing yet)". What should we store?
- Run UUID, status (pending/running/completed/failed), timestamps?
- Should we create a new database table `training_runs` with columns: id, dataset_key, model_id, status, started_at, completed_at, logs, exit_code?
- Or just return execution results without persistence for M13?

### 4. Endpoint Design
The plan shows `POST /api/tunix/run`. What should the request/response look like?

**Option A (Manifest-first):**
```json
Request: {
  "manifest_yaml": "...",
  "dataset_jsonl": "...",
  "dry_run": true
}
Response: {
  "run_id": "uuid",
  "status": "completed",
  "exit_code": 0,
  "stdout": "...",
  "stderr": "...",
  "duration_seconds": 120
}
```

**Option B (Parameters-first, generate manifest internally):**
```json
Request: {
  "dataset_key": "my_dataset-v1",
  "model_id": "google/gemma-2b-it",
  "dry_run": true
}
Response: (same as above)
```

Which approach do you prefer?

### 5. Local Execution Details
For local mode execution:
- Should we execute `tunix train` via subprocess.run() with timeout?
- What timeout should we use (e.g., 30 seconds for smoke tests, hours for real training)?
- Should execution be synchronous (blocking) or asynchronous (background task)?
- If async, how should the user poll for status? (GET /api/tunix/run/{run_id}?)

### 6. Frontend Integration
The plan says "Add minimal 'Run with Tunix (Local)' button". Where should this live?
- In the existing Tunix panel (next to Export/Manifest buttons)?
- Should we show logs in real-time or just final status?
- Should we have a "Training Runs" history section?

### 7. CI Workflow
The plan says "Add non-blocking tunix-runtime.yml, Manual or self-hosted only, Never block merge".
- Should this extend the existing `tunix-integration.yml` or be a separate workflow?
- What should it test? (Just that dry-run works? Or actually run a tiny training job?)
- Should it be manual-dispatch only, or also run on schedule (nightly)?

### 8. Stop Criteria
The plan says "Stop after: Successful dry-run path, One verified local execution path, CI green".

Does this mean:
- ✅ Dry-run mode works (validates manifest + dataset, returns status)
- ✅ Local mode works (executes Tunix CLI, captures logs, returns status)
- ✅ 501 response when Tunix not available + dry_run=false
- ✅ Default CI passes without Tunix installed
- ✅ Optional CI workflow runs dry-run test
- ❌ NO database persistence (deferred to M14?)
- ❌ NO result ingestion (checkpoints, metrics)
- ❌ NO evaluation metrics
- ❌ NO frontend training history

Is this correct?

---

## Testing Strategy

### 9. Test Coverage
Following UNGAR pattern:
- Default tests (no Tunix): Test dry-run mode, 501 responses
- Optional tests (`@pytest.mark.tunix`): Test local execution with Tunix installed
- Should we aim for similar coverage boost (~10-15 new tests)?

### 10. Local Execution Tests
For tests that actually run Tunix:
- Should we use a tiny toy dataset (1-2 traces)?
- Should we use a small model (e.g., gemma-2b-it with 1 epoch, 1 batch)?
- What timeout should we use? (5 seconds for smoke test?)

---

## Compatibility & Dependencies

### 11. Tunix Installation
For developers who want to test local execution:
- Should we document how to install Tunix? (Is it `pip install tunix`, or from Git like UNGAR?)
- Is Tunix publicly available or Google-internal only?
- Should we pin a specific Tunix version/commit like UNGAR does?

### 12. TPU Assumptions
The plan says "No TPU assumptions". Does this mean:
- Local execution should work on CPU/GPU only?
- No JAX/TPU-specific code in M13?
- Leave TPU orchestration for future milestones?

---

## Documentation

### 13. Documentation Scope
Should we create:
- `docs/M13_BASELINE.md` (pre-M13 state)
- `docs/M13_TUNIX_EXECUTION.md` (complete guide like M12_TUNIX_INTEGRATION.md)
- `docs/M13_SUMMARY.md` (post-M13 summary)
- Update `tunix-rt.md` with new endpoint
- Update `README.md` with Tunix execution examples

---

## Clarifications Needed Before Starting

Please confirm:
1. **Execution modes**: Dry-run (validate only) + Local (subprocess execution)?
2. **Persistence**: No database storage in M13, just return execution results?
3. **Endpoint**: Prefer manifest-first or parameters-first API design?
4. **Frontend**: Add button to existing Tunix panel, show logs inline?
5. **CI**: Extend `tunix-integration.yml` or create new workflow?
6. **Stop criteria**: Dry-run + one local execution + 501 handling = DONE?

Once you clarify these, I'll create the todo list and begin implementation.
