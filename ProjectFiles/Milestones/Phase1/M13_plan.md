Milestone M13: Tunix Runtime Execution (Phase 2)

Goal:
Enable optional, gated execution of Tunix runs using artifacts generated in M12,
without impacting default CI or requiring Tunix at runtime.

Constraints:
- Default CI must pass with no Tunix installed
- Tunix execution must be opt-in and fail gracefully
- No TPU assumptions
- No coupling in core code paths
- Follow the UNGAR optional-integration pattern exactly

Backend Tasks:
1. Introduce TunixExecutionService
   - Accept dataset JSONL + manifest YAML
   - Support modes: dry-run (default), local
   - Return structured run result (status, logs, metadata)

2. Add execution endpoint:
   POST /api/tunix/run
   - dry_run=true by default
   - If Tunix not available and dry_run=false → 501 with clear message

3. Capture stdout/stderr and exit code
4. Store execution metadata (no result parsing yet)

Frontend Tasks:
1. Add minimal “Run with Tunix (Local)” button
2. Display execution status and logs
3. Clear messaging when Tunix is unavailable

CI Tasks:
1. Add non-blocking tunix-runtime.yml
2. Manual or self-hosted only
3. Never block merge

Out of Scope:
- Training result ingestion
- Evaluation metrics
- TPU orchestration
- Benchmarking

Stop after:
- Successful dry-run path
- One verified local execution path
- CI green
