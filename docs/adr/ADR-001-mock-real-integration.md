# ADR-001: Mock/Real Mode RediAI Integration Pattern

**Status:** Accepted

**Date:** 2025-12-20 (M1 Milestone)

**Context:**

The tunix-rt application needs to integrate with RediAI for AI-powered reasoning trace analysis. However, we face several challenges:

1. **CI/CD Requirements**: CI pipelines should be deterministic and fast, without depending on external services
2. **Development Flexibility**: Developers need to test both with and without a running RediAI instance
3. **Testing Strategy**: Unit tests must be fast and reliable without network dependencies
4. **Integration Validation**: We still need confidence that real RediAI integration works

**Decision:**

Implement a **dual-mode integration pattern** using dependency injection and protocol-based design:

1. **Protocol Definition** (`RediClientProtocol`):
   - Defines the interface for RediAI clients
   - Enables type-safe dependency injection
   - Allows multiple implementations

2. **Two Implementations**:
   - **MockRediClient**: Returns deterministic responses, no external dependencies
   - **RediClient**: Makes actual HTTP requests to RediAI instance

3. **Runtime Mode Selection**:
   - Controlled by `REDIAI_MODE` environment variable ("mock" or "real")
   - Mode selection happens in `get_redi_client()` dependency provider
   - FastAPI dependency injection allows test overrides

4. **Usage Pattern**:
   ```python
   @app.get("/api/redi/health")
   async def redi_health(
       redi_client: Annotated[RediClientProtocol, Depends(get_redi_client)]
   ):
       return await redi_client.health()
   ```

**Consequences:**

### Positive

- ✅ **CI Determinism**: No external dependencies in CI, tests always pass
- ✅ **Fast Tests**: Unit tests complete in milliseconds
- ✅ **Easy Testing**: Override dependency in tests for custom scenarios
- ✅ **Development Flexibility**: Developers can work offline or with real RediAI
- ✅ **Type Safety**: Protocol ensures both implementations match interface
- ✅ **Future-Proof**: Easy to add new implementations (e.g., circuit breaker, retry logic)

### Negative

- ⚠️ **Mock Drift Risk**: Mock behavior may diverge from real RediAI over time
  - **Mitigation**: E2E tests can run in real mode locally
  - **Mitigation**: Contract tests planned for M2 (Schemathesis)
- ⚠️ **Extra Complexity**: Two implementations to maintain
  - **Assessment**: Minimal - MockRediClient is <10 lines
  - **Benefit**: Complexity pays for itself in test reliability

### Trade-offs Accepted

- We accept potential mock drift in exchange for CI speed and determinism
- We rely on E2E tests (optionally with real RediAI) to catch integration issues
- Mock mode is the default to make onboarding frictionless

**Alternatives Considered:**

1. **Always use real RediAI**:
   - Rejected: CI would be slow, flaky, and require RediAI deployment
   - Rejected: Developers would need RediAI running locally always

2. **Test containers (Testcontainers pattern)**:
   - Rejected: Adds Docker dependency to tests, slower CI
   - Rejected: Overkill for simple health check integration

3. **VCR/HTTP mocking (record/replay)**:
   - Rejected: More complex than our dual-mode approach
   - Rejected: Recorded cassettes can become stale

**References:**

- Hexagonal Architecture (Ports & Adapters): https://alistair.cockburn.us/hexagonal-architecture/
- FastAPI Dependency Injection: https://fastapi.tiangolo.com/tutorial/dependencies/
- Protocol-based design in Python: https://peps.python.org/pep-0544/

**Review:**

This pattern should be reviewed if:
- Mock drift becomes a significant problem (requires frequent sync with real API)
- RediAI integration becomes more complex (multiple endpoints, state management)
- We add caching/circuit breakers (may want to extract to separate layer)
