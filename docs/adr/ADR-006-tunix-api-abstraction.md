# ADR-006: Tunix API Abstraction Pattern

**Status:** Accepted  
**Date:** 2025-12-21  
**Context:** M11 - Stabilize + Complete Service Extraction  
**Decision Makers:** tunix-rt team  
**Related:** Training pipeline (M9), future Tunix integration (M12+)

---

## Context

The tunix-rt project needs to integrate with the Tunix API for production fine-tuning jobs. However:

1. **Tunix API is external** and subject to changes/deprecation
2. **Local development** requires testing without Tunix credentials/quota
3. **CI/CD** must run deterministically without external dependencies
4. **Training scripts** (`training/train_sft_tunix.py`) currently have no abstraction layer
5. **Future flexibility** needed for different training backends (local JAX, cloud TPU, etc.)

**Problem:** How do we build a training pipeline that:
- Works locally without Tunix
- Is testable in CI
- Can adapt to Tunix API changes
- Supports future training backends

---

## Decision

We will implement a **protocol-based abstraction layer** for Tunix API interactions using Python's structural typing (`Protocol`).

### Architecture

```python
# backend/tunix_rt_backend/integrations/tunix/client.py

from typing import Protocol
from tunix_rt_backend.training.schema import TrainingConfig, JobID, JobStatus

class TunixClientProtocol(Protocol):
    """Protocol defining the interface for Tunix API clients."""
    
    async def submit_job(self, config: TrainingConfig) -> JobID:
        """Submit a training job to Tunix."""
        ...
    
    async def get_status(self, job_id: JobID) -> JobStatus:
        """Get status of a submitted job."""
        ...
    
    async def cancel_job(self, job_id: JobID) -> None:
        """Cancel a running job."""
        ...
    
    async def download_model(self, job_id: JobID, output_dir: Path) -> Path:
        """Download trained model artifacts."""
        ...


class TunixClient(TunixClientProtocol):
    """Real Tunix API client for production use."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.tunix.ai"):
        self.api_key = api_key
        self.base_url = base_url
    
    async def submit_job(self, config: TrainingConfig) -> JobID:
        # Real HTTP calls to Tunix API
        ...


class MockTunixClient(TunixClientProtocol):
    """Local simulation client for testing."""
    
    async def submit_job(self, config: TrainingConfig) -> JobID:
        # Return deterministic job ID
        return JobID(f"mock-{uuid.uuid4()}")
    
    async def get_status(self, job_id: JobID) -> JobStatus:
        # Simulate completion after delay
        return JobStatus(state="completed", progress=1.0)
```

### Training Script Usage

```python
# training/train_sft_tunix.py

from tunix_rt_backend.integrations.tunix.client import (
    TunixClientProtocol,
    TunixClient,
    MockTunixClient,
)

async def main():
    # Dependency injection based on environment
    if os.getenv("TUNIX_MODE") == "mock":
        client: TunixClientProtocol = MockTunixClient()
    else:
        api_key = os.getenv("TUNIX_API_KEY")
        client: TunixClientProtocol = TunixClient(api_key=api_key)
    
    # Submit job (interface is the same)
    job_id = await client.submit_job(config)
    
    # Poll for completion
    while True:
        status = await client.get_status(job_id)
        if status.state == "completed":
            break
```

---

## Consequences

### Positive

1. **Testability:** Training scripts can be tested locally without Tunix credentials
2. **CI/CD Safety:** Smoke tests work in CI with mock client
3. **API Isolation:** Changes to Tunix API only affect `TunixClient`, not scripts
4. **Type Safety:** Protocol ensures all implementations have the same interface
5. **Future Flexibility:** Easy to add `LocalJAXClient`, `GCPTPUClient`, etc.
6. **No Runtime Dependencies:** Mock mode works without installing Tunix SDK

### Negative

1. **Abstraction Overhead:** Additional layer of indirection
2. **Maintenance:** Must keep mock behavior realistic
3. **Incomplete Mocking:** Mock can't catch all Tunix API edge cases
4. **Initial Setup:** Requires upfront design of protocol interface

### Risks

| Risk | Mitigation |
|------|------------|
| Mock diverges from real API | Periodic validation against real Tunix API |
| Protocol incomplete | Iterate protocol as new Tunix features needed |
| Performance overhead | Protocol is compile-time only (zero runtime cost) |

---

## Implementation Plan

### Phase 1 (M11 - Optional):
- ✅ Document pattern in ADR-006
- ❌ Implementation deferred to M12 (training scripts work locally today)

### Phase 2 (M12):
1. Create `backend/tunix_rt_backend/integrations/tunix/client.py`
2. Define `TunixClientProtocol`
3. Implement `MockTunixClient` (for testing)
4. Refactor `training/train_sft_tunix.py` to use protocol
5. Add unit tests for mock client
6. Update `TRAINING_PRODUCTION.md` with usage

### Phase 3 (M13+):
1. Implement `TunixClient` with real API calls
2. Add integration tests (manual/nightly only)
3. Document production deployment

---

## Alternatives Considered

### Alternative 1: Direct Tunix SDK Usage
**Rejected:** Hard to test, tightly coupled, no fallback for local dev.

### Alternative 2: Environment Variable Mocking
```python
# Training script directly checks env vars
if os.getenv("TUNIX_MODE") == "mock":
    # Mock behavior inline
    ...
else:
    # Real Tunix calls
    ...
```
**Rejected:** Mixes concerns, hard to test, violates SRP.

### Alternative 3: Dependency Injection via Function Args
```python
def submit_job(config, api_client=None):
    if api_client is None:
        api_client = TunixClient()
    ...
```
**Rejected:** Weaker type safety, requires passing client through call stack.

### Alternative 4: ABC (Abstract Base Class)
```python
class TunixClientABC(ABC):
    @abstractmethod
    async def submit_job(...): ...
```
**Chosen Alternative** uses `Protocol` instead because:
- No inheritance required (structural typing)
- More Pythonic for duck typing
- Better mypy support
- Less boilerplate

---

## Related Patterns

1. **ADR-001: Mock/Real Mode Pattern** - Same strategy used for RediAI
2. **Repository Pattern** - Similar separation between interface and implementation
3. **Adapter Pattern** - `TunixClient` adapts external API to our protocol

---

## References

- [PEP 544 – Protocols](https://peps.python.org/pep-0544/)
- [Tunix API Documentation](https://docs.tunix.ai) (future)
- [tunix-rt TRAINING_PRODUCTION.md](../TRAINING_PRODUCTION.md)
- [M09 Training Schema](../M09_DATASET_FORMAT.md)

---

## Decision Rationale

This pattern is **CRITICAL** for future-proofing the training pipeline. Without it:
- Tunix API changes break training scripts directly
- Local development requires Tunix credentials
- Testing becomes flaky/expensive
- Cannot experiment with alternative training backends

The protocol pattern provides the **minimum viable abstraction** to decouple training logic from Tunix API specifics while maintaining type safety and testability.

---

**Approved:** 2025-12-21  
**Next Review:** M12 (when implementing real Tunix client)

