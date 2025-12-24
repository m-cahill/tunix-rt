Below are **locked, implementation-ready answers** to the M18 clarifying questions. These are **decisions**, not options, so Cursor can proceed without further clarification.

---

## 1️⃣ GemmaJudge & LLM Provider

### **Decision**

**Use RediAI as the inference provider.**
Do **not** integrate `google-generativeai`, `ollama`, or `openai` directly in M18.

### **Why**

* This project is explicitly positioning RediAI as the orchestration + inference layer.
* M18 is about **judge abstraction + credibility**, not vendor sprawl.
* Using RediAI keeps:

  * inference auditable
  * prompts versioned
  * eval runs reproducible
* External SDKs introduce auth, rate limits, nondeterminism, and CI fragility.

### **Implementation Guidance**

* Extend `RediClient` with a **single minimal inference method**:

  ```python
  class RediClient:
      async def generate(
          self,
          model: str,
          prompt: str,
          *,
          temperature: float = 0.0,
          max_tokens: int = 1024,
      ) -> str:
          ...
  ```
* `GemmaJudge` calls `RediClient.generate(...)`
* Model name example: `"gemma-judge-v1"`

### **Contract Requirements**

* Deterministic by default (`temperature=0`)
* Timeout-bounded
* Fail-closed (raise → evaluation fails loudly)
* Return raw text only; parsing happens in `GemmaJudge`

### **API Spec**

You do **not** need a full RediAI API spec for M18.
Stub the client against an internal HTTP endpoint or mock provider.
Real inference wiring can be finalized in M19 without breaking contracts.

---

## 2️⃣ Regression Baseline Storage

### **Decision**

Create a **separate `RegressionBaseline` table**.

❌ Do **not** add `is_baseline` flags to `TunixRun` or `TunixRunEvaluation`.

### **Why**

* Baselines are **policy**, not intrinsic run properties
* A run may be:

  * baseline for one metric
  * non-baseline for another
* Flags don’t scale to:

  * multiple baselines
  * time-windowed baselines
  * dataset-specific baselines

### **Schema (Minimal)**

```sql
regression_baselines (
    id UUID PK,
    name TEXT UNIQUE,
    run_id UUID FK -> tunix_runs.id,
    metric TEXT,          -- e.g. "score"
    created_at TIMESTAMP
)
```

### **Behavior**

* One named baseline per metric (enforced in service logic)
* Regression check logic:

  * fetch baseline evaluation
  * compare current evaluation
  * return pass/fail + delta

### **Acceptance**

* Can define baseline: `"gemma-v1-initial"`
* Can compare future runs against it
* No schema churn later

---

## 3️⃣ Frontend Scope

### **Decision**

**Yes — frontend changes are in scope for M18.**

### **But keep them minimal.**

### **Required Frontend Work**

* Update Leaderboard page to:

  * consume paginated API (`limit` / `offset`)
  * show pagination controls (Prev / Next is sufficient)
* No new visualizations
* No filtering UI yet
* No regression UI yet

### **Why**

* Pagination is a **hard scalability boundary**
* Leaving it backend-only creates:

  * dead code
  * untested behavior
* This is a **small, safe UI change** that locks the contract

### **Acceptance**

* Existing leaderboard still works
* Default page size respected
* Tests updated as needed
* No UX polish required

---

## 🔒 Summary for Cursor

* **GemmaJudge**

  * Uses **RediAI inference only**
  * Extend `RediClient.generate(...)`
  * Deterministic, timeout-safe, fail-closed

* **Regression Baselines**

  * New `RegressionBaseline` table
  * No boolean flags on runs or evaluations

* **Frontend**

  * Pagination updates are **in scope**
  * Keep UI minimal and contractual

These decisions preserve:

* architectural cleanliness
* future flexibility
* evaluation trust
* CI stability

If you want next, I can:

* Draft the exact `Judge` interface + `GemmaJudge` skeleton
* Provide a minimal RediAI inference mock
* Write the regression comparison service logic
