# ADR-003: Coverage Strategy (Line + Branch Thresholds)

**Status:** Accepted

**Date:** 2025-12-20 (M1 Milestone)

**Context:**

Test coverage is a key quality metric, but different coverage types provide different insights:

1. **Line Coverage**: Measures which lines of code execute during tests
2. **Branch Coverage**: Measures which code paths (if/else branches) execute

**Problem:**

M0 achieved 82% line coverage but 0% branch coverage, exposing a critical gap:
- Line coverage can be high while missing important conditional logic
- Bugs often hide in untested branches (error handlers, edge cases)
- Many critical paths in `redi_client.py` and `app.py` had no branch tests

**Decision:**

Implement **dual-threshold coverage enforcement**:

1. **Line Coverage Gate**: ≥ 80%
2. **Branch Coverage Gate**: ≥ 68%

**Rationale for 68% Branch Gate:**

- **70% target** with **2% buffer** for small code changes
- M1 achieved 83.33% (exceeds gate by 15%)
- Buffer prevents flaky CI when refactoring shifts branch ratios slightly

**Implementation:**

### 1. Enable Branch Measurement

```bash
pytest --cov --cov-branch --cov-report=json:coverage.json
```

### 2. Custom Coverage Gate Script

```python
# backend/tools/coverage_gate.py
def main() -> int:
    data = json.load(open("coverage.json"))
    line_coverage = data["totals"]["percent_covered"]
    branch_coverage = data["totals"]["percent_branches_covered"]
    
    if line_coverage >= 80.0 and branch_coverage >= 68.0:
        print("[PASS] All coverage gates PASSED")
        return 0
    else:
        print("[FAIL] Coverage gates FAILED")
        return 1
```

**Why custom script instead of pytest-cov `--cov-fail-under`?**

- pytest-cov's `--cov-fail-under` only supports **one threshold**
- We need separate gates for line and branch coverage
- Custom script provides clearer error messages

### 3. CI Integration

```yaml
- name: pytest with coverage
  run: pytest --cov --cov-branch --cov-report=json:coverage.json

- name: Enforce coverage gates
  run: python tools/coverage_gate.py
```

**Consequences:**

### Positive

- ✅ **Real Quality Signal**: Branch coverage catches untested error paths
- ✅ **Clear Gates**: Separate thresholds for line vs branch
- ✅ **Buffer for Stability**: 2% buffer prevents flaky CI
- ✅ **Fast Feedback**: Gate check is instant (<100ms)
- ✅ **Visibility**: Reports both metrics clearly

### Negative

- ⚠️ **Extra Maintenance**: Custom script to maintain
  - **Assessment**: Script is 70 lines, well-tested
- ⚠️ **Learning Curve**: Developers need to understand branch coverage
  - **Mitigation**: Document in README, provide examples

**Coverage Results (M1):**

| Metric | M0 | M1 | Change |
|--------|----|----|--------|
| Line Coverage | 82% | 90.91% | +8.91% |
| Branch Coverage | 0% | 83.33% | +83.33% |
| Tests | 7 | 21 | +14 |
| Line Gate | 70% | 80% | +10% |
| Branch Gate | N/A | 68% | New |

**What Tests Were Added:**

1. **Error Path Tests** (`redi_client.py`):
   - Non-2xx HTTP responses
   - Timeout errors
   - Connection refused errors

2. **Mode Selection Tests** (`app.py`):
   - Mock mode returns MockRediClient
   - Real mode returns RediClient

3. **Settings Validation Tests** (`settings.py`):
   - Invalid REDIAI_MODE enum
   - Invalid URL format
   - Port out of range (0, 99999)

4. **Cache Tests** (`app.py`):
   - Cache hit (no additional RediAI call)
   - Cache expiry (fetches fresh data)

**Alternatives Considered:**

1. **Only Line Coverage**:
   - Rejected: Misses untested branches (M0 gap)

2. **100% Branch Coverage**:
   - Rejected: Too strict, diminishing returns
   - Example: Protocol definitions can't be covered (abstract methods)

3. **Same Threshold for Line and Branch**:
   - Rejected: Branch coverage is harder to achieve
   - 80% branch would block PRs unnecessarily

4. **pytest-cov only**:
   - Rejected: Can't enforce two separate thresholds

**Review:**

These thresholds should be reviewed if:
- Coverage consistently exceeds 90% → raise gates
- Difficulty hitting gates → lower branch gate to 60%
- Adding more complex conditional logic → may need higher branch target

**References:**

- pytest-cov branch coverage: https://pytest-cov.readthedocs.io/en/latest/config.html
- Coverage.py branch measurement: https://coverage.readthedocs.io/en/latest/branch.html
- Google Testing Blog on branch coverage: https://testing.googleblog.com/2020/08/code-coverage-best-practices.html

