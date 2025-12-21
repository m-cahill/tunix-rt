# M07 Clarifying Questions

## 1. UNGAR Repository Location & Integration

**Question:** Where is the UNGAR repository located? Should I:
- a) Assume UNGAR is installed as a pip package from a git URL (e.g., `git+https://github.com/m-cahill/ungar.git`)?
- b) Assume UNGAR is cloned separately as a sibling directory to `tunix-rt`?
- c) Use a local path for development (e.g., `pip install -e ../ungar`)?

**Context:** This affects how we specify the `[ungar]` extra in `pyproject.toml` and how the CI job will install UNGAR.

---

## 2. UNGAR Version/Commit Target

**Question:** Should I target:
- a) A specific UNGAR release/tag (e.g., `v1.0.0`)?
- b) The latest `main` branch?
- c) A specific commit SHA?

**Context:** This ensures reproducible builds and helps with dependency pinning.

---

## 3. High Card Duel Trace Conversion - Detail Level

**Question:** For the UNGAR→Tunix trace conversion (High Card Duel episodes), how sophisticated should the natural language generation be?

**Option A (Minimal - as M07 plan suggests):**
```
prompt: "Given my_hand=[AS], opponent_hand=[unknown], unseen=[51 cards], choose action: reveal"
steps:
  - "Legal moves: [reveal]"
  - "My hand contains: AS"
  - "Unseen count: 51"
final_answer: "reveal"
```

**Option B (More narrative):**
```
prompt: "You have been dealt one card. Decide when to reveal."
steps:
  - "I observe my hand: Ace of Spades"
  - "This is a high-ranking card (rank 14/14)"
  - "Legal moves: reveal"
  - "Decision: Reveal now"
final_answer: "reveal"
```

**Preference?** I'm inclined toward **Option A** (minimal) per M07 plan ("do not attempt perfect natural language"), but want to confirm.

---

## 4. CI Strategy for UNGAR Tests

**Question:** For the optional UNGAR integration tests, should the CI job be:

**Option A (Non-blocking optional job):**
- Runs on every PR/push
- Does NOT block merge if it fails
- Uses `continue-on-error: true` or similar

**Option B (Separate nightly/scheduled job):**
- Only runs on schedule (e.g., nightly)
- Separate workflow file (e.g., `.github/workflows/ungar-integration.yml`)

**Option C (Manual dispatch only):**
- Only runs when manually triggered
- Uses `workflow_dispatch`

**Preference?** I'm inclined toward **Option A** (non-blocking optional job) for faster feedback, but want to confirm.

---

## 5. JSONL Export Format - Tunix Specifics

**Question:** The M07 plan mentions exporting to "Tunix-friendly JSONL" format. Are there specific requirements from the Tunix library for this format, or should I design a reasonable structure based on the Kaggle "show your work" framing?

**Proposed structure:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "prompt": "What is 27 × 19?",
  "trace_steps": ["Parse the task", "27 × 19 = 513"],
  "final_answer": "513",
  "metadata": {
    "created_at": "2025-12-21T10:30:00Z",
    "source": "tunix-rt",
    "trace_version": "1.0"
  },
  "scores": {
    "baseline": 67.5
  }
}
```

**Does this look correct, or is there a specific Tunix JSONL schema I should follow?**

---

## 6. UNGAR Import Strategy - Module vs Package

**Question:** When implementing the optional UNGAR dependency, should the import be:

**Option A (Import specific UNGAR modules as needed):**
```python
try:
    from ungar.games.high_card_duel import make_high_card_duel_spec
    from ungar.runner import play_random_episode
    UNGAR_AVAILABLE = True
except ImportError:
    UNGAR_AVAILABLE = False
```

**Option B (Check UNGAR availability once at module level):**
```python
try:
    import ungar
    UNGAR_AVAILABLE = True
except ImportError:
    UNGAR_AVAILABLE = False

# Then use lazy imports inside functions
def generate_ungar_traces():
    if not UNGAR_AVAILABLE:
        raise ImportError("...")
    from ungar.games.high_card_duel import ...
```

**Preference?** I'm inclined toward **Option B** (lazy imports inside functions) per M07 plan guardrails.

---

## 7. Testing Without UNGAR - Mock Strategy

**Question:** For testing the UNGAR generator endpoint **without** UNGAR installed, should I:

**Option A:** Test only that it returns 501 with appropriate error message
**Option B:** Create a mock UNGAR interface for testing the conversion logic separately

**Preference?** I'm inclined toward **Option A** (just test the 501 response) to keep tests simple, but want to confirm.

---

## 8. Phase 0 Baseline - Documentation Format

**Question:** For Phase 0 baseline verification (documenting that M6 is stable), should the `docs/M07_BASELINE.md` be:

**Option A (Minimal):**
```markdown
# M07 Baseline Verification

**Date:** 2025-12-21
**Commit:** abc123def456
**Status:** ✅ All tests passing

## Backend (Python 3.11 + 3.12)
- Ruff: ✅ Pass
- mypy: ✅ Pass
- pytest: ✅ 56/56 tests passing
- Coverage: 90% line, 88% branch

## Frontend
- Tests: ✅ 11/11 passing
- Build: ✅ Success

## E2E
- Tests: ✅ 5/5 passing
```

**Option B (Detailed with full output):**
- Include full pytest output, coverage reports, etc.

**Preference?** I'm inclined toward **Option A** (minimal but clear) unless detailed output is needed.

---

**Please answer these questions, and I'll proceed with creating the todo list and implementation plan.**

