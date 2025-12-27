# M35 Summary: Quality Loop 1 — Eval Signal Hardening + Leaderboard Fidelity + Regression Guardrails

**Date:** December 26, 2025  
**Status:** ✅ Complete  
**Tests:** 370 passed, 11 skipped

## Overview

M35 focuses on making the evaluation and leaderboard loop **credible and decision-grade**. This means:
- Clear, deterministic scoring rules
- Robust normalization and aggregation
- UI/API that answers "is run B better than run A?" effectively
- Guardrails to prevent silent quality degradation

## Key Deliverables

### 1. Eval Set v2 (`training/evalsets/eval_v2.jsonl`)

- **100 items** with structured composition:
  - **Core (60%)**: Basic arithmetic, geometry, conversions, knowledge
  - **Trace-sensitive (25%)**: Multi-step word problems requiring reasoning
  - **Edge cases (15%)**: Formatting variations, boundary conditions

- **New fields**: `section`, `category`, `difficulty`
- **Stable**: Designed for regression testing and consistent scoring

### 2. Eval Set Validator (`backend/tools/validate_evalset.py`)

CLI tool for validating eval set schema:

```bash
python backend/tools/validate_evalset.py training/evalsets/eval_v2.jsonl
```

**Features:**
- Required field validation
- Duplicate ID detection
- Section/category/difficulty validation
- Summary statistics (composition percentages)

### 3. Scorecard Aggregator (`scoring.py`)

New `Scorecard` dataclass and `compute_scorecard()` function:

```python
from tunix_rt_backend.scoring import compute_scorecard

rows = [{"item_id": "001", "metrics": {"answer_correctness": 1.0}, "section": "core"}, ...]
card = compute_scorecard(rows)
# card.n_items, card.n_scored, card.n_skipped, card.primary_score, card.stddev
# card.section_scores, card.category_scores, card.difficulty_scores
```

**Outputs:**
- n_items, n_scored, n_skipped
- primary_score (mean), stddev
- Per-section/category/difficulty breakdowns

### 4. Leaderboard Filtering (API + UI)

**API**: `GET /api/tunix/evaluations?dataset_key=...&model_id=...&date_from=...`

**Filters (AND logic):**
- `dataset_key`: Exact match
- `model_id`: Contains match
- `config_path`: Contains match
- `date_from`, `date_to`: Date range

**UI (`Leaderboard.tsx`):**
- Inline filter inputs above table
- Scorecard summary (items/scored)
- Primary score as percentage
- Dataset column added

### 5. Run Comparison Enhancements (`RunComparison.tsx`)

- **Per-item diff table**: Collapsible section showing expected vs predicted
- Correct/Wrong indicators with color coding
- Diff column (+1/-1/=)
- Clear message when MockJudge doesn't produce predictions

### 6. Regression Baseline Enhancements

**Default metric changed to `primary_score`**:
```python
RegressionBaselineCreate(name="baseline", run_id=uuid, metric="primary_score")  # Default
```

**New fields for multi-baseline support:**
- `eval_set`: e.g., "eval_v2.jsonl"
- `dataset_key`: e.g., "dev-reasoning-v2"

### 7. Determinism Check (`backend/tools/check_determinism.py`)

Verifies evaluation pipeline produces identical results:

```bash
cd backend && uv run python tools/check_determinism.py --verbose
cd backend && uv run python tools/check_determinism.py --eval-set ../training/evalsets/eval_v2.jsonl
```

**Checks:**
- `compute_primary_score` determinism
- `compute_scorecard` determinism
- Ordering independence

### 8. Evidence Folder (`submission_runs/m35_v1/`)

Files:
- `run_manifest.json`: With `eval_set` field
- `eval_summary.json`: With `scorecard` object
- `kaggle_output_log.txt`: Template

### 9. Packaging Tool Update

- `eval_v2.jsonl` added to bundle
- Archive prefix changed to `m35`

## Test Coverage

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_evalset_validator.py` | 23 | Validator function + eval_v2 schema |
| `test_scoring.py` | 34 | compute_primary_score + compute_scorecard |
| `test_regression.py` | 14 | Regression service + primary_score default |
| `test_evidence_files.py` (M35) | 12 | Evidence schema validation |

**Total new tests:** 60+

## Usage Examples

### Validate Eval Set
```bash
python backend/tools/validate_evalset.py training/evalsets/eval_v2.jsonl
```

### Check Determinism
```bash
cd backend && uv run python tools/check_determinism.py --verbose
```

### API: Filtered Leaderboard
```bash
curl "http://localhost:8000/api/tunix/evaluations?dataset_key=dev-reasoning-v2&model_id=gemma"
```

### Python: Compute Scorecard
```python
from tunix_rt_backend.scoring import compute_scorecard

rows = [...evaluation_results...]
card = compute_scorecard(rows)
print(f"Score: {card.primary_score:.2%} ± {card.stddev:.4f}")
print(f"By section: {card.section_scores}")
```

## Next Steps (M36+)

1. **Real Kaggle run** with eval_v2.jsonl to populate evidence
2. **Regression CI workflow** (manual trigger, nightly)
3. **Per-item artifact storage** for full diff table support
4. **Baseline promotion API** ("promote best run to baseline")
