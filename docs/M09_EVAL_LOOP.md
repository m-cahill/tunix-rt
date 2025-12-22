# M09 Evaluation Loop Documentation

**Milestone:** M09 - Reproducible Training Loop v1  
**Last Updated:** December 21, 2025

---

## Overview

The evaluation loop enables quantitative comparison of model performance before and after training. It consists of:

1. **Static Eval Set** - Fixed, deterministic test questions
2. **Generation Scripts** - Produce model outputs
3. **Comparison Reports** - Compute deltas and metrics

---

## Evaluation Set

**Location:** `training/evalsets/eval_v1.jsonl`

**Format:**
```jsonl
{"id": "eval-001", "prompt": "What is 2+2?", "expected_answer": "4", "category": "arithmetic", "difficulty": "easy"}
```

**Statistics:**
- **Total Examples:** 25
- **Categories:** arithmetic, geometry, knowledge, word_problem, conversion, pattern, number_theory, probability
- **Difficulty:** easy (16), medium (9)

---

## Workflow

### Step 1: Generate Baseline Outputs

Run eval with **base model** (no training):

```bash
python training/eval_generate.py \
  --model base \
  --eval-set training/evalsets/eval_v1.jsonl \
  --output eval_baseline.jsonl \
  --seed 42
```

**Output:** JSONL file with generated traces
```jsonl
{
  "trace_version": "1.0",
  "prompt": "What is 2+2?",
  "final_answer": "[generated answer]",
  "steps": [...],
  "meta": {
    "source": "evaluation",
    "eval_id": "eval-001",
    "model": "base"
  }
}
```

---

### Step 2: Train Model

```bash
python training/train_sft_tunix.py \
  --config training/configs/sft_tiny.yaml \
  --data training_dataset.jsonl \
  --output artifacts/training_runs/my_run
```

---

### Step 3: Generate Post-Training Outputs

Run eval with **trained checkpoint**:

```bash
python training/eval_generate.py \
  --model artifacts/training_runs/my_run/checkpoint-final \
  --eval-set training/evalsets/eval_v1.jsonl \
  --output eval_trained.jsonl \
  --seed 42
```

---

### Step 4: Create Delta Report

```bash
python training/eval_report.py \
  --before eval_baseline.jsonl \
  --after eval_trained.jsonl \
  --output delta_report.md
```

**Report Contents:**
- Average score change
- Individual example comparisons
- Statistical summary
- Methodology notes

---

## Import to Database (Optional)

To visualize eval results in tunix-rt UI:

```bash
# Import baseline traces
curl -X POST http://localhost:8000/api/traces/batch \
  -H "Content-Type: application/json" \
  -d @eval_baseline.jsonl

# Import trained traces  
curl -X POST http://localhost:8000/api/traces/batch \
  -H "Content-Type: application/json" \
  -d @eval_trained.jsonl

# Compare in UI
# Navigate to http://localhost:5173 and use trace comparison feature
```

---

## Scoring Methodology

**Current (M09):** Placeholder scoring based on response length

**Future Enhancements:**
- Answer correctness (exact match or semantic similarity)
- Reasoning quality (step coherence, logical flow)
- Fluency metrics (perplexity, grammar)
- Multi-dimensional scores (correctness, explanation, clarity)

---

## Determinism

All evaluation is deterministic:

- ✅ **Fixed eval set** - Same 25 questions every time
- ✅ **Seeded generation** - Same seed = same outputs (modulo model stochasticity)
- ✅ **Versioned evalsets** - `eval_v1.jsonl` is immutable

**To ensure reproducibility:**
```bash
# Always use same seed
--seed 42

# Always use same eval set version
training/evalsets/eval_v1.jsonl

# Record in manifest
{
  "eval_set": "eval_v1.jsonl",
  "eval_seed": 42,
  "eval_timestamp": "2025-12-21T10:00:00Z"
}
```

---

## Evaluation Metrics

### Current Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Average Score** | Mean of all example scores | 0-100 |
| **Delta** | Change after training | -100 to +100 |

### Future Metrics (M10+)

- **Accuracy** - Exact match with expected answer
- **Reasoning Quality** - LLM-as-judge scoring
- **Step Count** - Number of reasoning steps
- **Coherence** - Logical flow score
- **Category Breakdown** - Per-category performance

---

## Example Report

```markdown
# Evaluation Delta Report

**Generated:** 2025-12-21T10:00:00Z  
**Eval Examples:** 25

## Summary

- **Before Training Avg Score:** 42.50
- **After Training Avg Score:** 58.75
- **Delta:** +16.25

✅ **Result:** Training improved scores by 16.25 points

## Individual Examples

### Example 1: `eval-001`

**Prompt:** What is 15 + 27?

**Before Score:** 35.00  
**After Score:** 50.00  
**Delta:** +15.00

...
```

---

## Best Practices

1. **Run eval before training** - Establish baseline
2. **Use same seed** - Ensures comparable outputs
3. **Save all artifacts** - Keep eval outputs with run
4. **Version eval sets** - Don't modify `eval_v1.jsonl`
5. **Document changes** - Note any eval set updates

---

## Integration with Training Runs

Evaluation outputs should be stored with training artifacts:

```
artifacts/training_runs/<run_id>/
├── run_manifest.json
├── metrics.jsonl
├── checkpoint-final/
├── eval_before.jsonl        # Pre-training eval
├── eval_after.jsonl         # Post-training eval
└── delta_report.md          # Comparison report
```

---

## See Also

- `docs/M09_TRAINING_QUICKSTART.md` - Training workflow
- `docs/M09_DATASET_FORMAT.md` - Data formats
- `training/evalsets/eval_v1.jsonl` - Evaluation set
- `training/eval_generate.py` - Generation script
- `training/eval_report.py` - Report script

