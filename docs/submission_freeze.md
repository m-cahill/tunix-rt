# Tunix RT - Submission Freeze

**Milestone:** M31 - Final Submission Package  
**Version:** `m31_v1`  
**Freeze Date:** 2025-12-25

---

## Frozen Configuration

This document records the exact configuration used for the final Kaggle submission.

### Commit Reference

| Field | Value |
|-------|-------|
| **Commit SHA** | `<TO_BE_FILLED_ON_FINAL_RUN>` |
| **Branch** | `milestone/M31-final-submission-package` |
| **Tag** | `v1.0.0-m31` (optional) |

---

### Dataset Configuration

| Field | Value |
|-------|-------|
| **Canonical Dataset** | `golden-v2` |
| **Dataset Location** | `backend/datasets/golden-v2/` |
| **Manifest** | `backend/datasets/golden-v2/manifest.json` |
| **Trace Count** | 100 |
| **Seed** | 42 (deterministic) |

**Alternative Dataset (for smoke testing):**
- `dev-reasoning-v1` (200 traces, 70% reasoning, 30% synthetic)

---

### Training Configuration

| Field | Value |
|-------|-------|
| **Config File** | `training/configs/submission_gemma3_1b.yaml` |
| **Base Model** | `google/gemma-3-1b-it` |
| **Alternative** | `google/gemma-2-2b` (via `submission_gemma2_2b.yaml`) |
| **Max Sequence Length** | 512 |
| **Seed** | 42 |

---

### Evaluation Configuration

| Field | Value |
|-------|-------|
| **Eval Set** | `training/evalsets/eval_v1.jsonl` |
| **Eval Items** | 25 |
| **Primary Metric** | `answer_correctness` (0-100, higher is better) |

---

### Commands Used

**Training:**
```bash
python training/train_jax.py \
  --config training/configs/submission_gemma3_1b.yaml \
  --output ./output/final_submission \
  --dataset golden-v2 \
  --device auto
```

**Evaluation:**
```bash
python training/eval_generate.py \
  --checkpoint ./output/final_submission \
  --eval_set training/evalsets/eval_v1.jsonl \
  --output ./output/final_submission/predictions.jsonl

python training/eval_report.py \
  --predictions ./output/final_submission/predictions.jsonl \
  --eval_set training/evalsets/eval_v1.jsonl
```

---

### Artifact Bundle

| Field | Value |
|-------|-------|
| **Archive Name** | `<TO_BE_FILLED_AFTER_PACKAGING>` |
| **Archive Location** | `./submission/` |
| **Checksum (SHA256)** | `<TO_BE_FILLED>` |

---

## Reproducibility Checklist

- [ ] Commit SHA matches archived version
- [ ] Dataset manifest hash verified
- [ ] Training config matches documented version
- [ ] Eval set unchanged from freeze point
- [ ] All dependencies pinned (see `backend/uv.lock`)

---

## Notes

- **Gemma Model Access:** Gemma models on Hugging Face require accepting the license agreement. Ensure access is granted before running in Kaggle.
- **Device Selection:** Use `--device auto` for automatic GPU/TPU detection, or `--device cpu` for local smoke testing.
- **Resume Training:** Orbax checkpoints support resume via `--resume_from` flag.

---

## Related Documentation

- [Submission Checklist](submission_checklist.md)
- [Kaggle Submission Guide](kaggle_submission.md)
- [Training End-to-End](training_end_to_end.md)
- [Submission Artifacts](submission_artifacts.md)
