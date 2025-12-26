# Tunix RT - Submission Freeze

**Milestone:** M33 - Kaggle Submission Rehearsal v1  
**Version:** `m33_v1`  
**Freeze Date:** 2025-12-26

---

## Frozen Configuration

This document records the exact configuration used for the final Kaggle submission.

### Commit Reference

| Field | Value |
|-------|-------|
| **Commit SHA** | `915254b` (M33 rehearsal) |
| **Branch** | `milestone/M33-kaggle-rehearsal-v1` |
| **Tag** | `v1.0.0-m33` (optional) |

---

### Dataset Configuration

| Field | Value |
|-------|-------|
| **Canonical Dataset** | `dev-reasoning-v2` |
| **Dataset Location** | `backend/datasets/dev-reasoning-v2/` |
| **Manifest** | `backend/datasets/dev-reasoning-v2/manifest.json` |
| **Trace Count** | 550 |
| **Seed** | 42 (deterministic) |

**Alternative Dataset (for quick sanity):**
- `golden-v2` (100 traces, calibration)

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
| **Archive Name** | `tunix_rt_m33_2025-12-26_915254b.zip` |
| **Archive Location** | `./submission/` |
| **Archive Size** | 70.3 KB |
| **Evidence Folder** | `submission_runs/m33_v1/` |

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
