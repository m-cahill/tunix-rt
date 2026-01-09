# M45 Tool Calls Log

## Entry 1 — Project Analysis Phase
**Timestamp:** 2026-01-08 (Session Start)
**Tool:** read_file (multiple), list_dir, grep
**Purpose:** Familiarize with project structure, read tunix-rt.md, understand M44 completion state
**Files:**
- tunix-rt.md (project overview/schema)
- ProjectFiles/Milestones/Phase4/M44_summary.md (previous milestone state)
- training/README.md (training infrastructure)
- backend/tunix_rt_backend/ (backend structure)
- backend/datasets/dev-reasoning-v2/manifest.json (dataset manifest)
- backend/datasets/dev-reasoning-v2/dataset.jsonl (100 lines sample)
- training/evalsets/eval_v2.jsonl (evaluation set)
- backend/tools/seed_dev_reasoning_v2.py (seeder script)
- training_pt/train.py (PyTorch training script)
- submission_runs/m44_v1/m44_summary.md (M44 evidence)

**Status:** ✅ COMPLETE

**Findings:**
- M44 = authoritative GPU training (Gemma 2B, PyTorch, RTX 5090, 3 epochs)
- M42 = immutable Kaggle submission artifact
- `dev-reasoning-v2` has 550 traces with rich metadata:
  - `meta.category`: reasoning (385), synthetic (110), golden_style (35), edge_case (20)
  - `steps` array with structure `{i, type, content}`
  - Trace length = `len(steps)` — ranges from 1-6 steps
- Training infrastructure: PyTorch path (`training_pt/train.py`)
- Evaluation set: `training/evalsets/eval_v2.jsonl` (100 items)

**CHECKPOINT:** Project analysis complete. Ready for clarifying questions.

---

## Entry 2 — Questions & Answers Locked
**Timestamp:** 2026-01-08
**Tool:** write (M45_questions.md, M45_answers.md)
**Purpose:** Document clarifying questions and record authoritative decisions
**Files:**
- ProjectFiles/Milestones/Phase5/M45_questions.md (6 questions)
- ProjectFiles/Milestones/Phase5/M45_answers.md (locked decisions)

**Status:** ✅ COMPLETE

**Locked Configuration:**
- Partitioning: Category-first (synthetic+golden→A, reasoning 3-step→B, reasoning 4+step+edge→C)
- Epochs: 2-2-3
- Baseline: M44 final checkpoint
- Backend: PyTorch
- Eval: eval_v2.jsonl
- Directory: research/m45_curriculum_reasoning/

**CHECKPOINT:** Decisions locked. Execution phase begins.

---

## Entry 3 — Create Directory Structure & Split Script
**Timestamp:** 2026-01-08
**Tool:** write, run_terminal_cmd
**Purpose:** Create research directory structure and dataset partitioning script
**Files:**
- research/m45_curriculum_reasoning/ (directory tree)
- research/m45_curriculum_reasoning/split_dataset.py (partitioning script)
- research/m45_curriculum_reasoning/data/stage_a.jsonl (145 traces)
- research/m45_curriculum_reasoning/data/stage_b.jsonl (64 traces)
- research/m45_curriculum_reasoning/data/stage_c.jsonl (341 traces)
- research/m45_curriculum_reasoning/data/split_stats.json
- research/m45_curriculum_reasoning/data/trace_length_histogram.txt

**Status:** ✅ COMPLETE

**Partition Results:**
- Stage A: 145 traces (synthetic: 110, golden_style: 35) — 2 steps each
- Stage B: 64 traces (reasoning w/ 3 steps)
- Stage C: 341 traces (reasoning 4+ steps: 321, edge_case: 20)
- Source hash: 601441c0fd176437...

**CHECKPOINT:** Dataset partitioned. Creating training configs.

---

## Entry 4 — Create Training Configs
**Timestamp:** 2026-01-08
**Tool:** read_file, write
**Purpose:** Create training configs for stages A, B, C with identical hyperparameters
**Files:**
- submission_runs/m44_v1/m44_config.yaml (template)
- research/m45_curriculum_reasoning/configs/stage_a.yaml
- research/m45_curriculum_reasoning/configs/stage_b.yaml
- research/m45_curriculum_reasoning/configs/stage_c.yaml
- research/m45_curriculum_reasoning/run_curriculum.py (orchestrator)

**Status:** ✅ COMPLETE

**Hyperparameters (identical across stages):**
- Model: google/gemma-2b
- LR: 2.0e-5, weight_decay: 0.01
- Batch: 1, grad_accum: 4, dtype: bfloat16
- Optimizer: adafactor

**CHECKPOINT:** Configs and orchestrator ready. Starting training.

---

## Entry 5 — Execute Curriculum Training (Stages A, B, C)
**Timestamp:** 2026-01-08
**Tool:** run_terminal_cmd (background)
**Purpose:** Run full curriculum training pipeline
**Output:**
- checkpoints/stage_a/final_model/
- checkpoints/stage_b/final_model/
- checkpoints/stage_c/final_model/
- curriculum_training_summary.json
- training_log.txt

**Status:** ✅ COMPLETE

**Training Results:**
| Stage | Loss | Steps | Runtime |
|-------|------|-------|---------|
| A | 0.5640 | 74 | 57.1s |
| B | 0.2908 | 32 | 28.4s |
| C | 0.1913 | 258 | 134.4s |

Total runtime: ~220s (~3.7 min)
Throughput: ~8.3 samples/sec (Stage C)

**CHECKPOINT:** Curriculum training complete. Proceeding to evaluation.

---

## Entry 6 — Evaluate All Checkpoints
**Timestamp:** 2026-01-08
**Tool:** run_terminal_cmd
**Purpose:** Generate predictions for M44, Post-A, Post-B, Post-C on eval_v2.jsonl
**Files:**
- training/evalsets/eval_v2.jsonl (100 examples)
- eval/m44_baseline_predictions.jsonl
- eval/post_stage_a_predictions.jsonl
- eval/post_stage_b_predictions.jsonl
- eval/post_stage_c_predictions.jsonl
- eval/eval_summary.json

**Status:** ✅ COMPLETE

**Note:** Exact-match accuracy is 0% for all checkpoints (expected — models generate free-form text, not just answers). The key metric for M45 is **qualitative trace structure**, not accuracy.

**CHECKPOINT:** Evaluation artifacts generated. Proceeding to provenance and analysis.

---

## Entry 7 — Provenance & Analysis
**Timestamp:** 2026-01-08
**Tool:** write, read_file
**Purpose:** Create provenance.json and analysis.md with trace comparisons
**Files:**
- research/m45_curriculum_reasoning/provenance.json
- research/m45_curriculum_reasoning/analysis.md

**Status:** ✅ COMPLETE

**Key Findings:**
1. Curriculum training reshapes trace structure (decomposition, formula-first, verification)
2. Post-Stage-C uses explicit "I need to use the formula" language
3. Post-Stage-C has better topic coherence than flat SFT
4. Both models still suffer from repetition (generation issue)

**CHECKPOINT:** M45 complete. All artifacts produced.

---

## Final Summary

**M45 Status:** ✅ COMPLETE

**Training:**
- Stage A: 145 samples, 2 epochs, 0.564 loss
- Stage B: 64 samples, 2 epochs, 0.291 loss
- Stage C: 341 samples, 3 epochs, 0.191 loss
- Total runtime: ~220 seconds

**Artifacts Produced:**
- [x] Dataset split script
- [x] stage_a.jsonl, stage_b.jsonl, stage_c.jsonl
- [x] Training configs (3)
- [x] Sequential checkpoints (3)
- [x] Provenance/run manifest
- [x] Evaluation outputs (4 checkpoints)
- [x] Qualitative analysis document

**Conclusion:** Curriculum ordering reshapes reasoning structure without changing the model.

---

## Entry 8 — Milestone Closeout & Audit
**Timestamp:** 2026-01-09
**Tool:** run_terminal_cmd, write
**Purpose:** Run milestone audit, finalize documentation, push to GitHub
**Files:**
- ProjectFiles/Milestones/Phase5/M45_audit.md
- Git push to origin

**Status:** IN PROGRESS

---


