# Training Readiness Checklist

Before launching a large-scale training or tuning run, verify the following gates are closed.

## 1. Evaluation Semantics Locked
- [x] **Primary Metric defined:** `answer_correctness` (Binary 0/1).
- [x] **Evaluator implemented:** `AnswerCorrectnessJudge` in `services/judges.py`.
- [x] **Versioning:** Evaluator output includes `judge_version`.

## 2. Dataset Canonicalization
- [x] **Golden Set:** `golden-v1` exists and is importable.
- [x] **Validation:** Dataset builder prevents empty datasets.
- [ ] **Ground Truth:** `golden-v1` manifest includes verified ground truth answers.

## 3. Metric Persistence
- [x] **Runs:** Metrics stored in `TunixRunEvaluation`.
- [x] **Registry:** Metrics promoted to `ModelVersion`.
- [x] **UI:** Metrics visible in Run History and Model Registry.

## 4. Guardrails
- [ ] **Tuning:** Block tuning if metric is undefined or judge is missing.
- [x] **Registry:** Prevent promoting failed runs.

## 5. CI/CD
- [x] **Tests:** Backend coverage restored.
- [x] **Frontend:** "Promote Best" workflow tested.

## Usage

To start a training run:
1. Ensure `golden-v1` (or your target dataset) is seeded: `python -m backend.tools.seed_golden_dataset`
2. Select `answer_correctness` as the metric.
3. Verify results in Leaderboard.
