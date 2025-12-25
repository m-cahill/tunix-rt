# M28: Tuning, Comparison, and Leaderboard

## 1. Hyperparameter Tuning (Ray Tune)

M28 enables hyperparameter sweeps using Ray Tune. The infrastructure supports `choice`, `uniform`, `loguniform`, and `randint` search spaces.

### Running a Sweep Locally

You can trigger a sweep using the API or the provided helper script.

**Prerequisites:**
- Backend running (`uvicorn ...`)
- `golden-v2` dataset seeded (`uv run python backend/tools/seed_golden_v2.py`)

**Using the Helper Script:**
```bash
cd backend
uv run python ../scripts/run_m28_sweep.py
```

**Using the API:**
POST `/api/tuning/jobs` with payload:
```json
{
  "name": "My Sweep",
  "dataset_key": "golden-v2",
  "base_model_id": "google/gemma-2b-it",
  "metric_name": "answer_correctness",
  "metric_mode": "max",
  "num_samples": 3,
  "max_concurrent_trials": 1,
  "search_space": {
    "learning_rate": { "type": "loguniform", "min": 1e-5, "max": 1e-4 },
    "batch_size": { "type": "choice", "values": [2, 4] }
  }
}
```
Then POST `/api/tuning/jobs/{id}/start`.

## 2. Run Comparison UI

You can compare two runs side-by-side to analyze performance differences.

### How to Compare
1. Go to the "Home" tab and expand "Run History".
2. Select two runs using the checkboxes on the left.
3. Click the **Compare** button that appears.

### Deep Linking
You can share a comparison view using the URL:
`http://localhost:5173/?runA={uuid}&runB={uuid}`

## 3. Evaluation Score

The Leaderboard and Tuning metrics use `answer_correctness` as the primary score.

- **Definition:** Mean of correctness (exact match, normalized) across the evaluation set.
- **Range:** 0.0 to 100.0 (displayed as percentage).
- **Source:** `AnswerCorrectnessJudge` compares model output against ground truth in `golden-v2` manifest.
