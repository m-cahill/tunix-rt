# M31 Clarifying Questions

Before finalizing the todo list and beginning implementation, I need clarification on the following:

---

## 1. Missing Eval Set: `golden_eval.jsonl`

The `kaggle_submission.py` and `docs/kaggle_submission.md` both reference `training/evalsets/golden_eval.jsonl`, but this file **does not exist**. Only `training/evalsets/eval_v1.jsonl` is present (25 evaluation items).

**Question:** Should I:
- (A) Rename/copy `eval_v1.jsonl` to `golden_eval.jsonl`?
- (B) Create a new `golden_eval.jsonl` tailored to the golden-v2 dataset?
- (C) Update references to use `eval_v1.jsonl` throughout?

---

## 2. Kaggle Notebook Execution Issues

The current `kaggle_submission.ipynb` has shell commands using Python f-string variables that **will not work** in Jupyter:

```python
!python training/train_jax.py --dataset {DATASET} --model_name {MODEL_NAME} ...
```

In Jupyter, `{}` in shell commands don't interpolate Python variables. They need `$variable` syntax or explicit cell magic.

**Question:** Should I refactor the notebook to:
- (A) Use `subprocess.run()` calls instead of shell commands (more robust)?
- (B) Use `%run` magic with arguments?
- (C) Create a thin wrapper that imports and calls `kaggle_submission.py` directly?

---

## 3. Project Name in Archive

The M31 plan specifies: `submission/tracehammer_m31_<YYYY-MM-DD>_<shortsha>.zip`

But the project is named **tunix-rt** throughout the codebase.

**Question:** Should the archive prefix be:
- (A) `tracehammer_m31_...` (as specified in the plan)?
- (B) `tunix_rt_m31_...` (matching the actual project name)?
- (C) Something else?

---

## 4. Model Selection for Final Submission

The `sft_tiny.yaml` config uses `distilgpt2` (a tiny model for smoke testing), while the competition requires **Gemma 2 2B** or **Gemma 3 1B**.

The existing `train_golden_v2.yaml` also uses `distilgpt2`:
```yaml
model:
  model_id: "distilgpt2"
```

**Question:** For the final submission notebook and packaging:
- (A) Should I update `train_golden_v2.yaml` to use `google/gemma-2-2b`?
- (B) Create a new config specifically for submission (e.g., `submission_config.yaml`)?
- (C) Leave as-is and let the user override via CLI args in the notebook?

---

## 5. Video Script Focus

The M31 plan mentions showing "what 'trace-first' means (show one trace)".

**Question:** Is there a specific example trace you'd like highlighted in the video script? Should I:
- (A) Pick a representative trace from `golden-v2`?
- (B) Create a curated "showcase" trace specifically for the video?
- (C) Leave it generic with placeholders for you to fill in?

---

## 6. Dataset for Final Run

The plan mentions both `dev-reasoning-v1` (200 traces) and `golden-v2` (100 traces).

**Question:** For the "submission freeze" documentation, which dataset should be designated as the **canonical submission dataset**?
- (A) `golden-v2` (seems to be the calibration set)
- (B) `dev-reasoning-v1` (larger, for development)
- (C) Both should be supported with clear guidance on when to use each?

---

## 7. Submission Folder Location

The plan suggests `./submission/` for the packaging tool output.

**Question:** Is the root of the repo (`D:\Coding\tunix-rt\submission\`) the correct location, or should it be inside `backend/` or elsewhere?

---

Once you provide answers to these questions, I'll finalize the implementation plan and begin work on M31.
