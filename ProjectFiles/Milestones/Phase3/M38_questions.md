# M38 Clarifying Questions

Based on my analysis of the M38 plan and project state, I have the following questions before proceeding:

---

## 1. Kaggle TPU Execution — Who Does What?

The M38 plan centers on executing a real TPU training run on Kaggle. Since Kaggle notebooks must be run interactively on kaggle.com:

- **Question:** Will **you** execute the notebook on Kaggle and provide me with the output logs and results?
- **Follow-up:** Do you already have:
  - Kaggle TPU quota available?
  - HF_TOKEN set up in Kaggle Secrets?
  - Gemma license accepted at https://huggingface.co/google/gemma-2b?

My role would then be:
- Verify notebook is ready for execution
- Prepare evidence folder structure
- Update evidence files once you provide real run data
- Handle frontend coverage work in parallel

---

## 2. Evidence Folder Versioning

The M38 plan says to populate `submission_runs/m37_v1/`. However:

- M37 is the milestone that *created* the infrastructure
- M38 is the milestone that *executes* the real run

**Question:** Should I:
- **Option A:** Update the existing `m37_v1/` folder with real values (treat M37+M38 as a single evidence unit)
- **Option B:** Create a new `m38_v1/` folder with real run data (cleaner separation)

The current `m37_v1/` contains templates with `null` values — ready to be populated.

---

## 3. Frontend Coverage — Target Components

The M38 plan says "identify lowest-coverage React components" as optional work.

Current frontend components:
| Component | Has Tests? |
|-----------|-----------|
| `App.tsx` | ✅ `App.test.tsx` |
| `Leaderboard.tsx` | ✅ `Leaderboard.test.tsx` |
| `LiveLogs.tsx` | ✅ `LiveLogs.test.tsx` |
| `ModelRegistry.tsx` | ✅ `ModelRegistry.test.tsx` |
| `RunComparison.tsx` | ✅ `RunComparison.test.tsx` |
| `Tuning.tsx` | ✅ `Tuning.test.tsx` |
| `api/client.ts` | ✅ `client.test.ts` |

All components appear to have test files. 

**Question:** Should I:
- **Option A:** Run the existing tests to generate a coverage report and identify gaps?
- **Option B:** Skip frontend coverage (all components already have test files)?
- **Option C:** Add specific test scenarios you know are missing?

---

## 4. Per-Item Predictions — Scope Clarity

The M38 plan mentions two things about per-item predictions:

> **In Scope:** "Optional: persist per-item predictions if trivial"

> **M37 Summary:** "Per-item artifact storage — persist predictions.jsonl to database"

But the current design already saves `predictions.jsonl` to the output folder.

**Question:** Does "persist per-item predictions" mean:
- **Option A:** Save to filesystem (already done)
- **Option B:** Save to database (new feature, likely not trivial)
- **Option C:** Skip entirely for M38

---

## 5. Timeline & Prioritization

Given that M38 is primarily about execution + evidence:

**Question:** What's your preferred order of operations?
- **Option A (Serial):** I prepare everything, you execute TPU run, then I update evidence
- **Option B (Parallel):** While you set up/run Kaggle, I work on frontend coverage
- **Option C (Evidence-first):** Focus only on execution — skip frontend coverage entirely

---

## Summary

Once you answer these questions, I'll finalize my task list and begin work. The critical path is:

1. ✅ Notebook is ready for TPU execution (verified by M37)
2. ⏳ You execute on Kaggle TPU v3-8
3. 📝 I populate evidence files with real values
4. 🔧 (Optional) Frontend coverage uplift
5. ✅ CI green, guardrails verified

Looking forward to your responses!
