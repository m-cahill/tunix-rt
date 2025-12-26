# Tunix RT - Video Script (M31)

**Target Duration:** 2:30 - 2:45 (safe buffer under 3:00 limit)  
**Format:** Screen recording with voiceover  
**Style:** Technical demo with clear narration

---

## Script (~150 words, ~2:30 at conversational pace)

### Opening (0:00 - 0:15)

> "Hi, I'm Michael, and this is Tunix RT — a trace-first training framework for reasoning models.
>
> The goal: train language models to show their work, step by step."

*[Show: Project README or title slide]*

---

### What is "Trace-First"? (0:15 - 0:45)

> "Traditional fine-tuning uses input-output pairs. Trace-first training captures the *reasoning process* between question and answer.
>
> Here's an example trace from our dataset:"

*[Show: One trace from golden-v2 dataset, highlight the reasoning steps]*

> "Each trace has a prompt, one or more reasoning steps, and a final answer. This teaches the model *how* to think, not just *what* to output."

---

### Reproducibility (0:45 - 1:15)

> "Reproducibility is critical for competition submissions. Every run in Tunix RT is deterministic.
>
> We use fixed seeds, versioned datasets with manifests, and pinned dependencies. The artifact bundle captures everything needed to recreate results."

*[Show: manifest.json, submission_freeze.md briefly]*

> "You can verify the exact commit, dataset version, and training config used for any run."

---

### Running the Notebook (1:15 - 2:00)

> "Let me show you the Kaggle notebook in action."

*[Show: Open kaggle_submission.ipynb]*

> "First, a smoke run validates the pipeline works — just two training steps."

*[Show: Smoke run cell executing, success message]*

> "For the full run, we train on the golden-v2 dataset with 100 traces. The notebook handles training, prediction generation, and evaluation automatically."

*[Show: Full run configuration cell]*

> "At the end, we get a submission summary with the model, dataset, training metrics, and evaluation score."

*[Show: Submission Summary output]*

---

### Results (2:00 - 2:30)

> "Our model achieves an answer correctness score of [X] on the evaluation set.
>
> The key takeaway: by training on reasoning traces instead of just answers, we get a model that shows its work — exactly what the Tunix Hackathon is about.
>
> All code is available in the repository. Thanks for watching!"

*[Show: Final results, fade to repository link]*

---

## Notes for Recording

1. **Pacing:** Aim for ~100 words per minute. Pause between sections.
2. **Editing:** Plan for ~15% cuts. Record 3:00 of content for 2:30 final.
3. **Visuals:** Keep code visible for at least 3 seconds before moving on.
4. **Audio:** Use a quiet environment. Consider noise reduction in post.
5. **Resolution:** Record at 1080p minimum. Kaggle requires clear, readable code.

---

## Time Budget

| Section | Start | Duration | Words |
|---------|-------|----------|-------|
| Opening | 0:00 | 0:15 | ~25 |
| Trace-First | 0:15 | 0:30 | ~50 |
| Reproducibility | 0:45 | 0:30 | ~50 |
| Notebook Demo | 1:15 | 0:45 | ~75 |
| Results | 2:00 | 0:30 | ~40 |
| **Total** | — | **2:30** | **~240** |

---

## Key Beats to Hit

1. ✅ What the project is
2. ✅ What "trace-first" means (show one trace)
3. ✅ How reproducibility works (seed + manifest + bundle)
4. ✅ Quick results (eval score + what it demonstrates)
5. ✅ How to run the notebook
