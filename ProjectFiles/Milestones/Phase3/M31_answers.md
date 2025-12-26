Here are my answers / preferences for the **M31 clarifying questions**, with concrete choices you can implement immediately.

---

## 1) Missing eval set: `golden_eval.jsonl`

**Pick: (C) Update references to use `training/evalsets/eval_v1.jsonl` everywhere.**

Rationale: it already exists, it’s small/curated, and it keeps us honest about what’s actually in the repo. If we later want an eval set explicitly derived from `golden-v2`, we can add it as a separate milestone without creating naming confusion now.

---

## 2) Kaggle notebook execution (`!python ... {DATASET}`)

Your concern is understandable, but **`!` commands in Jupyter/IPython *do* support Python variable substitution via `{var}` and `$var`**. So the pattern:

```python
DATASET = "golden-v2"
!python training/train_jax.py --dataset {DATASET}
```

is valid in IPython-based notebooks. ([IPython Documentation][1])

**Pick: (C) Create a thin wrapper that imports/calls `kaggle_submission.py` directly** (best “enterprise-grade” move), and keep `!python ... {VAR}` as a fallback for readability.

Concretely:

* Notebook becomes: set variables → call `kaggle_submission.main([...])` (or an exported `run()` function).
* If anything about import paths gets weird in Kaggle, fallback to `subprocess.run([...])`.

---

## 3) Project name in archive

**Pick: (B) `tunix_rt_m31_<YYYY-MM-DD>_<shortsha>.zip`**

Rationale: matches the repo/project name everywhere else; reduces ambiguity when you share artifacts.

---

## 4) Model selection for final submission (Gemma requirement)

The hackathon requires starting from **Gemma2 2B or Gemma3 1B**. ([Kaggle][2])

**Pick: (B) Create a new submission-specific config** (don’t overwrite your tiny/smoke configs).

Suggested:

* `training/configs/submission_gemma3_1b.yaml` using `google/gemma-3-1b-it` (or the base variant if you prefer). ([Hugging Face][3])
* Optionally also provide `training/configs/submission_gemma2_2b.yaml` using `google/gemma-2-2b`. ([Hugging Face][4])

Also note: Gemma weights on HF/Kaggle can require accepting a license / access gate, so documenting that in the submission doc is worth it. ([Hugging Face][5])

---

## 5) Video script focus

**Pick: (A) Use a representative trace from `golden-v2`** (don’t invent one).

If you want maximum control, you *can* add a “showcase” trace later, but for M31: pick a real one from the canonical set and walk through:

* prompt → steps → answer → eval result → how trace artifacts are stored/viewed.

(Separately: Kaggle hackathons commonly want a short video; I’ve seen explicit “under ~3 minutes / upload to Media Gallery / publish to YouTube” language on Kaggle pages for similar events, but if you want this perfectly exact for Tunix, we should treat the competition page as the source of truth.) ([Kaggle][2])

---

## 6) Dataset for “submission freeze”

**Pick: (C) Support both, with clear guidance:**

* **Canonical submission dataset:** `golden-v2` (the “known good / stable” set)
* **Dev / scale / sanity dataset:** `dev-reasoning-v1`

In the “submission freeze” doc, explicitly state:

* what dataset the packaged run used
* dataset hash/manifest version
* trace count

---

## 7) Submission folder location

**Pick: repo root `./submission/`**

Rationale: it’s packaging-oriented (zip, checklist, notebook/script, config snapshot) and avoids mixing with backend runtime code. Keep it in `.gitignore` *except* for templates/docs you want versioned.

---

If you want, I can also turn these decisions into a **Cursor-ready M31 prompt** with exact file edits (notebook refactor steps, new configs, and the packaging script behavior).

[1]: https://ipython.readthedocs.io/en/stable/interactive/tutorial.html?utm_source=chatgpt.com "Introducing IPython — IPython 9.8.0 documentation"
[2]: https://www.kaggle.com/competitions/google-tunix-hackathon?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
[3]: https://huggingface.co/google/gemma-3-1b-it?utm_source=chatgpt.com "google/gemma-3-1b-it"
[4]: https://huggingface.co/google/gemma-2-2b?utm_source=chatgpt.com "google/gemma-2-2b"
[5]: https://huggingface.co/google/gemma-2-2b/tree/main?utm_source=chatgpt.com "google/gemma-2-2b at main"
