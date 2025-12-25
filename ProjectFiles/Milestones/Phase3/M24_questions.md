# M24 Clarifying Questions

1.  **Dependency Management (`uv`):** `uv` is not currently installed in the environment.
    *   Should I install `uv`?
    *   If yes, should I strictly use it to generate/check `uv.lock` (preserving `pip` for other workflows), or should I fully migrate the project to use `uv` for development (e.g., `uv sync`, `uv run`)?

2.  **Training Reality:** `training/train_sft_tunix.py` is currently a simulation script (creates fake metrics/checkpoints).
    *   For Phase 3 ("Micro-train"), do you want **real** JAX/Flax training logic implemented now (e.g., fine-tuning GPT-2), or should the *training* step remain simulated/stubbed while we focus on making the **inference** (`generate_predictions`) real?
    *   *Context:* The M24 title highlights "Real Inference", but Phase 3 mentions "Trained Model". If training is stubbed, the "Trained Model" won't actually learn anything, but the pipeline will be proven.

3.  **Model Selection:** For Phase 2 (Real Inference), which model should be used as the default/baseline for smoke tests and CI?
    *   Recommendation: `gpt2` or `distilgpt2` (Hugging Face) for speed/CPU compatibility.
