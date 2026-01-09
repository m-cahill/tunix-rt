# M44 Clarifying Questions

**Date:** 2026-01-08  
**Status:** Awaiting Answers

---

## Context from M43

In M43, we successfully ran:
- **Model:** `google/gemma-2b` (base, not instruction-tuned)
- **Steps:** 138 (1 epoch on 550 samples)
- **Loss:** 2.21 → 0.76
- **Runtime:** 73.2 seconds
- **Evaluation:** Model learned training patterns but not Q&A behavior

---

## Questions

### Q1: Training Duration

M44 plan specifies **300-500 steps** (or 1 epoch if epoch-based).

With `dev-reasoning-v2` (550 samples) and batch_size=1, grad_accum=4:
- 1 epoch = 138 steps
- 300 steps ≈ 2.2 epochs
- 500 steps ≈ 3.6 epochs

**Options:**
- **A) 2 epochs** (~276 steps, ~2.5 minutes)
- **B) 3 epochs** (~414 steps, ~4 minutes)
- **C) Stick with 1 epoch** (same as M43, for consistency)

Which do you prefer?

---

### Q2: Model Variant

M43 used `google/gemma-2b` (base). The plan says "Gemma 2B (PyTorch weights)" but doesn't specify variant.

**Options:**
- **A) Base model** (`google/gemma-2b`) — Same as M43, consistent
- **B) Instruction-tuned** (`google/gemma-2b-it`) — Better Q&A behavior expected

Note: Instruction-tuned might require different prompt formatting.

**Recommendation:** Stick with base model for consistency with M43 evidence trail. The goal is GPU validation, not model quality.

---

### Q3: HuggingFace Token

In M43, we set `$env:HF_TOKEN` manually in the session.

Should I:
- **A) Assume it persists** (may not, depending on shell session)
- **B) Re-set it at start** (you'll provide the token again)
- **C) Use `huggingface-cli login`** (persistent storage)

---

### Q4: Evaluation Scoring

M43 generated predictions but didn't score them (just observed pattern learning).

For M44, should I:
- **A) Just generate predictions** (same as M43)
- **B) Run scoring** (compare to expected answers, compute accuracy)

Note: Scoring requires the `compute_scorecard()` function from the backend, which would need the backend environment activated.

---

## Default Answers (If No Response)

If you want me to proceed immediately, I'll use:
- **Q1:** 2 epochs (~276 steps)
- **Q2:** Base model (`google/gemma-2b`)
- **Q3:** Re-set token (you provide it)
- **Q4:** Just generate predictions (consistency with M43)

---

**Awaiting your answers before proceeding.**
