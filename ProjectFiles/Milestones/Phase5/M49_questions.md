# M49 Clarifying Questions — Observer Model for Error Detection

**Context:** M49 tests whether error detection works better as a separate observation task, validating M48's finding that generation lacks a state-difference operator.

---

## Q1: Observer Model Architecture

The plan lists 3 allowed approaches:
1. Logistic regression over embeddings
2. Small frozen LM + classification head
3. Linear probe on hidden states

**Question:** Which approach should I implement?

**Options:**
1. **Option 1 only** — Logistic regression (simplest, most interpretable)
2. **Option 2 only** — Frozen LM + head (better signal, still lightweight)
3. **Both 1 and 2** — Compare approaches for robustness
4. **Start with 1, escalate to 2 if needed** — Progressive complexity

**My Recommendation:** Option 4 (Start with logistic regression, escalate if needed). Logistic regression is maximally interpretable and proves the architectural separation claim. If it fails, a frozen LM + head would show if more capacity helps.

---

## Q2: Embedding Source

If using embeddings (for logistic regression or classification head):

**Question:** Which model should provide the embeddings?

**Options:**
1. **Existing Gemma checkpoint** — Use the M46/M47 fine-tuned model
2. **Base Gemma** — Use untuned google/gemma-2b
3. **Sentence-Transformers** — Use a general-purpose encoder (e.g., all-MiniLM-L6-v2)
4. **TF-IDF + n-grams** — Non-neural baseline

**My Recommendation:** Option 3 (Sentence-Transformers). Using the fine-tuned Gemma would conflate generator knowledge with observer capability. A general-purpose encoder provides clean separation and is lightweight.

---

## Q3: Dataset Composition

Available data:
- M47 training data: 307 traces (21 with errors, 286 clean)
- M47 holdout: 34 traces (all with errors injected)
- M46 holdout: 34 traces (clean)

**Question:** What data should I use for observer training?

**Options:**
1. **Holdout only** — 68 total samples (34 error + 34 clean), minimal but clean
2. **Training data only** — 307 samples (21 error + 286 clean), imbalanced
3. **Combined** — All data (~375 samples)
4. **Holdout + balanced subset of training** — ~100 samples with 50/50 split

**My Recommendation:** Option 4 (Holdout + balanced subset). This gives enough data for meaningful training while avoiding extreme imbalance. The holdout provides clean error/clean pairs.

---

## Q4: Class Balance

With natural distribution (~7% errors), the observer might learn to always predict "no error".

**Question:** How should I handle class imbalance?

**Options:**
1. **Natural distribution** — Let the observer learn the prior
2. **Balanced sampling** — 50/50 error/clean
3. **Weighted loss** — Use class weights to penalize misses
4. **Oversample errors** — Duplicate error examples

**My Recommendation:** Option 2 (Balanced sampling). For a demonstration milestone, we want to show the observer CAN detect errors, not that it calibrates to the prior. Balanced sampling is cleaner.

---

## Q5: Localization Task

The plan mentions optional `suspected_step_index` prediction.

**Question:** Should I include step-level localization?

**Options:**
1. **Binary only** — Just error_present (0/1)
2. **Binary + localization** — Also predict which step has the error
3. **Binary first, localization if successful** — Progressive complexity

**My Recommendation:** Option 1 (Binary only). The core hypothesis is about detection capability, not localization. Adding localization complicates the experiment without strengthening the main claim.

---

## Q6: Success Threshold

**Question:** What level of performance constitutes "success" for M49?

**Options:**
1. **Any improvement over random** — AUC > 0.5, accuracy > 50%
2. **Moderate signal** — AUC > 0.7, accuracy > 70%
3. **Strong signal** — AUC > 0.85, accuracy > 80%
4. **No threshold** — Report results honestly regardless of level

**My Recommendation:** Option 2 (Moderate signal). AUC > 0.7 would strongly support the architectural separation hypothesis. Even AUC = 0.6 would be meaningful given the generator's 0% detection rate.

---

## Q7: Contrastive Demonstration Format

The plan emphasizes "side-by-side examples" as the "money shot".

**Question:** How many contrastive examples should I produce?

**Options:**
1. **All test cases** — Full table of generator vs observer
2. **Representative sample (5-7)** — Curated examples
3. **Error cases only** — Show observer success on error traces
4. **Error + false positive analysis** — Show both successes and failures

**My Recommendation:** Option 4 (Error + false positive analysis). This shows both that the observer detects real errors AND where it makes mistakes, which is more honest and informative.

---

# Summary of Recommendations

| Question | My Recommendation |
|----------|-------------------|
| Q1: Model architecture | Start with logistic regression, escalate if needed |
| Q2: Embedding source | Sentence-Transformers (clean separation) |
| Q3: Dataset composition | Holdout + balanced subset (~100 samples) |
| Q4: Class balance | Balanced sampling (50/50) |
| Q5: Localization | Binary only (error_present) |
| Q6: Success threshold | Moderate (AUC > 0.7) |
| Q7: Contrastive format | Error + false positive analysis |

---

**Awaiting confirmation before proceeding.**

