# M48 Clarifying Questions — Reasoning Failure Topology

**Context:** M48 is an **analysis milestone** to map failure modes of verification behavior. Deliverable is understanding, not improvement.

---

## Q1: Taxonomy Refinement Scope

The plan provides 6 seed failure classes:
1. Ritual Verification
2. Computation-from-Scratch Reset
3. Local Error Blindness
4. Error Detection without Localization
5. Correction Hallucination
6. Verification Collapse

**Question:** How should I handle taxonomy evolution?

**Options:**
1. **Strict adherence** — Use exactly these 6 classes, no additions
2. **Refinement allowed** — May merge/split/relabel based on observed data, but keep similar scope
3. **Discovery mode** — Start from scratch, let classes emerge from data

**My Recommendation:** Option 2 (Refinement allowed). The seed classes are well-designed, but I may find that some are empty or some need subdivision based on actual M47 traces.

---

## Q2: Input Trace Selection

The plan lists multiple input sources:
- M45 Stage-C predictions
- M46 self-correct predictions  
- M47 error-injected predictions
- Error manifests from M47

**Question:** Which traces should be the primary focus for classification?

**Options:**
1. **M47 only** — Focus on error-injected predictions (68 traces: 34 holdout_error + 34 holdout_clean)
2. **M46 + M47** — Include M46 as behavioral baseline
3. **All three (M45/M46/M47)** — Full longitudinal view

**My Recommendation:** Option 2. M47 predictions are the primary failure analysis target, but M46 predictions provide the behavioral baseline (what verification looks like without errors). M45 is less relevant since it predates VERIFY/CORRECT structure.

---

## Q3: Classification Heuristic Complexity

The plan allows "rule-based (regex + structural checks)".

**Question:** What level of sophistication is appropriate?

**Options:**
1. **Pure regex** — Pattern matching on keywords only
2. **Structural + regex** — Check for VERIFY/CORRECT block presence, content patterns, step references
3. **Lightweight semantic** — Use simple NLP (sentence similarity, keyword extraction) in addition to rules

**My Recommendation:** Option 2 (Structural + regex). This matches the evidence-based approach without introducing model dependencies. Pure regex is too shallow for distinguishing "Ritual Verification" from "Verification Collapse".

---

## Q4: Contrastive Analysis Depth

M47 injected 21 errors. The plan asks for contrastive pair analysis.

**Question:** How many contrastive examples should the analysis cover?

**Options:**
1. **All 21** — Complete coverage of injected errors
2. **Representative sample (5-7)** — One per failure class, carefully curated
3. **Stratified selection (10-12)** — Cover all classes but skip redundant cases

**My Recommendation:** Option 2 (Representative sample). The goal is to illustrate failure modes, not exhaustively enumerate them. 5-7 well-chosen examples with detailed annotation is more valuable than 21 repetitive entries.

---

## Q5: Reasoning Graph Format

The plan allows ASCII, Mermaid, or annotated Markdown.

**Question:** Which format should I use for the reasoning graph sketch?

**Options:**
1. **ASCII** — Portable, no rendering dependencies
2. **Mermaid** — Renders in GitHub, cleaner visuals
3. **Both** — ASCII in analysis.md, Mermaid in separate file

**My Recommendation:** Option 2 (Mermaid). GitHub renders Mermaid natively, and it's more readable than ASCII for showing structural disconnection.

---

## Q6: Success Threshold

The stop conditions mention "failure modes collapse into one trivial class" or "heuristics cannot distinguish behaviors meaningfully."

**Question:** What is the positive success threshold?

**Options:**
1. **At least 3 distinct failure classes** with non-trivial membership (>5% each)
2. **At least 4 distinct failure classes** with meaningful differentiation
3. **No minimum** — Report whatever emerges, even if 2 classes dominate

**My Recommendation:** Option 3 (No minimum). This is a discovery milestone. If verification truly collapses to one or two patterns, that's a finding worth reporting. Forcing diversity would be dishonest.

---

## Q7: Cross-Model Comparison

M47 produced predictions from 3 checkpoints:
- M46 self_correct
- M47 clean
- M47 error_aware

**Question:** Should the failure analysis compare across these models?

**Options:**
1. **Single model** — Focus on M47 error_aware only (the experimental condition)
2. **All three** — Compare failure distributions across models
3. **Two-way** — M47 error_aware vs M46 self_correct (experimental vs baseline)

**My Recommendation:** Option 3 (Two-way). This shows whether error-aware training changed the *type* of failures, not just the *rate*.

---

# Summary of Recommendations

| Question | My Recommendation |
|----------|-------------------|
| Q1: Taxonomy | Refinement allowed (Option 2) |
| Q2: Input traces | M46 + M47 (Option 2) |
| Q3: Heuristics | Structural + regex (Option 2) |
| Q4: Contrastive depth | Representative sample of 5-7 (Option 2) |
| Q5: Graph format | Mermaid (Option 2) |
| Q6: Success threshold | No minimum (Option 3) |
| Q7: Cross-model | Two-way comparison (Option 3) |

---

**Awaiting confirmation before proceeding.**

