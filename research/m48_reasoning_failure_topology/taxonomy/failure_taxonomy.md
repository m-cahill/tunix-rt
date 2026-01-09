# M48 Failure Taxonomy — Reasoning Verification Failures

**Milestone:** M48 Reasoning Failure Topology  
**Version:** 1.0  
**Date:** 2026-01-09

---

## Purpose

This taxonomy classifies **how verification fails** when models attempt self-correction.
The goal is mechanistic understanding, not performance improvement.

---

## Seed Classes (6)

### 1. Ritual Verification

**Definition:**  
VERIFY block is present but contains only templated language with no reference to actual computation or prior reasoning steps.

**Detection Heuristic:**
- VERIFY block exists
- Content matches known templates (e.g., "Check by inverse", "Verify answer format")
- No numbers from the reasoning trace appear in VERIFY
- No step references (e.g., "Step 2", "the calculation")

**Example Pattern:**
```
VERIFY: Check by inverse: divide distance by time to verify speed
CORRECT: No correction needed
```

**Mechanistic Interpretation:**  
Verification operates as a post-hoc appendage without connecting to computational state.

---

### 2. Computation-from-Scratch Reset

**Definition:**  
The model re-solves the problem from the prompt instead of inspecting prior reasoning. Verification becomes redundant recalculation rather than comparison.

**Detection Heuristic:**
- Model output contains a complete reasoning chain
- No reference to "previous" or "earlier" steps
- Verification/correction does not compare two states
- The output is a fresh solve, not an inspection

**Example Pattern:**
```
I need to use the formula: distance = speed x time
Given: speed = 108 km/h, time = 6 hours
Distance = 108 x 6 = 648 km
VERIFY: ...
```
(The model computes from scratch; it never inspects its own prior work.)

**Mechanistic Interpretation:**  
The model lacks a "diff operator" — it cannot compare current state to prior state.

---

### 3. Local Error Blindness

**Definition:**  
The model detects global structure (e.g., "this is a distance problem") but misses specific local errors (e.g., arithmetic miscalculation).

**Detection Heuristic:**
- VERIFY mentions the problem type or formula
- VERIFY does not cite specific numbers
- An arithmetic error exists in the trace but is not mentioned
- CORRECT says "No correction needed" despite error

**Example Pattern:**
```
(Trace contains: 60 + 80 = 150 [should be 140])
VERIFY: Check using the distance formula
CORRECT: No correction needed
```

**Mechanistic Interpretation:**  
Verification operates at semantic/structural level without arithmetic grounding.

---

### 4. Error Detection without Localization

**Definition:**  
The model acknowledges something may be wrong but cannot identify which step or value is incorrect.

**Detection Heuristic:**
- VERIFY or CORRECT contains vague language: "something seems off", "double check", "let me verify"
- No specific step or value is identified
- Correction is either absent or vague

**Example Pattern:**
```
VERIFY: Let me double check the calculation
CORRECT: The answer might need adjustment
```

**Mechanistic Interpretation:**  
The model has weak error salience — it senses anomaly without localization.

---

### 5. Correction Hallucination

**Definition:**  
The model "corrects" something that was not wrong, potentially introducing a new error.

**Detection Heuristic:**
- CORRECT block contains a specific change
- The "correction" targets a value that was originally correct
- The original trace had no error at that location

**Example Pattern:**
```
(Original: 108 x 6 = 648 ← correct)
CORRECT: Step 2 is wrong: 108 x 6 = 658. Fixed.
```

**Mechanistic Interpretation:**  
False positive correction — the model applies correction behavior without proper grounding.

---

### 6. Verification Collapse

**Definition:**  
The VERIFY block degenerates into a restatement of the answer or problem, losing all checking semantics.

**Detection Heuristic:**
- VERIFY block content is semantically identical to final answer
- No comparison language ("check", "verify", "inverse")
- VERIFY is effectively a duplicate of conclusion

**Example Pattern:**
```
Final result: 648 km
VERIFY: The answer is 648 km
CORRECT: No correction needed
```

**Mechanistic Interpretation:**  
Verification has collapsed into echo — structure preserved, function lost.

---

## Classification Rules

1. Each trace receives **one primary class** based on the dominant failure pattern.
2. A **secondary class** may be assigned if two patterns clearly co-occur.
3. Traces with no VERIFY/CORRECT blocks are labeled `no_verification`.
4. Traces where verification correctly identifies an error are labeled `successful_detection`.

---

## Expected Distribution (Hypothesis)

Based on M47 results (0% error detection), we expect:
- **Ritual Verification** to be the dominant class (60-80%)
- **Computation-from-Scratch Reset** to be common (~20-30%)
- **Correction Hallucination** to be rare (<5%)
- **Successful Detection** to be zero or near-zero

---

## Revision Log

| Date | Change |
|------|--------|
| 2026-01-09 | Initial taxonomy with 6 seed classes |

