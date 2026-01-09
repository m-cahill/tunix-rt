# M48 Contrastive Pair Analysis

**Generated:** 2026-01-09T02:23:41.295940+00:00

This document shows representative examples of verification failures.
Each example contrasts the injected error with the model's verification behavior.

---

## Example 1: Arithmetic Error

### Injected Error

- **Error Type:** intermediate
- **Error Class:** arithmetic
- **Location:** Step 1
- **Original Value:** `649 km`
- **Injected Value:** `corrupted`
- **Expected Final:** `649 km`
- **Injected Final:** `error-injected`

### Prompt

```
A train travels at 108 km/h for 6 hours. How far does it travel?
```

### Model Output

```
I need to use the formula: distance = speed × time Given: speed = 108 km/h, time = 6 hours Distance = 108 × 6 = 648 km The train travels 648 km VERIFY: Check by inverse: divide distance by time to verify speed CORRECT: No correction needed
648 km
```

### Failure Analysis

✅ VERIFY block present

✅ CORRECT block present

**Key Observation:**

> The VERIFY block uses templated language ("Check by inverse...") but does not reference the specific calculation where the error was injected. Verification operates at a semantic level ("this is a distance problem") without connecting to computational state.

**Where Divergence Should Have Been Noticed:**

> At Step 1, the computation produced `corrupted` instead of `649 km`. A state-comparison operation would detect this mismatch. Instead, verification only references the problem *type* (e.g., "distance formula"), not the actual *values* produced.

---

## Example 2: Arithmetic Error

### Injected Error

- **Error Type:** intermediate
- **Error Class:** arithmetic
- **Location:** Step 1
- **Original Value:** `-21`
- **Injected Value:** `corrupted`
- **Expected Final:** `-21`
- **Injected Final:** `error-injected`

### Prompt

```
Calculate 20 - 41 showing your work.
```

### Model Output

```
Subtracting 41 from 20 Ones digit: handle 0 - 1 Tens digit: handle 2 - 4 Final result: -21 VERIFY: Check by inverse: add result to subtrahend CORRECT: No correction needed
-21
```

### Failure Analysis

✅ VERIFY block present

✅ CORRECT block present

**Key Observation:**

> The VERIFY block uses templated language ("Check by inverse...") but does not reference the specific calculation where the error was injected. Verification operates at a semantic level ("this is a distance problem") without connecting to computational state.

**Where Divergence Should Have Been Noticed:**

> At Step 1, the computation produced `corrupted` instead of `-21`. A state-comparison operation would detect this mismatch. Instead, verification only references the problem *type* (e.g., "distance formula"), not the actual *values* produced.

---

## Example 3: Arithmetic Error

### Injected Error

- **Error Type:** intermediate
- **Error Class:** arithmetic
- **Location:** Step 1
- **Original Value:** `802`
- **Injected Value:** `corrupted`
- **Expected Final:** `802`
- **Injected Final:** `error-injected`

### Prompt

```
Multiply 44 × 18 step by step.
```

### Model Output

```
Breaking down 44 × 18 44 × 8 (ones) = 352 44 × 10 (tens) = 440 Add partials: 352 + 440 = 792 VERIFY: Check by inverse: divide product by one factor CORRECT: No correction needed
792
```

### Failure Analysis

✅ VERIFY block present

✅ CORRECT block present

**Key Observation:**

> The VERIFY block uses templated language ("Check by inverse...") but does not reference the specific calculation where the error was injected. Verification operates at a semantic level ("this is a distance problem") without connecting to computational state.

**Where Divergence Should Have Been Noticed:**

> At Step 1, the computation produced `corrupted` instead of `802`. A state-comparison operation would detect this mismatch. Instead, verification only references the problem *type* (e.g., "distance formula"), not the actual *values* produced.

---

## Example 4: Arithmetic Error

### Injected Error

- **Error Type:** final
- **Error Class:** arithmetic
- **Location:** Step 1
- **Original Value:** `$37.0`
- **Injected Value:** `corrupted`
- **Expected Final:** `$37.0`
- **Injected Final:** `error-injected`

### Prompt

```
If 3 items cost $111, what is the cost per item?
```

### Model Output

```
To find cost per item, divide total cost by number of items Cost per item = total cost ÷ number of items $111 ÷ 3 = $37.0 Each item costs $37.0 VERIFY: Check by inverse: multiply result by count to verify total CORRECT: No correction needed
$37.0
```

### Failure Analysis

✅ VERIFY block present

✅ CORRECT block present

**Key Observation:**

> The VERIFY block uses templated language ("Check by inverse...") but does not reference the specific calculation where the error was injected. Verification operates at a semantic level ("this is a distance problem") without connecting to computational state.

**Where Divergence Should Have Been Noticed:**

> At Step 1, the computation produced `corrupted` instead of `$37.0`. A state-comparison operation would detect this mismatch. Instead, verification only references the problem *type* (e.g., "distance formula"), not the actual *values* produced.

---

## Example 5: Arithmetic Error

### Injected Error

- **Error Type:** final
- **Error Class:** arithmetic
- **Location:** Step 1
- **Original Value:** `👋`
- **Injected Value:** `corrupted`
- **Expected Final:** `👋`
- **Injected Final:** `error-injected`

### Prompt

```
Translate 'hello' to emoji: 👋
```

### Model Output

```
Hello translates to wave emoji 👋 VERIFY: Check: verify answer matches expected format CORRECT: No correction needed
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
👋
```

### Failure Analysis

✅ VERIFY block present

✅ CORRECT block present

**Key Observation:**

> The VERIFY block uses templated language ("Check by inverse...") but does not reference the specific calculation where the error was injected. Verification operates at a semantic level ("this is a distance problem") without connecting to computational state.

**Where Divergence Should Have Been Noticed:**

> At Step 1, the computation produced `corrupted` instead of `👋`. A state-comparison operation would detect this mismatch. Instead, verification only references the problem *type* (e.g., "distance formula"), not the actual *values* produced.

---

## Summary: The Pattern of Failure

Across all contrastive examples, the same pattern emerges:

1. **Verification is templated** — VERIFY blocks use formulaic language from training
2. **No state comparison** — The model never compares current values to prior values
3. **Semantic-level only** — Verification references problem *type* but not *values*
4. **No diff operator** — There is no mechanism to detect "before vs after"

This explains why M47's error-aware training failed: the model learned to produce verification *structure* but not verification *function*.
