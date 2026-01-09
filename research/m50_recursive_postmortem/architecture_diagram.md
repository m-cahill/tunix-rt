# Phase 5 Architecture Diagram — Reasoning System with Observer

This document provides a conceptual architecture diagram explaining:
1. Why self-correction fails in the generator
2. Where the observer succeeds
3. The architectural separation that enables error detection

---

## System Architecture Overview

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        Q[Problem/Prompt]
    end
    
    subgraph Generator["Generator (Autoregressive LM)"]
        direction TB
        R1[Step 1: Parse problem]
        R2[Step 2: Compute intermediate]
        R3[Step 3: Compute final]
        V[VERIFY: Template text]
        C[CORRECT: No correction needed]
        A[Final Answer]
        
        R1 --> R2 --> R3 --> V --> C --> A
    end
    
    subgraph Observer["Observer (External Classifier)"]
        direction TB
        E1[Extract generated answer]
        E2[Compare to expected]
        D[Detect mismatch?]
        O[Error signal]
        
        E1 --> E2 --> D --> O
    end
    
    Q --> R1
    A --> E1
    
    style V fill:#fbb,stroke:#333
    style C fill:#fbb,stroke:#333
    style D fill:#9f9,stroke:#333
    style O fill:#9f9,stroke:#333
```

---

## Annotated Failure Points

### Failure Point 1: Verification as Sequence Completion

```mermaid
flowchart LR
    subgraph Training["What Training Teaches"]
        T1["[reasoning]"] --> T2["VERIFY: [template]"] --> T3["CORRECT: No correction"]
    end
    
    subgraph Reality["What the Model Learns"]
        L1["After reasoning..."] --> L2["emit VERIFY template"] --> L3["emit default CORRECT"]
    end
    
    style T2 fill:#fbb
    style T3 fill:#fbb
    style L2 fill:#fbb
    style L3 fill:#fbb
```

**Annotation:** VERIFY and CORRECT are learned as positional tokens, not as inspection operations. The model learns "what comes next" rather than "what to check."

---

### Failure Point 2: Missing State-Comparison Operator

```mermaid
flowchart TB
    subgraph Expected["Expected Verification"]
        EX1[Computed Value: 648]
        EX2[Expected Value: 649]
        EX3[Compare: 648 ≠ 649]
        EX4[Error Detected!]
        
        EX1 --> EX3
        EX2 --> EX3
        EX3 --> EX4
    end
    
    subgraph Actual["Actual Generator Behavior"]
        AC1[Computed Value: 648]
        AC2["VERIFY: Check by inverse..."]
        AC3["CORRECT: No correction needed"]
        
        AC1 --> AC2 --> AC3
    end
    
    style EX3 fill:#9f9,stroke:#333,stroke-width:2px
    style EX4 fill:#9f9,stroke:#333
    style AC2 fill:#fbb,stroke:#333
    style AC3 fill:#fbb,stroke:#333
```

**Annotation:** The generator has no mechanism to compare its output to an expected value. The VERIFY block references the *type* of check ("inverse") but never *instantiates* it with actual values.

---

### Success Point: Observer as External Comparator

```mermaid
flowchart LR
    subgraph GeneratorOutput["Generator Output"]
        GO[Answer: 648 km]
    end
    
    subgraph GroundTruth["Ground Truth"]
        GT[Expected: 649 km]
    end
    
    subgraph Observer["Observer Model"]
        OB1[Extract numbers]
        OB2[648 ≠ 649]
        OB3[Mismatch detected]
        OB4[Confidence: 72%]
    end
    
    GO --> OB1
    GT --> OB1
    OB1 --> OB2 --> OB3 --> OB4
    
    style OB2 fill:#9f9,stroke:#333,stroke-width:2px
    style OB3 fill:#9f9,stroke:#333
```

**Annotation:** The observer succeeds because it *explicitly compares* two values. This comparison is the state-difference operator that the generator lacks.

---

## Architectural Insight

### Why Generation Cannot Self-Inspect

```mermaid
flowchart TB
    subgraph Autoregressive["Autoregressive Generation"]
        direction LR
        T1["token_1"] --> T2["token_2"] --> T3["..."] --> TN["token_n"]
    end
    
    subgraph Problem["The Problem"]
        P1["Each token depends only on previous tokens"]
        P2["No explicit 'working memory' for values"]
        P3["No comparison operator between states"]
    end
    
    Autoregressive --> Problem
```

**Key Insight:** Autoregressive generation is fundamentally *forward-only*. To verify, the model would need to:
1. Extract specific values from its reasoning
2. Hold them in working memory
3. Compare them to computed or expected values
4. Branch on the result

None of these operations are native to sequence-to-sequence generation.

---

### Why Observation Succeeds

```mermaid
flowchart TB
    subgraph Observer["Observer Architecture"]
        direction TB
        I[Input: generated + expected]
        F[Feature extraction]
        CMP[Comparison features]
        CLF[Classifier]
        OUT[Error/No-error]
        
        I --> F --> CMP --> CLF --> OUT
    end
    
    subgraph Key["Why This Works"]
        K1["Explicit access to both values"]
        K2["Comparison is a built-in operation"]
        K3["Classification, not generation"]
    end
    
    Observer --> Key
    
    style CMP fill:#9f9,stroke:#333
    style K2 fill:#9f9,stroke:#333
```

**Key Insight:** The observer succeeds because comparison is its *primary function*, not a side effect of generation.

---

## Summary Diagram

```mermaid
flowchart LR
    subgraph Phase5["Phase 5 Insight"]
        G["Generator"]
        V["Verification Block"]
        O["Observer"]
        
        G -->|produces| V
        V -->|"always: No correction"| X["0% detection"]
        G -->|output| O
        O -->|compares| Y["50% detection"]
    end
    
    style V fill:#fbb
    style X fill:#fbb
    style O fill:#9f9
    style Y fill:#9f9
```

**Conclusion:** Error detection is a separable function. The generator produces verification *structure* but not verification *function*. An external observer can provide the comparison capability that generation lacks.

---

**Generated:** 2026-01-09  
**Purpose:** Explain architectural separation between generation and observation

