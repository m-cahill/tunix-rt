# M48 Reasoning Graph — Structural Disconnection

**Purpose:** Visualize where verification *should* attach to reasoning but doesn't.

---

## The Expected vs. Actual Verification Model

### Expected: Verification with State Comparison

```mermaid
flowchart TB
    subgraph Reasoning["Reasoning Chain"]
        P[Prompt] --> S1[Step 1: Parse problem]
        S1 --> S2[Step 2: Compute intermediate]
        S2 --> S3[Step 3: Compute final]
        S3 --> A[Final Answer]
    end
    
    subgraph Verification["Expected Verification"]
        V1[VERIFY: Compare S3 to inverse check]
        V2[State Diff: actual vs expected]
        V3[CORRECT: Localize if mismatch]
    end
    
    S3 -.-> V1
    V1 --> V2
    V2 --> V3
    V3 -.-> S2
    
    style V2 fill:#f9f,stroke:#333,stroke-width:2px
    style V1 fill:#9f9,stroke:#333,stroke-width:2px
    style V3 fill:#9f9,stroke:#333,stroke-width:2px
```

**Key Element:** The `State Diff` node compares computed values to expected values.

---

### Actual: Verification as Post-Hoc Ritual

```mermaid
flowchart TB
    subgraph Reasoning["Reasoning Chain"]
        P[Prompt] --> S1[Step 1: Parse problem]
        S1 --> S2[Step 2: Compute intermediate]
        S2 --> S3[Step 3: Compute final]
        S3 --> A[Final Answer]
    end
    
    subgraph Verification["Actual Verification (Ritual)"]
        V1[VERIFY: Template text]
        V3[CORRECT: No correction needed]
    end
    
    A --> V1
    V1 --> V3
    
    style V1 fill:#fbb,stroke:#333,stroke-width:2px
    style V3 fill:#fbb,stroke:#333,stroke-width:2px
```

**Critical Observation:**  
- VERIFY attaches to the final answer, not to intermediate steps
- No state comparison occurs
- CORRECT defaults to "No correction needed"

---

## Where Comparison Should Occur But Doesn't

```mermaid
flowchart LR
    subgraph ErrorInjected["Error-Injected Trace"]
        I1[Step 2: 60 + 80 = 150] 
        I2[Step 3: 150 + 8 = 158]
        I3[Answer: 158]
    end
    
    subgraph CorrectTrace["Correct Computation"]
        C1[Step 2: 60 + 80 = 140]
        C2[Step 3: 140 + 8 = 148]
        C3[Answer: 148]
    end
    
    subgraph Missing["Missing Operation"]
        D[DIFF: 150 ≠ 140]
    end
    
    I1 -.->|should compare| D
    C1 -.->|should compare| D
    
    style D fill:#ff9,stroke:#f00,stroke-width:3px,stroke-dasharray: 5 5
    style I1 fill:#fbb
    style C1 fill:#9f9
```

**The Gap:**  
There is no mechanism for the model to compare `150` against `140`.  
It would need to either:
1. Re-compute and compare (expensive)
2. Have an internal consistency constraint (not learned)

---

## Failure Topology Schematic

```mermaid
flowchart TD
    subgraph Input["Input Layer"]
        P[Prompt with embedded error]
    end
    
    subgraph Processing["Reasoning Layer"]
        R1[Fresh computation from scratch]
        R2[Produces new answer]
    end
    
    subgraph Verification["Verification Layer"]
        V[Template: 'Check by inverse...']
        C[Default: 'No correction needed']
    end
    
    subgraph Output["Output Layer"]
        O[Final response with VERIFY/CORRECT]
    end
    
    P --> R1
    R1 --> R2
    R2 --> V
    V --> C
    C --> O
    
    P -.->|error should propagate| V
    R2 -.->|comparison should occur| V
    
    style V fill:#fbb,stroke:#333
    style C fill:#fbb,stroke:#333
```

**Key Insight:**  
The error-injected prompt flows through reasoning, but verification operates independently. There is no feedback loop from verification back to reasoning.

---

## Why This Topology Produces Ritual Verification

1. **Training Signal**: Verification text appears after reasoning in 100% of training data
2. **Position Encoding**: VERIFY/CORRECT are learned as sequence completions, not inspections
3. **No Contrastive Examples**: Model never sees (error, correction) paired with (clean, no-correction)
4. **Template Dominance**: 93% of training had "No correction needed" → becomes default

The model learns:  
> "After reasoning, emit VERIFY template, then CORRECT: No correction needed"

It does NOT learn:  
> "Compare my computation to an expected value and report discrepancies"

---

## Implications for Future Work

To achieve functional verification, the model would need:

1. **State-Comparison Training**: Explicit examples of (before, after, diff)
2. **Contrastive Pairs**: Same problem with/without errors
3. **Verification Grounding**: VERIFY must reference specific values, not templates
4. **Error Localization**: CORRECT must identify which step is wrong

This is the structural gap that M47 failed to bridge and that M48 has now characterized.

