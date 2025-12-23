# M09 Dataset Format Specification

**Version:** 1.0  
**Last Updated:** December 21, 2025  
**Milestone:** M09 - Reproducible Training Loop v1

---

## Overview

This document specifies the dataset formats used in tunix-rt for training and evaluation. M09 introduces three export formats optimized for different use cases.

---

## Export Formats

### 1. Trace Format (Raw Data)

**Usage:** Analysis, debugging, intermediate processing  
**Endpoint:** `GET /api/datasets/{dataset_key}/export.jsonl?format=trace`

**Structure:**
```jsonl
{
  "id": "uuid",
  "prompts": "original question",
  "trace_steps": ["step1 content", "step2 content", ...],
  "final_answer": "answer",
  "metadata": {
    "created_at": "ISO-8601 timestamp",
    "trace_version": "1.0",
    "source": "ungar|manual|eval",
    ...
  }
}
```

**Example:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "prompts": "What is 15 + 27?",
  "trace_steps": ["Parse the addition problem", "Add 15 and 27"],
  "final_answer": "42",
  "metadata": {
    "created_at": "2025-12-21T10:00:00Z",
    "trace_version": "1.0",
    "source": "ungar"
  }
}
```

---

### 2. Tunix SFT Format (Gemma Template)

**Usage:** Direct Tunix SFT training  
**Endpoint:** `GET /api/datasets/{dataset_key}/export.jsonl?format=tunix_sft`

**Structure:**
```jsonl
{
  "id": "uuid",
  "prompts": "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{reasoning}\n{answer}<end_of_turn>",
  "final_answer": "answer",
  "metadata": {
    "created_at": "ISO-8601 timestamp",
    "format": "tunix_sft",
    ...
  }
}
```

**Gemma Chat Template:**
```
<start_of_turn>user
{user_prompt}<end_of_turn>
<start_of_turn>model
Reasoning:
1. {step1}
2. {step2}
...
Answer: {final_answer}<end_of_turn>
```

**Example:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "prompts": "<start_of_turn>user\nWhat is 15 + 27?<end_of_turn>\n<start_of_turn>model\nReasoning:\n1. Parse the addition problem\n2. Add 15 and 27\nAnswer: 42<end_of_turn>",
  "final_answer": "42",
  "metadata": {
    "created_at": "2025-12-21T10:00:00Z",
    "format": "tunix_sft"
  }
}
```

---

### 3. Training Example Format (M09)

**Usage:** Abstract prompt/response pairs  
**Endpoint:** `GET /api/datasets/{dataset_key}/export.jsonl?format=training_example`

**Structure:**
```jsonl
{
  "id": "uuid",
  "prompt": "instruction + question",
  "response": "reasoning + answer",
  "metadata": {
    "source_trace_id": "original trace uuid",
    "created_at": "ISO-8601 timestamp",
    ...
  }
}
```

**Example:**
```json
{
  "id": "660f9500-f39c-52e5-b827-557766551111",
  "prompt": "What is 15 + 27?\n\nPlease show your reasoning steps.",
  "response": "Reasoning:\n1. Parse the addition problem\n2. Add 15 and 27\nAnswer: 42",
  "metadata": {
    "source_trace_id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2025-12-21T10:00:00Z"
  }
}
```

---

## Dataset Manifest

Every dataset has a manifest at `backend/datasets/{dataset_key}/manifest.json`:

```json
{
  "dataset_key": "name-version",
  "build_id": "uuid",
  "dataset_name": "name",
  "dataset_version": "version",
  "dataset_schema_version": "1.0",
  "created_at": "ISO-8601 timestamp",
  "filters": {"key": "value", ...},
  "selection_strategy": "latest|random",
  "seed": 42,
  "trace_ids": ["uuid1", "uuid2", ...],
  "trace_count": 100,
  "stats": {
    "avg_step_count": 4.2,
    "min_step_count": 1,
    "max_step_count": 10,
    "avg_total_chars": 245.7
  },
  "session_id": null,
  "parent_dataset_id": null,
  "training_run_id": null
}
```

---

## Format Comparison

| Feature | trace | tunix_sft | training_example |
|---------|-------|-----------|------------------|
| **Use Case** | Analysis | Direct SFT | Abstract training |
| **Prompt Field** | Raw question | Gemma-formatted | Instruction + question |
| **Response** | Separate steps | Embedded in prompt | Combined reasoning |
| **Token Ready** | No | Yes (Gemma) | Partial |
| **Deterministic** | Yes | Yes | Yes |

---

## Format Selection Guide

**Choose `trace` when:**
- Analyzing trace structure
- Debugging dataset quality
- Converting to other formats
- Human-readable review

**Choose `tunix_sft` when:**
- Training with Tunix SFT
- Using Gemma models
- Need chat-template formatting
- Direct tokenization

**Choose `training_example` when:**
- Model-agnostic training
- Custom tokenization pipeline
- Abstract prompt/response pairs
- Non-Gemma models

---

## Validation

All formats must:
- Be valid JSONL (one JSON object per line)
- Include `id`, `metadata` fields
- Have non-empty prompts/responses
- Follow deterministic ordering (manifest order)

**Validation Commands:**
```bash
# Check JSONL validity
cat dataset.jsonl | jq empty

# Count samples
wc -l dataset.jsonl

# Verify first sample
head -1 dataset.jsonl | jq .

# Check for required fields
cat dataset.jsonl | jq 'select(.id == null or .prompts == null or .metadata == null)' | wc -l
# Should be 0
```

---

## See Also

- `docs/M09_TRAINING_QUICKSTART.md` - Using datasets for training
- `docs/M08_SUMMARY.md` - Dataset build API
- `training/README.md` - Training scripts overview
