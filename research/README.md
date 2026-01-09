# Research Directory

This directory contains **exploratory research experiments** that are:
- NOT production code
- NOT part of the official submission (M42)
- Intended for differentiation and future development

## Contents

### M45: Curriculum Reasoning Training

**Location:** `m45_curriculum_reasoning/`

**Hypothesis:** Curriculum ordering of reasoning data produces qualitative improvements in reasoning trace structure compared to flat SFT — without changing the model or optimizer.

**Status:** ✅ Complete (2026-01-09)

**Key Finding:** Post-curriculum outputs exhibit more explicit step-by-step patterns, formula-first reasoning, and verification language.

### Artifacts

| Directory | Contents | Tracked in Git |
|-----------|----------|----------------|
| `data/` | Stage A/B/C dataset splits | ✅ Yes |
| `configs/` | Training YAML configs | ✅ Yes |
| `checkpoints/` | Model weights (~5GB each) | ❌ No (gitignored) |
| `eval/` | Prediction JSONL files | ✅ Yes |
| `*.py` | Training/eval scripts | ✅ Yes |
| `*.md` | Analysis and documentation | ✅ Yes |

## Reproduction

### Prerequisites

- GPU environment: `.venv-gpu` with PyTorch nightly cu128
- HuggingFace authentication: `huggingface-cli login`
- ~20GB free disk space for checkpoints

### Run M45 Curriculum Training

```powershell
# From project root
.\.venv-gpu\Scripts\Activate.ps1
python research/m45_curriculum_reasoning/run_curriculum.py
```

### Run M45 Evaluation

```powershell
python research/m45_curriculum_reasoning/eval_all_checkpoints.py
```

### View Results

- Training summary: `m45_curriculum_reasoning/checkpoints/curriculum_training_summary.json`
- Analysis: `m45_curriculum_reasoning/analysis.md`
- Provenance: `m45_curriculum_reasoning/provenance.json`

## Adding New Research

1. Create a new directory: `research/m{XX}_{name}/`
2. Include:
   - `provenance.json` for reproducibility
   - `analysis.md` for findings
   - Training/eval scripts
3. Add large binaries to `.gitignore`
4. Document in this README

## Relationship to Submission

**M42 is the authoritative submission artifact.**

Research experiments in this directory:
- Demonstrate additional capabilities
- Provide evidence for future directions
- Do NOT modify the submission package

