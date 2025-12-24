# Tunix Training

This directory contains the training infrastructure for Tunix RT.

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[training]"

# 2. Run smoke test
python training/sft_smoke.py
```

## Structure

- `train_sft_tunix.py`: Main SFT runner script (integrates with Tunix library)
- `eval_generate.py`: Generation script for evaluation
- `eval_report.py`: Comparison report generator
- `configs/`: YAML configuration files
- `evalsets/`: Static evaluation datasets

## Test Fixtures

- `backend/datasets/test-v1`: **Tracked test fixture** required for E2E tests. All other datasets are gitignored.
