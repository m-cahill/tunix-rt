# M26 Clarifying Questions

1. **Dependency Management**: `backend/pyproject.toml` lists `jax`, `flax`, and `optax` in the `training` extras, but `orbax` (required for Phase 2) is missing. Should I add `orbax-checkpoint` to the `training` optional dependencies?

2. **Dataset Seeding**: Phase 4 mentions creating `tools/seed_dataset.py`. Existing we have `tools/seed_golden_dataset.py` which is specific to the golden dataset. Should I:
   - Create a new generic `seed_dataset.py` that can handle `golden-v2` and other future datasets?
   - Or refactor/rename `seed_golden_dataset.py` to be the general purpose seeder?

3. **Benchmarking Script**: Phase 3 mentions adding `training/bench_jax.py` or a `--benchmark` mode to `train_jax.py`. I propose creating a standalone `training/bench_jax.py` to keep the training loop clean and focused. Is this preferred?

4. **Metrics Granularity**: Phase 5 asks to persist metrics. `tunix_runs` has a `metrics` JSON column. For long runs, storing every step's metrics in a single JSON column might become large. Should I:
   - Store all step metrics in the `metrics` JSON column for now (assuming run length is manageable)?
   - Or only store summary/final metrics in the DB, and rely on the `metrics.jsonl` artifact (already being written by `train_jax.py`) for the detailed time-series data? The UI/API would then read from the artifact for the chart.

5. **Existing Files**: I assume `training/train_jax.py` refers to the file at `d:\Coding\tunix-rt\training\train_jax.py` (root level training folder), not anything inside `backend/tunix_rt_backend`. Correct?
