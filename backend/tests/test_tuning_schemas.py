"""Unit tests for tuning schemas."""

import pytest
from pydantic import TypeAdapter, ValidationError

from tunix_rt_backend.schemas.tuning import SearchSpaceParam, TuningJobCreate


def test_tuning_job_create_valid() -> None:
    data = {
        "name": "My Experiment",
        "dataset_key": "ds-v1",
        "base_model_id": "model-v1",
        "metric_name": "score",
        "metric_mode": "max",
        "num_samples": 10,
        "max_concurrent_trials": 2,
        "search_space": {
            "lr": {"type": "loguniform", "min": 1e-5, "max": 1e-3},
            "batch_size": {"type": "choice", "values": [4, 8, 16]},
        },
    }
    model = TuningJobCreate(**data)
    assert model.name == "My Experiment"
    assert model.search_space["lr"].type == "loguniform"


def test_tuning_job_create_invalid_search_space() -> None:
    # Empty search space
    data = {
        "name": "My Experiment",
        "dataset_key": "ds-v1",
        "base_model_id": "model-v1",
        "search_space": {},
    }
    with pytest.raises(ValidationError):
        TuningJobCreate(**data)

    # Invalid param type
    data["search_space"] = {"lr": {"type": "unknown"}}  # type: ignore
    with pytest.raises(ValidationError):
        TuningJobCreate(**data)


def test_search_space_param_validation() -> None:
    # Choice requires values
    with pytest.raises(ValidationError):
        TypeAdapter(SearchSpaceParam).validate_python({"type": "choice"})  # Missing values

    # Uniform requires min/max
    with pytest.raises(ValidationError):
        TypeAdapter(SearchSpaceParam).validate_python(
            {"type": "uniform", "min": 0.1}
        )  # Missing max
