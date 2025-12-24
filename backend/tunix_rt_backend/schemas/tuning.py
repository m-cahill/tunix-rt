"""Tuning schemas (M19)."""

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator


# Search Space Validation
class SearchSpaceParamBase(BaseModel):
    type: str


class ChoiceParam(SearchSpaceParamBase):
    type: Literal["choice"]
    values: list[Any]


class UniformParam(SearchSpaceParamBase):
    type: Literal["uniform", "loguniform"]
    min: float
    max: float


class IntParam(SearchSpaceParamBase):
    type: Literal["randint"]
    min: int
    max: int


SearchSpaceParam = Annotated[
    Union[ChoiceParam, UniformParam, IntParam], Field(discriminator="type")
]


class TuningJobCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    dataset_key: str
    base_model_id: str
    metric_name: str = "score"  # Default to simple score
    metric_mode: Literal["max", "min"] = "max"
    num_samples: int = Field(1, ge=1, le=100)
    max_concurrent_trials: int = Field(1, ge=1, le=10)
    search_space: dict[str, SearchSpaceParam]

    @field_validator("search_space")
    @classmethod
    def validate_search_space(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("Search space cannot be empty")
        return v


class TuningTrialRead(BaseModel):
    id: str
    tuning_job_id: uuid.UUID
    run_id: uuid.UUID | None
    params_json: dict[str, Any]
    metric_value: float | None
    status: str
    error: str | None
    created_at: datetime
    completed_at: datetime | None


class TuningJobRead(BaseModel):
    id: uuid.UUID
    name: str
    status: str
    dataset_key: str
    base_model_id: str
    mode: str
    metric_name: str
    metric_mode: str
    num_samples: int
    max_concurrent_trials: int
    search_space_json: dict[str, Any]
    best_run_id: uuid.UUID | None
    best_params_json: dict[str, Any] | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    # Optional include (default None to avoid bloat in lists)
    trials: list[TuningTrialRead] | None = None


class TuningJobStartResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    message: str
