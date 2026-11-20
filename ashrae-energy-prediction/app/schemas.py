from typing import Any, Literal

from pydantic import BaseModel, Field

TaskType = Literal["binary", "multiclass", "regression"]


class TrainRequest(BaseModel):
    """학습 요청. records가 비어 있으면 sklearn 예제 데이터(breast_cancer)로 학습한다."""

    records: list[dict[str, Any]] | None = Field(default=None, description="학습 데이터 (행 단위 dict)")
    target: str = Field(default="target", description="타깃 컬럼명")
    task: TaskType = "binary"
    test_size: float = Field(default=0.2, gt=0, lt=1)
    params: dict[str, Any] | None = Field(default=None, description="LightGBM 파라미터 덮어쓰기")


class TuneRequest(TrainRequest):
    n_trials: int = Field(default=30, ge=1, le=1000)
    timeout: int | None = Field(default=None, description="튜닝 제한 시간(초)")


class TrainResponse(BaseModel):
    model_id: str
    mlflow_run_id: str
    task: TaskType
    metric: str
    score: float
    params: dict[str, Any]
    features: list[str]


class TuneResponse(TrainResponse):
    n_trials: int
    best_trial: int


class PredictRequest(BaseModel):
    model_id: str
    records: list[dict[str, Any]]


class PredictResponse(BaseModel):
    model_id: str
    predictions: list[Any]
