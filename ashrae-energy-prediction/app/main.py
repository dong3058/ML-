from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

from app import ml
from app.schemas import (
    PredictRequest, PredictResponse, TrainRequest, TrainResponse, TuneRequest, TuneResponse,
)

app = FastAPI(title="ML API", version="0.1.0", description="LightGBM baseline + Optuna tuning")


@app.get("/health")
def health():
    return {"status": "ok", "service": "ashrae-energy-prediction"}


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest):
    try:
        X, y = ml.load_data(req.records, req.target)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return await run_in_threadpool(ml.train_baseline, X, y, req.task, req.test_size, req.params)


@app.post("/tune", response_model=TuneResponse)
async def tune(req: TuneRequest):
    try:
        X, y = ml.load_data(req.records, req.target)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return await run_in_threadpool(
        ml.tune, X, y, req.task, req.test_size, req.n_trials, req.timeout, req.params
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        preds = ml.predict(req.model_id, req.records)
    except FileNotFoundError:
        raise HTTPException(404, f"model '{req.model_id}' not found")
    except KeyError as e:
        raise HTTPException(400, f"missing feature columns: {e}")
    return PredictResponse(model_id=req.model_id, predictions=preds)


@app.get("/models/{model_id}")
def model_meta(model_id: str):
    """모델 메타데이터 (피처 순서, task, ONNX 입력 이름 등)."""
    try:
        return ml.model_meta(model_id)
    except FileNotFoundError:
        raise HTTPException(404, f"model '{model_id}' not found")


@app.get("/models/{model_id}/onnx")
def model_onnx(model_id: str):
    """ONNX 모델 파일 다운로드 (Spring 로컬 추론용)."""
    try:
        path = ml.onnx_path(model_id)
    except FileNotFoundError:
        raise HTTPException(404, f"onnx model '{model_id}' not found")
    return FileResponse(path, media_type="application/octet-stream", filename=f"{model_id}.onnx")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8081, reload=True)
