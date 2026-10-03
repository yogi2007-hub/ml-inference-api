from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"

app = FastAPI(
    title="ML Inference API",
    description="Returns class probabilities from the trained model.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=1, description="Numeric features in the model's expected order"
    )


class PredictResponse(BaseModel):
    probabilities: list[float]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model file is not available")

    model = joblib.load(MODEL_PATH)

    try:
        probabilities = model.predict_proba([request.features])[0]
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(probabilities=[float(value) for value in probabilities])
