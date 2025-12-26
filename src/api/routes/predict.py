from fastapi import APIRouter, HTTPException
from src.api.schemas import PredictRequest, PredictResponse
from src.iris_classifier.model_io import load_model

router = APIRouter(tags=["prediction"])


@router.post("/predict")
def predict(req: PredictRequest):
    try:
        model = load_model(req.model_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")

    preds = model.predict(req.data)

    return {
        "model_name": req.model_name,
        "predictions": preds.tolist(),
    }

