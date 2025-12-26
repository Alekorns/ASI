from fastapi import APIRouter, HTTPException
from src.api.schemas import ContinueTrainRequest, ContinueTrainResponse
from src.iris_classifier.model_io import load_model, save_model


router = APIRouter(tags=["training"])


@router.post("/continue-train")
def continue_train(req: ContinueTrainRequest):
    try:
        model = load_model(req.model_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")

    metrics = model.train(req.X, req.y)
    save_model(model, req.new_model_name)

    return {
        "new_model_name": req.new_model_name,
        "metrics": metrics,
    }
