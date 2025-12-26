from fastapi import APIRouter
from src.api.schemas import ModelsResponse
from src.iris_classifier.model_io import list_models


router = APIRouter(tags=["models"])

@router.get("/models", response_model=ModelsResponse)
def get_models():
    return {"models": list_models()}

