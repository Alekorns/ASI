from fastapi import APIRouter
from iris_classifier.model_io import list_models

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[str])
def get_models():
    return list_models()
