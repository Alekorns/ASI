from pydantic import BaseModel
from typing import List, Dict


class PredictRequest(BaseModel):
    model_name: str
    data: List[List[float]]


class PredictResponse(BaseModel):
    model_name: str
    predictions: List[int]


class ContinueTrainRequest(BaseModel):
    model_name: str
    new_model_name: str
    X: List[List[float]]
    y: List[int]


class ContinueTrainResponse(BaseModel):
    new_model_name: str
    metrics: Dict[str, float]


class ModelsResponse(BaseModel):
    models: List[str]
